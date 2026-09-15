"""
Twelve equivariance-error metrics from the literature, all scored on the same
head-to-head, against known ground truth.

learned_vs_architectural_equivariance.py showed the Lie derivative (LEE) ranks a
"learned equivariant" plain CNN above an exactly-equivariant escnn C8 network,
contradicting behaviour, and that CKA reverses the verdict. The obvious question
is whether that is something peculiar to those two metrics. It is not. The split
runs along one axis: DOES THE METRIC HARDCODE rho(g)?

Same two models as that script (imported, trained identically), same probe. The
measurement vector is the hidden conv feature map, spatially mean-pooled, giving
96 dims for BOTH models -- matched dimensionality, so no metric is advantaged by
having more features to work with. Probe is a 90 degree rotation: it lies in C8
AND is a lattice symmetry of a square grid, so it needs no interpolation and
introduces no aliasing. That makes the ground truth unambiguous.

Ground truth at 90 degrees: the architectural model is EXACTLY invariant, by
construction. Its pooled vector satisfies v(gx) = P(g) v(x) for a known 96x96
permutation P(g) (regular-representation fields permute their 8 group channels;
spatial mean-pooling is exactly rotation-invariant on a square grid at quarter
turns). Verified numerically in-script. Any metric that ranks the plain CNN as
more equivariant is simply wrong.

  FAMILY A -- hardcode rho(g). The paper's own convention: a tensor that is not
              4D is assumed to transform trivially (be invariant). True for
              neither model's pooled vector, and badly false for the
              architectural one, whose vector permutes.
    A1  LEE-style relative error       Gruver et al. 2023 (arXiv:2210.02984)
    A2  EQ-R, PSNR in dB               Karras et al. 2021 (arXiv:2106.12423)

  FAMILY B -- fit rho(g) from data instead of guessing it. Fitted on one half of
              the test set, residual reported on the held-out half, so a metric
              cannot win by overfitting.
    B1  learned linear/affine map      Lenc & Vedaldi 2015 (arXiv:1411.5908)
    B2  best orthogonal map            Procrustes-constrained variant of B1

  FAMILY C -- assume no rho(g) at all; compare representational geometry.
    C1  linear CKA                     Kornblith et al. 2019
    C2  RBF CKA                        (via this repo's own EquivarianceTracker)
    C3  SVCCA                          Raghu et al. 2017
    C4  distance correlation           Szekely et al. 2007
    C5  RSA, Spearman of RDMs          Kriegeskorte et al. 2008

  FAMILY D -- behaviour. The ground truth the others are proxies for.
    D1  prediction consistency         Zhang 2019 (arXiv:1904.11486)
    D2  accuracy drop under rotation

FINDING (measured; see script output). The split is total and falls exactly on
the rho(g) axis:

    family A  hardcode rho(g)     0/2 agree with ground truth
    family B  fit rho(g)          2/2
    family C  no rho(g) at all    5/5

Both family-A metrics rank the plain CNN as the more equivariant model. They are
wrong: at this probe the architectural model's predictions are 100% consistent
under rotation (D1 = 1.00000, accuracy drop exactly 0.00000) and its residual
under the correct rho(g) is 4e-07. Every metric in families B and C gets it
right, most of them returning exactly 1.00000 / 0.00000 for the architectural
model.

Two details worth noting. First, the fitted maps in family B RECOVER the
permutation from data on held-out samples -- residual 0.00000 for the
architectural model, yet a non-degenerate 0.08 for the plain CNN, so this is
genuine recovery and not a metric that fits anything. Second, A2 is a monotone
function of A1's MSE, so it cannot disagree with A1; two metrics from different
literatures inherit one shared flaw, which is the point.

Also note the plain CNN is the BETTER CLASSIFIER here (higher clean accuracy)
and still the less equivariant one, so the family-A verdict cannot be explained
away as tracking model quality. And while its aggregate accuracy barely moves
under rotation (-0.006), 12.5% of its individual predictions flip -- aggregate
accuracy hides per-sample instability that D1 consistency exposes.

Run: cd src/escnn_experiments && python equivariance_metrics_comparison.py   (see README.md for the venv)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from escnn import nn as enn
from scipy.stats import spearmanr

# Shared with the rest of the repo: datasets live in src/data, one level up.
DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")
# ...as does utils.py, whose EquivarianceTracker is reused here for RBF CKA.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learned_vs_architectural_equivariance import (ArchitecturalCNN, LearnedCNN, train)
from utils import EquivarianceTracker


SEED = 0
BATCH = 256
SUBSAMPLE_ON = 1500   # for the O(n^2) metrics (C4, C5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUARTER_TURNS = 1     # 90 degrees: in C8 and a lattice symmetry -> alias-free


# ---------------------------------------------------------------- family A

def a1_lee_relative(X, Y):
    """||f(gx) - rho(g)f(x)|| / ||rho(g)f(x)|| with rho(g)=I (the img_like
    fallback for non-4D tensors). Lower is better."""
    return ((Y - X).norm() / X.norm()).item()


def a2_eq_psnr_db(X, Y):
    """Karras et al.'s EQ metric: PSNR in dB between the two, rho(g)=I.
    Higher is better. A monotone function of MSE, so it must agree with A1."""
    peak = (X.max() - X.min()).item()
    mse = (Y - X).pow(2).mean().item()
    return 10.0 * np.log10(peak ** 2 / max(mse, 1e-20))


# ---------------------------------------------------------------- family B

def _split(X, Y):
    n = X.shape[0] // 2
    return X[:n], Y[:n], X[n:], Y[n:]


def b1_learned_linear_map(X, Y):
    """Lenc & Vedaldi: fit an affine map E with E.phi(x) ~ phi(gx), report the
    held-out normalized residual. Lower is better."""
    Xf, Yf, Xe, Ye = _split(X, Y)
    aug = lambda A: torch.cat([A, torch.ones(A.shape[0], 1, dtype=A.dtype)], 1)
    E = torch.linalg.lstsq(aug(Xf), Yf).solution
    return ((Ye - aug(Xe) @ E).norm() / Ye.norm()).item()


def b2_procrustes_map(X, Y):
    """Same idea, but rho(g) constrained to be orthogonal (as a true group
    representation of a rotation is). Lower is better."""
    Xf, Yf, Xe, Ye = _split(X, Y)
    mx, my = Xf.mean(0, keepdim=True), Yf.mean(0, keepdim=True)
    U, _, Vh = torch.linalg.svd((Xf - mx).T @ (Yf - my))
    R = U @ Vh
    return (((Ye - my) - (Xe - mx) @ R).norm() / (Ye - my).norm()).item()


# ---------------------------------------------------------------- family C

def c1_linear_cka(X, Y):
    Xc, Yc = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    K, L = Xc @ Xc.T, Yc @ Yc.T
    hsic = lambda A, B: (A * B).sum()
    return (hsic(K, L) / torch.sqrt(hsic(K, K) * hsic(L, L))).item()


def c2_rbf_cka(X, Y):
    """Reuses this repo's own EquivarianceTracker (unbiased HSIC + median-
    heuristic bandwidth) rather than reimplementing it."""
    tracker = EquivarianceTracker("cpu")
    for i in range(0, X.shape[0] - 255, 256):     # tracker expects batches
        tracker.update(X[i:i + 256], Y[i:i + 256])
    return tracker.compute_stats()["rbf_cka"]


def c3_svcca(X, Y, k=20):
    """Mean of the top-k canonical correlations. Higher is better."""
    Xc, Yc = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    qx, _ = torch.linalg.qr(Xc)
    qy, _ = torch.linalg.qr(Yc)
    s = torch.linalg.svdvals(qx.T @ qy)
    return s[:k].clamp(0, 1).mean().item()


def _double_center(D):
    return D - D.mean(0, keepdim=True) - D.mean(1, keepdim=True) + D.mean()


def c4_distance_correlation(X, Y):
    A = _double_center(torch.cdist(X, X))
    B = _double_center(torch.cdist(Y, Y))
    dcov = (A * B).mean()
    dvx, dvy = (A * A).mean(), (B * B).mean()
    return (dcov / torch.sqrt(dvx * dvy).clamp_min(1e-20)).clamp_min(0).sqrt().item() ** 2


def c5_rsa(X, Y):
    """Spearman correlation between the two representational dissimilarity
    matrices. Higher is better."""
    n = X.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    dx = torch.cdist(X, X)[iu[0], iu[1]].numpy()
    dy = torch.cdist(Y, Y)[iu[0], iu[1]].numpy()
    return float(spearmanr(dx, dy).statistic)


# ---------------------------------------------------------------- collection

def collect(model, loader, architectural):
    """Pooled 96-dim features on clean and 90-degree-rotated inputs, plus
    behaviour. Also returns the residual under the CORRECT rho(g) when known."""
    Xs, Ys, logits_c, logits_r, labels = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            x_rot = torch.rot90(x, QUARTER_TURNS, dims=(-2, -1))
            lc, hc, _ = model(x)
            lr, hr, _ = model(x_rot)
            Xs.append(hc.mean((-2, -1)).cpu())      # spatial mean pool -> (B, 96)
            Ys.append(hr.mean((-2, -1)).cpu())
            logits_c.append(lc.cpu()); logits_r.append(lr.cpu()); labels.append(y)

    X, Y = torch.cat(Xs).double(), torch.cat(Ys).double()
    lc, lr, lab = torch.cat(logits_c), torch.cat(logits_r), torch.cat(labels)

    correct_residual = float("nan")
    if architectural:
        element = list(model.gspace.fibergroup.elements)[
            QUARTER_TURNS * (len(model.gspace.fibergroup.elements) // 4)]
        P = torch.as_tensor(np.asarray(model.hidden_type.representation(element))).double()
        correct_residual = ((Y - X @ P.T).norm() / Y.norm()).item()

    behaviour = {
        "consistency": (lc.argmax(-1) == lr.argmax(-1)).double().mean().item(),
        "acc": (lc.argmax(-1) == lab).double().mean().item(),
        "acc_rot": (lr.argmax(-1) == lab).double().mean().item(),
    }
    return X, Y, behaviour, correct_residual


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tr = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,
                                     transform=transforms.ToTensor())
    te = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True,
                                     transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(te, batch_size=BATCH, shuffle=False, num_workers=2)

    data = {}
    for name, ctor in [("learned", LearnedCNN), ("architectural", ArchitecturalCNN)]:
        model = train(ctor(), trainloader)
        arch = isinstance(model, ArchitecturalCNN)
        X, Y, behaviour, correct_res = collect(model, testloader, arch)
        data[name] = dict(X=X, Y=Y, behaviour=behaviour, correct_res=correct_res)
        print(f"trained {name:14s} dims={X.shape[1]}  n={X.shape[0]}  "
              f"acc={behaviour['acc']:.4f}  acc_rot={behaviour['acc_rot']:.4f}")

    cr = data["architectural"]["correct_res"]
    print(f"\nsanity: architectural residual under the CORRECT rho(g) "
          f"(known 96x96 permutation) = {cr:.2e}  -> exactly equivariant, as designed")

    # name, fn, higher_is_better, family
    metrics = [
        ("A1  LEE relative error   (Gruver+23)", lambda X, Y: a1_lee_relative(X, Y), False, "A"),
        ("A2  EQ-R PSNR dB         (Karras+21)", lambda X, Y: a2_eq_psnr_db(X, Y), True, "A"),
        ("B1  learned linear map   (Lenc+15) ", lambda X, Y: b1_learned_linear_map(X, Y), False, "B"),
        ("B2  Procrustes orthogonal map      ", lambda X, Y: b2_procrustes_map(X, Y), False, "B"),
        ("C1  linear CKA           (Kornblith)", c1_linear_cka, True, "C"),
        ("C2  RBF CKA              (repo impl)", c2_rbf_cka, True, "C"),
        ("C3  SVCCA                (Raghu+17)", c3_svcca, True, "C"),
        ("C4  distance correlation (Szekely+07)", lambda X, Y: c4_distance_correlation(X[:SUBSAMPLE_ON], Y[:SUBSAMPLE_ON]), True, "C"),
        ("C5  RSA Spearman         (Kriegeskorte)", lambda X, Y: c5_rsa(X[:SUBSAMPLE_ON], Y[:SUBSAMPLE_ON]), True, "C"),
    ]

    print("\n" + "=" * 92)
    print(f"{'metric':>40s} {'learned':>12s} {'architectural':>14s} {'verdict':>12s}  ground truth?")
    print("=" * 92)
    tally = {"A": [], "B": [], "C": []}
    for label, fn, higher_better, family in metrics:
        vl = fn(data["learned"]["X"], data["learned"]["Y"])
        va = fn(data["architectural"]["X"], data["architectural"]["Y"])
        arch_wins = (va > vl) if higher_better else (va < vl)
        tally[family].append(arch_wins)
        print(f"{label:>40s} {vl:12.5f} {va:14.5f} "
              f"{'architectural' if arch_wins else 'learned':>12s}  "
              f"{'correct' if arch_wins else 'WRONG'}")

    print("-" * 92)
    bl, ba = data["learned"]["behaviour"], data["architectural"]["behaviour"]
    for label, key, higher_better in [("D1  prediction consistency (Zhang19)", "consistency", True)]:
        arch_wins = (ba[key] > bl[key]) if higher_better else (ba[key] < bl[key])
        print(f"{label:>40s} {bl[key]:12.5f} {ba[key]:14.5f} "
              f"{'architectural' if arch_wins else 'learned':>12s}  ground truth")
    dl, da = bl["acc"] - bl["acc_rot"], ba["acc"] - ba["acc_rot"]
    print(f"{'D2  accuracy drop under rotation':>40s} {dl:12.5f} {da:14.5f} "
          f"{'architectural' if da < dl else 'learned':>12s}  ground truth")
    print("=" * 92)

    for family, name in [("A", "hardcode rho(g)"), ("B", "fit rho(g)"),
                         ("C", "no rho(g) at all")]:
        n_ok = sum(tally[family])
        print(f"  family {family} ({name:17s}): {n_ok}/{len(tally[family])} agree with ground truth")


if __name__ == "__main__":
    main()
