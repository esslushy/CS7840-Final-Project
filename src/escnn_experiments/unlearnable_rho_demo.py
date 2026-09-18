"""
What survives when rho(g) cannot be learned at all?

equivariance_metrics_comparison.py found a clean split: metrics that hardcode
rho(g) fail, metrics that FIT rho(g) (Lenc & Vedaldi, Procrustes) succeed, and
metrics that assume no rho(g) (CKA and friends) succeed. But family B only
succeeded because rho(g) was a linear map -- a 96x96 permutation -- so a
least-squares fit recovered it exactly.

This script removes that crutch. The underlying model is unchanged and still
EXACTLY equivariant (escnn C8; residual under the true permutation ~4e-07). We
only change the coordinate system it is *observed* in, by composing a fixed
nonlinear readout sigma onto the pooled features:

    observed clean    Z  = sigma(v(x))
    observed rotated  Zr = sigma(P v(x))

sigma is part of the readout, applied identically to both, so the model's
equivariance is untouched -- but rho(g) expressed in these coordinates is
sigma o P o sigma^-1, which is nonlinear. No linear map exists to fit.

Two choices of sigma, chosen to bracket the outcome:

  TWIST     u -> exp(theta(||u||) A) u, A skew-symmetric, theta = omega*||u||.
            Orthogonal for every sample, so norms are preserved to ~1e-15: the
            map is BIJECTIVE and information is exactly preserved. Equivariance
            is provably fully intact. It is a "swiss roll" in disguise --
            locally near-isometric, globally scrambled.

  COSINE    u -> cos(omega * W u), W a random rotation. Oscillatory, so linear
            correlation decays with omega while deterministic dependence
            remains. Not bijective (cos is not injective), so unlike TWIST this
            one does lose some information at large omega.

EVERY metric is reported alongside a NEGATIVE CONTROL: the same computation with
Zr's rows shuffled, which destroys the pairing and must score ~0 for any honest
metric. This matters more than it sounds -- see the bandwidth trap below.

FINDINGS (measured).

1. THE BANDWIDTH TRAP. Under TWIST, RBF CKA appears to score a perfect 1.0000
   at small bandwidth (0.05x the median heuristic). It is entirely spurious: the
   shuffled control ALSO scores 1.0000. As bandwidth shrinks, both kernel
   matrices approach the identity and CKA(I, I) = 1 regardless of the data. Any
   bandwidth at or below ~0.2x median is vacuous here. Reporting RBF CKA without
   a negative control at that bandwidth would have produced a completely wrong
   conclusion -- worth knowing, since utils.py's EquivarianceTracker picks its
   bandwidth by the median heuristic and never checks this.

2. TWIST DEFEATS EVERYTHING. At omega=2, no metric detects the equivariance that
   is provably, exactly present: linear fit 1.01, linear CKA 0.008, RBF CKA
   0.058 at median bandwidth, and nothing usable at smaller bandwidths once the
   control is accounted for. This is a real limit of the whole enterprise, not
   of one metric family: a bijective, information-preserving change of
   coordinates can hide exact equivariance from every metric tested here.

3. COSINE IS THE REGIME THAT WAS WANTED. At omega=2 the linear fit is destroyed
   (residual 0.92 -- rho(g) genuinely unlearnable) while CKA still detects the
   relationship well above its control. So "cannot learn rho(g), but CKA still
   works" does exist.

4. RBF HELPS, BUT "ONLY RBF" OVERSTATES IT. RBF CKA does beat linear CKA, and
   the margin is meaningful once the bandwidth drops below the median heuristic
   -- but only in the narrow band between "too coarse to gain anything" (1.0x)
   and "vacuous" (<=0.2x). Best control-corrected margins found:

       TWIST  omega=0.5    linear 0.525   RBF @0.5x median 0.665   (+0.14)
       COSINE omega=2.0    linear 0.339   RBF @0.5x median 0.453   (+0.11)

   So sub-median RBF CKA is the most sensitive metric here, and the median
   heuristic that utils.py uses leaves that sensitivity on the table. But linear
   CKA still detects the relationship in every regime where RBF does, and the
   two die together at large omega. Across five constructions tried (saturating
   tanh, bijective sine warp, piecewise-rigid, twist, cosine) no honest regime
   was found where RBF CKA succeeds and linear CKA outright fails.

5. MUTUAL INFORMATION LARGELY DEFEATS THE TWIST, AND THAT IS PRINCIPLED. For
   bijective sigma, I(sigma(X); sigma(Y)) = I(X; Y) *exactly* -- MI is invariant
   to any invertible change of coordinates, whereas CKA is invariant only to
   orthogonal maps plus isotropic scaling. TWIST is bijective, so the true MI is
   constant in omega and cannot be fooled even in principle. Measured under
   TWIST (KSG, k=5, n=4000):

       omega      0.0     0.5     2.0     6.0    20.0
       KSG MI    4.54    3.86    1.83    1.68    1.64      (control ~ -0.01)
       linCKA    0.976   0.562   0.025   0.005   0.003
       rbfCKA    0.977   0.610   0.053   0.032   0.031

   The estimator is not exactly invariant -- it decays 4.54 -> 1.64 nats,
   because KSG depends on local max-norm kNN geometry that the twist distorts,
   so its finite-sample behaviour is coordinate-dependent even though its
   estimand is not. But it decays GRACEFULLY and then plateaus, holding a
   ~1.6-nat margin over a cleanly-zero control exactly where both CKA variants
   have collapsed to ~0.003 and become indistinguishable from noise.

   (That sweep is n=4000; the script itself runs n=SUB and reports smaller
   magnitudes, since KSG is sample-size dependent. The pattern is the same.)

   Caveat on magnitude: for a deterministic bijection between continuous
   variables the true MI is infinite, and KSG saturates around log(n/k), so
   every number above is a severe underestimate. Useful for detection and
   ranking, not as absolute information content.

6. THE SHARPEST RESULT: MI SEPARATES THE TWO CASES CKA CONFLATES. Compare what
   MI does to each readout at the point where CKA has given up:

       TWIST  omega=2 (bijective, lossless)  KSG margin 1.41  -> DETECTS
       COSINE omega=5 (non-injective, lossy) KSG margin 0.06  -> fails

   CKA fails on both, identically and uninformatively. MI succeeds on exactly
   the one where the information is still there and fails on exactly the one
   where it has been destroyed -- which is the correct answer in both cases, not
   merely a more robust one. A scrambled coordinate system and a lossy readout
   are fundamentally different situations: the first hides recoverable
   equivariance, the second destroys it. MI tells them apart; CKA cannot.

   This is a concrete argument for the framing in this repo's title. The
   reparameterization invariance of MI is the specific property that survives an
   adversarial change of readout coordinates, and CKA does not have it.

   The weak point of the argument above is that every MI number here is a KSG
   ESTIMATE, so "decays gracefully then plateaus" has to be defended as estimator
   artifact rather than shown. src/metric_demos/unlearnable_rho_exact_demo.py redoes this
   comparison (LEE vs linear CKA vs MI) on the exhaustive 2^16 binary space, where
   the joint pmf is exact and the same invariance comes out as a printed
   +0.00e+00 instead of a 4.54 -> 1.64 decay.

Run: cd src/escnn_experiments && .venv/bin/python3 unlearnable_rho_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
from scipy.special import digamma

DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learned_vs_architectural_equivariance import ArchitecturalCNN, train  # noqa: E402

SEED = 0
DIM = 96
SUB = 2000          # subsample for the O(n^2) kernel metrics
BANDWIDTHS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ features

def pooled_features():
    """Trained escnn C8 model -> spatially mean-pooled features on clean and
    90-degree-rotated inputs, plus the exact permutation rho(g) relating them."""
    torch.manual_seed(SEED)
    tr = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,
                                     transform=transforms.ToTensor())
    te = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True,
                                     transform=transforms.ToTensor())
    model = train(ArchitecturalCNN(),
                  torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=2))
    V, Vr = [], []
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(te, batch_size=256):
            x = x.to(DEVICE)
            V.append(model(x)[1].mean((-2, -1)).cpu().double())
            Vr.append(model(torch.rot90(x, 1, dims=(-2, -1)))[1].mean((-2, -1)).cpu().double())
    V, Vr = torch.cat(V), torch.cat(Vr)
    element = list(model.gspace.fibergroup.elements)[len(model.gspace.fibergroup.elements) // 4]
    P = torch.as_tensor(np.asarray(model.hidden_type.representation(element))).double()
    return V, Vr, P


# ------------------------------------------------------------------ readouts

def make_readouts(V):
    mu, sd = V.mean(0, keepdim=True), V.std(0, keepdim=True) + 1e-9
    torch.manual_seed(SEED)
    Q = torch.linalg.qr(torch.randn(DIM, DIM).double())[0]
    W = torch.linalg.qr(torch.randn(DIM, DIM).double())[0]

    def twist(Z, omega):
        """Per-sample orthogonal => norm-preserving => bijective => lossless."""
        u = (Z - mu) / sd
        r = u.norm(dim=1, keepdim=True)
        w = (u @ Q).view(len(u), DIM // 2, 2)
        th = (omega * r).squeeze(1)
        c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
        w = torch.stack([c * w[..., 0] - s * w[..., 1], s * w[..., 0] + c * w[..., 1]], -1)
        return w.view(len(u), DIM) @ Q.T

    def cosine(Z, omega):
        return torch.cos(omega * (((Z - mu) / sd) @ W.T))

    # omega=0 is the identity for TWIST (a useful control) but collapses COSINE
    # to the constant vector cos(0)=1, whose zero variance makes CKA undefined.
    return {"TWIST (bijective)": (twist, [0.0, 0.5, 2.0]),
            "COSINE (lossy)": (cosine, [0.5, 2.0, 5.0])}


# ------------------------------------------------------------------- metrics

def fit_linear(X, Y):
    """Lenc & Vedaldi-style affine fit, held-out residual. Lower is better."""
    n = len(X) // 2
    aug = lambda A: torch.cat([A, torch.ones(len(A), 1, dtype=A.dtype)], 1)
    E = torch.linalg.lstsq(aug(X[:n]), Y[:n]).solution
    return ((Y[n:] - aug(X[n:]) @ E).norm() / Y[n:].norm()).item()


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.T, Y @ Y.T
    h = lambda A, B: (A * B).sum()
    return (h(K, L) / torch.sqrt(h(K, K) * h(L, L))).item()


def rbf_cka(X, Y, mult=1.0):
    def gram(Z):
        D = torch.cdist(Z, Z) ** 2
        s = D[D > 0].sqrt().median() * mult
        return torch.exp(-D / (2 * s ** 2))
    A, B = gram(X), gram(Y)
    n = len(A)
    H = torch.eye(n, dtype=A.dtype) - 1.0 / n
    A, B = H @ A @ H, H @ B @ H
    h = lambda a, b: (a * b).sum()
    return (h(A, B) / torch.sqrt(h(A, A) * h(B, B))).item()


def ksg_mi(X, Y, k=5):
    """Kraskov-Stogbauer-Grassberger mutual information estimate, in nats.

    Included because MI has the invariance CKA lacks: for any bijective sigma,
    I(sigma(X); sigma(Y)) = I(X; Y) exactly. TWIST is bijective, so the true MI
    it reports is invariant to omega by construction -- which is precisely the
    property that should make an information-theoretic measure immune to this
    attack. Whether the ESTIMATOR inherits that invariance is a separate
    question, and the answer below is "only partly"."""
    X, Y = np.asarray(X), np.asarray(Y)
    n = len(X)
    joint = np.hstack([X, Y])
    eps = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, k]
    nx = np.array([len(a) for a in cKDTree(X).query_ball_point(X, eps - 1e-12, p=np.inf)]) - 1
    ny = np.array([len(a) for a in cKDTree(Y).query_ball_point(Y, eps - 1e-12, p=np.inf)]) - 1
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def main():
    V, Vr, P = pooled_features()
    print(f"ground truth: ||Vr - V P^T|| / ||Vr|| = "
          f"{((Vr - V @ P.T).norm() / Vr.norm()).item():.2e}  -> exactly equivariant\n")

    readouts = make_readouts(V)
    shuffle = torch.randperm(SUB, generator=torch.Generator().manual_seed(1))

    for name, (sigma, omegas) in readouts.items():
        print("=" * 78)
        print(f"readout sigma = {name}")
        print("=" * 78)
        for omega in omegas:
            Z, Zr = sigma(V, omega), sigma(Vr, omega)
            Zs, Zrs = Z[:SUB], Zr[:SUB]
            drift = (Z.norm(dim=1) - ((V - V.mean(0, keepdim=True))
                                       / (V.std(0, keepdim=True) + 1e-9)).norm(dim=1)).abs().max()
            print(f"\n  omega={omega}   (max norm drift {drift.item():.1e}"
                  f"{'  <- lossless' if drift < 1e-9 else '  <- lossy'})")
            print(f"    {'linear fit residual (lower better)':<38s} {fit_linear(Z, Zr):8.4f}")
            print(f"    {'metric':<26s} {'true':>8s} {'shuffled':>9s} {'margin':>8s}  verdict")
            lt, lf = linear_cka(Zs, Zrs), linear_cka(Zs, Zrs[shuffle])
            rows = [("linear CKA", lt, lf)]
            for mult in BANDWIDTHS:
                rows.append((f"RBF CKA @{mult}x median",
                             rbf_cka(Zs, Zrs, mult), rbf_cka(Zs, Zrs[shuffle], mult)))
            for label, t, f in rows:
                margin = t - f
                verdict = ("VACUOUS (control high)" if f > 0.5
                           else "detects" if margin > 0.3 else "fails")
                print(f"    {label:<26s} {t:8.4f} {f:9.4f} {margin:8.4f}  {verdict}")
            # MI is on a different scale (nats, unbounded) so it gets its own row
            mt, mf = ksg_mi(Zs, Zrs), ksg_mi(Zs, Zrs[shuffle])
            print(f"    {'KSG MI (nats)':<26s} {mt:8.4f} {mf:9.4f} {mt - mf:8.4f}  "
                  f"{'detects' if mt - mf > 0.3 else 'fails'}")
        print()


if __name__ == "__main__":
    main()
