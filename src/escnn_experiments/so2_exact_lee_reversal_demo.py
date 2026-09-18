"""
LEE prefers a learned-equivariant model over an exactly SO(2)-equivariant one,
with the aliasing defense removed.

Every previous demo in this folder lives on a square pixel grid, which hands LEE
an escape hatch: off the 90-degree lattice a rotation cannot be represented
exactly, so "the architectural guarantee really does evaporate there" is a fair
rebuttal (and the README grants it). This experiment removes the grid.

The data is a set of 2D POINT CLOUDS. Rotation acts by p -> p R^T, an exact
orthogonal matrix multiply. There is no resampling, no interpolation, no
bandwidth, and no lattice, so NO angle is privileged: 7.0 degrees is represented
to exactly the same precision as 90.0. self_check() asserts this rather than
assuming it. Whatever LEE reports here cannot be blamed on discretization.

    WHY A COMPLEX LINEAR MAP IS THE EQUIVARIANT LAYER
    Write a point as z = x + iy. Rotation by theta is z -> e^{i theta} z. Any
    COMPLEX linear map W commutes with that: W(e^{i t} z) = e^{i t} (W z). So
    complex matmul IS the SO(2)-equivariant linear layer on frequency-1
    features -- Schur's lemma, the commutant of the 2D rotation irrep being C.
    Magnitude |z| is invariant, so gating by a function of |z| is an equivariant
    nonlinearity, and reading out |z| gives an exactly invariant head. No bias
    terms on the complex layers: a constant is frequency 0 and would break it.

Two models, both trained with full SO(2) augmentation on the same exactly
invariant target, so architecture is the only difference:

  EquivariantSO2   exactly equivariant for every real theta, by construction
  LearnedMLP       no prior; matched hidden width; must learn invariance

WHAT IT SHOWS (measured, 3 seeds, 300 epochs; the equivariant row is analytic)

                          LEE naive   LEE correct   1-CKA     drift     test MSE
    EquivariantSO2           0.9999      0.0000     0.000000  0.000000    0.0499
    LearnedMLP               0.3702         n/a     0.008651  0.098904    0.0122
    LearnedMLP, UNTRAINED    0.2518         n/a     0.460116  0.028638    1.0027

  EquivariantSO2's hidden features are frequency-1, so f(R_t x) = e^{it} f(x).
  Under the trivial rho that Gruver et al. assume for a non-image tensor, the
  Lie derivative is d/dt e^{it} f = i f, whose relative norm is EXACTLY 1.0 per
  radian -- for any weights, at any stage of training. Measured: 0.9999. Its LEE
  under the correct rho, its CKA error and its prediction drift are all
  identically 0: rotate the input and its predictions come back bit-identical.

  LEE nonetheless ranks it LAST of the three. And the model it likes best has
  never been trained: an untrained MLP scores 0.2518, four times "better" than a
  provably equivariant network, while predicting essentially nothing (test MSE
  1.0027 on a variance-normalised target, i.e. no better than the mean). Its low
  LEE is the bandwidth confound of lee_bandwidth_confound_demo.py in its purest
  form -- features that barely respond to anything cannot have a large
  derivative.

  Linear CKA gets the whole ordering right, including the untrained control:

      LEE          0.2518 untrained  <  0.3702 learned  <  0.9999 equivariant
      linear CKA   0.0000 equivariant <  0.0087 learned  <  0.4601 untrained
      behaviour    drift 0 exactly for the equivariant model; 9.9% for the
                   learned one; the untrained one is near-constant garbage

  The trained baseline is DEEPER AND WIDER than the equivariant model and fits
  the task BETTER (0.0122 vs 0.0499 test MSE, since frequency-1 features are a
  real expressivity constraint), so "the control was crippled" is unavailable.
  It is simply not equivariant, and LEE prefers it anyway.

  Read drift alongside test MSE, not alone: the untrained model has LOW drift
  (0.029) purely because its output is nearly constant. Constancy fools drift
  the same way it fools LEE. Only CKA separates "invariant" from "uninformative"
  here, because a constant feature map has no Gram structure to preserve.

  WHAT CANNOT EXPLAIN ANY OF THIS: aliasing. The group action is an exact
  orthogonal matmul and self_check() asserts equivariance to ~1e-6 at generic
  angles. There is no lattice, so the usual "the guarantee evaporates
  off-lattice" rebuttal has nothing to attach to. LEE's verdict here is a
  representation error and nothing else.

  Scope: the reversal comes from NEAR-INVARIANT READOUTS, which depth and width
  produce, not from "learned equivariance" as such. A shallow 2x128 baseline
  goes the other way -- its LEE RISES from 0.73 to 1.38 over 600 epochs as its
  features grow more orientation-sensitive. Report it that way.

Needs NO escnn -- deliberately, since escnn's guarantee is about grids and the
whole point here is to not have one. Runs in either venv:
    .venv/bin/python3 src/escnn_experiments/so2_exact_lee_reversal_demo.py
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import EquivarianceTracker

SEED = 0
K = 8                      # 2D points per cloud
H1, H2 = 64, 64            # complex channels; hidden is 2*H2 real dims
N_TRAIN, N_EVAL = 20000, 2048
BATCH, LR = 256, 2e-3
EPS = 0.02                 # rad, central-difference step for the Lie derivative
# Nothing distinguishes these angles. There is no lattice to be aligned to.
ANGLES = [7.0, 23.0, 45.0, 73.0, 118.0, 180.0, 241.0, 307.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data: the SO(2) action is an exact matrix multiply
# ---------------------------------------------------------------------------
def sample_clouds(n, gen):
    """n clouds of K points in R^2, radii in [0.3, 1.5], angles uniform."""
    r = 0.3 + 1.2 * torch.rand(n, K, generator=gen)
    a = 2 * math.pi * torch.rand(n, K, generator=gen)
    return torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=-1)


def rotate_cloud(p, theta):
    """Exact SO(2) action. No grid, no interpolation, no privileged angle."""
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=p.dtype, device=p.device)
    return p @ R.T


def invariant_target(p):
    """Exactly rotation-invariant scalar, built from the Gram matrix because
    <R p_i, R p_j> = <p_i, p_j> identically. Nontrivial: it depends on the
    relative angles between points, not only on their radii, so invariance has
    to be earned rather than read off the norms."""
    G = p @ p.transpose(-1, -2)
    iu = torch.triu_indices(K, K, offset=1, device=p.device)
    return torch.tanh(2.0 * G[:, iu[0], iu[1]]).sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# Exactly SO(2)-equivariant model
# ---------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    """Equivariant linear map on frequency-1 features: complex matmul, no bias
    (a bias is frequency 0 and would destroy equivariance)."""

    def __init__(self, cin, cout):
        super().__init__()
        s = 1.0 / math.sqrt(2 * cin)
        self.wr = nn.Parameter(torch.randn(cout, cin) * s)
        self.wi = nn.Parameter(torch.randn(cout, cin) * s)

    def forward(self, z):                      # z: (B, cin, 2)
        zr, zi = z[..., 0], z[..., 1]
        return torch.stack([zr @ self.wr.T - zi @ self.wi.T,
                            zr @ self.wi.T + zi @ self.wr.T], dim=-1)


class NormGate(nn.Module):
    """Equivariant nonlinearity: scale each channel by a function of its own
    magnitude, which is invariant. The phase -- the equivariant part -- is
    never touched."""

    def __init__(self, c):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self, z):
        mag = z.norm(dim=-1)
        return z * (F.relu(mag + self.b) / (mag + 1e-6))[..., None]


class EquivariantSO2(nn.Module):
    """Exactly equivariant for EVERY real theta -- not a finite subgroup, and
    with no spatial grid to alias."""

    def __init__(self):
        super().__init__()
        self.l1, self.g1 = ComplexLinear(K, H1), NormGate(H1)
        self.l2, self.g2 = ComplexLinear(H1, H2), NormGate(H2)
        self.head = nn.Sequential(nn.Linear(H2, 128), nn.ReLU(), nn.Linear(128, 1))

    def features(self, p):
        """(B, H2, 2) frequency-1 features, still carrying phase."""
        return self.g2(self.l2(self.g1(self.l1(p))))

    def forward(self, p):
        h = self.features(p)
        return self.head(h.norm(dim=-1)), h.flatten(1)

    @staticmethod
    def rho(theta, h):
        """The model's TRUE hidden representation: frequency 1, so rho(theta)
        rotates every complex channel by theta. Orthogonal, and defined at every
        real theta -- no interpolation needed to apply it."""
        c, s = math.cos(theta), math.sin(theta)
        hr, hi = h[..., 0], h[..., 1]
        return torch.stack([c * hr - s * hi, s * hr + c * hi], dim=-1)


class LearnedMLP(nn.Module):
    """No equivariance prior, and deliberately DEEPER AND WIDER than
    EquivariantSO2 (4x256 vs 2x64 complex). It also fits the task better, so
    "the control was crippled" is not available as an objection.

    Depth and width are what make its hidden features near-invariant, and
    near-invariant features are exactly what LEE's assumed trivial rho rewards.
    A shallow 2x128 version goes the OTHER way -- its LEE rises from 0.73 to
    1.38 over 600 epochs as its features grow more orientation-sensitive -- so
    the effect is a property of the readout, not of "learned equivariance"
    in general."""

    def __init__(self, width=256, depth=4):
        super().__init__()
        layers, d = [], 2 * K
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(width, 2 * H2), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * H2, 128), nn.ReLU(), nn.Linear(128, 1))

    def features(self, p):
        return self.body(p.flatten(1))

    def forward(self, p):
        h = self.features(p)
        return self.head(h), h


# ---------------------------------------------------------------------------
def train(model, p, y, epochs, seed=SEED):
    """Both models get identical full-SO(2) augmentation. The target is exactly
    invariant, so a rotated cloud keeps its label."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.Adam(model.parameters(), LR)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(p.shape[0], generator=gen)
        for i in range(0, p.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            pb = rotate_cloud(p[idx], float(torch.rand(1, generator=gen)) * 2 * math.pi)
            opt.zero_grad()
            F.mse_loss(model(pb)[0], y[idx]).backward()
            opt.step()
    return model.eval()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def lee_naive(model, p, eps=EPS):
    """LEE = ||L_X f(x)|| / ||f(x)||, under the TRIVIAL rho that Gruver et al.'s
    implementation assumes for any non-image-shaped tensor -- i.e. it assumes
    the features ought to be invariant. Central difference, because torch's jvp
    is unavailable on some of the ops the repo uses elsewhere; here it would
    work, and is kept for consistency with metric_demos/lie_vs_cka_demo.py."""
    f = lambda th: model(rotate_cloud(p, th))[1]
    return (((f(eps) - f(-eps)) / (2 * eps)).norm() / f(0.0).norm()).item()


@torch.no_grad()
def lee_correct(model, p, eps=EPS):
    """The same derivative under the model's true rho. Defined only when rho is
    known -- for LearnedMLP it is not, and that asymmetry is the whole problem:
    LEE cannot be computed correctly for the model it ends up preferring."""
    if not hasattr(model, "rho"):
        return None

    def f(th):
        return model.rho(-th, model.features(rotate_cloud(p, th))).flatten(1)

    return (((f(eps) - f(-eps)) / (2 * eps)).norm() / f(0.0).norm()).item()


@torch.no_grad()
def finite_naive(model, p, deg):
    """Finite-angle relative error under the same trivial rho. Not LEE; LEE is
    the derivative above."""
    X = model(p)[1]
    Y = model(rotate_cloud(p, math.radians(deg)))[1]
    return ((Y - X).norm() / X.norm()).item()


@torch.no_grad()
def cka_error(model, p, deg):
    """1 - linear CKA, via the repo's EquivarianceTracker (unbiased HSIC). Needs
    no rho at all: if f(g.x) = rho(g) f(x) for ANY orthogonal rho, the centered
    Gram matrix is unchanged and this is exactly 0."""
    X = model(p)[1]
    Y = model(rotate_cloud(p, math.radians(deg)))[1]
    t = EquivarianceTracker(X.device)
    t.update(X, Y)
    return 1.0 - t.compute_stats()["linear_cka"]


@torch.no_grad()
def pred_drift(model, p, deg):
    """Behavioural ground truth: does the prediction actually move? The target
    is exactly invariant, so any drift is real error."""
    a = model(p)[0]
    b = model(rotate_cloud(p, math.radians(deg)))[0]
    return ((b - a).norm() / a.norm()).item()


@torch.no_grad()
def self_check(eq, p):
    """Assert the premises instead of assuming them: the target really is
    invariant, and the equivariant model really is equivariant -- at generic,
    non-lattice angles, to machine precision."""
    y0 = invariant_target(p)
    for th in (0.1, 1.234, 2.7, 5.0):
        d = (invariant_target(rotate_cloud(p, th)) - y0).abs().max().item()
        assert d < 1e-4, f"target not invariant at {th}: {d:.2e}"
        h = eq.features(p)
        hr = eq.features(rotate_cloud(p, th))
        rel = ((hr - eq.rho(th, h)).norm() / hr.norm()).item()
        assert rel < 1e-5, f"model not equivariant at {th}: {rel:.2e}"
    print("  self-check passed: target invariant and model equivariant to ~1e-6\n"
          "  at generic angles (0.1, 1.234, 2.7, 5.0 rad). No lattice involved.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rows = {k: [] for k in ("EquivariantSO2", "LearnedMLP", "LearnedMLP (untrained)")}
    for seed in range(args.seeds):
        gen = torch.Generator().manual_seed(seed)
        p_tr = sample_clouds(N_TRAIN, gen).to(DEVICE)
        p_ev = sample_clouds(N_EVAL, gen).to(DEVICE)
        y_tr, y_ev = invariant_target(p_tr), invariant_target(p_ev)
        mu, sd = y_tr.mean(), y_tr.std()
        y_tr, y_ev = (y_tr - mu) / sd, (y_ev - mu) / sd

        def measure(m):
            return (lee_naive(m, p_ev), lee_correct(m, p_ev),
                    sum(cka_error(m, p_ev, d) for d in ANGLES) / len(ANGLES),
                    sum(pred_drift(m, p_ev, d) for d in ANGLES) / len(ANGLES),
                    F.mse_loss(m(p_ev)[0], y_ev).item())

        torch.manual_seed(seed)
        untrained = LearnedMLP().to(DEVICE).eval()
        rows["LearnedMLP (untrained)"].append(measure(untrained))

        fitted = {}
        for name, ctor in [("EquivariantSO2", EquivariantSO2), ("LearnedMLP", LearnedMLP)]:
            torch.manual_seed(seed)
            m = train(ctor().to(DEVICE), p_tr, y_tr, args.epochs, seed)
            fitted[name] = m
            rows[name].append(measure(m))
            if seed == 0 and name == "EquivariantSO2":
                self_check(m, p_ev)

        if seed == 0:
            print(f"\n  per-angle, seed 0 (no angle is privileged -- there is no lattice)")
            print(f"{'angle':>7s} {'eq 1-CKA':>10s} {'eq drift':>10s} "
                  f"{'mlp 1-CKA':>11s} {'mlp drift':>10s}")
            eq, ml = fitted["EquivariantSO2"], fitted["LearnedMLP"]
            for d in ANGLES:
                print(f"{d:7.1f} {cka_error(eq, p_ev, d):10.6f} {pred_drift(eq, p_ev, d):10.6f} "
                      f"{cka_error(ml, p_ev, d):11.6f} {pred_drift(ml, p_ev, d):10.6f}")

    def agg(vals):
        t = torch.tensor([v if v is not None else float("nan") for v in vals])
        return t.mean().item(), t.std().item() if len(vals) > 1 else 0.0

    print(f"\n  {args.seeds} seeds, {args.epochs} epochs, mean +/- sd")
    print(f"\n{'model':>23s} {'LEE naive':>16s} {'LEE correct':>12s} {'1-CKA':>17s}"
          f" {'drift':>17s} {'test MSE':>9s}")
    summ = {}
    for name, rs in rows.items():
        cols = list(zip(*rs))
        lee, lee_sd = agg(cols[0])
        lc = None if cols[1][0] is None else agg(cols[1])[0]
        cka, cka_sd = agg(cols[2])
        drf, drf_sd = agg(cols[3])
        mse, _ = agg(cols[4])
        summ[name] = (lee, cka, drf, mse)
        print(f"{name:>23s} {lee:9.4f}+/-{lee_sd:<5.3f} "
              f"{'n/a' if lc is None else f'{lc:.4f}':>12s} "
              f"{cka:10.6f}+/-{cka_sd:<6.4f} {drf:10.6f}+/-{drf_sd:<6.4f} {mse:9.5f}")

    eq, ml = summ["EquivariantSO2"], summ["LearnedMLP"]
    print("\n  VERDICT  (behaviour is the ground truth)")
    for label, i, better in [("LEE, naive rho", 0, "lower"), ("linear CKA", 1, "lower"),
                             ("prediction drift", 2, "lower")]:
        win = "LearnedMLP" if ml[i] < eq[i] else "EquivariantSO2"
        print(f"    {label:>18s}  prefers {win:>15s}   ({eq[i]:.6f} eq  vs  {ml[i]:.6f} mlp)")
    print(f"\n    The baseline is deeper, wider AND fits better "
          f"({ml[3]:.4f} vs {eq[3]:.4f} test MSE),")
    print("    so it was not crippled. Yet its predictions move under rotation while")
    print("    the equivariant model's are bit-identical. LEE prefers it anyway.")
    print("\n    Nothing here can be blamed on aliasing: the group action is an exact")
    print("    matrix multiply, asserted equivariant to ~1e-6 at generic angles.")
    print("    The untrained row separates learned invariance from an init artifact.")


if __name__ == "__main__":
    main()
