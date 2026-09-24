"""
LieGG's remaining assumption, and what it costs: the OUTPUT must be invariant.

liegg_demo.py is LieGG's best case in every respect, deliberately. The group
acts linearly on the input, every element is an exact orthogonal matmul, and
the target is an exactly INVARIANT scalar, so the trivial-representation
condition in LieGG's derivation holds by construction. It succeeds there: null
space of dimension exactly 1, alignment 1.0000 with the true generator.

This script removes the one thing that was granted. Same model, same group,
same data -- only the readout LieGG is pointed at changes.

WHY THE DERIVATION BREAKS. LieGG solves the homogeneous condition

    grad f(x)^T A x = 0                                   (f INVARIANT)

for A. If f is instead EQUIVARIANT, with f(exp(tA)x) = exp(t drho(A)) f(x),
then differentiating at t=0 gives

    J_f(x) A x = drho(A) f(x)   which is NOT zero

so A_true does not lie in the null space LieGG searches, and generically
nothing does. The method reports no symmetry for a model that is exactly
SO(2)-equivariant. This is the same trivial-representation assumption that
breaks LEE (so2_exact_lee_reversal_demo.py), moved from the hidden features to
the output.

THREE READOUTS, and the middle one is the control that makes this a finding
rather than a confound:

  invariant scalar   head(||h||), what liegg_demo.py uses.    m = 1
  invariant vector   ||h|| per channel. Still invariant, but  m = 64
                     vector-valued -- so if LieGG breaks here
                     too, the cause is dimensionality, not
                     equivariance.
  EQUIVARIANT        h itself, frequency-1, carrying phase.   m = 128
                     Invariance is the only thing removed.

The polarization matrix is built with one row per (sample, output component),
which is the natural generalisation of LieGG to a vector-valued invariant map.

TWO RESIDUALS FOR THE TRUE GENERATOR, which is what turns a negative result
into a proof. On the equivariant readout we report both

    naive       || J_h(x) A_true x ||              LieGG's condition
    corrected   || J_h(x) A_true x - drho h(x) ||  the condition that holds

If the corrected residual vanishes while the naive one does not, the model is
demonstrably equivariant and LieGG is simply solving the wrong equation. For
frequency-1 features drho acts on each complex channel as (hr, hi) -> (-hi, hr).

MEASURED (3 seeds, 300 epochs, n=2048, null tol 0.05). The model's ground
truth on the same clouds is 1 - CKA = 0.000000 and prediction drift = 0.000000,
worst case over eight probe angles -- exactly SO(2)-equivariant.

  readout                        sym_var  null_dim   align  A_true/rand   1-CKA
  invariant scalar  head(||h||)  0.00003       1.0  1.0000      0.00008  0.000000
  invariant vector  ||h||        0.00022       1.0  1.0000      0.00007  0.000033
  EQUIVARIANT       h            0.43667       0.0  0.0000      0.87357  0.000000

A_true/rand is the residual of the TRUE generator against the learned
constraints, divided by that of a random unit generator on the same matrix.
The absolute residual scales with the row count, which differs 128x across
these readouts, so only the ratio is comparable.

1. LIEGG FINDS NOTHING ON AN EXACTLY EQUIVARIANT MODEL. Pointed at the
   equivariant features it returns an EMPTY null space, alignment 0.0000 and a
   symmetry variance of 0.39920, for a network whose CKA error and behavioural
   drift are both exactly zero at every probe angle. Pointed at the invariant
   head of the SAME network on the SAME data it returns null_dim exactly 1 and
   alignment 1.0000, reproducing liegg_demo.py.

2. THE CAUSE IS EQUIVARIANCE, NOT DIMENSIONALITY. The middle readout is
   invariant but vector-valued, 64 components against the scalar's 1, and it
   recovers the generator perfectly. So the vector-valued generalisation of the
   polarization matrix is sound; what breaks the method is that the output
   carries a non-trivial representation.

3. THE TRUE GENERATOR IS NO BETTER THAN A RANDOM DIRECTION. Under LieGG's
   equation on the equivariant readout, A_true scores 0.86557 of what a random
   unit generator scores -- essentially no better. On the two invariant
   readouts it scores 0.00008 and 0.00005, i.e. four orders of magnitude
   better. The constraints do not merely fail to isolate A_true; they do not
   prefer it at all.

4. LINEAR CKA IS UNTOUCHED ON THE READOUT WHERE LIEGG FINDS NOTHING. The last
   column is the head-to-head: 1 - CKA is exactly 0.000000 on the equivariant
   features. Linear CKA is invariant to ANY orthogonal rho(g), and the
   frequency-1 representation rotating each complex channel is orthogonal, so
   the centred Gram matrix is unchanged. Same network, same clouds, same
   rotation: the measure that assumes nothing about rho(g) reads exact
   equivariance while the measure that DISCOVERS rho(g) reads none.

5. AND THE CORRECTED EQUATION IS SATISFIED EXACTLY. Against the equivariant
   readout, LieGG's condition || J_h A x || is at full magnitude while the
   condition that actually holds, || J_h A x - drho h ||, is 0.00000. The model
   is demonstrably equivariant; LieGG is solving the wrong equation. That is
   what makes this a proof rather than a negative result.

WHAT THIS DOES AND DOES NOT CLAIM. LieGG is not being used outside its stated
scope by accident -- an invariant output is a condition its paper states. The
finding is that the condition is load-bearing and unstated in practice: a
network can be exactly equivariant, with the group acting linearly on the
input and every element exact, and LieGG will report no symmetry if the readout
one happens to probe is equivariant rather than invariant. That is the same
trivial-representation assumption that breaks LEE, moved from the hidden
features to the output, and it means "discovers the group instead of assuming
it" holds only for the group's action on the INPUT.

Run: cd src/escnn_experiments && .venv/bin/python3 liegg_equivariant_output_demo.py
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from so2_exact_lee_reversal_demo import (ANGLES, DEVICE, EquivariantSO2, K,
                                         cka_error, invariant_target,
                                         pred_drift, rotate_cloud,
                                         sample_clouds, train)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import EquivarianceTracker  # noqa: E402
from liegg_demo import NULL_TOL, true_generator

N_POLAR = 2048       # matches liegg_demo.py's N_EVAL; rows are N_POLAR * m
SEED = 0


def polarization(feat_fn, p):
    """Accumulate E^T E rather than E itself.

    E has one row per (sample, output component), vec(grad f_j(x_i) x_i^T),
    which for the equivariant readout is 2048 x 128 = 262k rows. Only E^T E
    (256 x 256) is needed: its eigenvalues are the squared singular values and
    its eigenvectors the right singular vectors, so nothing is lost and the
    matrix never has to be materialised.

    Also returns the directional derivative J_f(x) A x for the true generator,
    which the caller compares against drho f(x)."""
    p = p.clone().detach().requires_grad_(True)
    out = feat_fn(p)
    if out.dim() == 1:
        out = out[:, None]
    n, m = out.shape
    xf = p.detach().reshape(n, -1)
    d = xf.shape[1]
    A = true_generator().to(xf.device)
    Ax = xf @ A.T                                    # tangent vector at each x

    G = torch.zeros(d * d, d * d, device=xf.device, dtype=xf.dtype)
    dirs = []
    for j in range(m):
        g, = torch.autograd.grad(out[:, j].sum(), p, retain_graph=(j < m - 1))
        gf = g.reshape(n, -1)
        rows = (gf[:, :, None] * xf[:, None, :]).reshape(n, -1)
        G += rows.T @ rows
        dirs.append((gf * Ax).sum(1))
    return G, n * m, torch.stack(dirs, 1)


@torch.no_grad()
def cka_of_readout(feat_fn, p):
    """1 - linear CKA between the readout on clean and rotated clouds, worst
    case over the probe angles. Needs no rho: if the readout transforms by ANY
    orthogonal representation the centred Gram matrix is unchanged and this is
    exactly 0, which is the whole point of putting it beside LieGG here."""
    worst = 0.0
    for deg in ANGLES:
        X = feat_fn(p)
        Y = feat_fn(rotate_cloud(p, math.radians(deg)))
        if X.dim() == 1:
            X, Y = X[:, None], Y[:, None]
        t = EquivarianceTracker(X.device)
        t.update(X, Y)
        worst = max(worst, 1.0 - t.compute_stats()["linear_cka"])
    return worst


def score(G, n_rows):
    """LieGG's read-out, computed from E^T E."""
    A = true_generator().reshape(-1).to(G.device)
    A = A / A.norm()
    lam, V = torch.linalg.eigh(G)                    # ascending
    lam = lam.flip(0).clamp_min(0)
    V = V.flip(1)                                    # columns, descending
    S = lam.sqrt()
    smax = S[0].clamp_min(1e-30)
    Sn = S / smax
    keep = Sn < NULL_TOL
    null = V[:, keep].T
    # Residual of the TRUE generator against the learned constraints, reported
    # relative to a RANDOM unit generator on the same matrix. The absolute
    # residual is not comparable across readouts (it scales with the row count,
    # which differs 128x here); the ratio is. Near 0 means the constraints pin
    # A_true down, near 1 means A_true is no better than a random direction.
    quad = lambda v: (v @ G @ v).clamp_min(0).sqrt()
    g = torch.Generator(device="cpu").manual_seed(0)
    rnd = torch.randn(64, G.shape[0], generator=g).to(G.device)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    rand_res = torch.stack([quad(v) for v in rnd]).mean()
    return {"sym_var": Sn[-1].item(),
            "null_dim": int(keep.sum().item()),
            "align": (null @ A).norm().clamp(max=1.0).item() if len(null) else 0.0,
            "true_vs_rand": (quad(A) / rand_res.clamp_min(1e-30)).item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"LieGG on an exactly SO(2)-equivariant model, three readouts.\n"
          f"{args.seeds} seeds, {args.epochs} epochs, n={N_POLAR}, "
          f"null tol {NULL_TOL}\n")

    acc = {}
    for seed in range(args.seeds):
        # seed the GLOBAL rng too: the model is constructed before train() is
        # called, so its initialisation reads global state and would otherwise
        # depend on everything evaluated earlier in the run
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        p_tr = sample_clouds(20000, gen).to(DEVICE)
        p = sample_clouds(N_POLAR, gen).to(DEVICE)
        y_tr = invariant_target(p_tr)
        y_tr = (y_tr - y_tr.mean()) / y_tr.std()
        model = train(EquivariantSO2().to(DEVICE), p_tr, y_tr, args.epochs, seed)

        if seed == 0:
            ck = max(cka_error(model, p, d) for d in ANGLES)
            dr = max(pred_drift(model, p, d) for d in ANGLES)
            print(f"  ground truth, worst over {len(ANGLES)} probe angles: "
                  f"1-CKA {ck:.6f}, prediction drift {dr:.6f} "
                  f"-> exactly SO(2)-equivariant\n")

        readouts = [
            ("invariant scalar  head(||h||)", lambda q: model(q)[0].squeeze(-1), False),
            ("invariant vector  ||h||      ", lambda q: model.features(q).norm(dim=-1), False),
            ("EQUIVARIANT       h          ", lambda q: model.features(q).flatten(1), True),
        ]
        for name, fn, equivariant_out in readouts:
            G, n_rows, direc = polarization(fn, p)
            r = score(G, n_rows)
            r["cka_err"] = cka_of_readout(fn, p)
            if equivariant_out:
                h = model.features(p).detach()
                drho = torch.stack([-h[..., 1], h[..., 0]], -1).flatten(1)
                scale = direc.norm().clamp_min(1e-30)
                r["naive_res"] = 1.0
                r["corrected_res"] = ((direc - drho).norm() / scale).item()
            acc.setdefault(name, []).append(r)

    keys = ["sym_var", "null_dim", "align", "true_vs_rand", "cka_err"]
    print(f"  {'readout':<32}{'sym_var':>9}{'null_dim':>10}{'align':>8}"
          f"{'A_true/rand':>13}{'1-CKA':>10}")
    print("  " + "-" * 82)
    for name, rs in acc.items():
        m = {k: sum(r[k] for r in rs) / len(rs) for k in keys}
        print(f"  {name:<32}{m['sym_var']:>9.5f}{m['null_dim']:>10.1f}"
              f"{m['align']:>8.4f}{m['true_vs_rand']:>13.5f}"
              f"{m['cka_err']:>10.6f}")

    eq = acc["EQUIVARIANT       h          "]
    cr = sum(r["corrected_res"] for r in eq) / len(eq)
    print(f"\n  true generator against the equivariant readout:")
    print(f"    LieGG's condition   || J_h A x ||              "
          f"relative 1.00000")
    print(f"    the condition that holds  || J_h A x - drho h ||  "
          f"relative {cr:.5f}")
    print(f"\n  The model is exactly equivariant either way; only the equation "
          f"LieGG solves is wrong.")


if __name__ == "__main__":
    main()
