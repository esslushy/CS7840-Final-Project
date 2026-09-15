"""
The complement case: where MI AND CKA both succeed and the rho(g)-based methods
both fail.

Across these experiments the metric families have separated cleanly:

    family A  hardcode rho(g)   LEE, EQ-R
    family B  fit rho(g)        Lenc & Vedaldi affine fit, Procrustes
    family C  assume no rho(g)  CKA, SVCCA, dCor, RSA
    (plus)    mutual information

equivariance_metrics_comparison.py found A failing while B and C succeeded, and
exact_mi_demo.py found a bijective scramble where only MI survived. This script
fills in the remaining cell: a readout on which A and B BOTH fail while CKA and
MI BOTH succeed.

THE CONSTRUCTION, and it is not contrived. The readout emits a linear block
alongside an ENERGY block -- squared magnitudes:

    sigma(v) = [ M v  |  (N v)^2 ]

Energy/power features are everywhere in real networks: fiber norms in steerable
CNNs, attention scores, variance and second-moment features, anything followed
by squaring or abs. Their transformation law under a group is QUADRATIC. If the
underlying representation obeys v(gx) = P v(x), then the energy block obeys

    (N P v)^2  =  (N P M^-1 · Mv)^2

which is quadratic in the linear block, not affine in it. So:

  * family A is simply wrong -- there is no fixed rho(g) matrix to assume, and
    the usual fallback (assume invariance) is badly off.
  * family B cannot express it -- an affine map has no access to the cross terms
    u_j u_k that a square requires, so the fit leaves the energy block
    unexplained no matter how much data you give it. Procrustes, being more
    constrained still, does worse.
  * CKA does not care: the point cloud's relational geometry is largely
    preserved, so the Gram matrices stay aligned.
  * MI does not care either, and here sigma is injective (the linear block alone
    determines v), so MI/H = 1.000 exactly -- no information is lost, which is
    what proves A and B are wrong rather than the representation being bad.

Everything runs on the exhaustive 2^16 binary 4x4 space from discretized.ipynb,
coarse-grained equivariantly (see exact_mi_demo.quadrant_coarse_grain) so the
alphabet is small enough for the MI null to be valid -- 486 symbols at ~135
samples each. Every detector reports its shuffled null.

The Sobel-x rows are the essential control: same energy readout, but on a
representation that is genuinely NOT equivariant. CKA and MI must drop there,
or they would merely be responding to the readout's structure rather than to
equivariance.

MEASURED.

  Laplacian (exactly equivariant), MI/H = 1.000 under BOTH readouts:

    metric                  identity readout    ENERGY readout
    A1  LEE, assumed rho=I      1.4933 FAILS      1.3640 FAILS
    B1  affine fit              0.0000 detects    0.6723 FAILS
    B2  Procrustes              0.0000 detects    0.7225 FAILS
    C1  linear CKA              1.0000 detects    0.7379 detects  (null 0.0029)
    C2  RBF CKA                 1.0000 detects    0.8634 detects  (null 0.0042)
    C3  SVCCA                   1.0000 detects    0.7802 detects  (null 0.0459)
    MI  exact (bits)            7.3205 detects    7.3205 detects  (null 0.8100)

  The identity column is the reference and shows family B working perfectly
  (residual 0.0000) when rho(g) happens to be linear. Switching only the readout
  -- same images, same kernel, same underlying representation, MI/H still
  exactly 1.000 so nothing was lost -- takes B from 0.0000 to 0.67/0.72 while
  every C metric and MI keep detecting. That isolates the cause precisely: the
  failure is family B's hypothesis class, not the representation.

  CONTROL (Sobel-x, genuinely not equivariant, same energy readout): linear CKA
  0.0071, RBF CKA 0.0091, SVCCA 0.0876 -- all collapse to their nulls. So the C
  metrics were responding to equivariance, not to the readout's structure.

  One honest note on the control: MI still reports a margin of 1.06 bits for
  Sobel-x, which the crude detect/fail threshold calls a detection. That is not
  a false positive -- a non-equivariant convolution genuinely does share some
  information between x and rot(x). The informative figure is MI/H: 1.000 for
  the equivariant kernel against 0.195 for Sobel-x. MI gives a graded answer
  where CKA gives a near-binary one (1.0000 vs 0.0023), which is a point in MI's
  favour but means MI should be read as a ratio, not thresholded.

Run: cd src/metric_demos && ../../.venv/bin/python3 energy_readout_demo.py
"""
import numpy as np

from exact_mi_demo import (KERNELS, NBITS, all_binary_images, binarize, circ_conv,
                           exact_entropy_bits, exact_mi_bits, linear_cka,
                           quadrant_coarse_grain, rbf_cka, svcca)

SEED = 0
CKA_SUBSAMPLE = 3000


def standardize(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)


def make_energy_readout(rng, d):
    """sigma(v) = [ M v | (N v)^2 ]. Injective, because the linear block alone
    determines v whenever M is invertible -- so no information is destroyed and
    any metric that reports 'not equivariant' is reporting its own limitation."""
    M = np.linalg.qr(rng.standard_normal((d, d)))[0]
    N = np.linalg.qr(rng.standard_normal((d, d)))[0]

    def sigma(V):
        return standardize(np.hstack([V @ M.T, (V @ N.T) ** 2]))
    return sigma


# ------------------------------------------------------- family A and family B

def lee_assumed_identity(X, Y):
    """A: hardcode rho(g). For a non-4D tensor the reference implementation
    assumes the trivial representation, i.e. that the readout is invariant."""
    return float(np.linalg.norm(Y - X) / np.linalg.norm(Y))


def affine_fit(X, Y):
    """B: Lenc & Vedaldi -- best affine map, held-out residual."""
    m = len(X) // 2
    aug = lambda T: np.hstack([T, np.ones((len(T), 1))])
    E = np.linalg.lstsq(aug(X[:m]), Y[:m], rcond=None)[0]
    return float(np.linalg.norm(Y[m:] - aug(X[m:]) @ E) / np.linalg.norm(Y[m:]))


def procrustes_fit(X, Y):
    """B: same, restricted to orthogonal maps (as a true rotation rep would be)."""
    m = len(X) // 2
    mx, my = X[:m].mean(0), Y[:m].mean(0)
    U, _, Vt = np.linalg.svd((X[:m] - mx).T @ (Y[:m] - my))
    R = U @ Vt
    return float(np.linalg.norm((Y[m:] - my) - (X[m:] - mx) @ R)
                 / np.linalg.norm(Y[m:] - my))


def main():
    rng = np.random.default_rng(SEED)
    images = all_binary_images()
    rot = np.rot90(images, k=1, axes=(-2, -1))
    n = len(images)
    mi_shuffle = rng.permutation(n)
    sub = rng.choice(n, CKA_SUBSAMPLE, replace=False)
    cka_shuffle = rng.permutation(CKA_SUBSAMPLE)
    sigma = make_energy_readout(rng, 4)

    for kname, kernel in KERNELS.items():
        equivariant = np.allclose(kernel, np.rot90(kernel))
        fa = circ_conv(images.astype(float), kernel)
        fb = circ_conv(rot.astype(float), kernel)
        thr = np.median(np.concatenate([fa.ravel(), fb.ravel()]))
        A = quadrant_coarse_grain(binarize(fa, thr)).astype(float)
        B = quadrant_coarse_grain(binarize(fb, thr)).astype(float)

        print("=" * 88)
        print(f"{kname}   ->  representation is "
              f"{'EXACTLY equivariant' if equivariant else 'NOT equivariant (control)'}")
        print("=" * 88)

        for rname, Za, Zb in [("identity readout (reference)", standardize(A), standardize(B)),
                              ("ENERGY readout [Mv | (Nv)^2]", sigma(A), sigma(B))]:
            h = exact_entropy_bits(Za)
            mi = exact_mi_bits(Za, Zb)
            mi_null = exact_mi_bits(Za, Zb[mi_shuffle])
            Xs, Ys = Za[sub], Zb[sub]
            rows = [
                ("A1  LEE, assumed rho=I", lee_assumed_identity(Za, Zb), None, False),
                ("B1  affine fit (Lenc+15)", affine_fit(Za, Zb), None, False),
                ("B2  Procrustes fit", procrustes_fit(Za, Zb), None, False),
                ("C1  linear CKA", linear_cka(Xs, Ys), linear_cka(Xs, Ys[cka_shuffle]), True),
                ("C2  RBF CKA", rbf_cka(Xs, Ys), rbf_cka(Xs, Ys[cka_shuffle]), True),
                ("C3  SVCCA", svcca(Xs, Ys), svcca(Xs, Ys[cka_shuffle]), True),
                ("MI  exact (bits)", mi, mi_null, True),
            ]
            print(f"\n  {rname}    H = {h:.4f} bits,  MI/H = {mi / h if h else 0:.3f}")
            print(f"  {'metric':<26s} {'value':>9s} {'null':>9s} {'margin':>8s}  verdict")
            for label, val, null, higher_better in rows:
                if null is None:                      # residual: lower is better
                    verdict = "FAILS" if val > 0.5 else "detects"
                    print(f"  {label:<26s} {val:9.4f} {'--':>9s} {'--':>8s}  {verdict}")
                else:
                    margin = val - null
                    verdict = "detects" if margin > 0.3 else "fails"
                    print(f"  {label:<26s} {val:9.4f} {null:9.4f} {margin:8.4f}  {verdict}")
        print()


if __name__ == "__main__":
    main()
