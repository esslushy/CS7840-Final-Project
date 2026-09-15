"""
Exact mutual information succeeds where CKA fails -- with no estimator error.

Follows the setup in discretized.ipynb: enumerate ALL 2^16 binary 4x4 images,
push them through a circular convolution, and binarize at a shared median. The
enumeration is exhaustive and every map is deterministic, so the joint
distribution is known EXACTLY. Mutual information is therefore computed, not
estimated -- no KSG, no neural bound, no sampling error, no bandwidth to choose.
That matters here, because the previous attempt at this question
(escnn_experiments/unlearnable_rho_demo.py) could only show MI degrading
"gracefully" from 4.54 to 1.64 nats and had to argue that the decay was
estimator artifact. On a discrete space the argument is unnecessary: the decay
is exactly zero, and you can read it off the output.

Two convolution kernels, both taken from discretized.ipynb's KERNELS:

    Laplacian  [[0,-1,0],[-1,4,-1],[0,-1,0]]   is rot90-SYMMETRIC, so circular
               convolution with it commutes exactly with rot90. The
               representation is EXACTLY C4-equivariant (verified in-script).
    Sobel-x    [[-1,0,1],[-2,0,2],[-1,0,1]]    is not rot90-symmetric, so it is
               not equivariant. Included as the contrast case.

Then the representation is observed through a fixed readout sigma, applied
identically to the clean and rotated branches, so the model's equivariance is
never touched -- only the coordinates it is reported in:

    identity        no change (reference)
    bit-permute     permute the 16 coordinates. Bijective AND orthogonal, so
                    CKA is invariant to it too. Both metrics should survive.
    alphabet-permute treat the 16-bit pattern as an integer in [0, 2^16) and
                    apply a fixed random permutation of that range. Perfectly
                    bijective, maximally nonlinear. This is the discrete
                    analogue of the TWIST readout that defeated every CKA
                    variant.
    lossy-8bit      keep only 8 of the 16 bits. Many-to-one, so information is
                    genuinely destroyed -- the control that separates "hidden"
                    from "lost".

WHAT TO EXPECT, and why it is a proof rather than a measurement:

  MI is invariant under any bijection applied to either variable, because it is
  a sum over the joint pmf and a bijection only relabels the terms. So MI under
  alphabet-permute must equal MI under identity to the last bit. CKA has no such
  invariance -- it reads the +-1 vectors geometrically, and relabeling the
  alphabet destroys that geometry completely.

  Under lossy-8bit, MI must genuinely DROP, because information really is gone.
  So MI is not merely more robust: it distinguishes a scrambled coordinate
  system from a lossy one, which is the distinction that actually matters and
  the one CKA cannot make.

Note the cost asymmetry, too: exact MI here is a histogram over 65536 rows,
while CKA needs an n x n Gram matrix and has to be subsampled to stay tractable.
On this space the information-theoretic quantity is both exact and cheaper.

WHY CKA FAILS: IT IS AN ISOMETRY-INVARIANT MEASURE, AND RBF DOES NOT CHANGE THAT.

  Linear CKA reads the data through inner products; RBF CKA reads it through
  pairwise Euclidean distances. Both therefore see only the METRIC structure of
  the point cloud, so both are invariant to exactly the isometries (orthogonal
  maps and translations, plus isotropic scaling once the median-heuristic
  bandwidth rescales with the data). Swapping the linear kernel for RBF changes
  WHAT DEPENDENCE STRUCTURE between X and Y it can detect -- it does not enlarge
  the set of reparameterizations of X or Y it is blind to. Those are two
  different properties, and a scrambled readout attacks the second one, which is
  why RBF buys nothing here.

  The readouts below are ordered to separate the three invariance groups, and the
  measured numbers pin each metric to its group exactly:

    readout            nature                          exact MI  linCKA  rbfCKA  SVCCA
    identity           reference                        15.1834  1.0000  1.0000  1.0000
    bit-permute        bijective, isometry              15.1834  1.0000  1.0000  1.0000
    ill-cond. linear   bijective, linear, NOT isometry  15.1834  0.0396  0.0525  1.0000
    alphabet-permute   bijective, NOT linear            15.1834  0.0044  0.0071  0.0559
    lossy-8bit         MANY-TO-ONE, info destroyed       3.1495  0.3892  0.3863  0.4352

  Read the third row: an invertible LINEAR map with condition number 1e3 is
  enough to take both CKA variants from 1.0000 to ~0.04, while SVCCA sits at
  1.0000 -- because CCA is invariant to every invertible linear map, a strictly
  larger group. And linCKA 0.0396 vs rbfCKA 0.0525 is the whole answer to
  "does RBF help": no, they fail together.

  So the hierarchy of invariance groups is nested and each metric sits in one:

      CKA, linear and RBF   isometries + isotropic scale   (smallest)
      CCA / SVCCA           all invertible linear maps
      distance corr., RSA   metric-based, so isometries again
      mutual information    ALL bijections                 (largest possible)

  MI's group is the maximal one for a dependence measure: it depends only on the
  joint distribution up to relabeling, with no metric at all. That is the entire
  reason it survives, and it is why no choice of kernel could rescue CKA -- the
  problem is not the kernel, it is that CKA is a statement about geometry.

  This also retro-explains every earlier result in this repo's experiments:
  bit-permutations are orthogonal so CKA survived them; the TWIST readout in
  escnn_experiments/unlearnable_rho_demo.py applied a per-sample rotation whose
  angle depended on the radius, which is NOT a global isometry, so CKA collapsed;
  distance correlation and RSA are metric-based and were fooled identically;
  Procrustes fits inside the orthogonal group and the linear fit inside the
  affine one, so each failed exactly when the true map left its class.

A TRAP IN THE MI MEASUREMENT ITSELF -- AND THE FIX.

  An earlier version of this script reported the full 16-bit pattern MI of
  15.1834 bits as evidence of perfect equivariance, WITHOUT a shuffled control.
  That was wrong, and the control shows why:

    granularity            symbols  samples/sym       MI   MI shuffled  margin
    FULL 16-bit pattern     45707          1.4   15.1834       14.3668   0.82
    quadrant coarse-grain     486        134.8    7.3205        0.8113   6.51

  With 45707 distinct symbols and only 65536 samples, nearly every symbol is
  unique -- so ANY pairing of rows is a near-bijection and therefore has
  near-maximal MI. Shuffling destroys the meaningful correspondence but
  preserves the structural fact that each x maps to exactly one y, which is all
  the plug-in estimate is really seeing. The null sits at 14.37 against a signal
  of 15.18: a 5% margin, not a detection.

  Worse, at that granularity the bias-corrected ordering INVERTS: the
  equivariant Laplacian scores a margin of 0.82 while the non-equivariant
  Sobel-x scores 2.00, because Sobel's lower-entropy readout (1225 symbols, 53.5
  samples each) is far better sampled. Read naively, the full-pattern numbers
  rank the wrong kernel first.

  This is the exact discrete twin of the RBF bandwidth trap in
  escnn_experiments/unlearnable_rho_demo.py: a metric saturating because its
  effective resolution exceeds what the sample can support. Note the enumeration
  being exhaustive does NOT save you -- the joint distribution over images is
  exact, but MI between two ~2^15-symbol representations still needs the joint
  to be populated, and 65536 samples cannot populate 45707 x 45707 cells.

  THE FIX is to coarse-grain in a way that RESPECTS THE GROUP ACTION, shrinking
  the alphabet without breaking the bijection -- see quadrant_coarse_grain().
  At 134.8 samples/symbol the null falls to 0.81 and the picture is correct:

    kernel                        MI      MI/H   margin over null
    Laplacian  (equivariant)   7.3205     1.000   6.51
    Sobel-x    (NOT equiv.)    1.2676     0.195   1.06

  MI/H = 1.000 exactly for the equivariant kernel (the bijection survives the
  coarse-graining), 0.195 for the non-equivariant one, and a 6x separation in
  the bias-corrected margin. Truncating bits would also shrink the alphabet but
  is NOT a valid fix: rot90 permutes positions, so a bit-truncated readout is no
  longer bijectively related to its rotated counterpart, and MI/H falls below 1
  for reasons unrelated to equivariance.

WHAT SURVIVES THE CORRECTION, AND WHAT DOES NOT.

  SURVIVES -- the invariance identity. MI(identity) - MI(alphabet-permute) =
  +0.00e+00 at BOTH granularities and for both kernels. This is immune to the
  sampling problem, because whatever bias the estimator carries is identical on
  the two sides and cancels exactly in the difference. So the central claim
  stands: MI is invariant to bijective reparameterization, CKA is not, and the
  invariance hierarchy above is unaffected.

  DOES NOT SURVIVE -- reading the absolute MI as a measure of how equivariant
  something is, without a null and without checking samples-per-symbol. Any
  MI-based equivariance metric must report (a) the shuffled null, (b) the
  alphabet size relative to n, and ideally (c) a null distribution over many
  permutations rather than the single one used here.

DOES THE SAME UNIQUENESS ARTIFACT INFLATE CKA? NO -- IT DEFLATES IT.

  Worth stating explicitly, because the trap above invites the misreading "MI is
  fragile, CKA is fine". On independent Gaussian noise (true MI = 0) with every
  point unique:

      n     linCKA null   rbfCKA null   plug-in MI null   log2(n)
     500        0.0596        0.0983            8.9658      8.97
    1000        0.0340        0.0549            9.9658      9.97
    2000        0.0164        0.0274           10.9658     10.97
    4000        0.0080        0.0137           11.9658     11.97

  CKA's null decays like 1/n; plug-in MI's null tracks log2(n) to three
  decimals. The reason is structural. With all values distinct the contingency
  table is an n x n permutation matrix -- n cells of count 1 -- so
  MI = sum (1/n) log2((1/n)/((1/n)(1/n))) = log2(n) BY CONSTRUCTION, independent
  of the data. It is an identifiability statistic and uniqueness maximises
  identifiability. CKA instead correlates two n x n similarity matrices; under
  shuffling L -> P L P^T and the Frobenius inner product becomes a sum of n^2
  randomly re-paired terms concentrating on zero. Distinctness of points says
  nothing about whether relational geometries agree.

  CKA does have the same disease, on a different knob. As the RBF bandwidth goes
  to zero every point is similar only to itself, K -> I, and CKA(I, I) = 1
  regardless of the data -- "every point unique in kernel space", structurally
  the same pathology. So the unifying rule is: BOTH metrics degenerate when
  their similarity structure collapses to the identity. For plug-in MI the knob
  is alphabet granularity; for RBF CKA it is bandwidth.

  Which recasts the median heuristic as a deliberate trade rather than a
  shortcoming: by putting sigma at the median pairwise distance it guarantees
  roughly half the pairs are within one bandwidth, so K is never near-identity.
  That is an automatic coarse-graining, and it is exactly the safeguard plug-in
  MI lacks -- which is why quadrant_coarse_grain() has to supply it by hand.
  Linear CKA has no bandwidth at all and so cannot be driven into this regime by
  resolution; its failure mode is the separate isometry-invariance one.

  MI/H = 1.000 says the representation carries EVERYTHING about its rotated
  counterpart -- perfect equivariance in the information sense -- and it says so
  identically in scrambled coordinates. CKA in those same coordinates reports
  0.0044, which is BELOW its own shuffled control of 0.0062: not merely degraded
  but actively asserting that no relationship exists, for a representation that
  is provably perfectly equivariant.

  Under lossy-8bit, MI correctly falls to 3.15 bits (MI/H = 0.399). It is not
  blindly invariant to everything -- it drops precisely when information is
  genuinely destroyed.

THE HEADLINE: CKA CONFLATES THREE SITUATIONS THAT MI SEPARATES.

  At the valid (quadrant coarse-grained) granularity, with nulls shown:

    situation                                  MI    null    CKA   CKA null
    exactly equivariant, natural coordinates  7.32   0.81  1.0000    0.0017
    exactly equivariant, scrambled coords     7.32   0.81  0.1113    0.0030
    genuinely NOT equivariant (Sobel-x)       1.27   0.20  0.0018    0.0009

  CKA scores the perfectly-equivariant-but-scrambled case (0.1113) in the same
  range as, and depending on the readout below, the genuinely non-equivariant
  one (0.0018) -- it cannot separate "equivariant in bad coordinates" from "not
  equivariant". MI separates them cleanly and on an interpretable scale: 7.32 of
  7.32 bits (MI/H = 1.000) versus 1.27 of 6.49 (MI/H = 0.195).

  This is the cleanest statement of the whole line of experiments in this
  directory and escnn_experiments/. Every geometric measure tested -- LEE under
  an assumed rho(g), EQ-R, fitted linear maps, Procrustes, linear CKA, RBF CKA,
  SVCCA, distance correlation, RSA -- is a statement about the coordinate system
  a representation happens to be written in. Mutual information is a statement
  about the representation.

Run: cd src/metric_demos && ../../.venv/bin/python3 exact_mi_demo.py
"""
import itertools

import numpy as np
from scipy.signal import convolve2d

SIDE = 4
NBITS = SIDE * SIDE
CKA_SUBSAMPLE = 4096          # CKA needs an n x n Gram; MI uses all 65536
SEED = 0

# Both from discretized.ipynb's KERNELS.
KERNELS = {
    "Laplacian (rot90-symmetric)": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float),
    "Sobel-x   (not symmetric)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
}


# ------------------------------------------------------------------ the space

def all_binary_images():
    """Every 4x4 image over {-1,+1}: 2^16 = 65536 of them, so the distribution
    over images is exactly uniform and exactly known."""
    combos = itertools.product([-1, 1], repeat=NBITS)
    return np.fromiter(itertools.chain(*combos), dtype=np.int8).reshape(-1, SIDE, SIDE)


def circ_conv(images, kernel):
    """Circular convolution, vectorized over the whole stack. Matches
    scipy.convolve2d(mode='same', boundary='wrap'), asserted below."""
    out = np.zeros(images.shape, dtype=float)
    kh, kw = kernel.shape
    for di in range(kh):
        for dj in range(kw):
            # True convolution (kernel flipped, as convolve2d does):
            # out[i,j] = sum_{di,dj} k[di,dj] * img[i-di+kh//2, j-dj+kw//2],
            # and roll(img, s)[i] = img[i-s], so s = (di-kh//2, dj-kw//2).
            out += kernel[di, dj] * np.roll(images, (di - kh // 2, dj - kw // 2), axis=(-2, -1))
    return out


def binarize(feature_maps, threshold):
    flat = feature_maps.reshape(len(feature_maps), -1)
    return np.where(flat >= threshold, 1, -1).astype(np.int8)


# -------------------------------------------------------- exact information

def quadrant_coarse_grain(binmaps):
    """Sum each 2x2 quadrant of the 4x4 binarized map.

    This is an EQUIVARIANT coarse-graining: a rot90 of the image cyclically
    permutes the four quadrants and rotates within each one, and a 2x2 sum is
    invariant to that inner rotation. So the bijective relation between clean and
    rotated readouts survives intact (MI/H stays exactly 1.000), while the
    alphabet shrinks from 2^16 to a few hundred -- which is what makes the
    shuffled null meaningful. Truncating bits instead would ALSO shrink the
    alphabet, but rot90 permutes positions, so a bit-truncated readout is no
    longer bijectively related to its rotated counterpart and MI/H drops below 1
    for reasons that have nothing to do with equivariance."""
    m = binmaps.reshape(-1, SIDE, SIDE)
    return np.stack([m[:, :2, :2].sum((1, 2)), m[:, :2, 2:].sum((1, 2)),
                     m[:, 2:, 2:].sum((1, 2)), m[:, 2:, :2].sum((1, 2))], 1)


def _symbols(X):
    """Assign each distinct row its own integer symbol. Works for any readout,
    including real-valued ones, and keeps the computation exact: a bijective
    readout just relabels symbols, and a lossy one merges them."""
    _, ids = np.unique(X, axis=0, return_inverse=True)
    return ids.ravel()


def exact_entropy_bits(X):
    p = np.bincount(_symbols(X))
    p = p[p > 0] / len(X)
    return float(-(p * np.log2(p)).sum())


def exact_mi_bits(A, B):
    """Exact I(A;B) in bits from the full joint histogram. No estimation:
    the image set is exhaustive and every map is deterministic, so these counts
    are the true probabilities."""
    a, b = _symbols(A), _symbols(B)
    n = len(a)
    nb = b.max() + 1
    pa, pb = np.bincount(a) / n, np.bincount(b) / n
    cells, cnt = np.unique(a * nb + b, return_counts=True)
    pab = cnt / n
    return float((pab * np.log2(pab / (pa[cells // nb] * pb[cells % nb]))).sum())


# ---------------------------------------------------------------- geometric

def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    K, L = X @ X.T, Y @ Y.T
    h = lambda A, B: (A * B).sum()
    denom = np.sqrt(h(K, K) * h(L, L))
    return float(h(K, L) / denom) if denom > 0 else float("nan")


def rbf_cka(X, Y):
    """Note what this depends on: pairwise Euclidean distances, and nothing
    else. That fixes its invariance group at the isometries, exactly as for
    linear CKA -- the kernel choice buys sensitivity to nonlinear DEPENDENCE
    between X and Y, not invariance to nonlinear REPARAMETERIZATION of either."""
    def gram(Z):
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        s = np.sqrt(np.median(d2[d2 > 0]))
        return np.exp(-d2 / (2 * s ** 2))
    A, B = gram(X), gram(Y)
    n = len(A)
    H = np.eye(n) - 1.0 / n
    A, B = H @ A @ H, H @ B @ H
    h = lambda a, b: (a * b).sum()
    denom = np.sqrt(h(A, A) * h(B, B))
    return float(h(A, B) / denom) if denom > 0 else float("nan")


def svcca(X, Y, k=NBITS):
    """Mean top-k canonical correlation. CCA is invariant to ALL invertible
    linear maps, which is a strictly larger group than CKA's isometries -- so
    this is the metric that separates 'not an isometry' from 'not linear'."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(s[:k], 0, 1).mean())


# ---------------------------------------------------------------- readouts

def make_readouts(rng):
    """Readouts are built lazily per input width, so the same set applies to the
    full 16-column pattern and to the 4-column coarse-grained readout."""
    cache = {}

    def params(d):
        if d not in cache:
            cache[d] = {
                "perm": rng.permutation(d),
                # invertible but badly non-isometric: condition number 1e3
                "ill": (np.linalg.qr(rng.standard_normal((d, d)))[0]
                        @ np.diag(np.logspace(0, 3, d))
                        @ np.linalg.qr(rng.standard_normal((d, d)))[0]),
                "keep": np.sort(rng.choice(d, max(1, d // 2), replace=False)),
            }
        return cache[d]

    def identity(P):
        return P.astype(float)

    def bit_permute(P):
        return P[:, params(P.shape[1])["perm"]].astype(float)

    def ill_conditioned_linear(P):
        return P.astype(float) @ params(P.shape[1])["ill"]

    def alphabet_permute(P):
        """Relabel the realized alphabet by a random permutation, then re-encode
        in binary. Bijective for any input, and maximally nonlinear."""
        ids = _symbols(P)
        m = ids.max() + 1
        key = ("alpha", m)
        if key not in cache:
            cache[key] = rng.permutation(m)
        new = cache[key][ids]
        width = max(1, int(np.ceil(np.log2(m))))
        bits = (new[:, None] >> np.arange(width)) & 1
        return np.where(bits > 0, 1.0, -1.0)

    def lossy_half(P):
        return P[:, params(P.shape[1])["keep"]].astype(float)

    return [("identity", identity, "reference"),
            ("bit-permute", bit_permute, "bijective, isometry"),
            ("ill-cond. linear", ill_conditioned_linear, "bijective, linear, NOT isometry"),
            ("alphabet-permute", alphabet_permute, "bijective, NOT linear"),
            ("lossy-half", lossy_half, "MANY-TO-ONE: info destroyed")]


def main():
    rng = np.random.default_rng(SEED)
    images = all_binary_images()
    print(f"enumerated {len(images)} images ({SIDE}x{SIDE} over +-1) "
          f"-> distribution is exact, MI is computed not estimated\n")

    # sanity: vectorized circular conv matches scipy on a sample
    probe = images[:8].astype(float)
    for k in KERNELS.values():
        ref = np.array([convolve2d(im, k, mode="same", boundary="wrap") for im in probe])
        assert np.allclose(circ_conv(probe, k), ref), "circ_conv disagrees with scipy"

    rot = np.rot90(images, k=1, axes=(-2, -1))
    sub = rng.choice(len(images), CKA_SUBSAMPLE, replace=False)
    shuffle = rng.permutation(CKA_SUBSAMPLE)
    readouts = make_readouts(rng)

    for kname, kernel in KERNELS.items():
        equivariant = np.allclose(kernel, np.rot90(kernel))
        fa, fb = circ_conv(images.astype(float), kernel), circ_conv(rot.astype(float), kernel)
        thr = np.median(np.concatenate([fa.ravel(), fb.ravel()]))
        A, B = binarize(fa, thr), binarize(fb, thr)

        print("=" * 104)
        print(f"kernel: {kname}   rot90-symmetric={equivariant}  "
              f"-> representation {'EXACTLY equivariant' if equivariant else 'NOT equivariant'}")
        print("=" * 104)

        mi_shuf_full = rng.permutation(len(A))
        for gname, grain in [("FULL 16-bit pattern", lambda P: P),
                             ("quadrant coarse-grain", quadrant_coarse_grain)]:
            Ag, Bg = grain(A), grain(B)
            nsym = len(np.unique(Ag, axis=0))
            print(f"\n  granularity: {gname}   "
                  f"symbols={nsym}  samples/symbol={len(A) / nsym:.1f}  "
                  f"H={exact_entropy_bits(Ag):.4f} bits"
                  f"{'   <- UNDERSAMPLED, null is invalid' if len(A) / nsym < 10 else ''}")
            print(f"  {'readout':<18s} {'nature':<32s} {'MI':>8s} {'MIshuf':>7s} "
                  f"{'margin':>7s} {'MI/H':>6s} {'linCKA':>7s} {'rbfCKA':>7s} "
                  f"{'rbfshuf':>8s} {'SVCCA':>7s}")
            mis = {}
            for rname, sigma, nature in readouts:
                Za, Zb = sigma(Ag), sigma(Bg)
                mi = exact_mi_bits(Za, Zb)
                mi_s = exact_mi_bits(Za, Zb[mi_shuf_full])
                h = exact_entropy_bits(Za)
                mis[rname] = mi
                Xs, Ys = Za[sub], Zb[sub]
                print(f"  {rname:<18s} {nature:<32s} {mi:8.4f} {mi_s:7.4f} "
                      f"{mi - mi_s:7.4f} {mi / h if h > 0 else 0:6.3f} "
                      f"{linear_cka(Xs, Ys):7.4f} {rbf_cka(Xs, Ys):7.4f} "
                      f"{rbf_cka(Xs, Ys[shuffle]):8.4f} {svcca(Xs, Ys):7.4f}")
            print(f"    invariance check:  MI(identity) - MI(alphabet-permute) = "
                  f"{mis['identity'] - mis['alphabet-permute']:+.2e}")
        print()


if __name__ == "__main__":
    main()
