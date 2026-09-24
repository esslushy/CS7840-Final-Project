"""
Shuffled nulls for the rho-ASSUMING and rho-FITTING families.

Every detector in this folder reports a shuffled null except two: LEE with an
assumed rho, and the fitted maps of Lenc & Vedaldi and Procrustes. That is an
asymmetry rather than an oversight of principle -- families C and MI are held to
a standard that A and B are not -- and it leaves their numbers uninterpretable.
energy_readout_demo.py prints "--" in those columns. This script fills them in.

WHAT A NULL MEANS FOR A FITTED MAP. Shuffle the row order of the rotated branch,
so sample i is paired with the rotated features of an unrelated sample j, and
leave everything else identical: same marginals, same dimensionality, same
sample count, same fit/held-out split. Then run the metric unchanged. Whatever
it reports is what it reports when there is demonstrably nothing to find. For
family B the fit is performed on the shuffled data too, which is the point --
the question is not "how well does the true map generalize" but "how well can
this hypothesis class fit noise".

WHY EACH FAMILY'S NULL SITS WHERE IT DOES.

  A1 LEE, rho = I.  With no relationship, ||Y_shuf - X|| is the distance between
  two independent standardized clouds, so the ratio lands near sqrt(2) ~ 1.41.
  Any LEE value near 1.41 is therefore indistinguishable from noise -- which
  matters, because several LEE readings in this repo sit at 1.36-1.49.

  B1 affine fit.  With no relationship the best affine map predicts the mean of
  Y, so the held-out residual approaches 1.0 from below. It cannot exceed 1 by
  much, because predicting the mean is always available.

  B2 Procrustes.  Restricted to orthogonal maps, it cannot even shrink to the
  mean -- it must rotate X onto Y at full scale -- so the residual of two
  independent centered clouds is ~sqrt(2) rather than 1. Procrustes nulls above
  1 are normal and are not evidence of anything.

That last point matters for reading the main table: B2's 1.4511 on the control
is not "worse than useless", it is simply AT its null.

N_PERM permutations are drawn rather than one, so each null comes with a spread
and a reader can see whether a margin is meaningful.

MEASURED (SEED=0, N_PERM=20, exhaustive 2^16 space, quadrant coarse-grained).
Values reproduce energy_readout_demo.py exactly; only the null columns are new.

  Laplacian, EXACTLY equivariant          value     null      sd    margin
    identity  A1 LEE, rho=I              1.4933   1.4145  0.0009   -0.0789  AT NULL
              B1 affine fit              0.0000   1.0001  0.0001    1.0001  clears
              B2 Procrustes              0.0000   1.4874  0.0020    1.4874  clears
    ENERGY    A1 LEE, rho=I              1.3640   1.4140  0.0009    0.0500  clears
              B1 affine fit              0.6723   1.0002  0.0001    0.3279  clears
              B2 Procrustes              0.7225   1.4749  0.0018    0.7524  clears

  Sobel-x, NOT equivariant (control)
    identity  A1 LEE, rho=I              1.3959   1.4150  0.0020    0.0191  clears
              B1 affine fit              1.0836   1.0001  0.0001   -0.0835  AT NULL
              B2 Procrustes              1.4884   1.4641  0.0024   -0.0243  AT NULL
    ENERGY    A1 LEE, rho=I              1.3958   1.4139  0.0012    0.0182  clears
              B1 affine fit              1.0438   1.0002  0.0001   -0.0436  AT NULL
              B2 Procrustes              1.4511   1.4422  0.0022   -0.0089  AT NULL

1. LEE SCORES WORSE THAN ITS OWN NULL ON A PROVABLY EQUIVARIANT MODEL. In
   natural coordinates on the Laplacian -- exactly C4-equivariant, verified --
   LEE reads 1.4933 against a null of 1.4145. It is 0.0789 BELOW chance. Handed
   the easiest case in the repo it does worse than randomly paired data, because
   the representation it assumes (trivial) is not the one the features carry
   (quadrant shift).

2. LEE HAS ALMOST NO DYNAMIC RANGE. Across all four cells -- exactly
   equivariant and not, natural coordinates and energy -- every reading lies in
   [1.3640, 1.4933], a band of 0.13 centred on its own null of ~1.414. The null
   is sqrt(2) to three decimals, which is what two independent standardized
   clouds must give. Margins of 0.02 to 0.05 are many null-sd's wide and so
   "significant", but the quantity they separate is not equivariance: LEE reads
   1.3959 on the NON-equivariant control and 1.4933 on the exactly equivariant
   model, i.e. it ranks the control BETTER.

3. FAMILY B IS A WORKING DETECTOR, AND THE NULLS ARE WHAT SHOW IT. On the
   equivariant kernel it clears its null by 1.00/1.49 (identity) and 0.33/0.75
   (energy); on the control it sits AT null in all four cells. So the energy
   readout degrades B without breaking it, and the earlier claim that B "fails"
   there was too strong -- it reports a large residual on a lossless
   representation, which is a failure of calibration, not of detection.

4. B2's 1.4884 ON THE CONTROL IS NOT A CATASTROPHIC SCORE, IT IS ITS NULL
   (1.4641). Procrustes is restricted to orthogonal maps and so cannot shrink
   towards the mean the way an affine fit can; the residual between two
   independent centred clouds is therefore ~sqrt(2), not 1. Procrustes values
   above 1 are normal and carry no information on their own.

Run: cd src/metric_demos && ../../.venv/bin/python3 ab_nulls_demo.py
"""
import numpy as np

from exact_mi_demo import (KERNELS, all_binary_images, binarize, circ_conv,
                           quadrant_coarse_grain)
from energy_readout_demo import (CKA_SUBSAMPLE, affine_fit, lee_assumed_identity,
                                 make_energy_readout, procrustes_fit, standardize)

SEED = 0
N_PERM = 20

METRICS = [("A1  LEE, rho=I   ", lee_assumed_identity),
           ("B1  affine fit   ", affine_fit),
           ("B2  Procrustes   ", procrustes_fit)]


def null_distribution(metric, X, Y, rng, n_perm=N_PERM):
    """The metric applied to X against a row-shuffled Y, n_perm times. For the
    fitted maps this refits on the shuffled data, so it measures how well the
    hypothesis class fits noise."""
    vals = [metric(X, Y[rng.permutation(len(Y))]) for _ in range(n_perm)]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    rng = np.random.default_rng(SEED)
    images = all_binary_images()
    rot = np.rot90(images, k=1, axes=(-2, -1))
    # Consume the RNG exactly as energy_readout_demo.main does, so sigma's M and
    # N are the same matrices and these nulls attach to the published values.
    _ = rng.permutation(len(images))
    _ = rng.choice(len(images), CKA_SUBSAMPLE, replace=False)
    _ = rng.permutation(CKA_SUBSAMPLE)
    sigma = make_energy_readout(rng, 4)

    print(f"shuffled nulls for the rho-assuming and rho-fitting families, "
          f"{N_PERM} permutations each")
    print("lower is better for all three; a value AT its null carries no "
          "information\n")

    for kname, kernel in KERNELS.items():
        equivariant = np.allclose(kernel, np.rot90(kernel))
        fa = circ_conv(images.astype(float), kernel)
        fb = circ_conv(rot.astype(float), kernel)
        thr = np.median(np.concatenate([fa.ravel(), fb.ravel()]))
        A = quadrant_coarse_grain(binarize(fa, thr)).astype(float)
        B = quadrant_coarse_grain(binarize(fb, thr)).astype(float)

        print("=" * 84)
        print(f"{kname}  ->  {'EXACTLY equivariant' if equivariant else 'NOT equivariant (control)'}")
        print("=" * 84)

        for rname, X, Y in [("identity readout", standardize(A), standardize(B)),
                            ("ENERGY readout  ", sigma(A), sigma(B))]:
            print(f"\n  {rname}")
            print(f"  {'metric':<20}{'value':>9}{'null':>9}{'sd':>8}"
                  f"{'margin':>9}   verdict")
            for label, fn in METRICS:
                v = fn(X, Y)
                mu, sd = null_distribution(fn, X, Y, rng)
                margin = mu - v          # lower-is-better: distance below null
                verdict = ("AT NULL" if margin < 3 * max(sd, 1e-6)
                           else "clears null")
                print(f"  {label:<20}{v:>9.4f}{mu:>9.4f}{sd:>8.4f}"
                      f"{margin:>9.4f}   {verdict}")


if __name__ == "__main__":
    main()
