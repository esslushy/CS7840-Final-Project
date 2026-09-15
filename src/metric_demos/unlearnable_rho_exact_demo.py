"""
LEE vs linear CKA vs mutual information -- three nested invariance groups, measured
exactly.

This is escnn_experiments/unlearnable_rho_demo.py redone on a discrete space. That
script asked whether an adversarial change of READOUT COORDINATES can hide exact
equivariance, and answered with KSG MI on 96-d pooled escnn features. The answer was
right but the evidence was soft: MI decayed 4.54 -> 1.64 nats under the bijective
TWIST readout and the write-up had to argue that the decay was estimator artifact,
since I(sigma(X); sigma(Y)) = I(X; Y) holds exactly for any bijective sigma. Here the
same experiment runs on the exhaustive 2^16 binary 4x4 space from discretized.ipynb
(shared with exact_mi_demo.py and energy_readout_demo.py), where the joint pmf is
known EXACTLY. So the argument is unnecessary: the decay is 0.00e+00, printed, and
the invariance claim stops being a trend and becomes an identity.

Three metrics, one per family, as requested by the comparison rather than for
coverage:

    LEE       family A, HARDCODES rho(g)   Gruver et al. 2023 (arXiv:2210.02984)
    lin. CKA  family C, assumes no rho(g)  Kornblith et al. 2019
    exact MI  no rho(g), no geometry       plug-in on the true joint pmf

LEE here is the FINITE-GROUP form, ||f(gx) - rho(g)f(x)|| / ||rho(g)f(x)||. Gruver
et al. measure the Lie derivative, the theta -> 0 limit of that quantity over theta;
C4 is discrete, so there is no derivative to take and the finite difference at the
generator is the whole content of the metric. What transfers exactly -- and is the
only property under test -- is that LEE is scored against a rho(g) fixed in advance.
Both variants are reported: LEE_rho with the TRUE rho(g) handed to it (the strongest
possible form of family A) and LEE_I with the usual fallback rho = I for a readout
carrying no declared field type.

THE MODEL is a circular convolution on the 4x4 torus, binarized at a shared median,
then quadrant coarse-grained. With the rot90-symmetric Laplacian kernel it is
EXACTLY C4-equivariant and rho(g) is the cyclic quadrant shift -- verified in-script
at LEE_rho = 0.00e+00, not approximately. Sobel-x is the non-equivariant control.

THE READOUTS are relabelings of the 486-symbol alphabet, sigma(z) = codebook[pi[z]]:

    identity        pi = id (reference)
    coord-permute   permutes the 4 quadrant coordinates. An ISOMETRY, so CKA is
                    blind to it by construction -- the row that isolates LEE.
    scramble p      permutes a fraction p of the alphabet. BIJECTIVE at every p, so
                    information is exactly preserved. p=0 is the identity and p=1 is
                    a full random relabeling: the discrete analogue of TWIST, with a
                    knob in place of TWIST's omega.
    merge q         collapses the alphabet to a fraction q of its symbols.
                    MANY-TO-ONE, so information is genuinely destroyed. The analogue
                    of COSINE.

Keeping every readout inside the codebook is what makes this a controlled
experiment: all rows live in the same 4-d space with the same marginal value
distribution, and the ONLY structural difference between the scramble and merge
families is injectivity. `mass` reports the fraction of samples the readout actually
relabels, so the two sweeps can be compared at matched disruption.

MEASURED (Laplacian, exactly equivariant; alphabet 486 symbols at 134.8
samples/symbol, H = 7.3205 bits):

  readout           ok  mass  LEE_rho   LEE_I  LEEnull | linCKA    null |     MI    null  MI/H
  identity         yes  0.00   0.0000  1.4786  1.3998  | 1.0000  0.0004 | 7.3205  0.8100 1.000
  coord-permute    yes  0.68   1.2554  1.4786  1.3993  | 1.0000  0.0004 | 7.3205  0.8100 1.000
  scramble p=0.10  yes  0.09   0.7710  1.4435  1.3970  | 0.5170  0.0005 | 7.3205  0.8100 1.000
  scramble p=0.25  yes  0.34   1.1351  1.3672  1.3862  | 0.1735  0.0012 | 7.3205  0.8100 1.000
  scramble p=0.50  yes  0.53   1.3469  1.3077  1.3933  | 0.0453  0.0022 | 7.3205  0.8100 1.000
  scramble p=0.75  yes  0.73   1.4182  1.3450  1.3932  | 0.0392  0.0003 | 7.3205  0.8100 1.000
  scramble p=1.00  yes  1.00   1.4175  1.3608  1.3923  | 0.0376  0.0009 | 7.3205  0.8100 1.000
  merge q=0.50      NO  1.00   1.4116  1.3486  1.3802  | 0.0380  0.0010 | 5.8681  0.2992 0.890
  merge q=0.25      NO  1.00   1.3694  1.3168  1.3794  | 0.0303  0.0010 | 5.0613  0.1478 0.818
  merge q=0.10      NO  1.00   1.3843  1.3564  1.3925  | 0.0431  0.0006 | 3.1574  0.0267 0.606

FINDINGS.

1. LEE'S INVARIANCE GROUP IS EMPTY -- IT SURVIVES NOTHING. Given the exact true
   rho(g) it scores 0.0000 in natural coordinates, and 1.2554 after nothing worse
   than permuting the four coordinates. Its own shuffled null is 1.3993, so at that
   point LEE cannot distinguish a provably exactly-equivariant representation in
   permuted coordinates from destroyed pairing. Both CKA (1.0000 -> 1.0000) and MI
   (7.3205 -> 7.3205) read the isometry as no change at all. Note also LEE_I =
   1.4786 in the natural coordinates: the standard fallback of assuming invariance
   is wrong before any adversarial readout is applied, because the readout is
   equivariant, not invariant.

2. CKA'S GROUP IS THE ISOMETRIES, AND THE SCRAMBLE SWEEP WALKS IT OUT. 1.0000 ->
   0.5170 -> 0.1735 -> 0.0453 -> 0.0392 -> 0.0376 as p goes 0 -> 1. Relabeling just
   9% of the probability mass already halves it; by p=0.5 it is at 0.0453 against a
   0.0022 null and has effectively given up. Exact MI is 7.3205 on every one of
   those rows, to the last bit.

3. THE MI INVARIANCE IS AN IDENTITY, NOT A MEASUREMENT.
   MI(identity) - MI(scramble p=1.00) = +0.00e+00. MI is a sum over the joint pmf,
   and a bijection only relabels its terms, so this is exact arithmetic rather than
   a robust estimate -- which is exactly what the KSG version of this experiment
   could not show. Any estimator bias is identical on the two sides and cancels.

4. THE CONFLATION, WHICH IS THE POINT. Two readouts that CKA scores the same:

       scramble p=1.00  lossless bijection          linCKA 0.0376   MI 7.3205 (MI/H 1.000)
       merge    q=0.10  90% of the alphabet gone    linCKA 0.0431   MI 3.1574 (MI/H 0.606)

   A 0.0056 CKA gap, both sitting on a ~0.0009 null -- and the sign is inverted:
   CKA ranks the information-DESTROYING readout slightly above the lossless one.
   MI separates them by 4.16 bits and on an interpretable scale. One of these
   situations hides recoverable equivariance and the other destroys it; that
   difference is the whole question, and CKA has no access to it.

5. MI IS GRADED WHERE IT SHOULD BE, NOT BLINDLY INVARIANT. Across the merge sweep it
   falls 7.3205 -> 5.8681 -> 5.0613 -> 3.1574 bits (MI/H 1.000 -> 0.890 -> 0.818 ->
   0.606), tracking how much of the alphabet was actually destroyed.

6. THE CONTROL CONFIRMS THIS IS DETECTION, NOT STRUCTURE. On Sobel-x, which is
   genuinely not equivariant, MI is still exactly invariant to every bijective
   readout (+0.00e+00) but invariant to a LOW value: 1.2676 bits, MI/H 0.195, versus
   1.000 for the Laplacian. LEE sits at 1.3095 against a 1.3284 null and linear CKA
   at 0.0021 against 0.0004 in every readout. So MI's constancy above is invariance
   to coordinates, not indifference to equivariance.

   The hierarchy the rows pin down, from smallest invariance group to largest:

       LEE (hardcoded rho)   exactly one coordinate system
       linear CKA            isometries + isotropic scale
       exact MI              ALL bijections

CAVEATS.

  * The MI granularity matters and is not free. At the FULL 16-bit pattern
    granularity the alphabet has ~45707 symbols for 65536 samples, the plug-in null
    rises to 14.37 against a signal of 15.18, and the kernel ranking inverts --
    see the trap section in exact_mi_demo.py. The quadrant coarse-graining used here
    is equivariant (rot90 permutes quadrants and a 2x2 sum is blind to the rotation
    inside each one), so it shrinks the alphabet to 486 symbols at 134.8
    samples/symbol without breaking the bijection, and the null falls to 0.8100.
    Every MI number above is reported against that null.

  * LEE here is the finite-difference form, not a Lie derivative; a discrete group
    has no derivative to take. The claim being tested is about hardcoding rho(g),
    which is common to both forms, and not about the limit.

  * The merge readout destroys information AND scrambles geometry, by design -- it is
    the same family of map as scramble, differing only in injectivity, which is what
    makes the pair a controlled comparison. It does not isolate "lossy readout in
    good coordinates"; energy_readout_demo.py covers the injective-but-nonlinear
    corner of that space.

  * CKA's decay under scramble is not monotone below ~0.05 (0.0453, 0.0392, 0.0376,
    and the merge rows wander between 0.0303 and 0.0431). Those rows are all at the
    null and the orderings among them carry no information -- which is the reason
    finding 4 leans on the MI gap rather than on the CKA sign.

Run: cd src/metric_demos && ../../.venv/bin/python3 unlearnable_rho_exact_demo.py
"""
import numpy as np

from exact_mi_demo import (KERNELS, all_binary_images, binarize, circ_conv,
                           exact_entropy_bits, exact_mi_bits, linear_cka,
                           quadrant_coarse_grain)

SEED = 0
CKA_SUBSAMPLE = 4096      # linear CKA needs an n x n Gram; exact MI uses all 65536
RHO = np.array([1, 2, 3, 0])   # rho(g) on quadrant coords; verified exact in main()


# --------------------------------------------------------------- the alphabet

def shared_alphabet(A, B):
    """One codebook and one symbol numbering for BOTH branches.

    Every readout below is a relabeling of this alphabet, so the two branches
    must be numbered together -- otherwise "the same readout" would mean two
    different maps on the clean and rotated sides and the comparison would be
    meaningless."""
    codebook, ids = np.unique(np.vstack([A, B]), axis=0, return_inverse=True)
    ids = ids.ravel()
    n = len(A)
    return codebook.astype(float), ids[:n], ids[n:]


# ---------------------------------------------------------------- the readouts

def make_readouts(rng, m):
    """Readouts as permutations/merges of the m symbols, expressed as an index
    map pi of length m. sigma(z) = codebook[pi[symbol(z)]].

    Staying inside the codebook keeps every readout on the same 4-d space with
    the same marginal value distribution, so nothing varies across rows except
    WHICH vector each symbol is reported as. pi injective => information exactly
    preserved; pi non-injective => information genuinely destroyed. That is the
    only structural difference between the SCRAMBLE and MERGE families, which is
    what makes them the right pair to test a metric against."""
    order = rng.permutation(m)      # nesting order, fixed so the sweeps are monotone
    keys = rng.random(m)            # fixed random keys -> a fixed permutation of any subset
    group = rng.integers(0, m, size=m)

    def scramble(p):
        """Bijective, maximally nonlinear. Discrete analogue of TWIST: permutes
        a fraction p of the alphabet, so p=0 is the identity readout and p=1 is
        a full random relabeling. Lossless at every p."""
        pi = np.arange(m)
        k = int(round(p * m))
        if k > 1:
            S = np.sort(order[:k])
            pi[S] = S[np.argsort(keys[S])]
        return pi

    def merge(q):
        """Many-to-one: collapses the alphabet to a fraction q of its symbols.
        Discrete analogue of COSINE -- information really is destroyed, and by a
        map of exactly the same kind as scramble(), just not injective."""
        n_keep = max(1, int(round(q * m)))
        reps = order[:n_keep]
        return reps[group % n_keep]

    # coord-permute is the one readout that acts on COORDINATES rather than on
    # symbols: an isometry of the readout space, which CKA is blind to by
    # construction and LEE is not. It is what isolates LEE's invariance group.
    rows = [("identity (reference)", "reference", scramble(0.0), None),
            ("coord-permute", "bijective, ISOMETRY", None, rng.permutation(4))]
    for p in [0.1, 0.25, 0.5, 0.75, 1.0]:
        rows.append((f"scramble p={p:.2f}", "bijective, NOT linear", scramble(p), None))
    for q in [0.5, 0.25, 0.1]:
        rows.append((f"merge q={q:.2f}", "MANY-TO-ONE: info destroyed", merge(q), None))
    return rows


def apply_readout(codebook, ids, pi, coord_perm):
    if pi is None:
        return codebook[ids][:, coord_perm]
    return codebook[pi[ids]]


# ------------------------------------------------------------------ the metrics

def lee(X, Y, rho):
    """LEE, finite-group form: the relative equivariance error

        ||f(gx) - rho(g) f(x)|| / ||rho(g) f(x)||

    with rho(g) HARDCODED. Gruver et al. (arXiv:2210.02984) measure the Lie
    derivative, the theta -> 0 limit of this quantity divided by theta; C4 is
    discrete so there is no derivative to take and the finite difference at the
    generator is the whole content of the metric. What matters here is the part
    that carries over exactly: LEE is evaluated against a rho(g) supplied in
    advance, so it is a statement about one specific coordinate system.

    rho=None means the usual fallback for a readout with no declared field type:
    assume the trivial representation, i.e. that the readout is invariant.
    Lower is better."""
    R = X if rho is None else X[:, rho]      # rho(g) f(x)
    return float(np.linalg.norm(Y - R) / (np.linalg.norm(R) + 1e-12))


def summarize(got, equivariant):
    """The three-way comparison, read off the rows just printed."""
    ref, iso = got["identity (reference)"], got["coord-permute"]
    hidden, lost = got["scramble p=1.00"], got["merge q=0.10"]
    if not equivariant:
        print(f"\n  control check: nothing to detect here, and every metric says so in every "
              f"readout --\n    LEE {ref['lee_rho']:.4f} against its {ref['lee_null']:.4f} null, "
              f"linCKA {ref['cka']:.4f} against {ref['cka_null']:.4f}.")
        print(f"    MI is still exactly invariant to the bijective readouts "
              f"({ref['mi'] - hidden['mi']:+.2e}), but it is invariant to a LOW value: "
              f"MI/H {ref['mih']:.3f}\n    here versus 1.000 for the equivariant kernel. "
              f"So MI's constancy above is invariance, not blindness.")
        return
    print("\n  invariance group, measured:")
    print(f"    LEE (hardcoded rho)  nothing beyond its own coordinates: exact "
          f"{ref['lee_rho']:.4f} in natural coords, {iso['lee_rho']:.4f} after a mere "
          f"ISOMETRY\n{'':25s}(its shuffled null is {iso['lee_null']:.4f} -- LEE cannot tell "
          f"the isometric readout from destroyed pairing)")
    print(f"    linear CKA           isometries: {ref['cka']:.4f} -> {iso['cka']:.4f} "
          f"unchanged, but {hidden['cka']:.4f} once the relabeling is nonlinear")
    print(f"    exact MI             ALL bijections: identity - scramble(p=1) = "
          f"{ref['mi'] - hidden['mi']:+.2e} exactly")
    print("\n  what CKA conflates and MI separates:")
    print(f"    scramble p=1.00  LOSSLESS bijection      linCKA {hidden['cka']:.4f}  "
          f"MI {hidden['mi']:.4f}  MI/H {hidden['mih']:.3f}")
    print(f"    merge    q=0.10  90% of alphabet DESTROYED linCKA {lost['cka']:.4f}  "
          f"MI {lost['mi']:.4f}  MI/H {lost['mih']:.3f}")
    if lost['cka'] >= hidden['cka']:
        print(f"    -> CKA ranks the LOSSY readout ABOVE the lossless one "
              f"({lost['cka']:.4f} > {hidden['cka']:.4f})")
    print(f"    CKA gap {abs(hidden['cka'] - lost['cka']):.4f} (both at their ~"
          f"{max(hidden['cka_null'], lost['cka_null']):.4f} null);  "
          f"MI gap {hidden['mi'] - lost['mi']:.4f} bits")


def main():
    rng = np.random.default_rng(SEED)
    images = all_binary_images()
    rot = np.rot90(images, k=1, axes=(-2, -1))
    n = len(images)
    mi_shuffle = rng.permutation(n)
    sub = rng.choice(n, CKA_SUBSAMPLE, replace=False)
    cka_shuffle = rng.permutation(CKA_SUBSAMPLE)

    print(f"enumerated {n} images (4x4 over +-1) -> the joint distribution is "
          f"exact and MI is computed, not estimated\n")

    for kname, kernel in KERNELS.items():
        equivariant = np.allclose(kernel, np.rot90(kernel))
        fa = circ_conv(images.astype(float), kernel)
        fb = circ_conv(rot.astype(float), kernel)
        thr = np.median(np.concatenate([fa.ravel(), fb.ravel()]))
        A = quadrant_coarse_grain(binarize(fa, thr)).astype(float)
        B = quadrant_coarse_grain(binarize(fb, thr)).astype(float)

        codebook, ida, idb = shared_alphabet(A, B)
        m = len(codebook)
        print("=" * 118)
        print(f"kernel: {kname}   rot90-symmetric={equivariant}  -> representation "
              f"{'EXACTLY equivariant' if equivariant else 'NOT equivariant (control)'}")
        print(f"  alphabet {m} symbols, {n / m:.1f} samples/symbol, "
              f"H = {exact_entropy_bits(A):.4f} bits"
              f"{'' if n / m >= 10 else '   <- UNDERSAMPLED'}")
        if equivariant:
            print(f"  ground truth: ||f(gx) - rho(g)f(x)|| / ||rho(g)f(x)|| = "
                  f"{lee(A, B, RHO):.2e} with the true rho(g) = quadrant shift "
                  f"-> exactly equivariant")
        print("=" * 118)
        print(f"  {'readout':<20s} {'nature':<28s} {'ok':>3s} {'mass':>5s} "
              f"{'LEE_rho':>8s} {'LEE_I':>7s} {'LEE_null':>8s} | "
              f"{'linCKA':>7s} {'null':>7s} | {'MI':>7s} {'null':>7s} {'MI/H':>6s}")

        got = {}
        for rname, nature, pi, coord_perm in make_readouts(
                np.random.default_rng(SEED), m):
            Za = apply_readout(codebook, ida, pi, coord_perm)
            Zb = apply_readout(codebook, idb, pi, coord_perm)
            bijective = pi is None or len(np.unique(pi)) == m
            mass = float((Za != codebook[ida]).any(1).mean())
            h = exact_entropy_bits(Za)
            mi = exact_mi_bits(Za, Zb)
            mi_null = exact_mi_bits(Za, Zb[mi_shuffle])
            Xs, Ys = Za[sub], Zb[sub]
            r = dict(lee_rho=lee(Za, Zb, RHO), lee_i=lee(Za, Zb, None),
                     lee_null=lee(Za, Zb[mi_shuffle], RHO),
                     cka=linear_cka(Xs, Ys), cka_null=linear_cka(Xs, Ys[cka_shuffle]),
                     mi=mi, mi_null=mi_null, mih=mi / h if h > 0 else 0.0)
            got[rname] = r
            print(f"  {rname:<20s} {nature:<28s} {'yes' if bijective else 'NO':>3s} {mass:5.2f} "
                  f"{r['lee_rho']:8.4f} {r['lee_i']:7.4f} {r['lee_null']:8.4f} | "
                  f"{r['cka']:7.4f} {r['cka_null']:7.4f} | "
                  f"{r['mi']:7.4f} {r['mi_null']:7.4f} {r['mih']:6.3f}")
        summarize(got, equivariant)
        print()


if __name__ == "__main__":
    main()
