"""
Fourth companion demo: equivariance genuinely LOST to aliasing at export.

The first three demos are all about a *measurement* problem: the equivariance
error was overstated because the metric assumed the wrong representation, while
CKA (which assumes none) was unaffected. This one is the control for that
claim -- here equivariance is really destroyed, so a good metric SHOULD report
it, and we check whether each metric does.

Setup: the same escnn R2Conv that is exactly SO(2)-equivariant by construction
(see lie_vs_cka_escnn_exact_demo.py). We then "export" its feature field at
reduced resolution three ways:

    full            no downsampling (reference)
    subsample       naive stride-2 subsampling
    blur_subsample  Gaussian low-pass, THEN stride-2 subsampling

Naive subsampling discards everything above the new Nyquist frequency by
folding it back onto lower frequencies -- aliasing. Gruver et al. (Sec. 2,
"Aliasing and equivariance") identify exactly this as a primary source of
equivariance violation in real CNNs, and the anti-aliased blur-then-subsample
fix is Zhang (2019)'s BlurPool, which they cite.

Two angle regimes are measured, and the contrast between them is the point:

  * GRID-ALIGNED angles (90, 180 deg) on an odd-sized grid. A quarter turn is a
    pure index permutation, and on an odd grid the even-index subsampling
    lattice maps exactly onto itself, so downsampling *commutes exactly* with
    the rotation. Error stays 0.00000 no matter how much aliasing there is --
    the aliasing is completely invisible to a C4-only probe.

  * GENERIC angles. Here the subsampling lattice is not preserved, aliasing
    bites, and naive subsampling inflates the equivariance error well above the
    full-resolution baseline -- which low-pass filtering then largely recovers.

That contrast matters for this repo specifically: every equivariance
measurement here (EquivarianceTracker in utils.py, and the ANGLES=(90,180,270)
loops in classification.py / gradient_field.py) uses only discrete rot90
rotations. This demo shows that is precisely the regime in which
downsampling-induced aliasing cannot be detected at all.

MEASURED, at theta = 2.0 rad (114.6 deg, off-lattice), 256 MNIST images:

  condition                      err       cka   cka null
  grid-aligned (theta = pi/2)  0.00000   1.0000     0.0729
  full resolution              0.13429   0.9994     0.0722
  subsample                    0.21735   0.9995     0.0727
  blur, then subsample         0.15250   0.9989     0.0432

FINDING. Naive subsampling inflates the representation-aware equivariance error
from 0.13 to 0.22, and blur-then-subsample brings it back to 0.15 -- a loss
identified by mechanism and then repaired. Linear CKA reads ~0.999 through all
three and does not register either the damage or the fix.

AND IT IS SATURATED, NOT VACUOUS -- which is why the null column is here. A
reading of 0.999 could mean either "these really are nearly identical" or "this
metric returns 0.999 on anything", the second being the failure mode RBF CKA
has at small bandwidth (unlearnable_rho_demo.py), where signal and shuffled
control both hit 1.0000. Row-shuffling the rotated branch here gives ~0.07, so
CKA sits about 14x above its own floor and the readings are real. The precise
failure is therefore one of RANGE rather than of validity: CKA spans 1.0000 to
0.9980 across the whole experiment while the quantity it tracks runs from
exactly 0 to 0.236. Note also that the null is lower for blur_subsample (0.043
vs 0.072): blurring reduces the effective rank of the features, and a
lower-rank Gram matrix has a lower chance agreement.

So this is the counterpoint to the other three demos, and the reason the two
metrics are complementary rather than one dominating the other:

    representation-aware equivariance error  catches aliasing, but is only as
                                             right as its assumed rho(g)
    linear CKA                               needs no rho(g) and so is never
                                             wrong about it, but is blind to
                                             aliasing-induced equivariance loss

CKA is a population-level statistic over a batch's similarity geometry. Aliasing
perturbs each sample's features without much disturbing which images resemble
which, so the Gram matrices -- and hence CKA -- survive nearly intact even
though genuine equivariance has degraded.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from escnn import gspaces, nn as enn

# Shared with the rest of the repo: datasets live in src/data, one level up.
DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")


SEED = 0
KERNEL_SIZE = 7
MAX_FREQUENCY = 4
BLUR_SIGMA = 1.2
BLUR_KSIZE = 7
GRID_ALIGNED = [np.pi / 2, np.pi]
GENERIC = [0.3, np.pi / 4, 0.7, 1.0, 2.0]
EXPORTS = ["full", "subsample", "blur_subsample"]


def gaussian_blur(t, sigma=BLUR_SIGMA, ksize=BLUR_KSIZE):
    ax = torch.arange(ksize).float() - ksize // 2
    k1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1).view(1, 1, ksize, ksize).expand(t.shape[1], 1, ksize, ksize)
    return F.conv2d(t, k2, padding=ksize // 2, groups=t.shape[1])


def export(t, mode):
    """Read the feature field out at reduced resolution."""
    if mode == "full":
        return t
    if mode == "subsample":
        return t[:, :, ::2, ::2]
    if mode == "blur_subsample":
        return gaussian_blur(t)[:, :, ::2, ::2]
    raise ValueError(mode)


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.t(), Y @ Y.t()
    hsic = lambda A, B: (A * B).sum()
    return (hsic(K, L) / torch.sqrt(hsic(K, K) * hsic(L, L))).item()


def main():
    torch.manual_seed(SEED)
    gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=MAX_FREQUENCY)
    so2 = gspace.fibergroup
    in_type = enn.FieldType(gspace, [gspace.trivial_repr])
    hid_type = enn.FieldType(gspace, [gspace.trivial_repr] * 2
                                      + [gspace.irrep(1)] * 2
                                      + [gspace.irrep(2)])
    conv = enn.R2Conv(in_type, hid_type, kernel_size=KERNEL_SIZE)
    conv.eval()

    testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True,
                                          transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    err = {(m, th): [] for m in EXPORTS for th in GRID_ALIGNED + GENERIC}
    cka = {(m, th): [] for m in EXPORTS for th in GRID_ALIGNED + GENERIC}
    # Shuffled control. This is the one experiment whose conclusion is "CKA does
    # not detect X", so it needs a floor: row-shuffling the rotated branch
    # destroys the pairing while leaving marginals, dimensionality and sample
    # count identical, and an honest detector must collapse to ~0 there.
    nul = {(m, th): [] for m in EXPORTS for th in GRID_ALIGNED + GENERIC}
    shuffle = None

    with torch.no_grad():
        for imgs, _ in loader:
            # pad 28 -> 29 so the conv output grid is odd: the even-index
            # subsampling lattice is then preserved by a quarter turn.
            imgs = F.pad(imgs, (0, 1, 0, 1))
            gx = enn.GeometricTensor(imgs, in_type)
            y = conv(gx)

            for th in GRID_ALIGNED + GENERIC:
                elem = so2.element(th)
                y_rot_in = conv(gx.transform(elem))  # model(rotate(x))
                for mode in EXPORTS:
                    lhs = export(y_rot_in.tensor, mode)
                    rhs = enn.GeometricTensor(export(y.tensor, mode),
                                              hid_type).transform(elem).tensor
                    err[(mode, th)].append(((lhs - rhs).norm() / rhs.norm()).item())
                    X = export(y.tensor, mode).flatten(1)
                    Y = lhs.flatten(1)
                    cka[(mode, th)].append(linear_cka(X, Y))
                    if shuffle is None or len(shuffle) != len(Y):
                        shuffle = torch.randperm(len(Y))
                    nul[(mode, th)].append(linear_cka(X, Y[shuffle]))

    mean = lambda v: sum(v) / len(v)
    for label, angles in [("GRID-ALIGNED (quarter turns)", GRID_ALIGNED),
                          ("GENERIC angles", GENERIC)]:
        print(f"\n=== {label} ===")
        print(f"{'theta':>8s} " + " ".join(f"{m:>24s}" for m in EXPORTS))
        print(f"{'':>8s} " + " ".join(f"{'err / cka / cka null':>24s}"
                                      for _ in EXPORTS))
        for th in angles:
            cells = " ".join(
                f"{mean(err[(m, th)]):7.5f} /{mean(cka[(m, th)]):7.4f} /{mean(nul[(m, th)]):7.4f}"
                for m in EXPORTS)
            print(f"{th:8.4f} {cells}")


if __name__ == "__main__":
    main()
