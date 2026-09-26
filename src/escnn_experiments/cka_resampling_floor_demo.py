"""
How much of CKA's off-angle reading is the ROTATION OPERATOR, not the model?

cn_angle_sweep_demo.py is this repo's one positive result for CKA: because it
needs no rho(g), theta can be swept continuously and a C_N model's error touches
exactly 0 at every multiple of 360/N and nowhere else, recovering the group
order from feature geometry alone. That result has an unexamined assumption.

THE ASSUMPTION. Rotating an image by a non-lattice angle is not a permutation
of pixels. It requires interpolation, which low-pass filters the image and
discards content at the corners. So rot(theta)x is not "x, rotated" -- it is a
rotated, resampled, slightly damaged x. Any metric comparing f(x) against
f(rot(theta)x) therefore sees two things at once:

    the model failing to be equivariant   +   the probe rotation destroying data

and a rising CKA error at off-lattice angles is attributed entirely to the
first. This script measures the second, so the attribution can be checked.

THE FLOOR, and why it is exactly the right quantity. Linear CKA is invariant to
any ORTHOGONAL map of the features. A quarter turn is a permutation of pixels,
hence orthogonal, so 1 - CKA(x, rot90(x)) is exactly 0 on raw pixels -- no model
involved. An interpolated rotation is NOT orthogonal: it is a smoothing operator
that loses rank. So

    input floor(theta) = 1 - CKA(x, rot(theta)x)          on RAW PIXELS

measures precisely how far the rotation operator departs from being an isometry,
with no network in the picture at all. It is zero where the lattice permits an
exact rotation and positive where it does not.

WHAT IS COMPARED. At each of a C16 model's own sixteen group elements -- angles
at which the ARCHITECTURE claims exact equivariance -- we report

    input floor   1 - CKA(x, rot(theta)x)             the operator alone
    model         1 - CKA(f(x), f(rot(theta)x))       operator + model
    ratio         model / floor

If the ratio is near 1, CKA's off-lattice reading is the resampling and not the
model, and the lobes in cn_angle_sweep_demo.py need a stated floor beneath them.
If the model sits well above the floor, the lobes are real and survive with the
floor reported. Part B then refines the grid: a quantity that is resampling
damage must converge to zero as the sampling gets finer, and one that is a
genuine property of the model must not.

This is deliberately self-critical. The confound it looks for, if present,
weakens the one experiment in this repo where CKA does something no other
metric can.

MEASURED (SEED=0, 128 MNIST images, C16 steerable conv). Both columns are
1 - linear CKA; the floor uses raw pixels and no model.

  Part A, at the C16 model's own group elements, 29x29 grid:

    theta                 on lattice?   input floor     model   model/floor
    0, 90, 180, 270 deg          yes        0.00000   0.00000       --
    22.5 deg and 3 repeats        no        0.00011   0.00032     2.83
    67.5 deg and 3 repeats        no        0.00011   0.00026     2.30
    45.0 deg and 3 repeats        no        0.00013   0.00054     4.10
    off-lattice mean                        0.00012   0.00037     3.08

  Part B, does either converge as the grid refines?

    grid     22.5 floor  22.5 model   45.0 floor  45.0 model   90 deg (both)
    29^2        0.00011     0.00032      0.00013     0.00054         0.00000
    57^2        0.00000     0.00004      0.00000     0.00005         0.00000
    113^2       0.00000     0.00000      0.00000     0.00000         0.00000
    225^2       0.00000     0.00000      0.00000     0.00000         0.00000

1. THE FLOOR IS EXACTLY ZERO ON THE LATTICE AND POSITIVE OFF IT, as the
   argument requires. A quarter turn is a permutation of pixels, hence
   orthogonal, and linear CKA is invariant to orthogonal maps, so the rotation
   operator is invisible there. An interpolated rotation is a smoothing
   operator, not an isometry, and shows up.

2. BUT THE CONFOUND IS NOT THERE AT A SCALE THAT MATTERS. The model's CKA error
   at off-lattice angles is 0.00037 and the resampling floor beneath it is
   0.00012 -- a ratio of about 3, so the convolution amplifies the input damage
   rather than merely inheriting it, but both quantities are minuscule. CKA is
   not losing resolution to lost data; it is barely registering that anything
   happened, which is the same blindness lie_vs_cka_aliasing_demo.py reports
   from the other direction.

3. ALL OF IT CONVERGES AWAY BY 113^2. Both the floor and the model error fall
   to zero as the sampling refines, so the entire off-lattice CKA signal at
   these angles is a sampling artifact and not a property of the model. Compare
   the TRUE equivariance error at the same angles, which is 0.13-0.16 and also
   converges -- the difference is that the true error is large enough to see.

4. WHAT THIS BUYS: A FLOOR FOR cn_angle_sweep_demo.py. That experiment sweeps
   theta continuously and finds C_N's error touching exactly 0 at multiples of
   360/N, with peaks of 0.04 (C16) to 0.51 (C1). Those peaks are 100x to 1000x
   the resampling floor measured here, so the group structure CKA recovers is
   genuine signal rather than interpolation artifact. This experiment was built
   to look for a confound in that result and did not find one at a scale that
   threatens it; the negative is reported because the floor had never been
   measured at the input, only at the model.

Run: cd src/escnn_experiments && .venv/bin/python3 cka_resampling_floor_demo.py
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from escnn import gspaces, nn as enn

DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")

SEED = 0
KERNEL_SIZE = 7
GROUP_N = 16                      # the model whose group elements we probe
MAX_FREQUENCY = 4                 # for the SO(2) gspace that supplies arbitrary theta
RESOLUTIONS = [29, 57, 113, 225]
BATCH = 128


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.t(), Y @ Y.t()
    hsic = lambda A, B: (A * B).sum()
    return (hsic(K, L) / torch.sqrt(hsic(K, K) * hsic(L, L))).item()


def build(size):
    """A C_GROUP_N steerable conv, plus an SO(2) gspace used only to produce
    arbitrary rotation elements for the INPUT."""
    torch.manual_seed(SEED)
    cn = gspaces.rot2dOnR2(N=GROUP_N)
    in_type = enn.FieldType(cn, [cn.trivial_repr])
    hid_type = enn.FieldType(cn, [cn.trivial_repr] * 2 + [cn.irrep(1)] * 2)
    conv = enn.R2Conv(in_type, hid_type, kernel_size=KERNEL_SIZE)
    conv.eval()

    so2_space = gspaces.rot2dOnR2(N=-1, maximum_frequency=MAX_FREQUENCY)
    so2_in = enn.FieldType(so2_space, [so2_space.trivial_repr])
    return in_type, hid_type, conv, so2_space.fibergroup, so2_in


def load_images(size):
    testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True,
                                         transform=transforms.ToTensor())
    imgs, _ = next(iter(torch.utils.data.DataLoader(testset, batch_size=BATCH,
                                                    shuffle=False)))
    if size == 29:
        return F.pad(imgs, (0, 1, 0, 1))          # 28 -> 29, odd grid
    return F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)


@torch.no_grad()
def measure(size, angles_deg):
    """For each angle: the input-only floor and the model's CKA error."""
    imgs = load_images(size)
    in_type, hid_type, conv, so2, so2_in = build(size)
    gx = enn.GeometricTensor(imgs, in_type)
    y = conv(gx).tensor
    x_flat = imgs.flatten(1)

    out = []
    for deg in angles_deg:
        elem = so2.element(math.radians(deg))
        # rotate the INPUT through the SO(2) gspace, so arbitrary theta is allowed
        x_rot = enn.GeometricTensor(imgs, so2_in).transform(elem).tensor
        floor = 1.0 - linear_cka(x_flat, x_rot.flatten(1))
        y_rot = conv(enn.GeometricTensor(x_rot, in_type)).tensor
        model = 1.0 - linear_cka(y.flatten(1), y_rot.flatten(1))
        out.append((deg, floor, model))
    return out


def main():
    angles = [360.0 * k / GROUP_N for k in range(GROUP_N)]
    lattice = lambda d: abs(d % 90.0) < 1e-9

    print(f"C{GROUP_N} steerable conv, {BATCH} MNIST images. Both columns are "
          f"1 - linear CKA.\n"
          f"The floor uses RAW PIXELS and no model: it is how far the rotation\n"
          f"operator itself departs from being an isometry.\n")

    print(f"=== Part A: the C{GROUP_N} model at its own group elements, 29x29 grid ===")
    print(f"  {'theta':>7}{'lattice':>9}{'input floor':>13}{'model':>10}"
          f"{'model/floor':>13}")
    rows = measure(29, angles)
    for deg, floor, model in rows:
        r = model / floor if floor > 1e-12 else float("nan")
        print(f"  {deg:>7.1f}{'yes' if lattice(deg) else 'no':>9}"
              f"{floor:>13.5f}{model:>10.5f}"
              f"{'' if floor <= 1e-12 else f'{r:>13.2f}'}")

    off = [(f, m) for d, f, m in rows if not lattice(d)]
    print(f"\n  off-lattice means: floor {sum(f for f, _ in off) / len(off):.5f}, "
          f"model {sum(m for _, m in off) / len(off):.5f}, "
          f"ratio {sum(m / f for f, m in off) / len(off):.2f}")

    print(f"\n=== Part B: does either converge as the grid refines? ===")
    probes = [22.5, 45.0, 90.0]
    print(f"  {'grid':>7}" + "".join(f"{f'{p} floor':>14}{f'{p} model':>14}"
                                     for p in probes))
    for size in RESOLUTIONS:
        r = {d: (f, m) for d, f, m in measure(size, probes)}
        print(f"  {size:>4}^2" + "".join(f"{r[p][0]:>14.5f}{r[p][1]:>14.5f}"
                                         for p in probes))
    print("\n  A quantity that is resampling damage must fall toward 0 as the\n"
          "  sampling gets finer. One that is a property of the model must not.")


if __name__ == "__main__":
    main()
