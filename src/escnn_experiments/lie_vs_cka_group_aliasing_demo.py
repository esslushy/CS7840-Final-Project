"""
Fifth demo: aliasing of the ROTATION GROUP itself.

lie_vs_cka_aliasing_demo.py subsampled the *spatial* grid. This one subsamples
the *group*: take the infinite rotation group SO(2) and discretize it to a
finite cyclic group C_N (r = 4, 8, 16, ...), which is what escnn's
gspaces.rot2dOnR2(N) does. A C_N-equivariant R2Conv carries an architectural
guarantee of exact equivariance to all N of its rotations.

On a square pixel grid that guarantee is only partly deliverable. A rotation by
a multiple of 90 degrees is a lattice symmetry -- a pure index permutation, no
resampling, no information lost. Every other rotation (45, 22.5, 67.5, ...)
has no exact representation on the grid: realizing it requires interpolation,
which aliases, and the guarantee silently fails.

So refining the group buys you nothing beyond C4:

    C4    4 of 4  group elements are lattice-exact   (100%)
    C8    4 of 8                                      (50%)
    C16   4 of 16                                     (25%)

The C4 subgroup is the only part of SO(2) a square grid can honor exactly, no
matter how finely the group is subsampled. Part B confirms the residual really
is aliasing rather than anything structural: it decays roughly by half per
doubling of the spatial sampling rate (vanishing in the continuum limit), while
the 90-degree error stays at exactly 0.00000 at every resolution.

This is also why the r=4 case looked "perfect" in the spatial-aliasing demo:
quarter turns are precisely the rotations that never need resampling.
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
GROUP_ORDERS = [4, 8, 16]
RESOLUTIONS = [29, 57, 113, 225]
BATCH = 128


def build(N):
    torch.manual_seed(SEED)
    gspace = gspaces.rot2dOnR2(N=N)
    in_type = enn.FieldType(gspace, [gspace.trivial_repr])
    # 2 scalar fields + 2 'vector' (irrep-1) fields = 6 channels, same for every N
    hid_type = enn.FieldType(gspace, [gspace.trivial_repr] * 2 + [gspace.irrep(1)] * 2)
    conv = enn.R2Conv(in_type, hid_type, kernel_size=KERNEL_SIZE)
    conv.eval()
    return gspace, in_type, hid_type, conv


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.t(), Y @ Y.t()
    hsic = lambda A, B: (A * B).sum()
    return (hsic(K, L) / torch.sqrt(hsic(K, K) * hsic(L, L))).item()


def equivariance_error(conv, gx, y, hid_type, element):
    """|| model(g.x) - rho(g).model(x) || / || rho(g).model(x) ||, with rho(g)
    taken from the model's own declared FieldType (exact ground truth)."""
    lhs = conv(gx.transform(element)).tensor
    rhs = enn.GeometricTensor(y.tensor, hid_type).transform(element).tensor
    err = ((lhs - rhs).norm() / rhs.norm()).item()
    return err, linear_cka(y.tensor.flatten(1), lhs.flatten(1))


def load_images(size):
    testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True,
                                          transform=transforms.ToTensor())
    imgs, _ = next(iter(torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)))
    if size == 29:
        return F.pad(imgs, (0, 1, 0, 1))  # 28 -> 29, odd grid
    return F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)


def part_a():
    print("=== Part A: each C_N model at its OWN group elements (29x29 grid) ===")
    imgs = load_images(29)
    for N in GROUP_ORDERS:
        gspace, in_type, hid_type, conv = build(N)
        gx = enn.GeometricTensor(imgs, in_type)
        exact = 0
        print(f"\n  C{N} model:")
        with torch.no_grad():
            y = conv(gx)
            for k, element in enumerate(gspace.fibergroup.elements):
                deg = 360.0 * k / N
                err, cka = equivariance_error(conv, gx, y, hid_type, element)
                lattice = abs(deg % 90.0) < 1e-9
                exact += lattice
                tag = "lattice-exact" if lattice else "needs interpolation"
                print(f"    {deg:6.1f} deg   err={err:.5f}  cka={cka:.4f}   {tag}")
        print(f"    -> {exact}/{N} group elements honored exactly "
              f"({100.0 * exact / N:.0f}% of the claimed group)")


def part_b():
    print("\n\n=== Part B: is it aliasing? C16 error vs spatial sampling rate ===")
    gspace, in_type, hid_type, conv = build(16)
    elements = list(gspace.fibergroup.elements)
    probes = [(1, "22.5"), (2, "45.0"), (3, "67.5"), (4, "90.0")]
    print(f"{'grid':>7s} " + " ".join(f"{d + 'deg':>10s}" for _, d in probes))
    for size in RESOLUTIONS:
        imgs = load_images(size)
        gx = enn.GeometricTensor(imgs, in_type)
        with torch.no_grad():
            y = conv(gx)
            row = [equivariance_error(conv, gx, y, hid_type, elements[k])[0] for k, _ in probes]
        print(f"{size:5d}^2 " + " ".join(f"{v:10.5f}" for v in row))
    print("\n  Non-lattice error roughly halves per doubling of resolution (-> 0 in the")
    print("  continuum limit); the 90-degree column stays exactly 0 at every rate.")


if __name__ == "__main__":
    part_a()
    part_b()
