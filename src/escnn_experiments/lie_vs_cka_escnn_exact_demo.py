"""
Third companion demo: instead of a trained (only approximately equivariant)
network, this one uses an escnn steerable CNN, which is EXACTLY equivariant to
continuous SO(2) rotations by construction (no training at all -- the guarantee
comes from the architecture, not from learning). That removes training noise
as a confound and lets us test the representation-mismatch story from
metric_demos/lie_vs_cka_demo.py / lie_vs_cka_vector_field_demo.py against a known-exact
ground truth instead of an approximation.

Each layer of an escnn network has a declared FieldType: a direct sum of group
representations (irreps) describing exactly how that layer's channels must
transform under rotation. A hidden layer mixing e.g. two trivial (scalar)
irreps, two copies of irrep(1) (2D "vector" pairs), and one irrep(2) has
channels that must mix in quite different ways under rotation -- nothing like
"rotate the grid, leave channels alone" (the naive img_like assumption from
Gruver et al.'s reference code, see metric_demos/lie_vs_cka_demo.py's docstring).

GeometricTensor.transform(g) applies the model's OWN declared representation
exactly, so instead of hand-deriving the correct transform law (as the vector-
field demo did), we can pull it straight from the architecture -- true ground
truth, not our own approximation of it. We compare that against a "naive"
version built by re-wrapping the same tensor in an all-trivial-irrep FieldType
of the same channel count, i.e. exactly the "treat every channel as an
independent scalar field" assumption.

Because the model is exactly equivariant and MNIST images are square, rotating
by an exact multiple of 90 degrees is alias-free (pure index permutation, no
interpolation) -- so the correctly-transformed prediction matches the model's
output on the actually-rotated input to floating-point precision (~0.000
relative error) at those angles. Generic continuous angles pick up a small
residual from interpolating the discretely-sampled input image itself -- the
same "aliasing" phenomenon the paper's own Section 2 identifies as the
fundamental obstruction to exact continuous equivariance on pixel grids. Both
are dwarfed by the naive assumption's error at every angle tested.
"""
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from escnn import gspaces, nn as enn

# Shared with the rest of the repo: datasets live in src/data, one level up.
DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")


SEED = 0
KERNEL_SIZE = 7
MAX_FREQUENCY = 4
HIDDEN_IRREPS = ["trivial", "trivial", "irrep1", "irrep1", "irrep2"]  # mixed FieldType
THETAS = [0.1, 0.5, 1.0, 2.0, np.pi / 2, np.pi]  # last two are grid-aligned (alias-free)


def build_field_type(gspace, spec):
    irreps = []
    for s in spec:
        if s == "trivial":
            irreps.append(gspace.trivial_repr)
        elif s.startswith("irrep"):
            irreps.append(gspace.irrep(int(s[len("irrep"):])))
        else:
            raise ValueError(s)
    return enn.FieldType(gspace, irreps)


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
    hid_type = build_field_type(gspace, HIDDEN_IRREPS)
    naive_hid_type = enn.FieldType(gspace, [gspace.trivial_repr] * hid_type.size)

    conv = enn.R2Conv(in_type, hid_type, kernel_size=KERNEL_SIZE)
    conv.eval()

    testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True,
                                          transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    cka_vals, correct_err, naive_err = [], {t: [] for t in THETAS}, {t: [] for t in THETAS}
    with torch.no_grad():
        for imgs, _ in loader:
            gx = enn.GeometricTensor(imgs, in_type)
            y = conv(gx)

            gx_rot90 = enn.GeometricTensor(torch.rot90(imgs, 1, dims=(-2, -1)), in_type)
            y_rot90 = conv(gx_rot90)
            cka_vals.append(linear_cka(y.tensor.flatten(1), y_rot90.tensor.flatten(1)))

            for theta in THETAS:
                elem = so2.element(theta)
                lhs = conv(gx.transform(elem))                                    # model(rotate(x))
                y_correct = y.transform(elem)                                     # correct fiber-mixing transform
                y_naive = enn.GeometricTensor(y.tensor, naive_hid_type).transform(elem)  # naive scalar-per-channel

                correct_err[theta].append(((lhs.tensor - y_correct.tensor).norm()
                                            / y_correct.tensor.norm()).item())
                naive_err[theta].append(((lhs.tensor - y_naive.tensor).norm()
                                          / y_naive.tensor.norm()).item())

    mean = lambda v: sum(v) / len(v)
    print(f"hidden FieldType: {HIDDEN_IRREPS}  ({hid_type.size} channels)")
    print(f"linear CKA (clean vs. discrete 90deg rotation): {mean(cka_vals):.4f}\n")
    print(f"{'theta (rad)':>12s} {'correct_rel_err':>16s} {'naive_rel_err':>14s}")
    for theta in THETAS:
        tag = "  (grid-aligned, alias-free)" if abs(theta % (np.pi / 2)) < 1e-6 else ""
        print(f"{theta:12.4f} {mean(correct_err[theta]):16.5f} {mean(naive_err[theta]):14.5f}{tag}")


if __name__ == "__main__":
    main()
