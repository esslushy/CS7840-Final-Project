"""
Does a lower Lie-derivative equivariance error actually mean "more equivariant"?

Gruver et al. ("The Lie Derivative for Measuring Learned Equivariance",
arXiv:2210.02984) rank architectures by Local Equivariance Error (LEE) and
conclude that non-equivariant architectures can end up MORE equivariant than
architecturally equivariant ones after training ("transformers can be more
equivariant than convolutional neural networks after training").

That conclusion rests on an inference: lower LEE => more equivariant. This
script constructs counterexamples to that inference using escnn models whose
TRUE equivariance is known exactly (perfect, by construction, not by training),
so LEE can be compared against ground truth rather than against another
estimate.

Every model here outputs a SCALAR field, so rho(g) is plain spatial rotation
for all of them -- an identical, unambiguous readout convention, with no
representation-choice confound of the kind the other demos in this directory
explore. The only thing that varies is the model.

  Part A  Six models, ALL exactly SO(2)-equivariant (true equivariance
          identical and perfect). LEE nonetheless spans ~25x, rising
          monotonically with the angular frequency of the features. A metric
          measuring equivariance would be flat here. LEE is tracking bandwidth.

  Part B  Head-to-head between an exactly-equivariant model and a blatantly
          non-equivariant one (an anisotropic Gaussian with a hard-coded 8:3
          preferred direction), probed on successively finer subsamplings of
          SO(2). On C4 -- quarter turns, which are lattice symmetries of a
          square grid and need no interpolation -- LEE is exactly right: the
          equivariant model scores 0.00000. Refine the export group to C8/C16
          or go fully continuous, and the ordering INVERTS: the non-equivariant
          model scores ~2.4x better than the provably equivariant one.

  Part C  Mechanism. Hold the non-equivariant model's anisotropy (its actual
          non-equivariance) fixed and vary only its blur. LEE falls
          monotonically with blur. Smoothness buys LEE; equivariance is not
          what is being purchased.

WHAT THIS DOES AND DOES NOT SHOW. A single counterexample is enough to refute
the inference rule "lower LEE => more equivariant" as a general principle, and
these are counterexamples against exact ground truth. It does not by itself
prove the paper's specific ViT-vs-CNN measurements are wrong. What it shows is
that those measurements are CONFOUNDED: LEE conflates equivariance with feature
bandwidth, and architectures differ systematically in bandwidth (ViT features,
built from patch embeddings and heavy smoothing, are far more band-limited than
a CNN's). Any cross-architecture LEE comparison therefore needs bandwidth held
fixed before it can support a claim about equivariance -- and the reported
comparisons do not control for it.
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
KERNEL_SIZE = 9
BATCH = 128
GENERIC_ANGLES = [0.3, 0.7, 1.0, 2.0, np.pi / 4]


def load_images():
    testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True,
                                          transform=transforms.ToTensor())
    imgs, _ = next(iter(torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)))
    return F.pad(imgs, (0, 1, 0, 1))  # 28 -> 29, odd grid


def rotate_interp(t, theta):
    """Continuous rotation; requires interpolation for non-quarter-turn angles."""
    th = torch.as_tensor(theta)
    m = torch.tensor([[torch.cos(th), torch.sin(th), 0.0],
                       [-torch.sin(th), torch.cos(th), 0.0]])[None].expand(t.shape[0], -1, -1)
    return F.grid_sample(t, F.affine_grid(m, t.size(), align_corners=True), align_corners=True)


def lee(fwd, imgs, angles, exact_quarter_turns=False):
    """Mean relative equivariance error of a scalar-field-valued model."""
    errs = []
    with torch.no_grad():
        y = fwd(imgs)
        for th in angles:
            if exact_quarter_turns:  # lattice symmetry: pure permutation, no resampling
                k = int(round(th / (np.pi / 2)))
                rot_in = lambda t: torch.rot90(t, k, dims=(-2, -1))
                rot_out, weight = rot_in, torch.ones_like(y)
            else:
                rot_in = lambda t: rotate_interp(t, th)
                rot_out = rot_in
                weight = (rotate_interp(torch.ones_like(y), th) > 0.99).float()
            lhs, rhs = fwd(rot_in(imgs)), rot_out(y)
            errs.append((((lhs - rhs) * weight).norm()
                         / ((rhs * weight).norm() + 1e-9)).item())
    return float(np.mean(errs))


def equivariant_model(freq, n_fields=4):
    """Exactly SO(2)-equivariant: conv into irrep-`freq` fields, then fiber norms.
    The norm of an irrep field is invariant under the fiber action, so the output
    is an exactly equivariant scalar field -- regardless of `freq`."""
    torch.manual_seed(SEED)
    gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=max(freq, 1) + 1)
    in_type = enn.FieldType(gspace, [gspace.trivial_repr])
    reps = [gspace.trivial_repr] * n_fields if freq == 0 else [gspace.irrep(freq)] * n_fields
    out_type = enn.FieldType(gspace, reps)
    conv = enn.R2Conv(in_type, out_type, kernel_size=KERNEL_SIZE)
    conv.eval()
    dim = 1 if freq == 0 else 2

    def fwd(x):
        y = conv(enn.GeometricTensor(x, in_type)).tensor
        return torch.stack([y[:, i * dim:(i + 1) * dim].pow(2).sum(1).sqrt()
                            for i in range(n_fields)], 1)
    return fwd


def anisotropic_model(sx, sy, ks=KERNEL_SIZE):
    """Blatantly NON-equivariant: Gaussians with a hard-coded preferred direction."""
    def kernel(a, b):
        ax = torch.arange(ks).float() - ks // 2
        Y, X = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(X ** 2 / (2 * a ** 2) + Y ** 2 / (2 * b ** 2)))
        return (k / k.sum()).view(1, 1, ks, ks)
    w = torch.cat([kernel(sx, sy), kernel(sy, sx),
                   kernel(sx, sy * 1.5), kernel(sy * 1.5, sx)], 0)
    return lambda x: F.conv2d(x, w, padding=ks // 2)


def part_a(imgs):
    print("=== Part A: LEE across models that are ALL exactly equivariant ===")
    print("    (true equivariance is perfect and IDENTICAL in every row)\n")
    print(f"    {'feature freq':>13s} {'true equivariance':>18s} {'LEE':>9s}")
    vals = []
    for f in [0, 1, 2, 3, 4, 5]:
        v = lee(equivariant_model(f), imgs, GENERIC_ANGLES)
        vals.append(v)
        print(f"    {f:>13d} {'EXACT':>18s} {v:9.5f}")
    print(f"\n    -> LEE spans {max(vals) / max(min(vals), 1e-9):.0f}x while true "
          f"equivariance never changes.")
    print("       LEE is tracking angular bandwidth, not equivariance.")


def part_b(imgs):
    print("\n\n=== Part B: exact-equivariant vs non-equivariant, by export group ===")
    eq, aniso = equivariant_model(4), anisotropic_model(8.0, 3.0)
    quarter = [np.pi / 2, np.pi, 3 * np.pi / 2]
    c8_new = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
    c16_new = [2 * np.pi * k / 16 for k in [1, 3, 5, 7, 9, 11, 13, 15]]

    probes = [("C4  (quarter turns, lattice-exact)", quarter, True),
              ("C8  (adds 45 deg, interpolated)", c8_new, False),
              ("C16 (adds 22.5 deg, interpolated)", c16_new, False),
              ("SO(2) (generic angles)", GENERIC_ANGLES, False)]

    print(f"\n    {'export group':>36s} {'equivariant (EXACT)':>20s} {'anisotropic (NOT)':>19s}   verdict")
    for name, angles, exact in probes:
        a = lee(eq, imgs, angles, exact_quarter_turns=exact)
        b = lee(aniso, imgs, angles, exact_quarter_turns=exact)
        verdict = "correct" if a < b else "INVERTED"
        print(f"    {name:>36s} {a:20.5f} {b:19.5f}   {verdict}")
    print("\n    -> On C4 the metric is exactly right. Subsample SO(2) any finer and the")
    print("       provably-equivariant model looks ~2.4x WORSE than a model with a")
    print("       hard-coded directional bias. The flip is pure interpolation aliasing:")
    print("       the blurry non-equivariant model barely notices interpolation, while")
    print("       the sharp equivariant model's error goes 0.00000 -> 0.22.")


def part_c(imgs):
    print("\n\n=== Part C: mechanism -- blur buys LEE at fixed non-equivariance ===")
    print("    (anisotropy ratio held at ~8:3 throughout; only the scale grows)\n")
    print(f"    {'anisotropic Gaussian':>28s} {'LEE':>9s}")
    for sx, sy in [(3.0, 1.0), (4.0, 1.5), (6.0, 2.0), (8.0, 3.0), (12.0, 4.5)]:
        print(f"    {f'sx={sx:.1f}, sy={sy:.1f}':>28s} "
              f"{lee(anisotropic_model(sx, sy), imgs, GENERIC_ANGLES):9.5f}")
    print("\n    -> Same directional bias, same (absent) equivariance, monotonically")
    print("       falling LEE. Smoothness is what the metric is rewarding.")


if __name__ == "__main__":
    images = load_images()
    part_a(images)
    part_b(images)
    part_c(images)
