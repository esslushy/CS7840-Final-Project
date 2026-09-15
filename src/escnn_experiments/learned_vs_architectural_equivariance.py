"""
Does a "learned equivariant" model really beat an architecturally equivariant
one at being equivariant -- and does that verdict survive switching metrics?

Gruver et al. (arXiv:2210.02984) rank models by Local Equivariance Error (LEE)
and conclude that architectures WITHOUT an equivariance prior can end up more
equivariant than ones with it ("transformers can be more equivariant than
convolutional neural networks after training"). This script sets up that exact
comparison for rotation, and scores it three ways.

Two models, trained identically (same data, same optimizer, same full 0-360
degree rotation augmentation, matched accuracy):

  learned       plain CNN. No equivariance prior at all. Whatever rotation
                robustness it has was learned from the augmentation.
  architectural escnn C8-steerable CNN with regular-representation fields and
                a GroupPooling readout. EXACTLY C8-invariant at its output, by
                construction, whether or not it is trained.

Three scorings:

  (1) BEHAVIOUR -- accuracy on clean vs rotated test images. The ground truth
      we actually care about: does the model's prediction survive rotation?

  (2) LEE, naive convention -- the paper's own reference code decides a
      tensor's group representation with `img_like = (len(z.shape) == 4)`:
      4D tensors get spatially rotated, everything else is assumed invariant.
      Applied to a hidden conv feature map, that is the SAME convention for
      both models -- apples to apples, exactly as the paper does it.

      It is also WRONG for the equivariant model. Regular-representation
      fields transform by *cyclically permuting the 8 group channels* as well
      as rotating space. The naive convention omits the permutation entirely,
      so it charges the equivariant model a large error for behaving exactly
      as designed.

  (3) CKA -- linear CKA between clean and rotated-input activations. Assumes
      no representation whatsoever, so it cannot make that mistake.

For reference, (2') also reports LEE under the CORRECT representation, taken
from escnn's own declared FieldType via GeometricTensor.transform() -- ground
truth available only for the architectural model.

Probe angles are reported separately because it matters:
  90 deg   in C8 AND a lattice symmetry of a square grid -> no interpolation,
           no aliasing. The clean measurement.
  45 deg   in C8, but not a lattice symmetry -> needs interpolation.
  generic  not in C8 at all.

FINDING, and it is more interesting than a flat refutation. The answer depends
on the probe angle, and BOTH metrics are wrong somewhere:

  at 90 deg (lattice-exact)   The architectural model is exactly invariant and
                              behaviour agrees exactly: accuracy drop 0.0000,
                              predictions 100% unchanged. LEE still ranks the
                              plain CNN above it. LEE is unambiguously WRONG
                              here, and CKA is right.

  at 45 deg / generic         The architectural model's guarantee is genuinely
                              destroyed -- by interpolation aliasing at 45 deg
                              (which is IN its group), and by simply not being
                              in its group at all otherwise. Behaviour says the
                              continuously-augmented plain CNN really is more
                              robust. Here LEE's verdict is right, and CKA --
                              which still prefers the architectural model -- is
                              WRONG.

So the paper's claim is not simply false: off-lattice, a learned-equivariant
model genuinely can beat an architecturally equivariant one. But the reason is
not that learning beats architecture. It is that a pixel grid cannot represent
a non-lattice rotation, so an architectural guarantee evaporates off-lattice
while augmentation-trained robustness does not -- the paper's own aliasing
story, applied to the comparison it draws rather than to the layers it studies.

And CKA is not a drop-in replacement. It tracks hidden-layer representational
consistency, which is not the same thing as end-task robustness; at off-lattice
angles it prefers the model that behaviour says is worse. The dependable
measurement here is D1/D2 (behaviour), with LEE-under-the-correct-rho(g) as the
only equivariance-error variant that is right at every probe.

Run: cd src/escnn_experiments && python learned_vs_architectural_equivariance.py   (see README.md for the venv)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from escnn import gspaces, nn as enn

# Shared with the rest of the repo: datasets live in src/data, one level up.
DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")


SEED = 0
EPOCHS = 6
BATCH = 128
GROUP_N = 8
GENERIC_ANGLES = [0.3, 0.7, 1.1, 2.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rotate(t, theta):
    """Continuous rotation. Exact (index permutation) only at quarter turns."""
    th = torch.as_tensor(theta, dtype=t.dtype, device=t.device)
    zero = torch.zeros((), dtype=t.dtype, device=t.device)
    m = torch.stack([torch.stack([torch.cos(th), torch.sin(th), zero]),
                     torch.stack([-torch.sin(th), torch.cos(th), zero])])[None]
    m = m.expand(t.shape[0], -1, -1)
    return F.grid_sample(t, F.affine_grid(m, t.size(), align_corners=True), align_corners=True)


class LearnedCNN(nn.Module):
    """Plain CNN: no equivariance prior. Rotation robustness must be learned."""

    def __init__(self, w1=48, w2=96):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(1, w1, 5, padding=2), nn.BatchNorm2d(w1)
        self.c2, self.b2 = nn.Conv2d(w1, w2, 5, padding=2), nn.BatchNorm2d(w2)
        self.fc = nn.Linear(w2, 10)

    def forward(self, x):
        h = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        h = F.max_pool2d(F.relu(self.b2(self.c2(h))), 2)
        v = h.mean((-2, -1))
        return self.fc(v), h, v


class ArchitecturalCNN(nn.Module):
    """escnn C8-steerable CNN. Exactly C8-invariant output, by construction."""

    def __init__(self, f1=6, f2=12, N=GROUP_N):
        super().__init__()
        gspace = gspaces.rot2dOnR2(N=N)
        self.gspace = gspace
        self.in_type = enn.FieldType(gspace, [gspace.trivial_repr])
        t1 = enn.FieldType(gspace, [gspace.regular_repr] * f1)
        t2 = enn.FieldType(gspace, [gspace.regular_repr] * f2)
        self.hidden_type = t2
        self.net = nn.Sequential(
            enn.R2Conv(self.in_type, t1, 5, padding=2), enn.InnerBatchNorm(t1), enn.ReLU(t1),
            enn.PointwiseMaxPool(t1, 2),
            enn.R2Conv(t1, t2, 5, padding=2), enn.InnerBatchNorm(t2), enn.ReLU(t2),
            enn.PointwiseMaxPool(t2, 2),
        )
        self.gpool = enn.GroupPooling(t2)
        self.fc = nn.Linear(f2, 10)

    def forward(self, x):
        y = self.net(enn.GeometricTensor(x, self.in_type))
        v = self.gpool(y).tensor.mean((-2, -1))
        return self.fc(v), y.tensor, v


def train(model, trainloader):
    torch.manual_seed(SEED)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(EPOCHS):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = rotate(x, float(np.random.uniform(0, 2 * np.pi)))  # full SO(2) augmentation
            opt.zero_grad()
            F.cross_entropy(model(x)[0], y).backward()
            opt.step()
    return model.eval()


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.t(), Y @ Y.t()
    hsic = lambda A, B: (A * B).sum()
    return (hsic(K, L) / torch.sqrt(hsic(K, K) * hsic(L, L))).item()


def rotate_hidden_naive(h, theta, exact):
    """The paper's img_like convention: 4D -> rotate space, leave channels alone."""
    return torch.rot90(h, int(round(theta / (np.pi / 2))), dims=(-2, -1)) if exact \
        else rotate(h, theta)


def rotate_hidden_correct(model, h, theta):
    """The true representation, read off escnn's declared FieldType (includes the
    regular-representation channel permutation). Only available for the
    architectural model -- that is the whole point."""
    k = int(round(theta / (2 * np.pi / GROUP_N))) % GROUP_N
    element = list(model.gspace.fibergroup.elements)[k]
    return enn.GeometricTensor(h, model.hidden_type).transform(element).tensor


def measure(model, loader, theta, exact, architectural):
    """Returns per-batch-averaged metrics at one probe angle."""
    acc = acc_rot = n = 0
    lee_naive, lee_correct, cka_hidden, cka_out = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_rot = torch.rot90(x, int(round(theta / (np.pi / 2))), dims=(-2, -1)) if exact \
                else rotate(x, theta)

            logits, h, v = model(x)
            logits_r, h_r, v_r = model(x_rot)

            acc += (logits.argmax(-1) == y).sum().item()
            acc_rot += (logits_r.argmax(-1) == y).sum().item()
            n += len(y)

            rel = lambda a, b: ((a - b).norm() / (b.norm() + 1e-9)).item()
            lee_naive.append(rel(h_r, rotate_hidden_naive(h, theta, exact)))
            if architectural:
                lee_correct.append(rel(h_r, rotate_hidden_correct(model, h, theta)))
            cka_hidden.append(linear_cka(h.flatten(1), h_r.flatten(1)))
            cka_out.append(linear_cka(v, v_r))

    mean = lambda a: float(np.mean(a)) if a else float("nan")
    return {"acc": acc / n, "acc_rot": acc_rot / n, "lee_naive": mean(lee_naive),
            "lee_correct": mean(lee_correct), "cka_hidden": mean(cka_hidden),
            "cka_out": mean(cka_out)}


def main():
    tr = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,
                                     transform=transforms.ToTensor())
    te = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True,
                                     transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(te, batch_size=256, shuffle=False, num_workers=2)

    models = {}
    for name, ctor in [("learned (plain CNN)", LearnedCNN),
                       ("architectural (escnn C8)", ArchitecturalCNN)]:
        m = train(ctor(), trainloader)
        models[name] = m
        print(f"trained {name:26s} params={sum(p.numel() for p in m.parameters()):,}")

    probes = [("90 deg  (in C8, lattice-exact)", np.pi / 2, True),
              ("45 deg  (in C8, interpolated)", np.pi / 4, False),
              ("generic (not in C8)", None, False)]

    for label, theta, exact in probes:
        print(f"\n\n=== probe: {label} ===")
        print(f"{'model':>26s} {'acc':>6s} {'acc_rot':>8s} "
              f"{'LEE naive':>10s} {'LEE correct':>12s} {'CKA hidden':>11s} {'CKA out':>8s}")
        rows = {}
        for name, m in models.items():
            arch = isinstance(m, ArchitecturalCNN)
            angles = GENERIC_ANGLES if theta is None else [theta]
            per = [measure(m, testloader, a, exact, arch) for a in angles]
            r = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            rows[name] = r
            lc = "     --   " if np.isnan(r["lee_correct"]) else f"{r['lee_correct']:12.5f}"
            print(f"{name:>26s} {r['acc']:6.4f} {r['acc_rot']:8.4f} "
                  f"{r['lee_naive']:10.5f} {lc} {r['cka_hidden']:11.5f} {r['cka_out']:8.5f}")

        le, la = (rows["learned (plain CNN)"], rows["architectural (escnn C8)"])
        print()
        print(f"    behaviour  (acc drop under rotation): "
              f"learned {le['acc'] - le['acc_rot']:+.4f}  vs  architectural "
              f"{la['acc'] - la['acc_rot']:+.4f}  -> "
              f"{'architectural' if (la['acc']-la['acc_rot']) < (le['acc']-le['acc_rot']) else 'learned'} more robust")
        print(f"    LEE naive  (lower=better): "
              f"{'LEARNED wins  <-- reproduces the paper' if le['lee_naive'] < la['lee_naive'] else 'architectural wins'}")
        print(f"    CKA hidden (higher=better): "
              f"{'architectural wins  <-- VERDICT REVERSED' if la['cka_hidden'] > le['cka_hidden'] else 'learned wins'}")


if __name__ == "__main__":
    main()
