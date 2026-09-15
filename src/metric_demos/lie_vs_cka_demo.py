"""
Faithful replication of the Lie-derivative equivariance test from Gruver et al.,
"The Lie Derivative for Measuring Learned Equivariance" (arXiv:2210.02984), compared
against linear CKA, on the CIFARNets CNN classifier using its two pre-trained
checkpoints already in src/models/ (no training needed).

Following the paper's own reference code (github.com/ngruver/lie-deriv, Fig. 3):
rotation is continuous (affine_grid + grid_sample), and the output representation
is chosen by `img_like = (len(z.shape) == 4)` -- if 4D, assume it rotates like the
input image; if not (e.g. a flattened FC vector or logits), assume the TRIVIAL
representation (it should be exactly invariant). That second branch is the one
this script probes: once CIFARNets.CNN flattens its feature map into fc1/fc2/fc3,
the reference method silently switches from "must rotate" to "must not change at
all," which is a much stronger and, empirically, much less discriminative
assumption than what CKA requires.

NOTE: the paper computes the derivative via torch.autograd.functional.jvp. That
call raises NotImplementedError in this environment because grid_sampler_2d has no
forward-mode/double-backward AD support in the installed PyTorch (2.10.0). A
central finite difference at small theta is used instead -- verified numerically
stable (~15% variation) across eps in [0.005, 0.1] rad, i.e. it is measuring the
same local derivative, not finite-difference noise.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Resolved from __file__, not the working directory, so the script runs from
# anywhere: it needs src/Models/ on the path plus the pretrained checkpoints and
# datasets that live under src/.
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from Models.Classification.CIFARNets import CNN  # noqa: E402

CHECKPOINTS = {
    "learned_equivariant": str(SRC_ROOT / "models/classification_learned_equivariant_cnn_dataset_cifar_model.pth"),
    "non_equivariant": str(SRC_ROOT / "models/classification_non_equivariant_cnn_dataset_cifar_model.pth"),
}
LAYERS = ["conv1", "pool1", "pool2", "fc1", "fc2", "fc3", "softmax"]
EPS = 0.02            # rotation step (rad) for the finite-difference Lie derivative
DISCRETE_ANGLE_K = 1  # 90 degrees, for CKA -- matches the rest of the repo's convention


def rotate(imgs, theta):
    """Continuous rotation by angle theta (rad) via bilinear-interpolated affine_grid,
    exactly matching the paper's Figure 3 `rotate` helper."""
    theta = torch.as_tensor(theta, dtype=imgs.dtype, device=imgs.device)
    m = torch.tensor([[torch.cos(theta), torch.sin(theta), 0.0],
                       [-torch.sin(theta), torch.cos(theta), 0.0]],
                      device=imgs.device)[None].expand(imgs.shape[0], -1, -1)
    grid = F.affine_grid(m, imgs.size(), align_corners=True)
    return F.grid_sample(imgs, grid, align_corners=True)


def forward_to(net, x, stop):
    """Mirrors CIFARNets.CNN.forward() layer-for-layer (without the .detach() calls
    used there for logging), stopping at the requested layer so it stays
    differentiable / usable for both the clean and rotated finite-difference passes."""
    x = net.conv1(x)
    if stop == "conv1": return x
    x = F.relu(x); x = net.pool(x)
    if stop == "pool1": return x
    x = net.conv2(x); x = F.relu(x); x = net.pool(x)
    if stop == "pool2": return x
    x = torch.flatten(x, 1)
    x = net.fc1(x)
    if stop == "fc1": return x
    x = F.relu(x); x = net.fc2(x)
    if stop == "fc2": return x
    x = F.relu(x); logits = net.fc3(x)
    if stop == "fc3": return logits
    return F.softmax(logits, dim=-1)


def relative_lee(net, stop, imgs, eps=EPS):
    """Relative Local Equivariance Error at one layer: ||L_X f(x)|| / ||f(x)||,
    using img_like ? rotate-correct : trivial-representation exactly as the paper's
    reference code does, and a central finite difference in place of jvp."""
    def f(theta):
        z = forward_to(net, rotate(imgs, theta), stop)
        return rotate(z, -theta) if z.dim() == 4 else z
    with torch.inference_mode():
        f0 = f(0.0)
        lie_deriv = (f(eps) - f(-eps)) / (2 * eps)
    rel = lie_deriv.flatten(1).norm(dim=1) / (f0.flatten(1).norm(dim=1) + 1e-8)
    return rel.mean().item()


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.t(), Y @ Y.t()
    hsic = lambda A, B: (A * B).sum()
    return (hsic(K, L) / torch.sqrt(hsic(K, K) * hsic(L, L))).item()


def evaluate(checkpoint_path, testloader):
    net = CNN(width1=120, width2=84)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net.eval()

    rel_lee = {l: [] for l in LAYERS}
    cka = {l: [] for l in LAYERS}
    for imgs, _ in testloader:
        imgs_rot = torch.rot90(imgs, DISCRETE_ANGLE_K, dims=(-2, -1))
        with torch.inference_mode():
            a_clean = {l: forward_to(net, imgs, l) for l in LAYERS}
            a_rot = {l: forward_to(net, imgs_rot, l) for l in LAYERS}
        for l in LAYERS:
            rel_lee[l].append(relative_lee(net, l, imgs))
            cka[l].append(linear_cka(a_clean[l].flatten(1), a_rot[l].flatten(1)))

    mean = lambda d: {l: sum(v) / len(v) for l, v in d.items()}
    return mean(rel_lee), mean(cka)


def main():
    testset = torchvision.datasets.CIFAR10(root=str(SRC_ROOT / "data"), train=False, download=True,
                                            transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    for name, path in CHECKPOINTS.items():
        rel_lee, cka = evaluate(path, testloader)
        print(f"\n=== {name} ===")
        print(f"{'layer':8s} {'rel_LEE':>9s} {'linear_cka':>11s}")
        for l in LAYERS:
            print(f"{l:8s} {rel_lee[l]:9.4f} {cka[l]:11.4f}")


if __name__ == "__main__":
    main()
