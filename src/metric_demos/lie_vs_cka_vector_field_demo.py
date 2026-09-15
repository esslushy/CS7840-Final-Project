"""
Companion to lie_vs_cka_demo.py: shows a second, distinct way the reference
Lie-derivative convention's "img_like = (len(z.shape) == 4)" branch (Gruver et al.,
"The Lie Derivative for Measuring Learned Equivariance", arXiv:2210.02984, Fig. 3)
can silently assume the wrong representation -- even when the tensor IS still a
genuine (B,C,H,W) spatial map, i.e. even *before* anything gets flattened.

img_like==True makes the paper's code apply `rotate(z, -theta)`: spatially rotate
the (H,W) grid, but leave every channel independent, exactly as if each channel
were its own scalar field. That's the right representation for e.g. an edge-
detector activation map. It's the WRONG representation for a genuine 2D vector
field, where rotating the image must also mix the channel pair via the 2x2
rotation matrix (new_vx = cos(t)*vx - sin(t)*vy, new_vy = sin(t)*vx + cos(t)*vy).

This repo already trains exactly such a task: gradient_field.py predicts a
2-channel (dI/dx, dI/dy) Sobel gradient field, a bona fide rotation-covariant
vector field by construction (rotating the image rotates the gradient vectors).
Using the two pre-trained UNet/MNIST checkpoints already in src/models/ (no
training needed), this script compares the naive scalar-per-channel assumption
against the channel-mixing-aware vector-field assumption at the network's final
output, plus linear CKA for reference.
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
from Models.GradientFieldNets import UNet  # noqa: E402

CHECKPOINTS = {
    "learned_equivariant": str(SRC_ROOT / "models/gradient_field_learned_equivariant_unet_dataset_mnist_kernel_linear_model.pth"),
    "non_equivariant": str(SRC_ROOT / "models/gradient_field_non_equivariant_unet_dataset_mnist_kernel_linear_model.pth"),
}
EPS = 0.02
DISCRETE_ANGLE_K = 1  # 90 degrees, for CKA and the task-loss sanity check


def rotate(imgs, theta):
    """Continuous spatial rotation by angle theta (rad), matching the paper's
    Figure 3 `rotate` helper. Channels are carried through untouched -- this is
    the "scalar field per channel" assumption."""
    theta = torch.as_tensor(theta, dtype=imgs.dtype, device=imgs.device)
    m = torch.tensor([[torch.cos(theta), torch.sin(theta), 0.0],
                       [-torch.sin(theta), torch.cos(theta), 0.0]],
                      device=imgs.device)[None].expand(imgs.shape[0], -1, -1)
    grid = F.affine_grid(m, imgs.size(), align_corners=True)
    return F.grid_sample(imgs, grid, align_corners=True)


def rotate_vector_field(z, theta):
    """Spatial rotation PLUS the correct 2x2 rotation of channels 0,1 -- the
    representation a genuine 2D vector field (e.g. an image gradient) obeys."""
    theta_t = torch.as_tensor(theta, dtype=z.dtype, device=z.device)
    zs = rotate(z, theta)
    vx, vy = zs[:, 0], zs[:, 1]
    c, s = torch.cos(theta_t), torch.sin(theta_t)
    return torch.stack([c * vx - s * vy, s * vx + c * vy], dim=1)


def sobel_gradient(images):
    """Same Sobel-gradient ground truth as gradient_field.py, duplicated here
    (read-only reuse of the same formula) only to report task loss for context."""
    C = images.shape[1]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=images.dtype).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=images.dtype).view(1, 1, 3, 3) / 8.0
    sobel_x, sobel_y = sobel_x.expand(C, 1, 3, 3), sobel_y.expand(C, 1, 3, 3)
    gx = F.conv2d(images, sobel_x, padding=1, groups=C).mean(dim=1, keepdim=True)
    gy = F.conv2d(images, sobel_y, padding=1, groups=C).mean(dim=1, keepdim=True)
    return torch.cat([gx, gy], dim=1)


def relative_lee(net, assume, imgs, eps=EPS):
    """Relative Local Equivariance Error at the network's output, under either
    the naive "scalar" (img_like, channel-independent) assumption or the
    "vector_field" (channel-mixing) assumption."""
    def out_only(x):
        out, _ = net(x)
        return out

    def f(theta):
        z = out_only(rotate(imgs, theta))
        return rotate(z, -theta) if assume == "scalar" else rotate_vector_field(z, -theta)

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
    net = UNet(in_channels=1, width1=128, width2=128)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net.eval()

    scalar_lee, vector_lee, cka = [], [], []
    mse_clean, mse_rot = [], []
    for imgs, _ in testloader:
        imgs_rot = torch.rot90(imgs, DISCRETE_ANGLE_K, dims=(-2, -1))
        with torch.inference_mode():
            out_clean, _ = net(imgs)
            out_rot, _ = net(imgs_rot)
            mse_clean.append(((out_clean - sobel_gradient(imgs)) ** 2).mean().item())
            mse_rot.append(((out_rot - sobel_gradient(imgs_rot)) ** 2).mean().item())
        scalar_lee.append(relative_lee(net, "scalar", imgs))
        vector_lee.append(relative_lee(net, "vector_field", imgs))
        cka.append(linear_cka(out_clean.flatten(1), out_rot.flatten(1)))

    mean = lambda v: sum(v) / len(v)
    return {
        "scalar_rel_LEE": mean(scalar_lee),
        "vector_field_rel_LEE": mean(vector_lee),
        "linear_cka": mean(cka),
        "task_mse_clean": mean(mse_clean),
        "task_mse_rot90": mean(mse_rot),
    }


def main():
    testset = torchvision.datasets.MNIST(root=str(SRC_ROOT / "data"), train=False, download=True,
                                          transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    for name, path in CHECKPOINTS.items():
        r = evaluate(path, testloader)
        print(f"\n=== {name} ===")
        print(f"  task MSE: clean={r['task_mse_clean']:.5f}  rot90={r['task_mse_rot90']:.5f}")
        print(f"  naive scalar-assumption   rel_LEE = {r['scalar_rel_LEE']:.4f}")
        print(f"  correct vector-field      rel_LEE = {r['vector_field_rel_LEE']:.4f}")
        print(f"  linear CKA (output)                = {r['linear_cka']:.4f}")


if __name__ == "__main__":
    main()
