import torch
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from Models.StressPredictionNets import UNet, CNN, ViT, NaiveNet
from utils import UnifiedEquivarianceTracker
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 200
BATCH_SIZE = 64
GRID_SIZE = 32


# ---------------------------------------------------------------------------
# Finite difference kernels
# ---------------------------------------------------------------------------

def _make_kernels(device, dx=1.0):
    """Second-derivative and first-derivative kernels for Airy → stress → force."""
    # ∂²/∂x²
    d2_dx2 = torch.tensor([[0, 0, 0],
                            [1, -2, 1],
                            [0, 0, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (dx ** 2)
    # ∂²/∂y²
    d2_dy2 = torch.tensor([[0, 1, 0],
                            [0, -2, 0],
                            [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (dx ** 2)
    # ∂²/∂x∂y
    d2_dxdy = torch.tensor([[ 1, 0, -1],
                             [ 0, 0,  0],
                             [-1, 0,  1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (4 * dx ** 2)
    # ∂/∂x
    d_dx = torch.tensor([[ 0, 0, 0],
                          [-1, 0, 1],
                          [ 0, 0, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (2 * dx)
    # ∂/∂y
    d_dy = torch.tensor([[0, -1, 0],
                          [0,  0, 0],
                          [0,  1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / (2 * dx)

    return d2_dx2, d2_dy2, d2_dxdy, d_dx, d_dy


def _apply_kernel(field, kernel):
    """Apply a convolution kernel to a (H, W) or (1, 1, H, W) field."""
    if field.dim() == 2:
        field = field.unsqueeze(0).unsqueeze(0)
    return Fn.conv2d(field, kernel, padding=1).squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def rotate_grid(grid_size, theta, device):
    """Build a sampling grid for spatial rotation by theta."""
    coords = torch.linspace(-1, 1, grid_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R_inv = torch.tensor([[c, s], [-s, c]], device=device)  # transpose of R
    xy_rot = (xy.view(-1, 2) @ R_inv.mT).view(grid_size, grid_size, 2)
    return xy_rot.unsqueeze(0)


def rotate_force_field(field, theta):
    """
    Rotate a 2D force (vector) field by continuous angle theta.
    Spatial rotation + vector component rotation.

    Args:
        field: (B, 2, H, W) — (fx, fy)
        theta: scalar tensor
    """
    B, _, H, W = field.shape
    device = field.device

    grid = rotate_grid(H, theta, device).expand(B, -1, -1, -1)
    rotated = Fn.grid_sample(field, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    c = torch.cos(theta)
    s = torch.sin(theta)
    fx, fy = rotated[:, 0:1], rotated[:, 1:2]
    new_fx = c * fx - s * fy
    new_fy = s * fx + c * fy
    return torch.cat([new_fx, new_fy], dim=1)


def rotate_stress_field(field, theta):
    """
    Rotate a 2D stress (rank-2 symmetric tensor) field by continuous angle theta.
    Spatial rotation + tensor transformation σ' = R σ R^T.

    Args:
        field: (B, 3, H, W) — (σ_xx, σ_yy, σ_xy)
        theta: scalar tensor
    """
    B, _, H, W = field.shape
    device = field.device

    grid = rotate_grid(H, theta, device).expand(B, -1, -1, -1)
    rotated = Fn.grid_sample(field, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    c = torch.cos(theta)
    s = torch.sin(theta)
    c2, s2, cs = c * c, s * s, c * s

    sxx, syy, sxy = rotated[:, 0:1], rotated[:, 1:2], rotated[:, 2:3]

    # σ' = R σ R^T
    new_sxx = c2 * sxx + s2 * syy - 2 * cs * sxy
    new_syy = s2 * sxx + c2 * syy + 2 * cs * sxy
    new_sxy = cs * (sxx - syy) + (c2 - s2) * sxy

    return torch.cat([new_sxx, new_syy, new_sxy], dim=1)


def rotate_force_field_single(field, theta):
    """Unbatched: (2, H, W)."""
    return rotate_force_field(field.unsqueeze(0), theta).squeeze(0)


def rotate_stress_field_single(field, theta):
    """Unbatched: (3, H, W)."""
    return rotate_stress_field(field.unsqueeze(0), theta).squeeze(0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class IsotropicStressDataset(Dataset):
    """
    Force field → stress tensor for an isotropic 2D elastic medium.

    Stress fields are generated from random Airy stress functions φ:
        σ_xx = ∂²φ/∂y²,  σ_yy = ∂²φ/∂x²,  σ_xy = -∂²φ/∂x∂y

    Body forces are derived from the stress via equilibrium:
        f = -div(σ)

    Input: (2, H, W) force field (fx, fy)
    Output: (3, H, W) stress tensor (σ_xx, σ_yy, σ_xy)

    The physics is fully rotation-equivariant: rotating the forces
    should rotate the stress tensor (as a rank-2 tensor, not a vector).
    """

    def __init__(self, n_samples=10000, grid_size=GRID_SIZE, rotate=False):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.rotate = rotate
        self.dx = 2.0 / (grid_size - 1)

        coords = torch.linspace(-1, 1, grid_size)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

        d2_dx2, d2_dy2, d2_dxdy, d_dx, d_dy = _make_kernels(torch.device('cpu'), self.dx)
        self.d2_dx2 = d2_dx2
        self.d2_dy2 = d2_dy2
        self.d2_dxdy = d2_dxdy
        self.d_dx = d_dx
        self.d_dy = d_dy

        self.forces = []
        self.stresses = []
        for _ in range(n_samples):
            phi = self._random_airy()
            stress = self._airy_to_stress(phi)
            force = self._stress_to_force(stress)
            self.forces.append(force)
            self.stresses.append(stress)

        self.forces = torch.stack(self.forces)
        self.stresses = torch.stack(self.stresses)

    def _random_airy(self):
        """Generate a random smooth Airy stress function."""
        phi = torch.zeros_like(self.xx)

        # Gaussian bumps
        n_bumps = torch.randint(3, 8, (1,)).item()
        for _ in range(n_bumps):
            cx = (torch.rand(1) * 2 - 1).item() * 0.7
            cy = (torch.rand(1) * 2 - 1).item() * 0.7
            sigma = 0.15 + torch.rand(1).item() * 0.25
            amplitude = (torch.rand(1) * 2 - 1).item() * 2.0
            phi += amplitude * torch.exp(
                -((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2 * sigma ** 2)
            )

        # Sinusoidal modes
        n_modes = torch.randint(1, 4, (1,)).item()
        for _ in range(n_modes):
            kx = (torch.rand(1) * 4 - 2).item() * np.pi
            ky = (torch.rand(1) * 4 - 2).item() * np.pi
            amplitude = (torch.rand(1) * 2 - 1).item() * 0.5
            phi += amplitude * torch.cos(kx * self.xx + ky * self.yy)

        return phi

    def _airy_to_stress(self, phi):
        """Compute stress tensor from Airy stress function."""
        sxx = _apply_kernel(phi, self.d2_dy2)
        syy = _apply_kernel(phi, self.d2_dx2)
        sxy = -_apply_kernel(phi, self.d2_dxdy)
        return torch.stack([sxx, syy, sxy], dim=0)  # (3, H, W)

    def _stress_to_force(self, stress):
        """Compute body force from divergence of stress: f = -div(σ)."""
        sxx, syy, sxy = stress[0], stress[1], stress[2]

        # fx = -(∂σ_xx/∂x + ∂σ_xy/∂y)
        fx = -(_apply_kernel(sxx, self.d_dx) + _apply_kernel(sxy, self.d_dy))
        # fy = -(∂σ_xy/∂x + ∂σ_yy/∂y)
        fy = -(_apply_kernel(sxy, self.d_dx) + _apply_kernel(syy, self.d_dy))

        return torch.stack([fx, fy], dim=0)  # (2, H, W)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        force = self.forces[idx]
        stress = self.stresses[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            force = rotate_force_field_single(force, theta)
            stress = rotate_stress_field_single(stress, theta)

        return force, stress


class AnisotropicStressDataset(Dataset):
    """
    Force field → stress tensor for an anisotropic 2D elastic medium.

    Same as IsotropicStressDataset but with a preferred material direction
    (fiber orientation along x-axis). The material is stiffer in x than y,
    so the stress response is direction-dependent.

    This breaks rotation symmetry: rotating the force doesn't simply rotate
    the stress, because the material stiffness is orientation-dependent.

    The anisotropy is implemented by scaling σ_xx by a stiffness ratio
    and adding a coupling term proportional to the loading direction.
    """

    def __init__(self, n_samples=10000, grid_size=GRID_SIZE, rotate=False, stiffness_ratio=2.0):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.rotate = rotate
        self.stiffness_ratio = stiffness_ratio
        self.dx = 2.0 / (grid_size - 1)

        coords = torch.linspace(-1, 1, grid_size)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

        d2_dx2, d2_dy2, d2_dxdy, d_dx, d_dy = _make_kernels(torch.device('cpu'), self.dx)
        self.d2_dx2 = d2_dx2
        self.d2_dy2 = d2_dy2
        self.d2_dxdy = d2_dxdy
        self.d_dx = d_dx
        self.d_dy = d_dy

        self.forces = []
        self.stresses = []
        for _ in range(n_samples):
            phi = self._random_airy()
            stress = self._airy_to_anisotropic_stress(phi)
            force = self._stress_to_force(stress)
            self.forces.append(force)
            self.stresses.append(stress)

        self.forces = torch.stack(self.forces)
        self.stresses = torch.stack(self.stresses)

    def _random_airy(self):
        phi = torch.zeros_like(self.xx)

        n_bumps = torch.randint(3, 8, (1,)).item()
        for _ in range(n_bumps):
            cx = (torch.rand(1) * 2 - 1).item() * 0.7
            cy = (torch.rand(1) * 2 - 1).item() * 0.7
            sigma = 0.15 + torch.rand(1).item() * 0.25
            amplitude = (torch.rand(1) * 2 - 1).item() * 2.0
            phi += amplitude * torch.exp(
                -((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2 * sigma ** 2)
            )

        n_modes = torch.randint(1, 4, (1,)).item()
        for _ in range(n_modes):
            kx = (torch.rand(1) * 4 - 2).item() * np.pi
            ky = (torch.rand(1) * 4 - 2).item() * np.pi
            amplitude = (torch.rand(1) * 2 - 1).item() * 0.5
            phi += amplitude * torch.cos(kx * self.xx + ky * self.yy)

        return phi

    def _airy_to_anisotropic_stress(self, phi):
        """
        Compute stress with anisotropy: material is stiffer along x-axis.

        The x-direction stress is amplified by the stiffness ratio,
        and a coupling term is added that depends on the x-gradient,
        breaking rotation symmetry.
        """
        sxx = _apply_kernel(phi, self.d2_dy2) * self.stiffness_ratio
        syy = _apply_kernel(phi, self.d2_dx2)
        sxy = -_apply_kernel(phi, self.d2_dxdy)

        # Additional directional coupling: fibers along x resist shear asymmetrically
        dphi_dx = _apply_kernel(phi, self.d_dx)
        sxy = sxy + 0.3 * dphi_dx

        return torch.stack([sxx, syy, sxy], dim=0)

    def _stress_to_force(self, stress):
        sxx, syy, sxy = stress[0], stress[1], stress[2]
        fx = -(_apply_kernel(sxx, self.d_dx) + _apply_kernel(sxy, self.d_dy))
        fy = -(_apply_kernel(sxy, self.d_dx) + _apply_kernel(syy, self.d_dy))
        return torch.stack([fx, fy], dim=0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        force = self.forces[idx]
        stress = self.stresses[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            force = rotate_force_field_single(force, theta)
            stress = rotate_stress_field_single(stress, theta)

        return force, stress


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "isotropic":
        trainset = IsotropicStressDataset(n_samples=50000, rotate=rotation)
        testset = IsotropicStressDataset(n_samples=5000, rotate=False)
    elif dataset == "anisotropic":
        trainset = AnisotropicStressDataset(n_samples=50000, rotate=rotation)
        testset = AnisotropicStressDataset(n_samples=5000, rotate=False)
    else:
        raise ValueError(f"No such dataset: {dataset}")

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print(f"[{dataset}] Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    # in_channels=2 (force field), out_channels=3 (stress tensor)
    if model == "vit":
        net = ViT(image_size=GRID_SIZE, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128, in_channels=2, out_channels=3)
    elif model == "naive":
        net = NaiveNet(in_channels=2, out_channels=3)
    elif model == "cnn":
        net = CNN(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=2, out_channels=3)
    elif model == "unet":
        net = UNet(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=2, out_channels=3)
    else:
        raise ValueError(f"No such model {model}")
    net = net.to(device)
    if finetune:
        net.load_state_dict(torch.load(finetune, weights_only=True))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    statistics = {
        "equivariant_loss": [],
        "train_loss": [],
        "test_loss": []
    }

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    update_statistics(net, criterion, statistics, trainloader, testloader, device)

    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            force, stress = data
            force = force.to(device)
            stress = stress.to(device)

            optimizer.zero_grad()

            output, _ = net(force)
            loss = criterion(output, stress)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}")
        net.eval()
        update_statistics(net, criterion, statistics, trainloader, testloader, device)

    print('Finished Training')

    tag = f"stress_prediction_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}{'_finetuned' if finetune else ''}"
    torch.save(net.state_dict(), f"models/{tag}_model.pth")
    with open(f"results/{tag}_statistics.json", "wt+") as f:
        json.dump(statistics, f)


def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    running_train_loss = 0.0
    for data in trainloader:
        force, stress = data
        force = force.to(device)
        stress = stress.to(device)

        output, layers = net(force)
        running_train_loss += criterion(output, stress).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = {k: [UnifiedEquivarianceTracker(device) for _ in range(3)] for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            force, stress = data
            force = force.to(device)
            stress = stress.to(device)

            output, layers = net(force)
            running_test_loss += criterion(output, stress).item()

            force_rot90 = torch.rot90(force, 1, dims=(-2, -1))
            force_rot180 = torch.rot90(force, 2, dims=(-2, -1))
            force_rot270 = torch.rot90(force, 3, dims=(-2, -1))

            *_, layers_rot90 = net(force_rot90)
            *_, layers_rot180 = net(force_rot180)
            *_, layers_rot270 = net(force_rot270)

            for key in layers.keys():
                for idx, layer in enumerate([layers_rot90, layers_rot180, layers_rot270]):
                    running_equivariant_error[key][idx].update(layers[key], layer[key])
    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: np.mean([x.compute() for x in v]) for k, v in running_equivariant_error.items()})


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("isotropic", "anisotropic"), default="isotropic")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)