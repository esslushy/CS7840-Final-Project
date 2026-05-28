import torch
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from HSIC import cka
from utils import split_array_randomly
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import os

NUM_EPOCHS = 200
BATCH_SIZE = 64
GRID_SIZE = 32
DT = 0.1
NUM_EVAL_ANGLES = 16  # number of evenly-spaced SO(2) elements for equivariance eval


def so2_eval_angles(n=NUM_EVAL_ANGLES):
    """
    Sample n evenly-spaced elements of SO(2) via the Lie algebra.

    SO(2) has a single generator J = [[0, -1], [1, 0]].
    The group elements are exp(t * J) = rotation by angle t.
    We sample t_k = 2π * k / (n+1) for k = 1, ..., n, which gives n
    uniformly spaced rotations excluding the identity (t=0).
    """
    return torch.tensor([2 * np.pi * k / (n + 1) for k in range(1, n + 1)])


# ---------------------------------------------------------------------------
# Continuous rotation utilities for vector fields on grids
# ---------------------------------------------------------------------------

def make_rotation_matrix(theta):
    """2x2 rotation matrix for angle theta (radians). Works on scalar or batch."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([c, -s, s, c], dim=-1).view(*theta.shape, 2, 2)


def rotate_grid(grid_size, theta, device):
    """
    Build a sampling grid that rotates a (grid_size x grid_size) image
    by angle theta around its center.

    Args:
        grid_size: int
        theta: scalar tensor, angle in radians
        device: torch device

    Returns:
        grid: (1, H, W, 2) sampling grid for F.grid_sample
    """
    coords = torch.linspace(-1, 1, grid_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    # grid_sample expects (x, y) ordering
    xy = torch.stack([xx, yy], dim=-1)  # (H, W, 2)

    R = make_rotation_matrix(theta)  # (2, 2)
    # Rotate coordinates: new_xy = R^{-1} @ xy (inverse rotation to find source)
    R_inv = R.mT  # orthogonal, so inverse = transpose
    xy_rot = (xy.view(-1, 2) @ R_inv.mT).view(grid_size, grid_size, 2)

    return xy_rot.unsqueeze(0)  # (1, H, W, 2)


def rotate_vector_field_continuous(field, theta):
    """
    Rotate a 2D vector field by a continuous angle theta.

    This does two things:
    1. Spatially rotates the grid (where each vector lives)
    2. Rotates the vectors themselves (their direction)

    Args:
        field: (B, 2, H, W) vector field
        theta: scalar tensor, angle in radians

    Returns:
        rotated: (B, 2, H, W)
    """
    B, _, H, W = field.shape
    device = field.device

    # 1. Spatial rotation via grid_sample
    grid = rotate_grid(H, theta, device).expand(B, -1, -1, -1)
    rotated = Fn.grid_sample(
        field, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    # 2. Rotate the vector components
    c = torch.cos(theta)
    s = torch.sin(theta)
    vx, vy = rotated[:, 0:1], rotated[:, 1:2]
    new_vx = c * vx - s * vy
    new_vy = s * vx + c * vy
    return torch.cat([new_vx, new_vy], dim=1)


def rotate_vector_field_single(field, theta):
    """Unbatched version: field is (2, H, W)."""
    return rotate_vector_field_continuous(field.unsqueeze(0), theta).squeeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FluidFlowDataset(Dataset):
    """
    Synthetic 2D fluid flow dataset.

    Each sample is a pair (velocity_t, velocity_t+dt) of 2D vector fields
    on a regular grid. Flows are built from superpositions of analytical
    primitives: vortices, uniform streams, sources, sinks, and shear flows.

    The time-stepping uses semi-Lagrangian advection: to find the velocity
    at position x at time t+dt, we trace backward along the current velocity
    to find where the fluid parcel came from, and read off the velocity there.
    """

    def __init__(self, n_samples=10000, grid_size=GRID_SIZE, dt=DT, rotate=False):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.dt = dt
        self.rotate = rotate

        # Build coordinate grid in [-1, 1]
        coords = torch.linspace(-1, 1, grid_size)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

        # Pre-generate all samples
        self.fields_t = []
        self.fields_tp1 = []
        for _ in range(n_samples):
            vx, vy = self._random_flow()
            field_t = torch.stack([vx, vy], dim=0)  # (2, H, W)

            # Semi-Lagrangian advection
            field_tp1 = self._advect(field_t)

            self.fields_t.append(field_t)
            self.fields_tp1.append(field_tp1)

        self.fields_t = torch.stack(self.fields_t)
        self.fields_tp1 = torch.stack(self.fields_tp1)

    def _random_flow(self):
        """Build a velocity field from random superposition of primitives."""
        vx = torch.zeros_like(self.xx)
        vy = torch.zeros_like(self.xx)

        n_primitives = torch.randint(1, 5, (1,)).item()
        for _ in range(n_primitives):
            kind = torch.randint(0, 5, (1,)).item()
            # Random center
            cx = (torch.rand(1) * 2 - 1).item() * 0.6
            cy = (torch.rand(1) * 2 - 1).item() * 0.6
            strength = (torch.rand(1) * 2 - 1).item() * 0.5

            dx = self.xx - cx
            dy = self.yy - cy
            r2 = dx ** 2 + dy ** 2 + 1e-4  # avoid singularity

            if kind == 0:
                # Vortex (rotation around center)
                vx += strength * (-dy) / r2.sqrt()
                vy += strength * dx / r2.sqrt()
            elif kind == 1:
                # Uniform flow
                angle = torch.rand(1).item() * 2 * np.pi
                vx += strength * np.cos(angle)
                vy += strength * np.sin(angle)
            elif kind == 2:
                # Source / sink
                vx += strength * dx / r2
                vy += strength * dy / r2
            elif kind == 3:
                # Rankine vortex (solid body rotation inside core)
                r = r2.sqrt()
                core = 0.3
                scale = torch.where(r < core, r / core, core / r) * strength
                vx += scale * (-dy) / (r + 1e-6)
                vy += scale * dx / (r + 1e-6)
            else:
                # Shear flow
                axis = torch.randint(0, 2, (1,)).item()
                if axis == 0:
                    vx += strength * self.yy
                else:
                    vy += strength * self.xx

        return vx, vy

    def _advect(self, field):
        """
        Semi-Lagrangian advection: trace each grid point backward by
        -dt * velocity, then interpolate the velocity at the departure point.
        """
        vx, vy = field[0], field[1]

        # Departure points (trace backward)
        dep_x = self.xx - self.dt * vx
        dep_y = self.yy - self.dt * vy

        # grid_sample expects (x, y) ordering
        grid = torch.stack([dep_x, dep_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        field_in = field.unsqueeze(0)  # (1, 2, H, W)

        advected = Fn.grid_sample(
            field_in, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        return advected.squeeze(0)  # (2, H, W)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        field_t = self.fields_t[idx]
        field_tp1 = self.fields_tp1[idx]

        if self.rotate:
            # Continuous random rotation
            theta = torch.rand(1) * 2 * np.pi
            field_t = rotate_vector_field_single(field_t, theta)
            field_tp1 = rotate_vector_field_single(field_tp1, theta)

        return field_t, field_tp1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, kernel: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from Models.FluidFlowNets import UNet, CNN, ViT, NaiveNet

    trainset = FluidFlowDataset(n_samples=50000, rotate=rotation)
    testset = FluidFlowDataset(n_samples=5000, rotate=False)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print(f"Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    if model == "vit":
        net = ViT(image_size=GRID_SIZE, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128)
    elif model == "naive":
        net = NaiveNet()
    elif model == "cnn":
        net = CNN(width1=256 if thicker else 120, width2=256 if thicker else 84)
    elif model == "unet":
        net = UNet(width1=256 if thicker else 120, width2=256 if thicker else 84)
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
        "test_loss": [],
        "baseline_cka": []
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    update_statistics(kernel, net, criterion, statistics, trainloader, testloader, device)

    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            field_t, field_tp1 = data
            field_t = field_t.to(device)
            field_tp1 = field_tp1.to(device)

            optimizer.zero_grad()

            pred, _ = net(field_t)
            loss = criterion(pred, field_tp1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}")
        net.eval()
        update_statistics(kernel, net, criterion, statistics, trainloader, testloader, device)

    print('Finished Training')

    tag = f"fluid_flow_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_kernel_{kernel}{'_finetuned' if finetune else ''}"
    torch.save(net.state_dict(), f"models/{tag}_model.pth")
    with open(f"results/{tag}_statistics.json", "wt+") as f:
        json.dump(statistics, f)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def update_statistics(kernel, net, criterion, statistics, trainloader, testloader, device):
    running_train_loss = 0.0
    for data in trainloader:
        field_t, field_tp1 = data
        field_t = field_t.to(device)
        field_tp1 = field_tp1.to(device)

        pred, _ = net(field_t)
        running_train_loss += criterion(pred, field_tp1).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = defaultdict(float)
    running_cka_baseline = defaultdict(float)

    # Deterministic angles: evenly-spaced SO(2) elements via Lie algebra
    eval_angles = so2_eval_angles()

    with torch.inference_mode():
        for data in testloader:
            field_t, field_tp1 = data
            field_t = field_t.to(device)
            field_tp1 = field_tp1.to(device)

            pred, layers = net(field_t)
            running_test_loss += criterion(pred, field_tp1).item()

            # Evaluate equivariance at evenly-spaced continuous angles
            rotated_layers_list = []
            for theta in eval_angles:
                theta_t = theta.to(device)
                field_rotated = rotate_vector_field_continuous(field_t, theta_t)
                _, layers_rotated = net(field_rotated)
                rotated_layers_list.append(layers_rotated)

            for key in layers.keys():
                cka_scores = []
                for layers_rotated in rotated_layers_list:
                    cka_scores.append(
                        cka(layers[key], layers_rotated[key], kernel=kernel).item()
                    )
                running_equivariant_error[key] += np.mean(cka_scores)

                layers_x, layers_y = split_array_randomly(layers[key])
                running_cka_baseline[key] += cka(layers_x, layers_y, kernel=kernel).item()

    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: v / len(testloader) for k,v in running_equivariant_error.items()})
    statistics["baseline_cka"].append({k: v / len(testloader) for k,v in running_cka_baseline.items()})


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--kernel", help="Which kernel to use for CKA", type=str, choices=["rbf", "linear"], default="linear")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.kernel, args.rotation, args.thicker, args.finetune)