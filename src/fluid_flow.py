import torch
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from utils import UnifiedEquivarianceTracker
from torch.utils.data import Dataset, DataLoader
from Models.FluidFlowNets import UNet, CNN, ViT, NaiveNet
import os

NUM_EPOCHS = 200
BATCH_SIZE = 64
GRID_SIZE = 32
DT = 0.1
GRAVITY = 5.0
DIFFUSIVITY = 0.01
NUM_BUOYANT_STEPS = 5  # accumulate multiple steps so gravity dominates
NUM_EVAL_ANGLES = 16


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
# Continuous rotation utilities
# ---------------------------------------------------------------------------

def make_rotation_matrix(theta):
    """2x2 rotation matrix for angle theta (radians)."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([c, -s, s, c], dim=-1).view(*theta.shape, 2, 2)


def rotate_grid(grid_size, theta, device):
    """Build a sampling grid that rotates by theta around center."""
    coords = torch.linspace(-1, 1, grid_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)

    R = make_rotation_matrix(theta)
    R_inv = R.mT
    xy_rot = (xy.view(-1, 2) @ R_inv.mT).view(grid_size, grid_size, 2)

    return xy_rot.unsqueeze(0)


def rotate_vector_field_continuous(field, theta):
    """
    Rotate a 2D vector field by a continuous angle theta.

    Spatially rotates the grid and rotates the vector components.

    Args:
        field: (B, 2, H, W) vector field
        theta: scalar tensor, angle in radians

    Returns:
        rotated: (B, 2, H, W)
    """
    B, _, H, W = field.shape
    device = field.device

    grid = rotate_grid(H, theta, device).expand(B, -1, -1, -1)
    rotated = Fn.grid_sample(
        field, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    c = torch.cos(theta)
    s = torch.sin(theta)
    vx, vy = rotated[:, 0:1], rotated[:, 1:2]
    new_vx = c * vx - s * vy
    new_vy = s * vx + c * vy
    return torch.cat([new_vx, new_vy], dim=1)


def rotate_vector_field_single(field, theta):
    """Unbatched version: field is (2, H, W)."""
    return rotate_vector_field_continuous(field.unsqueeze(0), theta).squeeze(0)


def rotate_buoyant_field_continuous(field, theta):
    """
    Rotate a buoyant flow state (vx, vy, T) by a continuous angle theta.

    Velocity (channels 0-1): spatially rotated + vector components rotated.
    Temperature (channel 2): scalar field, only spatially rotated.

    Args:
        field: (B, 3, H, W)
        theta: scalar tensor

    Returns:
        rotated: (B, 3, H, W)
    """
    B, _, H, W = field.shape
    device = field.device

    grid = rotate_grid(H, theta, device).expand(B, -1, -1, -1)
    rotated = Fn.grid_sample(
        field, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    c = torch.cos(theta)
    s = torch.sin(theta)
    vx, vy = rotated[:, 0:1], rotated[:, 1:2]
    new_vx = c * vx - s * vy
    new_vy = s * vx + c * vy

    T = rotated[:, 2:3]
    return torch.cat([new_vx, new_vy, T], dim=1)


def rotate_buoyant_field_single(field, theta):
    """Unbatched version: field is (3, H, W)."""
    return rotate_buoyant_field_continuous(field.unsqueeze(0), theta).squeeze(0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class IsotropicFlowDataset(Dataset):
    """
    Synthetic 2D isotropic fluid flow. No preferred direction.

    Each sample: (velocity_t, velocity_t+dt) as (2, H, W) fields.
    Flows built from superpositions of vortices, uniform streams,
    sources/sinks, Rankine vortices, and shear flows.
    Time-stepped via semi-Lagrangian advection.
    """

    def __init__(self, n_samples=10000, grid_size=GRID_SIZE, dt=DT, rotate=False):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.dt = dt
        self.rotate = rotate

        coords = torch.linspace(-1, 1, grid_size)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

        self.fields_t = []
        self.fields_tp1 = []
        for _ in range(n_samples):
            vx, vy = self._random_flow()
            field_t = torch.stack([vx, vy], dim=0)
            field_tp1 = self._advect(field_t)
            self.fields_t.append(field_t)
            self.fields_tp1.append(field_tp1)

        self.fields_t = torch.stack(self.fields_t)
        self.fields_tp1 = torch.stack(self.fields_tp1)

    def _random_flow(self):
        vx = torch.zeros_like(self.xx)
        vy = torch.zeros_like(self.xx)

        n_primitives = torch.randint(1, 5, (1,)).item()
        for _ in range(n_primitives):
            kind = torch.randint(0, 5, (1,)).item()
            cx = (torch.rand(1) * 2 - 1).item() * 0.6
            cy = (torch.rand(1) * 2 - 1).item() * 0.6
            strength = (torch.rand(1) * 2 - 1).item() * 0.5

            dx = self.xx - cx
            dy = self.yy - cy
            r2 = dx ** 2 + dy ** 2 + 1e-4

            if kind == 0:
                vx += strength * (-dy) / r2.sqrt()
                vy += strength * dx / r2.sqrt()
            elif kind == 1:
                angle = torch.rand(1).item() * 2 * np.pi
                vx += strength * np.cos(angle)
                vy += strength * np.sin(angle)
            elif kind == 2:
                vx += strength * dx / r2
                vy += strength * dy / r2
            elif kind == 3:
                r = r2.sqrt()
                core = 0.3
                scale = torch.where(r < core, r / core, core / r) * strength
                vx += scale * (-dy) / (r + 1e-6)
                vy += scale * dx / (r + 1e-6)
            else:
                axis = torch.randint(0, 2, (1,)).item()
                if axis == 0:
                    vx += strength * self.yy
                else:
                    vy += strength * self.xx

        return vx, vy

    def _advect(self, field):
        vx, vy = field[0], field[1]
        dep_x = self.xx - self.dt * vx
        dep_y = self.yy - self.dt * vy
        grid = torch.stack([dep_x, dep_y], dim=-1).unsqueeze(0)
        field_in = field.unsqueeze(0)
        advected = Fn.grid_sample(
            field_in, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        return advected.squeeze(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        field_t = self.fields_t[idx]
        field_tp1 = self.fields_tp1[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            field_t = rotate_vector_field_single(field_t, theta)
            field_tp1 = rotate_vector_field_single(field_tp1, theta)

        return field_t, field_tp1


class BuoyantFlowDataset(Dataset):
    """
    Synthetic 2D buoyant flow with gravity. Strongly breaks rotation symmetry.

    Key differences from IsotropicFlowDataset:
    - Background stratification: warm at bottom (y=-1), cold at top (y=+1)
    - Temperature blobs are biased: warm blobs placed low, cold blobs placed high
    - Multiple time steps are accumulated so gravity dominates the dynamics
    - Stronger gravity constant

    A model cannot learn rotation equivariance from this dataset because
    gravity picks out the y-axis, and the initial conditions themselves
    have a strong vertical bias.
    """

    def __init__(self, n_samples=10000, grid_size=GRID_SIZE, dt=DT, rotate=False):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.dt = dt
        self.rotate = rotate
        self.gravity = GRAVITY
        self.diffusivity = DIFFUSIVITY
        self.n_steps = NUM_BUOYANT_STEPS

        coords = torch.linspace(-1, 1, grid_size)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")
        self.dx = 2.0 / (grid_size - 1)

        self.laplacian_kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3) / (self.dx ** 2)

        self.states_t = []
        self.states_tp1 = []
        for _ in range(n_samples):
            state_t = self._random_state()
            # Accumulate multiple time steps so gravity reshapes the flow
            state_tp1 = state_t
            for _ in range(self.n_steps):
                state_tp1 = self._step(state_tp1)
            self.states_t.append(state_t)
            self.states_tp1.append(state_tp1)

        self.states_t = torch.stack(self.states_t)
        self.states_tp1 = torch.stack(self.states_tp1)

    def _random_state(self):
        """
        Generate initial state with strong vertical bias.

        Temperature has background stratification (warm bottom, cold top)
        plus biased blobs: warm blobs placed in the lower half,
        cold blobs in the upper half.
        """
        vx = torch.zeros_like(self.xx)
        vy = torch.zeros_like(self.xx)

        # Small random velocity perturbations (weak compared to gravity)
        n_vel = torch.randint(1, 3, (1,)).item()
        for _ in range(n_vel):
            kind = torch.randint(0, 3, (1,)).item()
            cx = (torch.rand(1) * 2 - 1).item() * 0.6
            cy = (torch.rand(1) * 2 - 1).item() * 0.6
            strength = (torch.rand(1) * 2 - 1).item() * 0.15

            dx = self.xx - cx
            dy = self.yy - cy
            r2 = dx ** 2 + dy ** 2 + 1e-4

            if kind == 0:
                vx += strength * (-dy) / r2.sqrt()
                vy += strength * dx / r2.sqrt()
            elif kind == 1:
                angle = torch.rand(1).item() * 2 * np.pi
                vx += strength * np.cos(angle)
                vy += strength * np.sin(angle)
            else:
                vx += strength * dx / r2
                vy += strength * dy / r2

        # Background stratification: warm at bottom (-y), cold at top (+y)
        T = -self.yy.clone() * 0.5

        # Biased blobs: warm blobs in lower half, cold in upper half
        n_blobs = torch.randint(3, 7, (1,)).item()
        for _ in range(n_blobs):
            cx = (torch.rand(1) * 2 - 1).item() * 0.7
            is_warm = torch.rand(1).item() > 0.5
            if is_warm:
                # Warm blob biased toward bottom (y < 0)
                cy = -0.2 - torch.rand(1).item() * 0.6
                amplitude = 0.3 + torch.rand(1).item() * 0.7
            else:
                # Cold blob biased toward top (y > 0)
                cy = 0.2 + torch.rand(1).item() * 0.6
                amplitude = -(0.3 + torch.rand(1).item() * 0.7)

            sigma = 0.08 + torch.rand(1).item() * 0.15
            blob = amplitude * torch.exp(
                -((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2 * sigma ** 2)
            )
            T += blob

        return torch.stack([vx, vy, T], dim=0)

    def _step(self, state):
        vx, vy, T = state[0], state[1], state[2]

        dep_x = self.xx - self.dt * vx
        dep_y = self.yy - self.dt * vy
        grid = torch.stack([dep_x, dep_y], dim=-1).unsqueeze(0)
        state_in = state.unsqueeze(0)
        advected = Fn.grid_sample(
            state_in, grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0)

        vx_new, vy_new, T_new = advected[0], advected[1], advected[2]

        # Buoyancy: gravity breaks rotation symmetry
        vy_new = vy_new + self.gravity * T_new * self.dt

        # Thermal diffusion
        T_padded = T_new.unsqueeze(0).unsqueeze(0)
        lap_T = Fn.conv2d(T_padded, self.laplacian_kernel, padding=1).squeeze(0).squeeze(0)
        T_new = T_new + self.diffusivity * lap_T * self.dt

        return torch.stack([vx_new, vy_new, T_new], dim=0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        state_t = self.states_t[idx]
        state_tp1 = self.states_tp1[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            state_t = rotate_buoyant_field_single(state_t, theta)
            state_tp1 = rotate_buoyant_field_single(state_tp1, theta)

        return state_t, state_tp1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "isotropic":
        trainset = IsotropicFlowDataset(n_samples=50000, rotate=rotation)
        testset = IsotropicFlowDataset(n_samples=5000, rotate=False)
        rotate_fn = rotate_vector_field_continuous
        channels = 2
    elif dataset == "buoyant":
        trainset = BuoyantFlowDataset(n_samples=50000, rotate=rotation)
        testset = BuoyantFlowDataset(n_samples=5000, rotate=False)
        rotate_fn = rotate_buoyant_field_continuous
        channels = 3
    else:
        raise Exception("Unknown Dataset")

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print(f"[{dataset}] Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    if model == "vit":
        net = ViT(image_size=GRID_SIZE, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128, channels=channels)
    elif model == "naive":
        net = NaiveNet(channels=channels)
    elif model == "cnn":
        net = CNN(width1=256 if thicker else 120, width2=256 if thicker else 84, channels=channels)
    elif model == "unet":
        net = UNet(width1=256 if thicker else 120, width2=256 if thicker else 84, channels=channels)
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

    update_statistics(net, criterion, statistics, trainloader, testloader, device, rotate_fn)

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
        update_statistics(net, criterion, statistics, trainloader, testloader, device, rotate_fn)

    print('Finished Training')

    tag = f"fluid_flow_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}{'_finetuned' if finetune else ''}"
    torch.save(net.state_dict(), f"models/{tag}_model.pth")
    with open(f"results/{tag}_statistics.json", "wt+") as f:
        json.dump(statistics, f)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def update_statistics(net, criterion, statistics, trainloader, testloader, device, rotate_fn):
    net.eval()
    running_train_loss = 0.0
    for data in trainloader:
        field_t, field_tp1 = data
        field_t = field_t.to(device)
        field_tp1 = field_tp1.to(device)

        pred, layers = net(field_t)
        running_train_loss += criterion(pred, field_tp1).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = {k: [UnifiedEquivarianceTracker(device) for _ in range(NUM_EVAL_ANGLES)] for k in layers.keys()}

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
                field_rotated = rotate_fn(field_t, theta_t)
                _, layers_rotated = net(field_rotated)
                rotated_layers_list.append(layers_rotated)

            for key in layers.keys():
                for idx, layers_rotated in enumerate(rotated_layers_list):
                    running_equivariant_error[key][idx].update(layers[key], layers_rotated[key])

    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: np.mean([x.compute() for x in v]) for k,v in running_equivariant_error.items()})

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("isotropic", "buoyant"), default="isotropic")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)