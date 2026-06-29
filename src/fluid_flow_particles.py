import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from utils import UnifiedEquivarianceTracker
from torch.utils.data import Dataset, DataLoader
import os

NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_PARTICLES = 128
DT = 0.1
GRAVITY = 5.0
NUM_BUOYANT_STEPS = 5
NUM_EVAL_ANGLES = 8


def so2_eval_angles(n=NUM_EVAL_ANGLES):
    """
    Sample n evenly-spaced elements of SO(2) via the Lie algebra.

    SO(2) has a single generator J = [[0, -1], [1, 0]].
    We sample t_k = 2π * k / (n+1) for k = 1, ..., n, excluding identity.
    """
    return torch.tensor([2 * np.pi * k / (n + 1) for k in range(1, n + 1)])


# ---------------------------------------------------------------------------
# Rotation utilities for particle states
# ---------------------------------------------------------------------------

def rotate_2d(vecs, theta):
    """
    Rotate 2D vectors by angle theta.

    Args:
        vecs: (..., 2)
        theta: scalar tensor

    Returns:
        rotated: (..., 2)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    x, y = vecs[..., 0], vecs[..., 1]
    new_x = c * x - s * y
    new_y = s * x + c * y
    return torch.stack([new_x, new_y], dim=-1)


def rotate_isotropic_particles(particles, theta):
    """
    Rotate isotropic particle state: (B, N, 4) = (x, y, vx, vy).

    Both position (x,y) and velocity (vx,vy) are 2D vectors that rotate.
    """
    pos = rotate_2d(particles[..., 0:2], theta)
    vel = rotate_2d(particles[..., 2:4], theta)
    return torch.cat([pos, vel], dim=-1)


def rotate_buoyant_particles(particles, theta):
    """
    Rotate buoyant particle state: (B, N, 5) = (x, y, vx, vy, T).

    Position and velocity rotate. Temperature is a scalar — unchanged.
    """
    pos = rotate_2d(particles[..., 0:2], theta)
    vel = rotate_2d(particles[..., 2:4], theta)
    T = particles[..., 4:5]
    return torch.cat([pos, vel, T], dim=-1)


# ---------------------------------------------------------------------------
# Analytical flow primitives (evaluate at arbitrary positions)
# ---------------------------------------------------------------------------

def eval_flow_at_points(positions, primitives):
    """
    Evaluate analytical velocity field at given positions.

    Args:
        positions: (N, 2) particle positions
        primitives: list of (kind, cx, cy, strength, extra) tuples

    Returns:
        velocities: (N, 2)
    """
    vx = torch.zeros(positions.shape[0], device=positions.device)
    vy = torch.zeros(positions.shape[0], device=positions.device)

    for kind, cx, cy, strength, extra in primitives:
        dx = positions[:, 0] - cx
        dy = positions[:, 1] - cy
        r2 = dx ** 2 + dy ** 2 + 1e-4

        if kind == "vortex":
            vx += strength * (-dy) / r2.sqrt()
            vy += strength * dx / r2.sqrt()
        elif kind == "uniform":
            angle = extra
            vx += strength * np.cos(angle)
            vy += strength * np.sin(angle)
        elif kind == "source":
            vx += strength * dx / r2
            vy += strength * dy / r2
        elif kind == "rankine":
            r = r2.sqrt()
            core = 0.3
            scale = torch.where(r < core, r / core, core / r) * strength
            vx += scale * (-dy) / (r + 1e-6)
            vy += scale * dx / (r + 1e-6)
        elif kind == "shear":
            axis = int(extra)
            if axis == 0:
                vx += strength * positions[:, 1]
            else:
                vy += strength * positions[:, 0]

    return torch.stack([vx, vy], dim=-1)


def random_primitives(n_primitives=None, max_strength=0.5):
    """Generate a random set of flow primitives."""
    if n_primitives is None:
        n_primitives = torch.randint(1, 5, (1,)).item()
    prims = []
    kinds = ["vortex", "uniform", "source", "rankine", "shear"]
    for _ in range(n_primitives):
        kind = kinds[torch.randint(0, len(kinds), (1,)).item()]
        cx = (torch.rand(1) * 2 - 1).item() * 0.6
        cy = (torch.rand(1) * 2 - 1).item() * 0.6
        strength = (torch.rand(1) * 2 - 1).item() * max_strength
        extra = torch.rand(1).item() * 2 * np.pi if kind == "uniform" else torch.randint(0, 2, (1,)).item()
        prims.append((kind, cx, cy, strength, extra))
    return prims


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class IsotropicParticleDataset(Dataset):
    """
    Lagrangian particle version of isotropic fluid flow.

    Each particle has state (x, y, vx, vy). Velocities are computed from
    analytical flow primitives. Particles are advected: new_pos = pos + vel * dt,
    then velocities are recomputed at the new positions.

    Input/output: (N, 4) per sample.
    """

    def __init__(self, n_samples=10000, num_particles=NUM_PARTICLES, dt=DT, rotate=False):
        self.n_samples = n_samples
        self.num_particles = num_particles
        self.dt = dt
        self.rotate = rotate

        self.states_t = []
        self.states_tp1 = []
        for _ in range(n_samples):
            prims = random_primitives()
            pos = (torch.rand(num_particles, 2) * 2 - 1) * 0.8
            vel = eval_flow_at_points(pos, prims)
            state_t = torch.cat([pos, vel], dim=-1)  # (N, 4)

            # Advect
            new_pos = pos + vel * dt
            new_pos = new_pos.clamp(-1, 1)
            new_vel = eval_flow_at_points(new_pos, prims)
            state_tp1 = torch.cat([new_pos, new_vel], dim=-1)

            self.states_t.append(state_t)
            self.states_tp1.append(state_tp1)

        self.states_t = torch.stack(self.states_t)
        self.states_tp1 = torch.stack(self.states_tp1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        state_t = self.states_t[idx]
        state_tp1 = self.states_tp1[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            state_t = rotate_isotropic_particles(state_t, theta)
            state_tp1 = rotate_isotropic_particles(state_tp1, theta)

        return state_t, state_tp1


class BuoyantParticleDataset(Dataset):
    """
    Lagrangian particle version of buoyant flow with gravity.

    Each particle has state (x, y, vx, vy, T). Temperature drives
    buoyancy: vy += gravity * T * dt. Background stratification and
    biased blob placement break rotation symmetry.

    Multiple steps are accumulated so gravity dominates.

    Input/output: (N, 5) per sample.
    """

    def __init__(self, n_samples=10000, num_particles=NUM_PARTICLES, dt=DT, rotate=False):
        self.n_samples = n_samples
        self.num_particles = num_particles
        self.dt = dt
        self.rotate = rotate
        self.gravity = GRAVITY
        self.n_steps = NUM_BUOYANT_STEPS

        self.states_t = []
        self.states_tp1 = []
        for _ in range(n_samples):
            state_t = self._random_state()
            state_tp1 = state_t.clone()
            for _ in range(self.n_steps):
                state_tp1 = self._step(state_tp1)
            self.states_t.append(state_t)
            self.states_tp1.append(state_tp1)

        self.states_t = torch.stack(self.states_t)
        self.states_tp1 = torch.stack(self.states_tp1)

    def _random_state(self):
        """Generate particles with stratified temperature and biased blobs."""
        prims = random_primitives(max_strength=0.15)
        pos = (torch.rand(self.num_particles, 2) * 2 - 1) * 0.8
        vel = eval_flow_at_points(pos, prims)

        # Background stratification: warm at bottom, cold at top
        T = -pos[:, 1:2] * 0.5  # (N, 1)

        # Biased blobs
        n_blobs = torch.randint(3, 7, (1,)).item()
        for _ in range(n_blobs):
            cx = (torch.rand(1) * 2 - 1).item() * 0.7
            is_warm = torch.rand(1).item() > 0.5
            if is_warm:
                cy = -0.2 - torch.rand(1).item() * 0.6
                amplitude = 0.3 + torch.rand(1).item() * 0.7
            else:
                cy = 0.2 + torch.rand(1).item() * 0.6
                amplitude = -(0.3 + torch.rand(1).item() * 0.7)

            sigma = 0.08 + torch.rand(1).item() * 0.15
            dist2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
            T[:, 0] += amplitude * torch.exp(-dist2 / (2 * sigma ** 2))

        return torch.cat([pos, vel, T], dim=-1)  # (N, 5)

    def _step(self, state):
        pos = state[:, 0:2]
        vel = state[:, 2:4]
        T = state[:, 4:5]

        # Buoyancy: gravity breaks rotation symmetry
        buoyancy = torch.zeros_like(vel)
        buoyancy[:, 1] = self.gravity * T[:, 0] * self.dt
        vel = vel + buoyancy

        # Advect
        new_pos = pos + vel * self.dt
        new_pos = new_pos.clamp(-1, 1)

        return torch.cat([new_pos, vel, T], dim=-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        state_t = self.states_t[idx]
        state_tp1 = self.states_tp1[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            state_t = rotate_buoyant_particles(state_t, theta)
            state_tp1 = rotate_buoyant_particles(state_tp1, theta)

        return state_t, state_tp1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from Models.FluidFlowParticlesNets import PointNet, MLP, NaiveNet, SetTransformer

    if dataset == "isotropic":
        trainset = IsotropicParticleDataset(n_samples=50000, rotate=rotation)
        testset = IsotropicParticleDataset(n_samples=5000, rotate=False)
        rotate_fn = rotate_isotropic_particles
        channels = 4
    elif dataset == "buoyant":
        trainset = BuoyantParticleDataset(n_samples=50000, rotate=rotation)
        testset = BuoyantParticleDataset(n_samples=5000, rotate=False)
        rotate_fn = rotate_buoyant_particles
        channels = 5
    else:
        raise Exception("Unknown Dataset")

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print(f"[{dataset}] Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    if model == "pointnet":
        net = PointNet(width1=256 if thicker else 128, width2=256 if thicker else 64, channels=channels)
    elif model == "mlp":
        net = MLP(width1=256 if thicker else 128, width2=256 if thicker else 64, channels=channels)
    elif model == "naive":
        net = NaiveNet(channels=channels)
    elif model == "transformer":
        net = SetTransformer(dim=256 if thicker else 128, depth=1, heads=4, mlp_dim=256 if thicker else 128, channels=channels)
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
            state_t, state_tp1 = data
            state_t = state_t.to(device)
            state_tp1 = state_tp1.to(device)

            optimizer.zero_grad()

            pred, _ = net(state_t)
            loss = criterion(pred, state_tp1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}")
        update_statistics(net, criterion, statistics, trainloader, testloader, device, rotate_fn)

    print('Finished Training')

    tag = f"fluid_flow_particles_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}{'_finetuned' if finetune else ''}"
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
        state_t, state_tp1 = data
        state_t = state_t.to(device)
        state_tp1 = state_tp1.to(device)

        pred, layers = net(state_t)
        running_train_loss += criterion(pred, state_tp1).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = {k: [UnifiedEquivarianceTracker(device) for _ in range(NUM_EVAL_ANGLES)] for k in layers.keys()}

    eval_angles = so2_eval_angles()

    with torch.inference_mode():
        for data in testloader:
            state_t, state_tp1 = data
            state_t = state_t.to(device)
            state_tp1 = state_tp1.to(device)

            pred, layers = net(state_t)
            running_test_loss += criterion(pred, state_tp1).item()

            rotated_layers_list = []
            for theta in eval_angles:
                theta_t = theta.to(device)
                state_rotated = rotate_fn(state_t, theta_t)
                _, layers_rotated = net(state_rotated)
                rotated_layers_list.append(layers_rotated)

            for key in layers.keys():
                cka_scores = []
                for idx, layers_rotated in enumerate(rotated_layers_list):
                    running_equivariant_error[key][idx].update(layers[key], layers_rotated[key])

    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: np.mean([x.compute() for x in v]) for k,v in running_equivariant_error.items()})

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("pointnet", "mlp", "naive", "transformer"), default="pointnet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("isotropic", "buoyant"), default="isotropic")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)