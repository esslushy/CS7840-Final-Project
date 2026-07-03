import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from utils import EquivarianceTracker
from torch.utils.data import Dataset, DataLoader
from Models.StressPredictionParticlesNets import PointNet, MLP, NaiveNet, SetTransformer

NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_PARTICLES = 128
NUM_EVAL_ANGLES = 8
STIFFNESS_RATIO = 2.0


def so2_eval_angles(n=NUM_EVAL_ANGLES):
    """
    Sample n evenly-spaced elements of SO(2) via the Lie algebra.

    t_k = 2π * k / (n+1) for k = 1, ..., n, excluding the identity.
    Returns (radians_tensor, degree_labels), the labels being integer degrees
    used as JSON-friendly keys for the per-angle statistics.
    """
    ks = range(1, n + 1)
    radians = torch.tensor([2 * np.pi * k / (n + 1) for k in ks])
    degrees = [int(round(360.0 * k / (n + 1))) for k in ks]
    return radians, degrees


# ---------------------------------------------------------------------------
# Rotation utilities for particle states
# ---------------------------------------------------------------------------

def rotate_2d(vecs, theta):
    """Rotate 2D vectors by angle theta. vecs: (..., 2)."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    x, y = vecs[..., 0], vecs[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def rotate_stress_state(state, theta):
    """
    Rotate the input particle state (..., 4) = (x, y, fx, fy).

    Position and force are both 2D vectors and rotate together.
    """
    pos = rotate_2d(state[..., 0:2], theta)
    force = rotate_2d(state[..., 2:4], theta)
    return torch.cat([pos, force], dim=-1)


def rotate_stress_tensor(stress, theta):
    """
    Rotate the per-particle stress tensor (..., 3) = (σ_xx, σ_yy, σ_xy)
    as a rank-2 symmetric tensor: σ' = R σ R^T.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    c2, s2, cs = c * c, s * s, c * s
    sxx, syy, sxy = stress[..., 0], stress[..., 1], stress[..., 2]
    new_sxx = c2 * sxx + s2 * syy - 2 * cs * sxy
    new_syy = s2 * sxx + c2 * syy + 2 * cs * sxy
    new_sxy = cs * (sxx - syy) + (c2 - s2) * sxy
    return torch.stack([new_sxx, new_syy, new_sxy], dim=-1)


# ---------------------------------------------------------------------------
# Analytical stress/force primitives (evaluate at arbitrary positions)
# ---------------------------------------------------------------------------

def eval_stress_force_at_points(positions, primitives, aniso=None):
    """
    Evaluate the analytic stress tensor and its equilibrium body force at points.

    Each primitive is a Gaussian-weighted constant symmetric tensor:
        σ(r) = Σ_p A_p g_p(r) M_p,   g_p(r) = exp(-|r - c_p|² / 2 s_p²)
    The body force follows in closed form from f = -div(σ). Since M_p is constant,
        div(σ)_i = Σ_j (∂_j g) M_ij = -(g / s²) (M (r - c))_i
    so  f_i = (g / s²) (M (r - c))_i .

    This is analytic (no grid, no finite differences) and exactly rotation-
    equivariant: rotating c -> Rc and M -> R M R^T rotates f as a vector and
    σ as a rank-2 tensor.

    Args:
        positions:  (N, 2)
        primitives: list of (cx, cy, A, s, Mxx, Myy, Mxy)
        aniso:      optional (a, b) fixed diagonal scaling applied as M -> S M S^T
                    with S = diag(a, b). A non-identity, non-rotated S breaks
                    rotation equivariance (anisotropic material).

    Returns:
        force:  (N, 2) = (fx, fy)
        stress: (N, 3) = (σ_xx, σ_yy, σ_xy)
    """
    N = positions.shape[0]
    device = positions.device
    sxx = torch.zeros(N, device=device)
    syy = torch.zeros(N, device=device)
    sxy = torch.zeros(N, device=device)
    fx = torch.zeros(N, device=device)
    fy = torch.zeros(N, device=device)

    for (cx, cy, A, s, Mxx, Myy, Mxy) in primitives:
        dx = positions[:, 0] - cx
        dy = positions[:, 1] - cy
        g = A * torch.exp(-(dx * dx + dy * dy) / (2 * s * s))

        M00, M01, M11 = Mxx, Mxy, Myy
        if aniso is not None:
            a, b = aniso
            # M -> S M S^T with S = diag(a, b)
            M00, M01, M11 = a * a * M00, a * b * M01, b * b * M11

        # stress = g * M
        sxx = sxx + g * M00
        syy = syy + g * M11
        sxy = sxy + g * M01

        # force = -div(sigma) = (g / s^2) * M (r - c)
        inv_s2 = 1.0 / (s * s)
        fx = fx + g * inv_s2 * (M00 * dx + M01 * dy)
        fy = fy + g * inv_s2 * (M01 * dx + M11 * dy)

    force = torch.stack([fx, fy], dim=-1)          # (N, 2)
    stress = torch.stack([sxx, syy, sxy], dim=-1)  # (N, 3)
    return force, stress


def random_stress_primitives(n_primitives=None):
    """Generate a random set of stress primitives (isotropic distribution)."""
    if n_primitives is None:
        n_primitives = torch.randint(3, 8, (1,)).item()
    prims = []
    for _ in range(n_primitives):
        cx = (torch.rand(1) * 2 - 1).item() * 0.7
        cy = (torch.rand(1) * 2 - 1).item() * 0.7
        s = 0.15 + torch.rand(1).item() * 0.25
        A = (torch.rand(1) * 2 - 1).item() * 2.0
        Mxx = (torch.rand(1) * 2 - 1).item()
        Myy = (torch.rand(1) * 2 - 1).item()
        Mxy = (torch.rand(1) * 2 - 1).item()
        prims.append((cx, cy, A, s, Mxx, Myy, Mxy))
    return prims


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _StressParticleDataset(Dataset):
    """
    Force field -> stress tensor for a 2D elastic medium, sampled at particles.

    Input:  (N, 4) per sample = (x, y, fx, fy)
    Output: (N, 3) per sample = (σ_xx, σ_yy, σ_xy)

    aniso=None gives the isotropic (rotation-equivariant) medium; a fixed
    (a, b) scaling gives an anisotropic medium that breaks rotation symmetry.
    """

    def __init__(self, n_samples=10000, num_particles=NUM_PARTICLES, rotate=False, aniso=None):
        self.n_samples = n_samples
        self.num_particles = num_particles
        self.rotate = rotate
        self.aniso = aniso

        self.states = []
        self.stresses = []
        for _ in range(n_samples):
            prims = random_stress_primitives()
            pos = (torch.rand(num_particles, 2) * 2 - 1) * 0.8
            force, stress = eval_stress_force_at_points(pos, prims, aniso=aniso)
            self.states.append(torch.cat([pos, force], dim=-1))  # (N, 4)
            self.stresses.append(stress)                          # (N, 3)

        self.states = torch.stack(self.states)
        self.stresses = torch.stack(self.stresses)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        state = self.states[idx]
        stress = self.stresses[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            state = rotate_stress_state(state, theta)
            stress = rotate_stress_tensor(stress, theta)

        return state, stress


class IsotropicStressParticleDataset(_StressParticleDataset):
    def __init__(self, n_samples=10000, num_particles=NUM_PARTICLES, rotate=False):
        super().__init__(n_samples, num_particles, rotate, aniso=None)


class AnisotropicStressParticleDataset(_StressParticleDataset):
    def __init__(self, n_samples=10000, num_particles=NUM_PARTICLES, rotate=False,
                 stiffness_ratio=STIFFNESS_RATIO):
        # fixed (non-rotated) scaling S = diag(sqrt(ratio), 1) breaks equivariance
        super().__init__(n_samples, num_particles, rotate,
                         aniso=(float(np.sqrt(stiffness_ratio)), 1.0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "isotropic":
        trainset = IsotropicStressParticleDataset(n_samples=50000, rotate=rotation)
        testset = IsotropicStressParticleDataset(n_samples=5000, rotate=False)
    elif dataset == "anisotropic":
        trainset = AnisotropicStressParticleDataset(n_samples=50000, rotate=rotation)
        testset = AnisotropicStressParticleDataset(n_samples=5000, rotate=False)
    else:
        raise ValueError(f"No such dataset: {dataset}")

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print(f"[{dataset}] Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    # in_channels=4 (x, y, fx, fy), out_channels=3 (stress tensor)
    if model == "pointnet":
        net = PointNet(width1=256 if thicker else 128, width2=256 if thicker else 64, in_channels=4, out_channels=3)
    elif model == "mlp":
        net = MLP(width1=256 if thicker else 128, width2=256 if thicker else 64, in_channels=4, out_channels=3)
    elif model == "naive":
        net = NaiveNet(in_channels=4, out_channels=3)
    elif model == "transformer":
        net = SetTransformer(dim=256 if thicker else 128, depth=1, heads=4, mlp_dim=256 if thicker else 128, in_channels=4, out_channels=3)
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
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tag = (f"stress_prediction_particles_{'learned_equivariant' if rotation else 'non_equivariant'}"
           f"_{model}{'_thicker' if thicker else ''}_dataset_{dataset}"
           f"{'_finetuned' if finetune else ''}")

    # baseline measurement before any training, then persist immediately
    update_statistics(net, criterion, statistics, trainloader, testloader, device)
    save_all(net, statistics, tag)

    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            state, stress = data
            state = state.to(device)
            stress = stress.to(device)

            optimizer.zero_grad()

            pred, _ = net(state)
            loss = criterion(pred, stress)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}")

        # measure and persist EVERY epoch so an interrupted run keeps all stats so far
        update_statistics(net, criterion, statistics, trainloader, testloader, device)
        save_all(net, statistics, tag)

    print('Finished Training')
    # already saved each epoch; final save is just belt-and-suspenders
    save_all(net, statistics, tag)


def save_all(net, statistics, tag):
    """Persist statistics and model weights atomically (write-temp-then-rename),
    so an interruption mid-write cannot leave a corrupt file. Called every epoch."""
    stats_path = f"results/{tag}_statistics.json"
    tmp_stats = stats_path + ".tmp"
    with open(tmp_stats, "wt") as f:
        json.dump(statistics, f)
    os.replace(tmp_stats, stats_path)

    model_path = f"models/{tag}_model.pth"
    tmp_model = model_path + ".tmp"
    torch.save(net.state_dict(), tmp_model)
    os.replace(tmp_model, model_path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    net.eval()
    running_train_loss = 0.0
    for data in trainloader:
        state, stress = data
        state = state.to(device)
        stress = stress.to(device)

        pred, layers = net(state)
        running_train_loss += criterion(pred, stress).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    eval_angles, angle_labels = so2_eval_angles()

    # per-angle test loss (0 = unrotated). The force->stress map is EQUIVARIANT:
    # the input state (position + force) rotates as vectors, the target stress
    # as a rank-2 tensor.
    running_test_loss = {0: 0.0}
    for deg in angle_labels:
        running_test_loss[deg] = 0.0
    # per-layer, PER-ANGLE equivariance tracker (kept separate, not averaged)
    running_equivariant = {k: {deg: EquivarianceTracker(device) for deg in angle_labels}
                           for k in layers.keys()}

    with torch.inference_mode():
        for data in testloader:
            state, stress = data
            state = state.to(device)
            stress = stress.to(device)

            # unrotated prediction + activations (equivariance reference)
            pred, layers = net(state)
            running_test_loss[0] += criterion(pred, stress).item()

            for theta, deg in zip(eval_angles, angle_labels):
                theta_t = theta.to(device)
                state_rot = rotate_stress_state(state, theta_t)      # vectors: pos + force
                stress_rot = rotate_stress_tensor(stress, theta_t)   # rank-2 tensor
                pred_rot, layers_rot = net(state_rot)                # one forward, reused for loss + equivariance

                running_test_loss[deg] += criterion(pred_rot, stress_rot).item()

                for key in layers.keys():
                    running_equivariant[key][deg].update(layers[key], layers_rot[key])

    n_test = len(testloader)
    statistics["test_loss"].append({angle: v / n_test for angle, v in running_test_loss.items()})
    # store the full tracker stats (z, p, floor_std, ...) per layer, per angle
    statistics["equivariant_loss"].append({
        key: {deg: running_equivariant[key][deg].compute_stats() for deg in angle_labels}
        for key in running_equivariant
    })


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("pointnet", "mlp", "naive", "transformer"), default="pointnet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("isotropic", "anisotropic"), default="isotropic")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)