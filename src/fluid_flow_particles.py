import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from utils import EquivarianceTracker, set_seed, save_all, so2_eval_angles, rotate_2d, update_cycle, loader_kwargs
from torch.utils.data import Dataset, DataLoader
import published_data
import os

NUM_EPOCHS = 400   # published data is smaller than the old 50k synthetic sets; 400 epochs puts the
                   # optimizer-step count in line with the image tasks at 200 (see slurm/launch_*)
BATCH_SIZE = 64
NUM_PARTICLES = 128   # random subset of the 2500 (TGV) / ~5740 (dam) SPH particles
NUM_EVAL_ANGLES = 16


# ---------------------------------------------------------------------------
# Rotation utilities for particle states
# ---------------------------------------------------------------------------

def rotate_particles(particles, theta):
    """
    Rotate a particle state (..., N, 4) = (x, y, vx, vy) by theta.

    Position and velocity are both 2D vectors and rotate together. Positions
    are centred on the simulation box (published_data.py), so this is rotation
    about the box centre.
    """
    pos = rotate_2d(particles[..., 0:2], theta)
    vel = rotate_2d(particles[..., 2:4], theta)
    return torch.cat([pos, vel], dim=-1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
#
# Both are LagrangeBench 2D SPH simulations (Toshev et al. 2023), downloaded and
# cached by published_data.py on first use. Each sample is a fixed random subset
# of NUM_PARTICLES fluid particles, followed from frame t to frame t+1:
#   lagrangebench_tgv  Taylor-Green vortex, periodic box, no external force
#                      (isotropic control). The periodic square is symmetric
#                      under 90-degree rotations only, not under the 16 SO(2)
#                      probe angles.
#   lagrangebench_dam  dam break under gravity in a walled box: gravity breaks
#                      rotation symmetry.

DATASETS = ("lagrangebench_tgv", "lagrangebench_dam")
ISOTROPIC_DATASETS = ("lagrangebench_tgv",)


class ParticlePairDataset(Dataset):
    """Next-step pairs of (N, 4) = (x, y, vx, vy) particle states."""

    def __init__(self, states_t, states_tp1, rotate=False):
        self.states_t = states_t
        self.states_tp1 = states_tp1
        self.rotate = rotate

    def __len__(self):
        return self.states_t.shape[0]

    def __getitem__(self, idx):
        state_t = self.states_t[idx]
        state_tp1 = self.states_tp1[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            state_t = rotate_particles(state_t, theta)
            state_tp1 = rotate_particles(state_tp1, theta)

        return state_t, state_tp1


def make_dataset(dataset, split, rotate):
    cases = {"lagrangebench_tgv": "tgv", "lagrangebench_dam": "dam"}
    if dataset not in cases:
        raise ValueError(f"No such dataset: {dataset}")
    states_t, states_tp1 = published_data.load_lagrangebench(cases[dataset], split, NUM_PARTICLES)
    return ParticlePairDataset(states_t, states_tp1, rotate=rotate)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path, resume: bool, seed: int):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from Models.FluidFlowParticlesNets import PointNet, MLP, NaiveNet, SetTransformer

    trainset = make_dataset(dataset, "train", rotate=rotation)
    testset = make_dataset(dataset, "test", rotate=False)
    rotate_fn = rotate_particles
    channels = 4

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, **loader_kwargs(device))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, **loader_kwargs(device))

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
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tag = (f"fluid_flow_particles_{'learned_equivariant' if rotation else 'non_equivariant'}"
           f"_{model}{'_thicker' if thicker else ''}_dataset_{dataset}"
           f"{'_finetuned' if finetune else ''}_seed_{seed}")

    if resume:
        net.load_state_dict(torch.load(f"models/{tag}_model.pth", weights_only=True))
        with open(f"results/{tag}_statistics.json") as f:
            statistics = json.load(f)
        start = len(statistics["equivariant_loss"]) - 1
    else:
        update_statistics(net, criterion, statistics, trainloader, testloader, device, rotate_fn)
        save_all(net, statistics, tag)
        start = 0

    for epoch in range(start, NUM_EPOCHS):
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

        # measure and persist EVERY epoch so an interrupted run keeps all stats so far
        update_statistics(net, criterion, statistics, trainloader, testloader, device, rotate_fn)
        save_all(net, statistics, tag)

    print('Finished Training')
    # already saved each epoch; final save is just belt-and-suspenders
    save_all(net, statistics, tag)


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

    eval_angles, angle_labels = so2_eval_angles(NUM_EVAL_ANGLES)

    # per-angle test loss (0 = unrotated). The dynamics map is EQUIVARIANT, so the
    # target for a rotated input is the ROTATED next state.
    running_test_loss = {0: 0.0}
    for deg in angle_labels:
        running_test_loss[deg] = 0.0
    # per-layer, PER-ANGLE equivariance tracker (kept separate, not averaged)
    running_equivariant = {k: {deg: EquivarianceTracker(device) for deg in angle_labels}
                           for k in layers.keys()}

    with torch.inference_mode():
        for data in testloader:
            state_t, state_tp1 = data
            state_t = state_t.to(device)
            state_tp1 = state_tp1.to(device)

            # unrotated prediction + activations: orientation 0 of the cycle
            pred, layers = net(state_t)
            running_test_loss[0] += criterion(pred, state_tp1).item()
            layers_by_rot = [layers]

            for theta, deg in zip(eval_angles, angle_labels):
                theta_t = theta.to(device)
                state_rotated = rotate_fn(state_t, theta_t)
                target_rotated = rotate_fn(state_tp1, theta_t)
                pred_rot, layers_rotated = net(state_rotated)   # one forward, reused for loss + equivariance

                running_test_loss[deg] += criterion(pred_rot, target_rotated).item()
                layers_by_rot.append(layers_rotated)

            # cycle pairing over C_{N+1} (identity + the N eval angles, which are
            # the multiples of 360/(N+1)): every orientation vs. the one `step` on
            for key in layers.keys():
                update_cycle([running_equivariant[key][a] for a in angle_labels],
                             [L[key] for L in layers_by_rot])

    n_test = len(testloader)
    statistics["test_loss"].append({angle: v / n_test for angle, v in running_test_loss.items()})
    # store the full tracker stats per layer, per angle (cycle pairing, see utils.update_cycle)
    statistics["equivariance_pairing"] = "cycle"
    statistics["equivariant_loss"].append({
        key: {deg: running_equivariant[key][deg].compute_stats() for deg in angle_labels}
        for key in running_equivariant
    })


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("pointnet", "mlp", "naive", "transformer"), default="pointnet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=DATASETS, default="lagrangebench_tgv")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args.add_argument("--resume", help="Resume training", action="store_true")
    args.add_argument("--seed", help="Random seed for reproducibility", type=int, default=0)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune, args.resume, args.seed)