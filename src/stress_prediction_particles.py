import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from utils import EquivarianceTracker, set_seed, save_all, so2_eval_angles, rotate_2d, update_cycle, loader_kwargs
from torch.utils.data import Dataset, DataLoader
from Models.StressPredictionParticlesNets import PointNet, MLP, NaiveNet, SetTransformer
import published_data

NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_PARTICLES = 128   # random subset of each geometry's 972 mesh points
NUM_EVAL_ANGLES = 16


# ---------------------------------------------------------------------------
# Rotation utilities for particle states
# ---------------------------------------------------------------------------

def rotate_positions(state, theta):
    """
    Rotate the input particle state (..., N, 2) = (x, y) by theta.

    Positions are centred on the unit cell (published_data.py), so this is
    rotation about the cell centre.
    """
    return rotate_2d(state[..., 0:2], theta)


# The target is a scalar stress per point (Geo-FNO's stress field: positive,
# heavy-tailed near the void, most likely von Mises). A scalar is unchanged by
# rotation, so the rotated target is the stress itself: the map positions ->
# stress should be rotation-INVARIANT per point, and it is not, because the
# loading direction is fixed.


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
#
#   geofno_elasticity  Geo-FNO Elasticity point clouds (Li et al. 2022): a
#                      hyperelastic unit cell with a random void, clamped at the
#                      bottom and pulled at the top. Downloaded and cached by
#                      published_data.py on first use. Each sample is a random
#                      NUM_PARTICLES-point subset of one geometry's 972 points.
#                      Several subsets are drawn per geometry (ELASTICITY_SUBSETS),
#                      because there are only 1000 train geometries.
#
# There is no published isotropic counterpart, so this task has no isotropic
# control dataset.

DATASETS = ("geofno_elasticity",)
ISOTROPIC_DATASETS = ()

ELASTICITY_SUBSETS = {"train": 10, "test": 5}   # 1000 x 10 train, 200 x 5 test samples


class PointStressDataset(Dataset):
    """Input (N, 2) = (x, y); output (N, 1) = scalar stress."""

    def __init__(self, pos, stress, rotate=False):
        self.pos = pos
        self.stress = stress
        self.rotate = rotate

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        pos = self.pos[idx]
        stress = self.stress[idx]

        if self.rotate:
            theta = torch.rand(1) * 2 * np.pi
            pos = rotate_positions(pos, theta)   # stress is a scalar: unchanged

        return pos, stress


def make_dataset(dataset, split, rotate):
    if dataset != "geofno_elasticity":
        raise ValueError(f"No such dataset: {dataset}")
    pos, stress = published_data.load_geofno_elasticity(split, NUM_PARTICLES, ELASTICITY_SUBSETS)
    return PointStressDataset(pos, stress, rotate=rotate)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path, resume: bool, seed: int):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainset = make_dataset(dataset, "train", rotate=rotation)
    testset = make_dataset(dataset, "test", rotate=False)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, **loader_kwargs(device))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, **loader_kwargs(device))

    print(f"[{dataset}] Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    # in_channels=2 (x, y), out_channels=1 (scalar stress)
    if model == "pointnet":
        net = PointNet(width1=256 if thicker else 128, width2=256 if thicker else 64, in_channels=2, out_channels=1)
    elif model == "mlp":
        net = MLP(width1=256 if thicker else 128, width2=256 if thicker else 64, in_channels=2, out_channels=1)
    elif model == "naive":
        net = NaiveNet(in_channels=2, out_channels=1)
    elif model == "transformer":
        net = SetTransformer(dim=256 if thicker else 128, depth=1, heads=4, mlp_dim=256 if thicker else 128, in_channels=2, out_channels=1)
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
           f"{'_finetuned' if finetune else ''}_seed_{seed}")

    if resume:
        net.load_state_dict(torch.load(f"models/{tag}_model.pth", weights_only=True))
        with open(f"results/{tag}_statistics.json") as f:
            statistics = json.load(f)
        start = len(statistics["equivariant_loss"]) - 1
    else:
        update_statistics(net, criterion, statistics, trainloader, testloader, device)
        save_all(net, statistics, tag)
        start = 0

    for epoch in range(start, NUM_EPOCHS):
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

    eval_angles, angle_labels = so2_eval_angles(NUM_EVAL_ANGLES)

    # per-angle test loss (0 = unrotated). Positions rotate as vectors; the
    # target is a scalar stress, so it is the same at every angle.
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

            # unrotated prediction + activations: orientation 0 of the cycle
            pred, layers = net(state)
            running_test_loss[0] += criterion(pred, stress).item()
            layers_by_rot = [layers]

            for theta, deg in zip(eval_angles, angle_labels):
                theta_t = theta.to(device)
                state_rot = rotate_positions(state, theta_t)         # vectors: positions
                pred_rot, layers_rot = net(state_rot)                # one forward, reused for loss + equivariance

                running_test_loss[deg] += criterion(pred_rot, stress).item()   # scalar target: unrotated
                layers_by_rot.append(layers_rot)

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
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=DATASETS, default="geofno_elasticity")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args.add_argument("--resume", help="Resume training", action="store_true")
    args.add_argument("--seed", help="Random seed for reproducibility", type=int, default=0)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune, args.resume, args.seed)