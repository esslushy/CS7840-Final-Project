import torch
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from utils import EquivarianceTracker, set_seed, save_all, update_cycle, loader_kwargs
from torch.utils.data import Dataset, DataLoader
from Models.FluidFlowNets import UNet, CNN, ViT, NaiveNet
import published_data
import os

NUM_EPOCHS = 400   # published data is smaller than the old 50k synthetic sets; 400 epochs puts the
                   # optimizer-step count in line with the image tasks at 200 (see slurm/launch_*)
BATCH_SIZE = 64
GRID_SIZE = 32   # both datasets are 128 x 128, average-pooled 4x

ANGLES = (90, 180, 270)   # non-identity C4 rotations tracked for equivariance


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------
#
# Grid data is only ever rotated by multiples of 90 degrees, via torch.rot90.
# That's an exact permutation of pixels (no resampling), unlike grid_sample at
# an arbitrary angle, which blurs across pixels and confounds the equivariance
# measurement with interpolation error.

def rotate_flow_state(field, k):
    """
    Rotate a batched 2D flow state by k * 90 degrees (k in {0, 1, 2, 3}).

    Channels 0-1 are the velocity (vx, vy): spatially rotated + vector
    components rotated. Every further channel (smoke density, fluid density,
    pressure) is a scalar field and is only spatially rotated.

    Args:
        field: (B, C, H, W), C >= 2, rows indexing y and columns indexing x

    Returns:
        rotated: (B, C, H, W)
    """
    field_r = torch.rot90(field, k, dims=(-2, -1))
    vx, vy = field_r[:, 0:1], field_r[:, 1:2]
    c = [1, 0, -1, 0][k]
    # Sign convention: torch.rot90 on dims (-2,-1) maps the row axis toward the
    # column axis, and rows index y and columns index x (published_data.py
    # transposes both datasets into that layout). The component mixing must match
    # that handedness. The conjugate table [0, 1, 0, -1] negates the field at 90
    # and 270 degrees.
    s = [0, -1, 0, 1][k]
    return torch.cat([c * vx - s * vy, s * vx + c * vy, field_r[:, 2:]], dim=1)


def rotate_flow_state_single(field, k):
    """Unbatched version: field is (C, H, W)."""
    return rotate_flow_state(field.unsqueeze(0), k).squeeze(0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
#
# Both are published simulations, downloaded and cached by published_data.py on
# first use:
#   pdebench_cfd    PDEBench 2D compressible Navier-Stokes, periodic, random ICs,
#                   no forcing: exactly symmetric under 90-degree rotations
#                   (isotropic control). Channels (vx, vy, density, pressure).
#   pdearena_smoke  PDEArena NavierStokes-2D PhiFlow smoke plume, buoyancy along
#                   +y: gravity breaks rotation symmetry. Channels (vx, vy, smoke).

DATASETS = ("pdebench_cfd", "pdearena_smoke")
ISOTROPIC_DATASETS = ("pdebench_cfd",)

# All published data: PDEArena train + valid (2,600 trajectories, 33,800 pairs) /
# test (1,300, 16,900), see published_data.PDEARENA_SOURCES. PDEBench 9,000 / 1,000
# of its 10,000 trajectories (180,000 / 20,000 pairs), a 55 GB remote read.
PDEBENCH_TRAJ = {"train": 9000, "test": 1000}   # 20 pairs per trajectory


class TrajectoryPairDataset(Dataset):
    """
    Next-step pairs (state_t, state_t+1) from (N_traj, T, C, H, W) trajectories.

    Every consecutive pair of stored frames is one sample. Pairs are indexed
    rather than materialised, so memory holds each frame once.
    """

    def __init__(self, traj, rotate=False):
        self.traj = traj
        self.rotate = rotate
        self.pairs_per_traj = traj.shape[1] - 1

    def __len__(self):
        return self.traj.shape[0] * self.pairs_per_traj

    def __getitem__(self, idx):
        i, t = divmod(idx, self.pairs_per_traj)
        field_t = self.traj[i, t]
        field_tp1 = self.traj[i, t + 1]

        if self.rotate:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                field_t = rotate_flow_state_single(field_t, k)
                field_tp1 = rotate_flow_state_single(field_tp1, k)

        return field_t, field_tp1


def make_dataset(dataset, split, rotate):
    if dataset == "pdebench_cfd":
        traj = published_data.load_pdebench_cfd(split, PDEBENCH_TRAJ, GRID_SIZE)
    elif dataset == "pdearena_smoke":
        traj = published_data.load_pdearena_smoke(split, GRID_SIZE)
    else:
        raise ValueError(f"No such dataset: {dataset}")
    return TrajectoryPairDataset(traj, rotate=rotate)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path, resume: bool, seed: int):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainset = make_dataset(dataset, "train", rotate=rotation)
    testset = make_dataset(dataset, "test", rotate=False)
    rotate_fn = rotate_flow_state
    channels = trainset.traj.shape[2]

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, **loader_kwargs(device))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, **loader_kwargs(device))

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
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tag = (f"fluid_flow_{'learned_equivariant' if rotation else 'non_equivariant'}"
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
        field_t, field_tp1 = data
        field_t = field_t.to(device)
        field_tp1 = field_tp1.to(device)

        pred, layers = net(field_t)
        running_train_loss += criterion(pred, field_tp1).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    # per-angle test loss (0 = unrotated). The dynamics map is EQUIVARIANT, so the
    # target for a rotated input is the ROTATED next state.
    running_test_loss = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    # per-layer, PER-ANGLE equivariance tracker (kept separate, not averaged)
    running_equivariant = {k: {a: EquivarianceTracker(device) for a in ANGLES}
                           for k in layers.keys()}

    with torch.inference_mode():
        for data in testloader:
            field_t, field_tp1 = data
            field_t = field_t.to(device)
            field_tp1 = field_tp1.to(device)

            # unrotated prediction + activations: orientation 0 of the cycle
            pred, layers = net(field_t)
            running_test_loss[0] += criterion(pred, field_tp1).item()
            layers_by_rot = [layers]

            for angle in ANGLES:
                k = angle // 90
                field_rotated = rotate_fn(field_t, k)
                target_rotated = rotate_fn(field_tp1, k)
                pred_rot, layers_rotated = net(field_rotated)   # one forward, reused for loss + equivariance

                running_test_loss[angle] += criterion(pred_rot, target_rotated).item()
                layers_by_rot.append(layers_rotated)

            # cycle pairing: every orientation vs. the one `step` quarter turns on
            for key in layers.keys():
                update_cycle([running_equivariant[key][a] for a in ANGLES],
                             [L[key] for L in layers_by_rot])

    n_test = len(testloader)
    statistics["test_loss"].append({angle: v / n_test for angle, v in running_test_loss.items()})
    # store the full tracker stats per layer, per angle (cycle pairing, see utils.update_cycle)
    statistics["equivariance_pairing"] = "cycle"
    statistics["equivariant_loss"].append({
        key: {angle: running_equivariant[key][angle].compute_stats() for angle in ANGLES}
        for key in running_equivariant
    })


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=DATASETS, default="pdebench_cfd")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args.add_argument("--resume", help="Resume training", action="store_true")
    args.add_argument("--seed", help="Random seed for reproducibility", type=int, default=0)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune, args.resume, args.seed)