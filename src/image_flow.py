import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from argparse import ArgumentParser
from pathlib import Path
from Models.ImageFlowNets import UNet, CNN, ViT, NaiveNet
from utils import EquivarianceTracker, set_seed, save_all
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 200
BATCH_SIZE = 64
GRID_SIZE = 32
WAVE_DT = 0.3

ANGLES = (90, 180, 270)   # non-identity C4 rotations tracked for equivariance


def rotate_flow(flow, k):
    """
    Rotate a batched 2D vector flow field (B, 2, H, W) by k * 90 degrees.

    A velocity field transforms as a vector field: the grid is spatially
    rotated AND the (vx, vy) components are rotated. This matches the
    rotation convention used in WaterSurfaceDataset.__getitem__.
    """
    flow_r = torch.rot90(flow, k, dims=(-2, -1))
    fx, fy = flow_r[:, 0:1], flow_r[:, 1:2]
    c = [1, 0, -1, 0][k]
    s = [0, 1, 0, -1][k]
    return torch.cat([c * fx - s * fy, s * fx + c * fy], dim=1)


# ---------------------------------------------------------------------------
# Water surface dataset
# ---------------------------------------------------------------------------

class WaterSurfaceDataset(Dataset):
    """
    Synthetic water surface images with analytically computed flow fields.

    Each sample is a superposition of linear deep-water wave components.
    The input is two consecutive surface elevation snapshots (t=0 and t=dt),
    so the model can see which direction each wave is moving.
    The target is the surface velocity field.

    For each wave component η_n = A_n cos(k_n · r + φ_n):
        η(t=0)  = A_n cos(k_n · r + φ_n)
        η(t=dt) = A_n cos(k_n · r + φ_n - ω_n * dt)
        u_n     = A_n ω_n k̂_n cos(k_n · r + φ_n)

    where ω_n = sqrt(|k_n|) (deep water dispersion, normalised g=1).

    Input:  (2, H, W) — surface elevation at t=0 and t=dt
    Output: (2, H, W) — surface velocity field (vx, vy)
    """

    def __init__(self, n_samples=10000, grid_size=GRID_SIZE, dt=WAVE_DT, directional=False, rotate=False):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.dt = dt
        self.directional = directional
        self.rotate = rotate

        coords = torch.linspace(-np.pi, np.pi, grid_size)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

        self.images = []
        self.flows = []
        for _ in range(n_samples):
            eta_t0, eta_t1, vx, vy = self._generate_waves()
            self.images.append(torch.stack([eta_t0, eta_t1], dim=0))  # (2, H, W)
            self.flows.append(torch.stack([vx, vy], dim=0))           # (2, H, W)

        self.images = torch.stack(self.images)
        self.flows = torch.stack(self.flows)

    def _generate_waves(self):
        eta_t0 = torch.zeros_like(self.xx)
        eta_t1 = torch.zeros_like(self.xx)
        vx = torch.zeros_like(self.xx)
        vy = torch.zeros_like(self.xx)

        n_waves = torch.randint(5, 15, (1,)).item()

        for _ in range(n_waves):
            if self.directional:
                angle = 0.0 + torch.randn(1).item() * 0.4
            else:
                angle = torch.rand(1).item() * 2 * np.pi

            k_mag = 1.0 + torch.rand(1).item() * 5.0
            kx = k_mag * np.cos(angle)
            ky = k_mag * np.sin(angle)

            A = torch.rand(1).item() * 0.3
            phi = torch.rand(1).item() * 2 * np.pi
            omega = np.sqrt(k_mag)

            phase = kx * self.xx + ky * self.yy + phi
            eta_t0 += A * torch.cos(phase)
            eta_t1 += A * torch.cos(phase - omega * self.dt)

            khat_x = kx / k_mag
            khat_y = ky / k_mag
            vx += A * omega * khat_x * torch.cos(phase)
            vy += A * omega * khat_y * torch.cos(phase)

        return eta_t0, eta_t1, vx, vy

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        flow = self.flows[idx]

        if self.rotate:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = torch.rot90(image, k, dims=(-2, -1))
                # Rotate flow spatially and rotate vector components
                flow = torch.rot90(flow, k, dims=(-2, -1))
                fx, fy = flow[0:1], flow[1:2]
                cos_vals = [1, 0, -1, 0]
                sin_vals = [0, 1, 0, -1]
                c, s = cos_vals[k], sin_vals[k]
                flow = torch.cat([c * fx - s * fy, s * fx + c * fy], dim=0)

        return image, flow


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path, seed: int):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "isotropic":
        trainset = WaterSurfaceDataset(n_samples=50000, directional=False, rotate=rotation)
        testset = WaterSurfaceDataset(n_samples=5000, directional=False, rotate=False)
    elif dataset == "directional":
        trainset = WaterSurfaceDataset(n_samples=50000, directional=True, rotate=rotation)
        testset = WaterSurfaceDataset(n_samples=5000, directional=True, rotate=False)
    else:
        raise ValueError(f"No such dataset: {dataset}")

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print(f"[{dataset}] Train: {len(trainset)} samples  |  Test: {len(testset)} samples")

    if model == "vit":
        net = ViT(image_size=GRID_SIZE, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128, in_channels=2, out_channels=2)
    elif model == "naive":
        net = NaiveNet(in_channels=2, out_channels=2)
    elif model == "cnn":
        net = CNN(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=2, out_channels=2)
    elif model == "unet":
        net = UNet(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=2, out_channels=2)
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

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    tag = (f"image_flow_{'learned_equivariant' if rotation else 'non_equivariant'}"
           f"_{model}{'_thicker' if thicker else ''}_dataset_{dataset}"
           f"{'_finetuned' if finetune else ''}_seed_{seed}")

    # baseline measurement before any training, then persist immediately
    update_statistics(net, criterion, statistics, trainloader, testloader, device)
    save_all(net, statistics, tag)

    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            image, flow = data
            image = image.to(device)
            flow = flow.to(device)

            optimizer.zero_grad()

            output, _ = net(image)
            loss = criterion(output, flow)
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


def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    net.eval()
    running_train_loss = 0.0
    for data in trainloader:
        image, flow = data
        image = image.to(device)
        flow = flow.to(device)

        output, layers = net(image)
        running_train_loss += criterion(output, flow).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    # per-angle test loss (0 = unrotated). Flow prediction is EQUIVARIANT, so the
    # target for a rotated input is the rotated flow (spatial + vector components).
    running_test_loss = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    # per-layer, PER-ANGLE equivariance tracker (kept separate, not averaged)
    running_equivariant = {k: {a: EquivarianceTracker(device) for a in ANGLES}
                           for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            image, flow = data
            image = image.to(device)
            flow = flow.to(device)

            # unrotated prediction + activations (equivariance reference)
            output, layers = net(image)
            running_test_loss[0] += criterion(output, flow).item()

            for angle in ANGLES:
                k = angle // 90
                image_rot = torch.rot90(image, k, dims=(-2, -1))
                flow_rot = rotate_flow(flow, k)
                output_rot, layers_rot = net(image_rot)   # one forward, reused for loss + equivariance

                running_test_loss[angle] += criterion(output_rot, flow_rot).item()

                for key in layers.keys():
                    running_equivariant[key][angle].update(layers[key], layers_rot[key])

    n_test = len(testloader)
    statistics["test_loss"].append({angle: v / n_test for angle, v in running_test_loss.items()})
    # store the full tracker stats (z, p, floor_std, ...) per layer, per angle
    statistics["equivariant_loss"].append({
        key: {angle: running_equivariant[key][angle].compute_stats() for angle in ANGLES}
        for key in running_equivariant
    })


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("isotropic", "directional"), default="directional")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args.add_argument("--seed", help="Random seed for reproducibility", type=int, default=0)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune, args.seed)