import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from Models.ImageFlowNets import UNet, CNN, ViT, NaiveNet
from utils import Random90Rotation, UnifiedEquivarianceTracker
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 200
BATCH_SIZE = 64
GRID_SIZE = 32


# ---------------------------------------------------------------------------
# Water surface dataset
# ---------------------------------------------------------------------------

class WaterSurfaceDataset(Dataset):
    """
    Synthetic water surface images with analytically computed flow fields.

    Each sample is a superposition of linear deep-water wave components.
    The rendered image is the surface elevation (height map).
    The target is the surface velocity field, derived from wave theory:

        For each wave component η_n = A_n cos(k_n · r + φ_n):
            u_n = A_n ω_n k̂_n cos(k_n · r + φ_n)

        where ω_n = sqrt(|k_n|) (deep water dispersion, normalised g=1)
        and k̂_n is the unit wave propagation direction.

    Input:  (1, H, W) surface elevation image
    Output: (2, H, W) surface velocity field (vx, vy)
    """

    def __init__(self, n_samples=10000, grid_size=GRID_SIZE, directional=False, rotate=False):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.directional = directional
        self.rotate = rotate

        coords = torch.linspace(-np.pi, np.pi, grid_size)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

        self.images = []
        self.flows = []
        for _ in range(n_samples):
            eta, vx, vy = self._generate_waves()
            self.images.append(eta.unsqueeze(0))          # (1, H, W)
            self.flows.append(torch.stack([vx, vy], dim=0))  # (2, H, W)

        self.images = torch.stack(self.images)
        self.flows = torch.stack(self.flows)

    def _generate_waves(self):
        eta = torch.zeros_like(self.xx)
        vx = torch.zeros_like(self.xx)
        vy = torch.zeros_like(self.xx)

        n_waves = torch.randint(5, 15, (1,)).item()

        for _ in range(n_waves):
            # Wave direction
            if self.directional:
                # Preferred direction: waves come from the left (positive x)
                # with spread — mimics wind-driven ocean swell
                angle = 0.0 + torch.randn(1).item() * 0.4
            else:
                angle = torch.rand(1).item() * 2 * np.pi

            k_mag = 1.0 + torch.rand(1).item() * 5.0
            kx = k_mag * np.cos(angle)
            ky = k_mag * np.sin(angle)

            A = torch.rand(1).item() * 0.3
            phi = torch.rand(1).item() * 2 * np.pi
            omega = np.sqrt(k_mag)  # deep water dispersion

            wave = A * torch.cos(kx * self.xx + ky * self.yy + phi)
            eta += wave

            # Surface velocity from linear wave theory
            khat_x = kx / k_mag
            khat_y = ky / k_mag
            vx += A * omega * khat_x * torch.cos(kx * self.xx + ky * self.yy + phi)
            vy += A * omega * khat_y * torch.cos(kx * self.xx + ky * self.yy + phi)

        return eta, vx, vy

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

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path):
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
        net = ViT(image_size=GRID_SIZE, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128, in_channels=1, out_channels=2)
    elif model == "naive":
        net = NaiveNet(in_channels=1, out_channels=2)
    elif model == "cnn":
        net = CNN(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=1, out_channels=2)
    elif model == "unet":
        net = UNet(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=1, out_channels=2)
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
        net.eval()
        update_statistics(net, criterion, statistics, trainloader, testloader, device)

    print('Finished Training')

    tag = f"image_flow_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}{'_finetuned' if finetune else ''}"
    torch.save(net.state_dict(), f"models/{tag}_model.pth")
    with open(f"results/{tag}_statistics.json", "wt+") as f:
        json.dump(statistics, f)


def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    running_train_loss = 0.0
    for data in trainloader:
        image, flow = data
        image = image.to(device)
        flow = flow.to(device)

        output, layers = net(image)
        running_train_loss += criterion(output, flow).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = {k: [UnifiedEquivarianceTracker(device) for _ in range(3)] for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            image, flow = data
            image = image.to(device)
            flow = flow.to(device)

            output, layers = net(image)
            running_test_loss += criterion(output, flow).item()

            image_rot90 = torch.rot90(image, 1, dims=(-2, -1))
            image_rot180 = torch.rot90(image, 2, dims=(-2, -1))
            image_rot270 = torch.rot90(image, 3, dims=(-2, -1))

            *_, layers_rot90 = net(image_rot90)
            *_, layers_rot180 = net(image_rot180)
            *_, layers_rot270 = net(image_rot270)

            for key in layers.keys():
                for idx, layer in enumerate([layers_rot90, layers_rot180, layers_rot270]):
                    running_equivariant_error[key][idx].update(layers[key], layer[key])
    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: np.mean([x.compute() for x in v]) for k, v in running_equivariant_error.items()})


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("isotropic", "directional"), default="directional")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)