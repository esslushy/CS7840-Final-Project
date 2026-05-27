import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from Models.FaceSwapNets import SmileToFrownGenerator, PatchDiscriminator
from utils import split_array_randomly, Random90Rotation
from HSIC import cka
from PIL import Image
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 200
BATCH_SIZE = 64

def download_celeba(root="./data"):
    """Download CelebA via torchvision if it doesn't already exist."""
    celeba_dir = os.path.join(root, "celeba")
    img_dir = os.path.join(celeba_dir, "img_align_celeba")
    attr_file = os.path.join(celeba_dir, "list_attr_celeba.txt")

    if os.path.isdir(img_dir) and os.path.isfile(attr_file):
        n = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        print(f"CelebA already exists: {n} images found")
        return

    print("Downloading CelebA (this may take a while)...")
    try:
        torchvision.datasets.CelebA(root=root, split="all", download=True)
        print("Download complete")
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\nPlease download manually from:")
        print("  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("  or https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        print(f"\nPlace files at:")
        print(f"  {img_dir}/          (face images)")
        print(f"  {attr_file}   (attribute annotations)")
        raise SystemExit(1)


class CelebASmiles(Dataset):
    """CelebA images filtered by Smiling attribute, respecting official splits."""

    def __init__(self, root, smiling=True, split="train", transform=None):
        self.img_dir = os.path.join(root, "img_align_celeba")
        self.transform = transform
        self.files = []

        # Load split assignments: 0=train, 1=val, 2=test
        split_map = {"train": "0", "val": "1", "test": "2"}
        target_split = split_map[split]
        split_path = os.path.join(root, "list_eval_partition.txt")
        splits = {}
        with open(split_path) as f:
            for line in f:
                parts = line.split()
                splits[parts[0]] = parts[1]

        # Filter by smiling attribute
        attr_path = os.path.join(root, "list_attr_celeba.txt")
        with open(attr_path) as f:
            f.readline()
            header = f.readline().split()
            si = header.index("Smiling")
            for line in f:
                parts = line.split()
                fname = parts[0]
                if splits.get(fname) != target_split:
                    continue
                is_smile = int(parts[1 + si]) == 1
                if is_smile == smiling:
                    self.files.append(fname)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.files[i % len(self.files)])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class PairedDomainLoader:
    """Yields (smile_batch, frown_batch) pairs, cycling the shorter domain."""

    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b

    def __iter__(self):
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        for _ in range(max(len(self.loader_a), len(self.loader_b))):
            try:
                a = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                a = next(iter_a)
            try:
                b = next(iter_b)
            except StopIteration:
                iter_b = iter(self.loader_b)
                b = next(iter_b)
            yield a, b

    def __len__(self):
        return max(len(self.loader_a), len(self.loader_b))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(kernel: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "./data"
    celeba_root = os.path.join(data_root, "celeba")
    download_celeba(data_root)

    img_size = 128
    transform_operations = [
        transforms.Resize(int(img_size * 1.12)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    if rotation:
        transform_operations.append(Random90Rotation())
    transform_train = transforms.Compose(transform_operations)

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_smile = CelebASmiles(celeba_root, smiling=True, split="train", transform=transform_train)
    train_frown = CelebASmiles(celeba_root, smiling=False, split="train", transform=transform_train)
    test_smile = CelebASmiles(celeba_root, smiling=True, split="test", transform=transform_test)
    test_frown = CelebASmiles(celeba_root, smiling=False, split="test", transform=transform_test)

    trainloader = PairedDomainLoader(
        DataLoader(train_smile, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True),
        DataLoader(train_frown, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True),
    )
    testloader = PairedDomainLoader(
        DataLoader(test_smile, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True),
        DataLoader(test_frown, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True),
    )

    num_features = 128 if thicker else 64
    G = SmileToFrownGenerator(num_features=num_features).to(device)
    F_net = SmileToFrownGenerator(num_features=num_features).to(device)
    D = PatchDiscriminator().to(device)

    if finetune:
        ckpt = torch.load(finetune, weights_only=True)
        G.load_state_dict(ckpt["G"])
        F_net.load_state_dict(ckpt["F"])
        D.load_state_dict(ckpt["D"])

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    opt_G = optim.Adam(
        list(G.parameters()) + list(F_net.parameters()), lr=2e-4, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    statistics = {
        "equivariant_loss": [],
        "train_loss": [],
        "test_loss": [],
        "baseline_cka": [],
    }

    os.mkdir("models", exist_ok=True)
    os.mkdir("results", exist_ok=True)

    update_statistics(kernel, G, criterion_cycle, statistics, trainloader, testloader, device)

    for epoch in range(NUM_EPOCHS):
        G.train()
        F_net.train()
        D.train()
        running_loss = 0.0

        for real_smile, real_frown in trainloader:
            real_smile = real_smile.to(device)
            real_frown = real_frown.to(device)

            # Generator step
            opt_G.zero_grad()

            fake_frown, _ = G(real_smile)
            pred_fake = D(fake_frown)
            loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))

            recon_smile, _ = F_net(fake_frown)
            loss_cycle = criterion_cycle(recon_smile, real_smile) * 10.0

            ident_frown, _ = G(real_frown)
            loss_identity = criterion_identity(ident_frown, real_frown) * 5.0

            loss_G = loss_gan + loss_cycle + loss_identity
            loss_G.backward()
            opt_G.step()

            # Discriminator step
            opt_D.zero_grad()

            loss_real = criterion_gan(D(real_frown), torch.ones_like(D(real_frown)))
            loss_fake = criterion_gan(D(fake_frown.detach()), torch.zeros_like(D(fake_frown.detach())))
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            running_loss += loss_G.item()

        print(f"[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}")
        update_statistics(kernel, G, criterion_cycle, statistics, trainloader, testloader, device)

    print("Finished Training")

    tag = f"face_swap_{'learned_equivariant' if rotation else 'non_equivariant'}{'_thicker' if thicker else ''}_kernel_{kernel}{'_finetuned' if finetune else ''}"
    torch.save({"G": G.state_dict(), "F": F_net.state_dict(), "D": D.state_dict()}, f"models/{tag}_model.pth")
    with open(f"results/{tag}_statistics.json", "wt+") as f:
        json.dump(statistics, f)


def update_statistics(kernel, net, criterion, statistics, trainloader, testloader, device):
    net.eval()

    running_train_loss = 0.0
    with torch.inference_mode():
        for real_smile, real_frown in trainloader:
            real_smile = real_smile.to(device)
            real_frown = real_frown.to(device)

            fake_frown, _ = net(real_smile)
            running_train_loss += criterion(fake_frown, real_frown).item()

    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = defaultdict(float)
    running_cka_baseline = defaultdict(float)

    with torch.inference_mode():
        for real_smile, real_frown in testloader:
            real_smile = real_smile.to(device)
            real_frown = real_frown.to(device)

            fake_frown, acts = net(real_smile)
            running_test_loss += criterion(fake_frown, real_frown).item()

            smile_rot90 = torch.rot90(real_smile, 1, dims=(-2, -1))
            smile_rot180 = torch.rot90(real_smile, 2, dims=(-2, -1))
            smile_rot270 = torch.rot90(real_smile, 3, dims=(-2, -1))

            _, acts_rot90 = net(smile_rot90)
            _, acts_rot180 = net(smile_rot180)
            _, acts_rot270 = net(smile_rot270)

            for key in acts.keys():
                running_equivariant_error[key] += np.mean([
                    cka(acts[key], acts_rot90[key], kernel=kernel).item(),
                    cka(acts[key], acts_rot180[key], kernel=kernel).item(),
                    cka(acts[key], acts_rot270[key], kernel=kernel).item(),
                ])
                acts_x, acts_y = split_array_randomly(acts[key])
                running_cka_baseline[key] += cka(acts_x, acts_y, kernel=kernel).item()

    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: v / len(testloader) for k, v in running_equivariant_error.items()})
    statistics["baseline_cka"].append({k: v / len(testloader) for k, v in running_cka_baseline.items()})


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--kernel", type=str, choices=["rbf", "linear"], default="linear")
    args.add_argument("--rotation", action="store_true")
    args.add_argument("--thicker", action="store_true")
    args.add_argument("--finetune", type=Path)
    args = args.parse_args()

    main(args.kernel, args.rotation, args.thicker, args.finetune)