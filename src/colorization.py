import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from Models.ColorizationNets import UNet, CNN, NaiveNet, ViT
from utils import Random90Rotation, EquivarianceTracker, set_seed, save_all
import numpy as np

NUM_EPOCHS = 200
BATCH_SIZE = 64
ANGLES = (90, 180, 270)


def rgb_to_grayscale(rgb):
    """
    Convert RGB to grayscale using luminance weights.

    Args:
        rgb: (B, 3, H, W) in [0, 1]

    Returns:
        gray: (B, 1, H, W)
    """
    return 0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path, resume: bool, seed: int):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_operations = [transforms.ToTensor()]
    if rotation:
        transform_operations.append(Random90Rotation())
    transform_train = transforms.Compose(transform_operations)

    transform_test = transforms.ToTensor()

    if dataset == "cifar":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
    elif dataset == "stl10":
        transform_operations_stl = [transforms.Resize(32), transforms.ToTensor()]
        if rotation:
            transform_operations_stl.append(Random90Rotation())
        trainset = torchvision.datasets.STL10(root='./data', split='train+unlabeled',
                                              download=True, transform=transforms.Compose(transform_operations_stl))
        testset = torchvision.datasets.STL10(root='./data', split='test',
                                             download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    else:
        raise ValueError(f"No such dataset: {dataset}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    if model == "vit":
        net = ViT(image_size=32, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1,
                  mlp_dim=256 if thicker else 128, in_channels=1, out_channels=3)
    elif model == "naive":
        net = NaiveNet(in_channels=1, out_channels=3)
    elif model == "cnn":
        net = CNN(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=1, out_channels=3)
    elif model == "unet":
        net = UNet(width1=256 if thicker else 120, width2=256 if thicker else 84, in_channels=1, out_channels=3)
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

    tag = (f"colorization_{'learned_equivariant' if rotation else 'non_equivariant'}"
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
            inputs, _ = data
            inputs = inputs.to(device)

            gray = rgb_to_grayscale(inputs)

            optimizer.zero_grad()

            output, _ = net(gray)
            loss = criterion(output, inputs)
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
        inputs, _ = data
        inputs = inputs.to(device)

        gray = rgb_to_grayscale(inputs)
        output, layers = net(gray)

        running_train_loss += criterion(output, inputs).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    running_equivariant = {k: {a: EquivarianceTracker(device) for a in ANGLES}
                           for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            inputs, _ = data
            inputs = inputs.to(device)
            gray = rgb_to_grayscale(inputs)

            # unrotated activations, used as the equivariance reference
            _, layers = net(gray)

            for angle in (0, 90, 180, 270):
                k = angle // 90
                gray_rot = torch.rot90(gray, k, dims=(-2, -1))
                target_rot = torch.rot90(inputs, k, dims=(-2, -1))
                output, layers_rot = net(gray_rot)   # one forward, reused for loss + equivariance

                running_test_loss[angle] += criterion(output, target_rot).item()

                if angle != 0:
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
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("cifar", "stl10"), default="cifar")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args.add_argument("--resume", help="Resume training", action="store_true")
    args.add_argument("--seed", help="Random seed for reproducibility", type=int, default=0)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune, args.resume, args.seed)