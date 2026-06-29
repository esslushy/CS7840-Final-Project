import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from Models.ColorizationNets import UNet, CNN, NaiveNet, ViT
from utils import Random90Rotation, UnifiedEquivarianceTracker
import numpy as np

NUM_EPOCHS = 200
BATCH_SIZE = 64


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

def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path):
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
        net = ViT(image_size=32, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128, in_channels=1, out_channels=3)
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
        "test_loss": []
    }

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    update_statistics(net, criterion, statistics, trainloader, testloader, device)

    for epoch in range(NUM_EPOCHS):
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
        net.eval()
        update_statistics(net, criterion, statistics, trainloader, testloader, device)

    print('Finished Training')

    tag = f"colorization_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}{'_finetuned' if finetune else ''}"
    torch.save(net.state_dict(), f"models/{tag}_model.pth")
    with open(f"results/{tag}_statistics.json", "wt+") as f:
        json.dump(statistics, f)


def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    running_train_loss = 0.0
    for data in trainloader:
        inputs, _ = data
        inputs = inputs.to(device)

        gray = rgb_to_grayscale(inputs)
        output, layers = net(gray)

        running_train_loss += criterion(output, inputs).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = {k: [UnifiedEquivarianceTracker(device) for _ in range(3)] for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            inputs, _ = data
            inputs = inputs.to(device)

            gray = rgb_to_grayscale(inputs)
            output, layers = net(gray)

            running_test_loss += criterion(output, inputs).item()

            gray_rot90 = torch.rot90(gray, 1, dims=(-2, -1))
            gray_rot180 = torch.rot90(gray, 2, dims=(-2, -1))
            gray_rot270 = torch.rot90(gray, 3, dims=(-2, -1))

            *_, layers_rot90 = net(gray_rot90)
            *_, layers_rot180 = net(gray_rot180)
            *_, layers_rot270 = net(gray_rot270)

            for key in layers.keys():
                for idx, layer in enumerate([layers_rot90, layers_rot180, layers_rot270]):
                    running_equivariant_error[key][idx].update(layers[key], layer[key])
    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: np.mean([x.compute() for x in v]) for k, v in running_equivariant_error.items()})


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("cifar", "stl10"), default="cifar")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)