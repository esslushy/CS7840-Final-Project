import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from HSIC import cka
from utils import split_array_randomly, Random90Rotation
from collections import defaultdict
from Models.GradientFieldNets import UNet, CNN, ViT, NaiveNet

NUM_EPOCHS = 20
BATCH_SIZE = 64

def sobel_gradient(images):
    """
    Compute the spatial gradient of a batch of images using Sobel filters.

    Works for any number of input channels by applying the filter per-channel
    and averaging the result, producing a single 2-channel gradient field.

    Args:
        images: (B, C, H, W) clean images

    Returns:
        grad: (B, 2, H, W) where channel 0 = dI/dx, channel 1 = dI/dy
    """
    C = images.shape[1]

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=images.dtype, device=images.device
    ).view(1, 1, 3, 3) / 8.0

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], dtype=images.dtype, device=images.device
    ).view(1, 1, 3, 3) / 8.0

    sobel_x = sobel_x.expand(C, 1, 3, 3)
    sobel_y = sobel_y.expand(C, 1, 3, 3)
    grad_x = nn.functional.conv2d(images, sobel_x, padding=1, groups=C).mean(dim=1, keepdim=True)
    grad_y = nn.functional.conv2d(images, sobel_y, padding=1, groups=C).mean(dim=1, keepdim=True)
    return torch.cat([grad_x, grad_y], dim=1)

def main(model: str, dataset: str, kernel: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "cifar":
        trainset, testset = load_cifar(rotation)
        in_channels = 3
    elif dataset == "mnist":
        trainset, testset = load_mnist(rotation)
        in_channels = 1
    else:
        raise Exception("Unknown Dataset")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    if model == "vit":
        net = ViT(image_size=32 if dataset == "cifar" else 28, patch_size=4, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128)
    elif model == "naive":
        net = NaiveNet(in_channels=in_channels)
    elif model == "cnn":
        net = CNN(in_channels=in_channels, width1=256 if thicker else 128, width2=256 if thicker else 128)
    elif model == "unet":
        net = UNet(in_channels=in_channels, width1=256 if thicker else 128, width2=256 if thicker else 128)
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

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    update_statistics(kernel, net, criterion, statistics, trainloader, testloader, device)

    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)

            target_grad = sobel_gradient(inputs)

            optimizer.zero_grad()

            pred_grad, _ = net(inputs)
            loss = criterion(pred_grad, target_grad)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}")
        net.eval()
        update_statistics(kernel, net, criterion, statistics, trainloader, testloader, device)

    print('Finished Training')

    tag = f"gradient_field_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}_kernel_{kernel}{'_finetuned' if finetune else ''}"
    torch.save(net.state_dict(), f"models/{tag}_model.pth")
    with open(f"results/{tag}_statistics.json", "wt+") as f:
        json.dump(statistics, f)


def load_cifar(rotation):
    transform_operations = [transforms.ToTensor()]
    if rotation:
        transform_operations.append(Random90Rotation())

    transform_train = transforms.Compose(transform_operations)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)

    transform_test = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    return trainset, testset


def load_mnist(rotation):
    transform_operations = [transforms.ToTensor()]
    if rotation:
        transform_operations.append(Random90Rotation())

    transform_train = transforms.Compose(transform_operations)
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform_train)

    transform_test = transforms.ToTensor()
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform_test)

    return trainset, testset


def update_statistics(kernel, net, criterion, statistics, trainloader, testloader, device):
    running_train_loss = 0.0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)

        target_grad = sobel_gradient(inputs)
        pred_grad, _ = net(inputs)

        running_train_loss += criterion(pred_grad, target_grad).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    running_test_loss = 0.0
    running_equivariant_error = defaultdict(float)
    running_cka_baseline = defaultdict(float)
    with torch.inference_mode():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)

            target_grad = sobel_gradient(inputs)

            pred_grad, layers = net(inputs)
            running_test_loss += criterion(pred_grad, target_grad).item()

            inputs_rot90 = torch.rot90(inputs, 1, dims=(-2, -1))
            inputs_rot180 = torch.rot90(inputs, 2, dims=(-2, -1))
            inputs_rot270 = torch.rot90(inputs, 3, dims=(-2, -1))

            _, layers_rot90 = net(inputs_rot90)
            _, layers_rot180 = net(inputs_rot180)
            _, layers_rot270 = net(inputs_rot270)

            for key in layers.keys():
                running_equivariant_error[key] += np.mean([
                    cka(layers[key], layers_rot90[key], kernel=kernel).item(),
                    cka(layers[key], layers_rot180[key], kernel=kernel).item(),
                    cka(layers[key], layers_rot270[key], kernel=kernel).item()
                ])
                layers_x, layers_y = split_array_randomly(layers[key])
                running_cka_baseline[key] += cka(layers_x, layers_y, kernel=kernel).item()
    statistics["test_loss"].append(running_test_loss / len(testloader))
    statistics["equivariant_loss"].append({k: v / len(testloader) for k,v in running_equivariant_error.items()})
    statistics["baseline_cka"].append({k: v / len(testloader) for k,v in running_cka_baseline.items()})


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "unet", "naive", "vit"), default="unet")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("cifar", "mnist"), default="mnist")
    args.add_argument("--kernel", help="Which kernel to use for CKA", type=str, choices=["rbf", "linear"], default="linear")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.kernel, args.rotation, args.thicker, args.finetune)