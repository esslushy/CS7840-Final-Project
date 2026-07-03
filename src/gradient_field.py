import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from utils import EquivarianceTracker, Random90Rotation
from Models.GradientFieldNets import UNet, CNN, ViT, NaiveNet

NUM_EPOCHS = 20
BATCH_SIZE = 64

ANGLES = (90, 180, 270)   # non-identity C4 rotations tracked for equivariance


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


def main(model: str, dataset: str, rotation: bool, thicker: bool, finetune: Path):
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
    }

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    tag = (f"gradient_field_{'learned_equivariant' if rotation else 'non_equivariant'}"
           f"_{model}{'_thicker' if thicker else ''}_dataset_{dataset}"
           f"{'_finetuned' if finetune else ''}")

    # baseline measurement before any training, then persist immediately
    update_statistics(net, criterion, statistics, trainloader, testloader, device)
    save_all(net, statistics, tag)

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


def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    net.eval()
    running_train_loss = 0.0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)

        target_grad = sobel_gradient(inputs)
        pred_grad, layers = net(inputs)

        running_train_loss += criterion(pred_grad, target_grad).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    # per-angle test loss (0 = unrotated). Gradient prediction is EQUIVARIANT, so
    # the target for a rotated input is the gradient OF the rotated input.
    running_test_loss = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    # per-layer, PER-ANGLE equivariance tracker (kept separate, not averaged)
    running_equivariant = {k: {a: EquivarianceTracker(device) for a in ANGLES}
                           for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)

            # unrotated prediction + activations (equivariance reference)
            pred_grad, layers = net(inputs)
            running_test_loss[0] += criterion(pred_grad, sobel_gradient(inputs)).item()

            for angle in ANGLES:
                k = angle // 90
                inputs_rot = torch.rot90(inputs, k, dims=(-2, -1))
                target_grad_rot = sobel_gradient(inputs_rot)
                pred_rot, layers_rot = net(inputs_rot)   # one forward, reused for loss + equivariance

                running_test_loss[angle] += criterion(pred_rot, target_grad_rot).item()

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
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("cifar", "mnist"), default="mnist")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)