import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
from Models.InvertNets import CNN
from utils import Random90Rotation, EquivarianceTracker
import numpy as np

NUM_EPOCHS = 200
BATCH_SIZE = 64

ANGLES = (90, 180, 270)   # non-identity C4 rotations tracked for equivariance


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
        in_channels = 3
    elif dataset == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=transform_train)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform_test)
        in_channels = 1
    else:
        raise ValueError(f"No such dataset: {dataset}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,  # Get all of the images.
                                             shuffle=False, num_workers=2)

    if model == "vit":
        raise NotImplementedError()
    elif model == "naive":
        raise NotImplementedError()
    elif model == "cnn":
        net = CNN(in_channels=in_channels, num_features=32 if thicker else 16)
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

    tag = (f"image_inversion_{'learned_equivariant' if rotation else 'non_equivariant'}"
           f"_{model}{'_thicker' if thicker else ''}_dataset_{dataset}"
           f"{'_finetuned' if finetune else ''}")

    # baseline measurement before any training, then persist immediately
    update_statistics(net, criterion, statistics, trainloader, testloader, device)
    save_all(net, statistics, tag)

    for epoch in range(NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            inputs, _ = data
            inputs = inputs.to(device)
            labels = 1.0 - inputs

            optimizer.zero_grad()

            output, _ = net(inputs)
            loss = criterion(output, labels)
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


def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    net.eval()
    running_train_loss = 0.0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = 1.0 - inputs

        output, layers = net(inputs)

        running_train_loss += criterion(output, labels).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))

    # per-angle test loss (0 = unrotated). Inversion is EQUIVARIANT, so the target
    # for a rotated input is the inverted rotated input: 1 - rot90(input).
    running_test_loss = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    # per-layer, PER-ANGLE equivariance tracker (kept separate, not averaged)
    running_equivariant = {k: {a: EquivarianceTracker(device) for a in ANGLES}
                           for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            inputs, _ = data
            inputs = inputs.to(device)

            # unrotated prediction + activations (equivariance reference)
            output, layers = net(inputs)
            running_test_loss[0] += criterion(output, 1.0 - inputs).item()

            for angle in ANGLES:
                k = angle // 90
                inputs_rot = torch.rot90(inputs, k, dims=(-2, -1))
                target_rot = 1.0 - inputs_rot
                output_rot, layers_rot = net(inputs_rot)   # one forward, reused for loss + equivariance

                running_test_loss[angle] += criterion(output_rot, target_rot).item()

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
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "naive", "vit"), default="cnn")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("cifar", "mnist"), default="cifar")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.thicker, args.finetune)