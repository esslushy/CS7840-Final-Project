import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.optim as optim
import json
from argparse import ArgumentParser
from pathlib import Path
import sys
from utils import Random90Rotation, EquivarianceTracker, set_seed, save_all
from print_digit import load_mnist_font_dataset

if "mnist_font" in sys.argv:
    CLASSES = tuple(range(10))
else:
    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if "mnist_font" in sys.argv:
    NUM_EPOCHS = 400
else:
    NUM_EPOCHS = 200
BATCH_SIZE = 64
ANGLES = (90, 180, 270) 


def main(model: str, dataset: str, rotation: bool, holdout: str, thicker: bool, finetune: Path, resume: bool, seed: int):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "cifar":
        from Models.Classification.CIFARNets import CNN, ViT, NaiveNet
        trainset, testset = load_cifar(rotation, holdout)
    elif dataset == "mnist_font":
        from Models.Classification.MNISTNets import CNN, ViT, NaiveNet
        trainset, testset = load_mnist_font(rotation, holdout)
    else:
        raise Exception("Unknown Dataset")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    if model == "vit":
        net = ViT(image_size=32 if dataset == "cifar" else 28, patch_size=4, num_classes=10,
                  dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128)
    elif model == "naive":
        net = NaiveNet()
    elif model == "cnn":
        net = CNN(width1=256 if thicker else 120, width2=256 if thicker else 84)
    else:
        raise ValueError(f"No such model {model}")
    net = net.to(device)
    if finetune:
        net.load_state_dict(torch.load(finetune, weights_only=True))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    statistics = {
        "equivariant_loss": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    tag = (f"classification_{'learned_equivariant' if rotation else 'non_equivariant'}"
           f"_{model}{'_thicker' if thicker else ''}_dataset_{dataset}"
           f"{f'_holdout_{holdout}' if holdout else ''}{'_finetuned' if finetune else ''}"
           f"_seed_{seed}")

    if holdout:
        random_rot = Random90Rotation(degrees=(0, 360))

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
        running_accuracy = 0.0
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            if holdout:
                # Apply random rotations to all but holdout
                inputs[labels != CLASSES.index(holdout)] = random_rot(inputs[labels != CLASSES.index(holdout)])

            optimizer.zero_grad()

            output, logits, _ = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy(output, labels).item()
        print(f"[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}")

        # measure and persist EVERY epoch so an interrupted run keeps all stats so far
        update_statistics(net, criterion, statistics, trainloader, testloader, device)
        save_all(net, statistics, tag)

    print('Finished Training')
    # already saved each epoch; final save is just belt-and-suspenders
    save_all(net, statistics, tag)


def load_cifar(rotation, holdout):
    transform_operations = [transforms.ToTensor()]
    if rotation and not holdout:
        transform_operations.append(Random90Rotation())

    transform_train = transforms.Compose(transform_operations)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)

    transform_test = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    return trainset, testset


def load_mnist_font(rotation, holdout):
    transform_operations = [transforms.ToTensor()]
    if rotation and not holdout:
        transform_operations.append(transforms.RandomRotation(degrees=(0, 360)))
    return load_mnist_font_dataset(
        transforms.Compose(transform_operations),
        transforms.ToTensor()
    )


def update_statistics(net, criterion, statistics, trainloader, testloader, device):
    net.eval()
    running_train_loss = 0.0
    running_train_accuracy = 0.0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        output, logits, layers = net(inputs)

        running_train_loss += criterion(logits, labels).item()
        running_train_accuracy += accuracy(output, labels).item()
    statistics["train_loss"].append(running_train_loss / len(trainloader))
    statistics["train_accuracy"].append(running_train_accuracy / len(trainloader))

    running_test_loss = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    running_test_accuracy = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    running_equivariant = {k: {a: EquivarianceTracker(device) for a in ANGLES}
                           for k in layers.keys()}
    with torch.inference_mode():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # unrotated activations, used as the equivariance reference
            output, logits, layers = net(inputs)
            running_test_loss[0] += criterion(logits, labels).item()
            running_test_accuracy[0] += accuracy(output, labels).item()

            for angle in ANGLES:
                k = angle // 90
                inputs_rot = torch.rot90(inputs, k, dims=(-2, -1))
                output_rot, logits_rot, layers_rot = net(inputs_rot)   # one forward, reused for loss + equivariance

                running_test_loss[angle] += criterion(logits_rot, labels).item()
                running_test_accuracy[angle] += accuracy(output_rot, labels).item()

                for key in layers.keys():
                    running_equivariant[key][angle].update(layers[key], layers_rot[key])

    n_test = len(testloader)
    statistics["test_loss"].append({angle: v / n_test for angle, v in running_test_loss.items()})
    statistics["test_accuracy"].append({angle: v / n_test for angle, v in running_test_accuracy.items()})
    # store the full tracker stats (z, p, floor_std, ...) per layer, per angle
    statistics["equivariant_loss"].append({
        key: {angle: running_equivariant[key][angle].compute_stats() for angle in ANGLES}
        for key in running_equivariant
    })


def accuracy(output, labels):
    return (torch.argmax(output, dim=-1) == labels).sum() / len(output)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "naive", "vit"), default="cnn")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("cifar", "mnist_font"), default="cifar")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--holdout", help="The class to hold out from rotation", type=str, choices=CLASSES)
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args.add_argument("--resume", help="Resume training", action="store_true")
    args.add_argument("--seed", help="Random seed for reproducibility", type=int, default=0)
    args = args.parse_args()

    if args.holdout and not args.rotation:
        raise Exception("Can't hold out class from rotation if not doing rotation.")

    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")

    main(args.model, args.dataset, args.rotation, args.holdout, args.thicker, args.finetune, args.resume, args.seed)