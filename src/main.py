import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from Models import CNN, ViT, NaiveNet
from utils import equiv_error_calc, baseline_cka_computation
import json
from argparse import ArgumentParser
from pathlib import Path

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_EPOCHS = 200
BATCH_SIZE = 64

def main(model: str, kernel: str, rotation: bool, holdout: str, thicker: bool, finetune: Path):
    transform_operations = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if rotation and not holdout:
        transform_operations.append(transforms.RandomRotation(degrees=(0, 360)))
    transform_train = transforms.Compose(transform_operations)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)
    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000, # Get all of the images.
                                            shuffle=False, num_workers=2)
    test_images, test_labels = next(iter(testloader))
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    
    if model == "vit":
        net = ViT(image_size=test_images.shape[2], patch_size=4, num_classes=10, dim=256 if thicker else 128, depth=1, heads=1, mlp_dim=256 if thicker else 128)
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
        "equivariant_losses": [],
        "test_losses": [],
        "baseline_cka": []
    }

    net.eval()
    statistics["equivariant_losses"].append(equiv_error_calc(net, test_images, kernel))
    statistics["test_losses"].append(criterion(net(test_images)[1], test_labels).item())
    statistics["baseline_cka"].append(baseline_cka_computation(net, test_images, kernel))

    if holdout:
        random_rot = transforms.RandomRotation(degrees=(0, 360))

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            if holdout:
                # Apply random rotations to all but holdout
                inputs[labels!=CLASSES.index(holdout)] = random_rot(inputs[labels!=CLASSES.index(holdout)])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}')
        net.eval()
        statistics["equivariant_losses"].append(equiv_error_calc(net, test_images, kernel))
        statistics["test_losses"].append(criterion(net(test_images)[1], test_labels).item())
        statistics["baseline_cka"].append(baseline_cka_computation(net, test_images, kernel))

    print('Finished Training')

    torch.save(net.state_dict(), f"models/cifar_{model}.pth")

    with open(f"results/{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_kernel_{kernel}{f'_holdout_{holdout}' if holdout else ''}{'_finetuned' if finetune else ''}_statistics.json", "wt+") as f:
        json.dump(statistics, f)

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "naive", "vit"), default="cnn")
    args.add_argument("--kernel", help="Which kernel to use for CKA", type=str, choices=["rbf", "linear"], default="linear")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--holdout", help="The class to hold out from rotation", type=str, choices=CLASSES)
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()

    if args.holdout and not args.rotation:
        raise Exception("Can't hold out class from rotation if not doing rotation.")
    
    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")
    
    main(args.model, args.kernel, args.rotation, args.holdout, args.thicker, args.finetune)