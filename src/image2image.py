import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from utils import equiv_error_calc, baseline_cka_computation
import json
from argparse import ArgumentParser
from pathlib import Path
from InvertNets import CNN

NUM_EPOCHS = 200
BATCH_SIZE = 64

def main(model: str, dataset: str, kernel: str, rotation: bool, thicker: bool, finetune: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_operations = [transforms.ToTensor()]
    if rotation:
        transform_operations.append(transforms.RandomRotation(degrees=(0, 360)))
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset) // 4, # Get all of the images.
                                            shuffle=False, num_workers=2)
    test_images, _ = next(iter(testloader))
    test_images = test_images.to(device)
    test_labels = 1.0 - test_images
    
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
        "baseline_cka": []
    }

    update_statistics(kernel, test_images, test_labels, net, criterion, statistics, trainloader, device)

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
        update_statistics(kernel, test_images, test_labels, net, criterion, statistics, trainloader, device)

    print('Finished Training')

    torch.save(net.state_dict(), f"models/classification_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}_kernel_{kernel}{'_finetuned' if finetune else ''}_model.pth")

    with open(f"results/image2image_{'learned_equivariant' if rotation else 'non_equivariant'}_{model}{'_thicker' if thicker else ''}_dataset_{dataset}_kernel_{kernel}{'_finetuned' if finetune else ''}_statistics.json", "wt+") as f:
        json.dump(statistics, f)    


def update_statistics(kernel, test_images, test_labels, net, criterion, statistics, trainloader, device):
    net.train()
    running_loss = 0.0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = 1.0 - inputs

        output, _ = net(inputs)

        running_loss += criterion(output, labels).item()
    statistics["train_loss"].append(running_loss / len(trainloader))
    net.eval()
    statistics["equivariant_loss"].append(equiv_error_calc(net, test_images, kernel))
    test_output, _ = net(test_images)
    statistics["test_loss"].append(criterion(test_output, test_labels).item())
    statistics["baseline_cka"].append(baseline_cka_computation(net, test_images, kernel))


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--model", help="Which model to use", type=str, choices=("cnn", "naive", "vit"), default="cnn")
    args.add_argument("--dataset", help="The dataset to train on.", type=str, choices=("cifar", "mnist"), default="cifar")
    args.add_argument("--kernel", help="Which kernel to use for CKA", type=str, choices=["rbf", "linear"], default="linear")
    args.add_argument("--rotation", help="Whether to train with rotation applied", action="store_true")
    args.add_argument("--thicker", help="Whether to make the dimension of the models thicker or not", action="store_true")
    args.add_argument("--finetune", help="The model to load for extra finetuning", type=Path)
    args = args.parse_args()
    
    if args.model == "naive" and args.thicker:
        raise Exception("Can't make a thicker naive model.")
    
    main(args.model, args.dataset, args.kernel, args.rotation, args.thicker, args.finetune)