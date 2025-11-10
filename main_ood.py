import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from net import Net
from project_parameters import PATH
from utils import equiv_error_calc
import json

def main():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    batch_size = 10000

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                            shuffle=False, num_workers=2)
    
    test_images, test_labels = next(iter(testloader))
    test_images = test_images.to(device)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net = Net()
    net = net.to(device)

    random_rot = transforms.RandomRotation(degrees=(0, 360))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    equivariant_losses = []

    equivariant_losses.append(equiv_error_calc(net, test_images))

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Apply random rotations to all but class 0
            inputs[labels!=0] = random_rot(inputs[labels!=0])

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
        equivariant_losses.append(equiv_error_calc(net, test_images))

    print('Finished Training')

    torch.save(net.state_dict(), f"{PATH}_ood.pth")

    with open("ood_equivariant_model_equivariant_losses.json", "wt+") as f:
        json.dump(equivariant_losses, f)

if __name__ == "__main__":
    main()