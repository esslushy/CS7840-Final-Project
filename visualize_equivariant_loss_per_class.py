import torch
from net import Net
from project_parameters import PATH
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import equiv_error_calc

def main():
    batch_size = 10000

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    images, labels = next(iter(testloader))

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

    net = Net()
    net.load_state_dict(torch.load(f"{PATH}_ood.pth", weights_only=True))
    for i, c in enumerate(classes):
        net_error = equiv_error_calc(net, images[labels==i])

        indices = list(range(len(net_error)))

        plt.plot(indices, net_error, label=f"{i}: {c}", linewidth=2, marker='o')

    plt.legend()

    # Add titles and labels
    plt.title("Delta InfoNCE Between Rotated and Non Rotated Images Per Class")
    plt.xlabel("Layer Number")
    plt.ylabel("Delta InfoNCE Loss (Lower is Better)")

    plt.tight_layout()

    # Show the chart
    plt.savefig(f"ood_equivariant_loss_per_class.pdf")

if __name__ == "__main__":
    main()