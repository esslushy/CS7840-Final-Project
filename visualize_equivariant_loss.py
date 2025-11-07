import torch
from net import Net
from project_parameters import PATH
import torchvision
import torchvision.transforms as transforms
from entropy import mutual_info
import numpy as np
import matplotlib.pyplot as plt
from MI import info_nce, gaussian_mi

def equiv_error_calc(net, images):
    images_rot90 = torch.rot90(images, dims=(-2, -1))
    images_rot180 = torch.rot90(images_rot90, dims=(-2, -1))
    images_rot270 = torch.rot90(images_rot180, dims=(-2, -1))
    net_errors = []
    
    with torch.inference_mode():
        output, _, layers = net(images)
        output_rot90, _, layers_rot90 = net(images_rot90)
        output_rot180, _, layers_rot180 = net(images_rot180)
        output_rot270, _, layers_rot270 = net(images_rot270)

    for rep, rep90, rep180, rep270 in zip(layers, layers_rot90, layers_rot180, layers_rot270):
        baseline = info_nce(rep, rep)
        net_errors.append(np.mean([
            info_nce(rep90, rep) - baseline,
            info_nce(rep180, rep) - baseline,
            info_nce(rep270, rep) - baseline,
        ]))
    return net_errors

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
    

    net = Net()
    net.load_state_dict(torch.load(f"{PATH}.pth", weights_only=True))
    net_error = equiv_error_calc(net, images)

    net.load_state_dict(torch.load(f"{PATH}_rotate.pth", weights_only=True))
    net_rotate_error = equiv_error_calc(net, images)

    indices = list(range(len(net_rotate_error)))

    plt.plot(indices, net_error, label="Original Model", color='blue', linewidth=2, marker='o')
    plt.plot(indices, net_rotate_error, label="Learned Equivariant Model", color='orange', linewidth=2, marker='s')

    plt.legend()

    # Add titles and labels
    plt.title("Delta InfoNCE Between Rotated and Non Rotated Images")
    plt.xlabel("Layer Number")
    plt.ylabel("Delta InfoNCE Loss (Lower is Better)")

    plt.tight_layout()

    # Show the chart
    plt.savefig(f"equivariant_loss.pdf")

if __name__ == "__main__":
    main()