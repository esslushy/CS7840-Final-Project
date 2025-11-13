import torch
from net import Net
from project_parameters import PATH
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import equiv_error_calc
from vit import ViT
from argparse import ArgumentParser

def main(vit):
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
    

    if vit:
        net = ViT(image_size=images.shape[2], patch_size=4, num_classes=10, dim=128, depth=1, heads=1, mlp_dim=128)
    else:
        net = Net()
    net.load_state_dict(torch.load(f"{PATH}{'_vit' if vit else ''}.pth", weights_only=True))
    net_error = equiv_error_calc(net, images)

    net.load_state_dict(torch.load(f"{PATH}{'_vit' if vit else ''}_rotate.pth", weights_only=True))
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
    plt.savefig(f"equivariant_loss{'_vit' if vit else ''}.pdf")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--vit", help="Use ViT model", action="store_true")
    args = args.parse_args()
    main(args.vit)