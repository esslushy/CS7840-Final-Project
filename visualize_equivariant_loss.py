import torch
from net import Net
from project_parameters import PATH
import torchvision
import torchvision.transforms as transforms
from entropy import mutual_info
import numpy as np
import matplotlib.pyplot as plt
from MI import info_nce, gaussian_mi

# Discretization of continous values from the layers
def discretization(activations_list,bins):
    n_bins = bins

    bins = torch.linspace(torch.min(activations_list),
                        torch.max(activations_list), n_bins+1)
    activations_list = torch.bucketize(activations_list, bins)
            
    return activations_list

def main():
    net = Net()
    net.load_state_dict(torch.load(f"{PATH}.pth", weights_only=True))

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 10000

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    images, labels = next(iter(testloader))
    images_rot90 = torch.rot90(images, dims=(-2, -1))
    images_rot180 = torch.rot90(images_rot90, dims=(-2, -1))
    images_rot270 = torch.rot90(images_rot180, dims=(-2, -1))
    with torch.inference_mode():
        output, middle = net(images)
        output_rot90, middle_rot90 = net(images_rot90)
        output_rot180, middle_rot180 = net(images_rot180)
        output_rot270, middle_rot270 = net(images_rot270)
    output = torch.nn.functional.softmax(output, dim=-1)
    output_rot90 = torch.nn.functional.softmax(output_rot90, dim=-1)
    images = torch.flatten(images, 1)
    images_rot90 = torch.flatten(images_rot90, 1)
    images_rot180 = torch.flatten(images_rot180, 1)
    images_rot270 = torch.flatten(images_rot270, 1)
    #Minimizing info nce maximizes MI
    data = {
        "0 Degrees": info_nce(middle, middle),
        "90 Degrees": info_nce(middle, middle_rot90),
        "180 Degrees": info_nce(middle, middle_rot180),
        "-90 Degrees": info_nce(middle, middle_rot270)
    }

    labels = list(data.keys())
    values = list(data.values())

    # Create the bar chart
    plt.figure(figsize=(8, 5))

    ref_label = "0 Degrees"
    ref_value = data[ref_label]

    
    for i, (label, val) in enumerate(data.items()):
        if val >= ref_value:
            # Draw the base portion up to the reference value
            plt.bar(label, ref_value, color="blue")
            # Draw the extra difference on top
            plt.bar(label, val - ref_value, bottom=ref_value, color="orange")
        else:
            # Draw the smaller bar
            plt.bar(label, val, color="blue")
            # Draw the missing portion (difference below reference)
            plt.bar(label, ref_value - val, bottom=val, color="salmon", alpha=0.6, hatch="//")

    plt.axhline(ref_value, color="black", linestyle="--", label=f"{ref_label} = {ref_value}")

    # Add titles and labels
    plt.title("InfoNCE Mutual Information Meausure Between Non-Rotated and Rotated Images")
    plt.xlabel("Degrees of Rotation")
    plt.ylabel("InfoNCE Loss (Lower is Better)")

    plt.tight_layout()

    # Show the chart
    plt.savefig("equivariant_loss.pdf")

if __name__ == "__main__":
    main()