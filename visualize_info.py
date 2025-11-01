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
    net_rot = Net()
    net_rot.load_state_dict(torch.load(f"{PATH}_rotate.pth", weights_only=True))

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 10000

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    images, labels = next(iter(testloader))
    images_rot = torch.rot90(images, dims=(-2, -1))
    output, middle = net(images)
    output_rot, middle_rot = net(images_rot)
    output = torch.nn.functional.softmax(output)
    output_rot = torch.nn.functional.softmax(output_rot)
    images = torch.flatten(images, 1)
    images_rot = torch.flatten(images_rot, 1)
    # print(info_nce(middle, labels.reshape(-1, 1)))
    # print(info_nce(middle_rot, labels.reshape(-1, 1)))
    print(info_nce(middle, middle))
    print(info_nce(middle, middle_rot))
    print(info_nce(images, images))
    print(info_nce(images, images_rot))
    print(info_nce(output, output_rot))
    # print(info_nce(images, labels.reshape(-1, 1)))
    # discretized = discretization(middle, 30)
    # discretized_rot = discretization(middle_rot, 30)
    # print(mutual_info(discretized, discretized))
    # print(mutual_info(discretized_rot, discretized))
    exit()
    p1 = np.array([mutual_info(discretized, images), mutual_info(discretized, labels.reshape(-1, 1))])
    p2 = np.array([mutual_info(discretized_rot, images_rot), mutual_info(discretized_rot, labels.reshape(-1, 1))])
    # Compute the distance
    distance = np.linalg.norm(p2 - p1)

    # Plot the points
    plt.figure(figsize=(6, 6))
    plt.scatter(*p1, color='blue', label='Original Image')
    plt.scatter(*p2, color='red', label='Rotated Image')

    # Draw a line between the points
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', label=f'Distance = {distance:.2f}')

    midpoint = (p1 + p2) / 2

    plt.text(midpoint[0] + 0.1, midpoint[1] + 0.1, "Equivariance Error")

    # Style the plot
    plt.title('Distance Between Two Points')
    plt.xlabel('I(X, T)')
    plt.ylabel('I(T, Y)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()