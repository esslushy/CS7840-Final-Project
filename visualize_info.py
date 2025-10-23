import torch
from net import Net
from project_parameters import PATH
import torchvision
import torchvision.transforms as transforms
from entropy import mutual_info

# Discretization of continous values from the layers
def discretization(activations_list,bins):
    n_bins = bins

    bins = torch.linspace(min(torch.min(activations_list,axis=1)),
                        max(torch.max(activations_list,axis=1)), n_bins+1)
    activations_list[0] = torch.bucketize(activations_list, bins)
            
    return activations_list

def main():
    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    images, labels = next(iter(testloader))
    images_rot = torch.rot90(images, dims=(-2, -1))
    output, middle = net(images)
    output, middle_rot = net(images_rot)
    images = torch.flatten(images, 1)
    images_rot = torch.flatten(images_rot, 1)

    print(mutual_info(middle, images))
    print(mutual_info(middle_rot, images_rot))
    print(mutual_info(middle, labels.reshape(-1, 1)))
    print(mutual_info(middle_rot, labels.reshape(-1, 1)))

if __name__ == "__main__":
    main()