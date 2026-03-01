import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
from vit import ViT
from naive_net import NaiveNet
from net import Net
import torchvision.transforms as transforms
import torchvision

def get_equivariance_mask(labels, heldout_classes):
    """
    Returns mask of which samples should receive equivariance loss
    """

    mask = torch.ones_like(labels, dtype=torch.bool)

    for c in heldout_classes:
        mask &= (labels != c)

    return mask

def random_rotation_matrix(d, device):
    A = torch.randn(d, d, device=device)
    Q, R = torch.linalg.qr(A)
    return Q

def rotate_batch(x):

    d = x.shape[-1]
    R = random_rotation_matrix(d, x.device)

    return x @ R, R

def equivariance_loss_masked(features, features_rot, mask):

    if mask.sum() == 0:
        return torch.tensor(0.0, device=features[0].device)

    loss = 0

    for f, f_rot in zip(features, features_rot):

        f = f.flatten(1)
        f_rot = f_rot.flatten(1)

        f = f[mask]
        f_rot = f_rot[mask]

        loss += F.mse_loss(f, f_rot)

    return loss / len(features)

def variance_loss(features, gamma=1.0, eps=1e-4):

    loss = 0

    for f in features:

        f = f.flatten(1)

        std = torch.sqrt(f.var(dim=0) + eps)

        loss += torch.mean(F.relu(gamma - std))

    return loss / len(features)

def covariance_loss(features):

    loss = 0

    for f in features:

        f = f.flatten(1)

        f = f - f.mean(dim=0)

        cov = (f.T @ f) / (f.shape[0] - 1)

        off_diag = cov - torch.diag(torch.diag(cov))

        loss += (off_diag**2).mean()

    return loss / len(features)

def train_step(
    model,
    optimizer,
    x,
    labels,
    heldout_classes,
    lambda_eq=1.0,
    lambda_var=1.0,
):

    model.train()

    x_rot, _ = rotate_batch(x)

    labels_pred, logits, features = model(x)
    labels_pred_rot, logits_rot, features_rot = model(x_rot)

    task_loss = F.cross_entropy(logits, labels)

    equiv_mask = get_equivariance_mask(labels, heldout_classes)

    eq_loss = equivariance_loss_masked(
        features,
        features_rot,
        equiv_mask
    )

    var_loss = variance_loss(features)

    loss = task_loss + (lambda_eq * eq_loss) + (lambda_var * var_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"total": loss.item(), "equiv": eq_loss.item()}


def train(model, dataloader, optimizer, epochs, holdout_classes):

    for epoch in range(epochs):

        stats = []

        for x, y in dataloader:

            x = x.cuda()
            y = y.cuda()

            s = train_step(model, optimizer, x, y, holdout_classes)

            stats.append(s)

        print(
            epoch,
            "loss:",
            sum(d["total"] for d in stats)/len(stats),
            "equiv:",
            sum(d["equiv"] for d in stats)/len(stats),
        )

def evaluate_equivariance_by_class(model, dataloader):

    model.eval()

    class_errors = {}

    with torch.no_grad():

        for x, labels in dataloader:

            x = x.cuda()
            labels = labels.cuda()

            x_rot, _ = rotate_batch(x)

            labels_pred, _, features = model(x)
            labels_pred_rot, _, features_rot = model(x_rot)

            final_f = features[-1].flatten(1)
            final_f_rot = features_rot[-1].flatten(1)

            errors = ((final_f - final_f_rot)**2).mean(dim=1)

            for label, err in zip(labels, errors):

                label = label.item()

                if label not in class_errors:
                    class_errors[label] = []

                class_errors[label].append(err.item())

    for k in class_errors:
        class_errors[k] = sum(class_errors[k]) / len(class_errors[k])

    return class_errors

def main(vit: bool, naive: bool):
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
    
    if vit:
        model = ViT(image_size=test_images.shape[2], patch_size=4, num_classes=10, dim=128, depth=1, heads=1, mlp_dim=128)
    elif naive:
        model = NaiveNet()
    else:
        model = Net()
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    heldout_classes = ['plane']  # class 0 never trained with equivariance

    train(model, trainloader, optimizer, 200, heldout_classes)

    errors = evaluate_equivariance_by_class(model, testloader)

    print("Equivariance error per class:")
    for c, err in errors.items():
        print(f"class {classes[c]} {'(held out)' if classes[c] in heldout_classes else ''}: {err:.4f}")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--vit", help="Use ViT model", action="store_true")
    args.add_argument("--naive", help="Use naive model", action="store_true")
    args = args.parse_args()
    main(args.vit, args.naive)