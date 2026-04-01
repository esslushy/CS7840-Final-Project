import torch
from HSIC import cka
import numpy as np

def equiv_error_calc(net, images):
    """
    Computes layerwise equivariance score using
    normalized HSIC with automatic sigma^2.

    Returns:
        List of HSIC scores per layer.
        Higher = more dependence (more equivariant).
    """

    images_rot90 = torch.rot90(images, 1, dims=(-2, -1))
    images_rot180 = torch.rot90(images, 2, dims=(-2, -1))
    images_rot270 = torch.rot90(images, 3, dims=(-2, -1))

    net_errors = []

    with torch.inference_mode():
        _, _, layers = net(images)
        _, _, layers_rot90 = net(images_rot90)
        _, _, layers_rot180 = net(images_rot180)
        _, _, layers_rot270 = net(images_rot270)

    for rep, rep90, rep180, rep270 in zip(
        layers, layers_rot90, layers_rot180, layers_rot270
    ):
        score = np.mean([
            cka(rep, rep90, False).item(),
            cka(rep, rep180, False).item(),
            cka(rep, rep270, False).item(),
        ])
        net_errors.append(score)

    return net_errors