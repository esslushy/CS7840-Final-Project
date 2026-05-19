import torch
from HSIC import cka
import numpy as np
from typing import Literal

def equiv_error_calc(net, images, kernel: Literal["linear", "rbf"] = "linear"):
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
        *_, layers = net(images)
        *_, layers_rot90 = net(images_rot90)
        *_, layers_rot180 = net(images_rot180)
        *_, layers_rot270 = net(images_rot270)

    for rep, rep90, rep180, rep270 in zip(
        layers, layers_rot90, layers_rot180, layers_rot270
    ):
        score = np.mean([
            cka(rep, rep90, False, kernel).item(),
            cka(rep, rep180, False, kernel).item(),
            cka(rep, rep270, False, kernel).item(),
        ])
        net_errors.append(score)

    return net_errors

def split_array_randomly(arr):
    n = len(arr)
    indices = np.random.permutation(n)
    return arr[indices[:n // 2]], arr[indices[n // 2:]]

def baseline_cka_computation(net, images, kernel: Literal["linear", "rbf"] = "linear"):
    cka_baselines = []

    images_x, images_y = split_array_randomly(images)

    with torch.inference_mode():
        *_, layers_x = net(images_x)
        *_, layers_y = net(images_y)

    for rep_x, rep_y in zip(layers_x, layers_y):
        cka_baselines.append(cka(rep_x, rep_y, False, kernel).item())

    return cka_baselines