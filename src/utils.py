import numpy as np
import torch

def split_array_randomly(arr):
    n = len(arr)
    indices = np.random.permutation(n)
    return arr[indices[:n // 2]], arr[indices[n // 2:]]

class Random90Rotation:
    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=(-2, -1))