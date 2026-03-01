import torch
from MI import info_nce
from HSIC import normalized_hsic

def gaussian_mi(x: torch.Tensor, y: torch.Tensor):
    """
    Computes MI assuming (x,y) are jointly Gaussian:
    I = -1/2 log(1 - rho^2)
    """
    x = x.squeeze()
    y = y.squeeze()

    rho = torch.corrcoef(torch.stack([x, y]))[0, 1]
    return -0.5 * torch.log(1 - rho**2 + 1e-12) # https://math.nyu.edu/~kleeman/infolect7.pdf

def stochastic_map(x: torch.Tensor, sigma: float = 1.0):
    return x + (sigma * torch.randn_like(x))

def deterministic_map(x: torch.Tensor):
    return x + 3

def equivariant_map(x: torch.Tensor):
    return 2 * x

def main():
    torch.manual_seed(0)
    x = torch.randn((2000, 1))

    print("Total:", gaussian_mi(x, x))
    print("Total (HSIC):", normalized_hsic(x, x))

    print("Stochastic:", gaussian_mi(stochastic_map(x), stochastic_map(-x)))
    print("Stochastic (HSIC):", normalized_hsic(stochastic_map(x), stochastic_map(-x)))

    noise = torch.randn_like(x)
    print("Consistent Noise:", gaussian_mi(x + noise, -x + noise))
    print("Consistent Noise (HSIC):", normalized_hsic(x + noise, -x + noise))

    print("Deterministic:", gaussian_mi(deterministic_map(x), deterministic_map(-x)))
    print("Deterministic (HSIC):", normalized_hsic(deterministic_map(x), deterministic_map(-x)))

    print("Equivariant:", gaussian_mi(equivariant_map(x), equivariant_map(-x)))
    print("Equivariant (HSIC):", normalized_hsic(equivariant_map(x), equivariant_map(-x)))



if __name__ == "__main__":
    main()
