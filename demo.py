import torch
from MI import info_nce

def gaussian_mi(x: torch.Tensor, y: torch.Tensor):
    """
    Computes MI assuming (x,y) are jointly Gaussian:
    I = -1/2 log(1 - rho^2)
    """
    x = x.squeeze()
    y = y.squeeze()

    rho = torch.corrcoef(torch.stack([x, y]))[0, 1]
    return -0.5 * torch.log(1 - rho**2 + 1e-12) # https://math.nyu.edu/~kleeman/infolect7.pdf


def stochastic_map(x: torch.Tensor, noise: torch.Tensor):
    return x + noise


def deterministic_map(x: torch.Tensor):
    return x + 3


def equivariant_map(x: torch.Tensor):
    return 2 * x


def main():
    torch.manual_seed(0)
    x = torch.randn((2000, 1))
    sigma = 1.0
    noise = torch.randn_like(x) * sigma

    print("Total:", gaussian_mi(x, x))

    print("Stochastic:", gaussian_mi(x, stochastic_map(x, noise)))
    print("Stochastic(-x):", gaussian_mi(x, stochastic_map(-x, noise)))

    print("Deterministic:", gaussian_mi(x, deterministic_map(x)))
    print("Deterministic(-x):", gaussian_mi(x, deterministic_map(-x)))

    print("Negation:", gaussian_mi(x, -x))

    print("Equivariant:", gaussian_mi(x, equivariant_map(x)))
    print("Equivariant(-x):", gaussian_mi(x, equivariant_map(-x)))


if __name__ == "__main__":
    main()
