import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Center a kernel matrix: K_c = HKH where H = I - (1/n)11^T."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    return H @ K @ H


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute the linear kernel matrix K = XX^T."""
    return X @ X.T


def hsic(X: torch.Tensor, Y: torch.Tensor, unbiased: bool = True) -> float:
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC)
    with a linear kernel.

    Args:
        X: Representation matrix of shape (n_samples, features_x).
        Y: Representation matrix of shape (n_samples, features_y).
        unbiased: If True, use the unbiased estimator (Song et al., 2012).
                  If False, use the biased estimator.

    Returns:
        HSIC value (float).
    """
    n = X.shape[0]
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

    X = torch.flatten(X, 1)
    Y = torch.flatten(Y, 1)

    K = linear_kernel(X)
    L = linear_kernel(Y)

    if unbiased:
        # Unbiased HSIC estimator
        # Zero out diagonals
        K.fill_diagonal_(0.0)
        L.fill_diagonal_(0.0)

        trace_KL = torch.trace(K @ L)
        sum_K = K.sum()
        sum_L = L.sum()
        sum_KL = (K.sum(axis=1) * L.sum(axis=1)).sum()  # sum of element-wise row sums product

        # Unbiased formula: 1/n(n-3) * [tr(KL) + sum(K)*sum(L)/((n-1)(n-2)) - 2*sum(K.*L row sums)/(n-2)]
        result = (
            trace_KL
            + (sum_K * sum_L) / ((n - 1) * (n - 2))
            - 2.0 * sum_KL / (n - 2)
        )
        return result / (n * (n - 3))
    else:
        # Biased HSIC estimator: (1/n^2) * tr(KHLH)
        Kc = center_kernel(K)
        Lc = center_kernel(L)
        return torch.trace(Kc @ Lc) / (n * n)


def cka(X: torch.Tensor, Y: torch.Tensor, unbiased: bool = True) -> float:
    """
    Compute Centered Kernel Alignment (CKA) with a linear kernel.

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    Args:
        X: Representation matrix of shape (n_samples, features_x).
        Y: Representation matrix of shape (n_samples, features_y).
        unbiased: If True, use the unbiased HSIC estimator.

    Returns:
        CKA similarity score in [0, 1] (though unbiased can slightly exceed bounds).
    """
    hsic_xy = hsic(X, Y, unbiased=unbiased)
    hsic_xx = hsic(X, X, unbiased=unbiased)
    hsic_yy = hsic(Y, Y, unbiased=unbiased)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return torch.tensor(0.0, dtype=torch.float32, device=X.device)
    return hsic_xy / denom

# ------------------------------------------------------------
# Equivariance Tracking
# ------------------------------------------------------------

if __name__ == "__main__":
    # Vectors
    vectors = torch.randn(100, 3)

    u1, u2, u3 = torch.rand(3)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    # Quaternion to rotation matrix
    R = torch.tensor([
        [1 - 2*(q3**2 + q4**2),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [    2*(q2*q3 + q1*q4), 1 - 2*(q2**2 + q4**2),     2*(q3*q4 - q1*q2)],
        [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2)]
    ])

    rotated_vectors = vectors @ R.T

    print(cka(vectors, vectors))
    print(cka(vectors, rotated_vectors))
    print(cka(vectors, torch.randn(100, 3)))