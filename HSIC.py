import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def flatten_rep(rep):
    """
    Flattens representation to (batch, d).
    Works for linear or convolutional layers.
    """
    if rep.dim() > 2:
        return rep.view(rep.size(0), -1)
    return rep


def normalize_features(X, eps=1e-8):
    """
    Per-feature standardization:
    zero mean, unit variance.
    """
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True) + eps
    return (X - mean) / std


def compute_sigma2(X):
    """
    Automatic sigma^2 = 2d rule.
    Assumes X is normalized and shape (batch, d).
    """
    d = X.shape[1]
    return 2.0 * d

def compute_sigma2_median(X):
    with torch.no_grad():
        D2 = torch.cdist(X, X, p=2) ** 2
        median = torch.median(D2[D2 > 0])
    return median

def rbf_kernel(X, sigma2):
    """
    RBF kernel matrix using sigma^2.
    """
    # Pairwise squared distances
    XX = torch.cdist(X, X, p=2) ** 2
    K = torch.exp(-XX / (2.0 * sigma2))
    return K


def normalized_hsic(X, Y, eps=1e-8):
    """
    Computes normalized HSIC (CKA-style).

    Steps:
    1) Flatten
    2) Normalize features
    3) Compute sigma^2 = 2d automatically
    4) Compute centered RBF kernels
    5) Normalize
    """

    X = flatten_rep(X)
    Y = flatten_rep(Y)

    X = normalize_features(X)
    Y = normalize_features(Y)

    sigma2_x = compute_sigma2_median(X)
    print(sigma2_x)
    sigma2_y = compute_sigma2_median(Y)
    print(sigma2_y)

    K = rbf_kernel(X, sigma2_x)
    L = rbf_kernel(Y, sigma2_y)

    K.fill_diagonal_(0)
    L.fill_diagonal_(0)

    hsic = (K * L).sum()

    norm_x = torch.sqrt((K * K).sum() + eps)
    norm_y = torch.sqrt((L * L).sum() + eps)

    return hsic / (norm_x * norm_y + eps)


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

    print(normalized_hsic(vectors, vectors))
    print(normalized_hsic(vectors, rotated_vectors))
    print(normalized_hsic(vectors, torch.randn(100, 3)))