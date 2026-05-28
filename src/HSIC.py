import torch
from typing import Literal, Optional


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Center a kernel matrix: K_c = HKH where H = I - (1/n)11^T."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    return H @ K @ H

def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute the linear kernel matrix K = XX^T."""
    return X @ X.T

def rbf_kernel(X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """RBF kernel with median-heuristic bandwidth.

    The median heuristic adapts sigma to the intrinsic scale of the
    activations, making it consistent across layers and architectures.
    Computed per-call so each layer gets its own bandwidth.
    """
    # Pairwise squared distances via expansion trick (avoids pdist)
    sq_norms = (X ** 2).sum(dim=1)
    sq_dists = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * X @ X.T
    sq_dists = sq_dists.clamp(min=0.0)  # numerical safety

    if sigma is None:
        # Median heuristic on upper-triangle distances
        n = X.shape[0]
        mask = torch.triu(torch.ones(n, n, device=X.device, dtype=torch.bool), diagonal=1)
        median_sq = sq_dists[mask].median()
        sigma = (median_sq / 2.0).clamp(min=1e-10).sqrt()

    return torch.exp(-sq_dists / (2.0 * sigma ** 2))

def hsic(X: torch.Tensor, Y: torch.Tensor, unbiased: bool = True, kernel: Literal["linear", "rbf"] = "linear") -> float:
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

    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"No such kernel: {kernel}")

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


def cka(X: torch.Tensor, Y: torch.Tensor, unbiased: bool = True, kernel: Literal["linear", "rbf"] = "linear") -> float:
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
    hsic_xy = hsic(X, Y, unbiased=unbiased, kernel=kernel)
    hsic_xx = hsic(X, X, unbiased=unbiased, kernel=kernel)
    hsic_yy = hsic(Y, Y, unbiased=unbiased, kernel=kernel)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return torch.tensor(0.0, dtype=torch.float32, device=X.device)
    return torch.clamp(hsic_xy / denom, 0.0, 1.0)

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
    new_random_vectors = torch.randn(100, 3)

    print(cka(vectors, vectors))
    print(cka(vectors, rotated_vectors))
    print(cka(vectors, new_random_vectors))

    print(cka(vectors, vectors, kernel="rbf"))
    print(cka(vectors, rotated_vectors, kernel="rbf"))
    print(cka(vectors, new_random_vectors, kernel="rbf"))

    def random_vector_field_batch(batch_size=8, height=64, width=64, n_modes=5, seed=None):
        """
        Generate a batch of smooth random vector fields as image tensors.

        Returns:
            fields: (B, 2, H, W) tensor — channel 0 is U, channel 1 is V.
                    Ready to feed into a Conv2d layer.
        """
        if seed is not None:
            torch.manual_seed(seed)

        x = torch.linspace(-2, 2, width)
        y = torch.linspace(-2, 2, height)
        Y, X = torch.meshgrid(y, x, indexing="ij")  # (H, W)

        # Expand for batch: (B, H, W)
        X = X.unsqueeze(0).expand(batch_size, -1, -1)
        Y = Y.unsqueeze(0).expand(batch_size, -1, -1)

        U = torch.zeros(batch_size, height, width)
        V = torch.zeros(batch_size, height, width)

        for _ in range(n_modes):
            kx = torch.empty(batch_size, 1, 1).uniform_(-2, 2)
            ky = torch.empty(batch_size, 1, 1).uniform_(-2, 2)
            phase_u = torch.empty(batch_size, 1, 1).uniform_(0, 2 * torch.pi)
            phase_v = torch.empty(batch_size, 1, 1).uniform_(0, 2 * torch.pi)
            amp_u = torch.empty(batch_size, 1, 1).uniform_(-1, 1)
            amp_v = torch.empty(batch_size, 1, 1).uniform_(-1, 1)
            U += amp_u * torch.sin(kx * X + ky * Y + phase_u)
            V += amp_v * torch.sin(kx * X + ky * Y + phase_v)

        # Stack into (B, 2, H, W) image tensor
        fields = torch.stack([U, V], dim=1)
        return fields


    def rotate_vector_field_batch(fields, angle_deg):
        """
        Rotate every vector in a (B, 2, H, W) field by a fixed angle.

        Returns:
            rotated: (B, 2, H, W) tensor
        """
        theta = torch.tensor(angle_deg * torch.pi / 180.0)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        U, V = fields[:, 0], fields[:, 1]
        U_rot = cos_t * U - sin_t * V
        V_rot = sin_t * U + cos_t * V

        return torch.stack([U_rot, V_rot], dim=1)
    
    fields = random_vector_field_batch(batch_size=200, height=64, width=64)
    fields_rot = rotate_vector_field_batch(fields, 45)
    new_random_fields = random_vector_field_batch(200, 64, 64)

    print(cka(fields, fields))
    print(cka(fields, fields_rot))
    print(cka(fields, new_random_fields))
    print(cka(fields, fields, kernel="rbf"))
    print(cka(fields, fields_rot, kernel="rbf"))
    print(cka(fields, new_random_fields, kernel="rbf"))

    # Test Residual Equivariant (which should be low)
    fields_rot_90 = rotate_vector_field_batch(fields, 90)
    fields_rot_180 = rotate_vector_field_batch(fields, 180)
    fields_rot_270 = rotate_vector_field_batch(fields, 270)

    rotation_reps = ([torch.pi/2] * 200) + ([torch.pi] * 200) + ([3*torch.pi/2] * 200)
    rotation_reps = torch.tensor(rotation_reps).reshape(-1, 1)

    print(cka(rotation_reps, torch.vstack([fields - fields_rot_90, fields - fields_rot_180, fields - fields_rot_270])))
