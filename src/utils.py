import json
import os
import random
import re
import torch
import numpy as np


def save_all(net, statistics, tag):
    """Persist statistics and model weights atomically (write-temp-then-rename),
    so an interruption mid-write cannot leave a corrupt file. Called every epoch."""
    stats_path = f"results/{tag}_statistics.json"
    tmp_stats = stats_path + ".tmp"
    with open(tmp_stats, "wt") as f:
        json.dump(statistics, f)
    os.replace(tmp_stats, stats_path)

    model_path = f"models/{tag}_model.pth"
    tmp_model = model_path + ".tmp"
    torch.save(net.state_dict(), tmp_model)
    os.replace(tmp_model, model_path)


def so2_eval_angles(n):
    """
    Sample n evenly-spaced elements of SO(2) via the Lie algebra.

    SO(2) has a single generator J = [[0, -1], [1, 0]].
    The group elements are exp(t * J) = rotation by angle t.
    We sample t_k = 2π * k / (n+1) for k = 1, ..., n, which gives n
    uniformly spaced rotations excluding the identity (t=0).

    Returns (radians_tensor, degree_labels) where degree_labels are integer
    degrees used as JSON-friendly keys for the per-angle statistics.
    """
    ks = range(1, n + 1)
    radians = torch.tensor([2 * np.pi * k / (n + 1) for k in ks])
    degrees = [int(round(360.0 * k / (n + 1))) for k in ks]
    return radians, degrees


def rotate_2d(vecs, theta):
    """Rotate 2D vectors by angle theta (radians). vecs: (..., 2)."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    x, y = vecs[..., 0], vecs[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def mean_cka_per_layer(epoch_dict, stat):
    """
    epoch_dict maps layer -> {angle: compute_stats_dict}.
    Return a list of mean cka-scores (averaged over angles) per layer,
    in the layer order given by the dict keys.
    """
    means = []
    for layer, per_angle in epoch_dict.items():
        cka_vals = [stats[stat] for stats in per_angle.values()]
        means.append(float(np.mean(cka_vals)))
    return means


def set_seed(seed: int):
    """Seed every RNG a training run touches (python, numpy, torch CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_seed_suffix(stem: str) -> str:
    """Strip '_seed_<n>' from a statistics/model-file stem, e.g. for grouping multiple
    seeded runs of the same experiment under one output name. The seed marker sits
    before the trailing '_statistics'/'_model' suffix (tag = "..._seed_{n}", then
    save_all() appends "_statistics"/"_model"), so it isn't always at the very end."""
    return re.sub(r"_seed_\d+(?=$|_statistics$|_model$)", "", stem)


class Random90Rotation:
    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=(-2, -1))

class EquivarianceTracker:
    def __init__(self, device):
        self.device = device

        # RBF accumulators
        self.rbf_num = 0.0
        self.rbf_den_X = 0.0
        self.rbf_den_Y = 0.0

        # Linear accumulators
        self.lin_num = 0.0
        self.lin_den_X = 0.0
        self.lin_den_Y = 0.0

        self.total_batches = 0
        self.last_sigma = 0.0

    def _unbiased_hsic(self, K, L):
        """Unbiased HSIC estimator (kernel-agnostic). Zeros the diagonal."""
        n = K.size(0)
        if n < 4:
            raise ValueError("Batch size must be >= 4 to compute unbiased HSIC.")

        K = K.clone()
        L = L.clone()
        K.fill_diagonal_(0.0)
        L.fill_diagonal_(0.0)

        kl_trace = torch.trace(K @ L)
        k_sum = torch.sum(K)
        l_sum = torch.sum(L)

        row_sum_k = torch.sum(K, dim=1)
        row_sum_l = torch.sum(L, dim=1)
        row_vectors_dot = torch.dot(row_sum_k, row_sum_l)

        hsic = (kl_trace + (k_sum * l_sum) / ((n - 1) * (n - 2))
                - 2.0 * row_vectors_dot / (n - 2)) / (n * (n - 3))
        return hsic

    def update(self, feat_clean, feat_rot):
        X = feat_clean.flatten(start_dim=1).detach()
        Y = feat_rot.flatten(start_dim=1).detach()

        # --- sigma via pooled median heuristic ---
        pooled = torch.cat([X, Y], dim=0)
        pairwise_dists = torch.cdist(pooled, pooled, p=2)
        triu_idx = torch.triu_indices(
            row=pairwise_dists.size(0), col=pairwise_dists.size(1),
            offset=1, device=self.device
        )
        valid_dists = pairwise_dists[triu_idx[0], triu_idx[1]]
        sigma = torch.median(valid_dists).item()
        if sigma == 0:
            sigma = 1e-6
        self.last_sigma = sigma

        # --- RBF Grams ---
        dist_X = torch.cdist(X, X, p=2) ** 2
        dist_Y = torch.cdist(Y, Y, p=2) ** 2
        K_rbf = torch.exp(-dist_X / (2 * (sigma ** 2)))
        L_rbf = torch.exp(-dist_Y / (2 * (sigma ** 2)))

        # --- Linear Grams ---
        K_lin = X @ X.t()
        L_lin = Y @ Y.t()

        # --- accumulate RBF ---
        self.rbf_num   += self._unbiased_hsic(K_rbf, L_rbf).item()
        self.rbf_den_X += self._unbiased_hsic(K_rbf, K_rbf).item()
        self.rbf_den_Y += self._unbiased_hsic(L_rbf, L_rbf).item()

        # --- accumulate Linear ---
        self.lin_num   += self._unbiased_hsic(K_lin, L_lin).item()
        self.lin_den_X += self._unbiased_hsic(K_lin, K_lin).item()
        self.lin_den_Y += self._unbiased_hsic(L_lin, L_lin).item()

        self.total_batches += 1

    def _resolve(self, num, den_X, den_Y):
        if self.total_batches == 0 or den_X <= 0 or den_Y <= 0:
            return 1.0
        denom = (den_X * den_Y) ** 0.5
        score = num / denom
        return max(0.0, min(1.0, score))

    def compute_stats(self):
        return {
            "rbf_cka":    self._resolve(self.rbf_num, self.rbf_den_X, self.rbf_den_Y),
            "linear_cka": self._resolve(self.lin_num, self.lin_den_X, self.lin_den_Y),
            "cka_gap":    None if self.total_batches == 0 else
                          self._resolve(self.lin_num, self.lin_den_X, self.lin_den_Y)
                          - self._resolve(self.rbf_num, self.rbf_den_X, self.rbf_den_Y),
            "calibrated_sigma": self.last_sigma,
        }