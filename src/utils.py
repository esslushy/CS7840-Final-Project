import torch
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