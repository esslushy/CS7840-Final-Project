import torch
class Random90Rotation:
    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=(-2, -1))

class EquivarianceTracker:
    def __init__(self, device):
        """
        Memory-free tracker matching your loop's exact signature.
        Calculates sigma and unbiased HSIC on the fly per batch.
        """
        self.device = device
        
        # Accumulators for global dataset-wide averaging
        self.hsic_num = 0.0
        self.hsic_den_X = 0.0
        self.hsic_den_Y = 0.0
        self.total_batches = 0
        self.last_sigma = 0.0 # Stored for monitoring/logs

    def _unbiased_hsic(self, K, L):
        """
        Computes the mathematically unbiased HSIC estimator for a batch.
        Removes the diagonal bias so scores can be averaged over a dataset.
        """
        n = K.size(0)
        if n < 4:
            raise ValueError("Batch size must be >= 4 to compute unbiased HSIC.")
            
        # Set diagonal entries to zero
        K.fill_diagonal_(0.0)
        L.fill_diagonal_(0.0)
        
        # Unbiased calculation components
        kl_trace = torch.trace(K @ L)
        k_sum = torch.sum(K)
        l_sum = torch.sum(L)
        
        row_sum_k = torch.sum(K, dim=1)
        row_sum_l = torch.sum(L, dim=1)
        row_vectors_dot = torch.dot(row_sum_k, row_sum_l)
        
        # Unbiased HSIC equation
        hsic = (kl_trace + (k_sum * l_sum) / ((n - 1) * (n - 2)) 
                - 2.0 * row_vectors_dot / (n - 2)) / (n * (n - 3))
        return hsic

    def update(self, feat_clean, feat_rot):
        """
        Processes a mini-batch immediately and throws away the raw activations.
        Matches signature: tracker.update(layers[key], layers_rot[key])
        """
        X = feat_clean.flatten(start_dim=1).detach()
        Y = feat_rot.flatten(start_dim=1).detach()
        
        # 1. Compute batch-specific sigma using the pooled median trick
        pooled = torch.cat([X, Y], dim=0)
        pairwise_dists = torch.cdist(pooled, pooled, p=2)
        
        triu_idx = torch.triu_indices(row=pairwise_dists.size(0), col=pairwise_dists.size(1), offset=1, device=self.device)
        valid_dists = pairwise_dists[triu_idx[0], triu_idx[1]]
        
        sigma = torch.median(valid_dists).item()
        if sigma == 0:
            sigma = 1e-6
        self.last_sigma = sigma

        # 2. Compute the RBF Kernels for this batch
        dist_X = torch.cdist(X, X, p=2)**2
        dist_Y = torch.cdist(Y, Y, p=2)**2
        
        K = torch.exp(-dist_X / (2 * (sigma ** 2)))
        L = torch.exp(-dist_Y / (2 * (sigma ** 2)))
        
        # 3. Compute unbiased components
        num = self._unbiased_hsic(K.clone(), L.clone())
        den_X = self._unbiased_hsic(K.clone(), K.clone())
        den_Y = self._unbiased_hsic(L.clone(), L.clone())
        
        # 4. Accumulate scalar sums (Uses zero RAM)
        self.hsic_num += num.item()
        self.hsic_den_X += den_X.item()
        self.hsic_den_Y += den_Y.item()
        self.total_batches += 1

    def compute_stats(self):
        """
        Resolves the global unbiased CKA score at the end of the epoch loop.
        Matches signature: tracker.compute_stats() -> returns dictionary
        """
        if self.total_batches == 0 or self.hsic_den_X <= 0 or self.hsic_den_Y <= 0:
            # Safely handle Angle 0 or empty validation checks
            return {"rbf_cka": 1.0, "calibrated_sigma": self.last_sigma}
            
        denominator = torch.sqrt(torch.tensor(self.hsic_den_X * self.hsic_den_Y))
        cka_score = self.hsic_num / denominator.item()
        
        # Clamp bounds to handle minor floating-point fluctuations near perfect alignment
        cka_score = max(0.0, min(1.0, cka_score))
        
        return {
            "rbf_cka": cka_score,
            "calibrated_sigma": self.last_sigma
        }
