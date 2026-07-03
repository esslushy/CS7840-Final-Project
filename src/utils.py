import torch
class Random90Rotation:
    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=(-2, -1))

class EquivarianceTracker:

    def __init__(self, device=None, store_device="cpu", dtype=torch.float32,
                 sigma_x: float = None, sigma_y: float = None,
                 max_samples: int = None, n_shuffles: int = 200, seed: int = 0,
                 block_size: int = 1024):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.store_device = torch.device(store_device)
        self.dtype = dtype
        self.sigma_x = sigma_x            # fixed bandwidth if given; else per-call median
        self.sigma_y = sigma_y
        self.max_samples = max_samples    # None (default) -> use ALL buffered samples
        self.n_shuffles = n_shuffles
        self.seed = seed
        self.block_size = block_size      # tiled compute; memory O(block_size * N)
        self.reset()

    def reset(self):
        self.all_x = []
        self.all_y = []
        self.total_samples = 0

    @torch.no_grad()
    def update(self, X: torch.Tensor, Y: torch.Tensor):
        X_flat = X.detach().flatten(start_dim=1).to(self.store_device, self.dtype)
        Y_flat = Y.detach().flatten(start_dim=1).to(self.store_device, self.dtype)
        self.all_x.append(X_flat)
        self.all_y.append(Y_flat)
        self.total_samples += X_flat.size(0)

    def _gram_norm(self, full, sigma):
        """Trace-normalized RBF Gram on self.device. Returns (A, sigma) with tr(A)=1."""
        d2 = torch.cdist(full, full, p=2) ** 2
        if sigma is None:
            n = full.size(0)
            mask = ~torch.eye(n, dtype=torch.bool, device=full.device)
            off = d2[mask]
            off = off[off > 0]
            sigma = torch.sqrt(0.5 * off.median() + 1e-8)
        else:
            sigma = torch.as_tensor(sigma, device=full.device, dtype=full.dtype)
        K = torch.exp(-d2 / (2.0 * sigma ** 2))
        return K / torch.trace(K), sigma               # normalize by ACTUAL trace

    @staticmethod
    def _S2(A):
        return -torch.log2((A * A).sum() + 1e-12)

    def _mi(self, A_x, A_y):
        Had = A_x * A_y
        A_xy = Had / torch.trace(Had)                   # Hadamard joint, diagonal-sum norm
        return self._S2(A_x) + self._S2(A_y) - self._S2(A_xy)

    def _resolve_sigma(self, data, sigma):
        """Bandwidth for the tiled path: use fixed if given, else median from a
        capped random subsample (full pairwise median would need an N×N matrix)."""
        if sigma is not None:
            return torch.as_tensor(sigma, device=data.device, dtype=data.dtype)
        m = min(2048, data.size(0))
        g = torch.Generator(device="cpu").manual_seed(self.seed + 7)
        idx = torch.randperm(data.size(0), generator=g)[:m].to(data.device)
        sub = data[idx]
        d2 = torch.cdist(sub, sub, p=2) ** 2
        off = d2[~torch.eye(m, dtype=torch.bool, device=data.device)]
        return torch.sqrt(0.5 * off[off > 0].median() + 1e-8)

    def _tiled_sums(self, X, Y, sigma_x, sigma_y):
        """Accumulate Sx=||Kx||_F^2, Sy=||Ky||_F^2, Sxy=sum Kx^2 Ky^2 over row
        blocks, materializing only block×N at a time. K_ii=1 => trace(K)=N."""
        N = X.size(0)
        block = self.block_size or N
        Sx = Sy = Sxy = torch.zeros((), device=self.device, dtype=torch.float64)
        for s in range(0, N, block):
            xi, yi = X[s:s + block], Y[s:s + block]
            Kx = torch.exp(-torch.cdist(xi, X, p=2) ** 2 / (2.0 * sigma_x ** 2))
            Ky = torch.exp(-torch.cdist(yi, Y, p=2) ** 2 / (2.0 * sigma_y ** 2))
            kx2, ky2 = (Kx * Kx).double(), (Ky * Ky).double()
            Sx = Sx + kx2.sum()
            Sy = Sy + ky2.sum()
            Sxy = Sxy + (kx2 * ky2).sum()
        return Sx, Sy, Sxy

    def _tiled_mi_parts(self, X, Y, sigma_x, sigma_y):
        """Returns (MI, H_X, H_Y) via tiled sums.  MI = log2(Sxy N^2/(Sx Sy))."""
        N = X.size(0)
        Sx, Sy, Sxy = self._tiled_sums(X, Y, sigma_x, sigma_y)
        log2N = torch.log2(torch.tensor(float(N), dtype=torch.float64, device=self.device))
        H_X = -torch.log2(Sx + 1e-12) + 2 * log2N
        H_Y = -torch.log2(Sy + 1e-12) + 2 * log2N
        MI = torch.log2(Sxy + 1e-12) + 2 * log2N - torch.log2(Sx + 1e-12) - torch.log2(Sy + 1e-12)
        return MI, H_X, H_Y

    @torch.no_grad()
    def compute_stats(self, reset_after: bool = False):
        """Noise-floor statistics for the raw matrix-based Rényi-2 MI.

        Returns a dict:
          raw_mi        : observed I(X;Y) in bits
          floor_mean    : mean MI over shuffled (independent) pairs  (bias floor)
          floor_std     : std of the shuffle distribution           (noise scale)
          debiased_mi   : raw_mi - floor_mean  (bits above the floor)
          z             : (raw_mi - floor_mean) / floor_std
                          -> signal in units of noise std; comparable across
                             layers WITHOUT entropy normalization, since it is
                             scaled by each layer's own noise.
          p             : permutation p-value = (1 + #{shuffle >= raw}) / (S+1)
                          -> smallest resolvable p is 1/(n_shuffles+1); raise
                             n_shuffles for finer resolution / tighter floor_std.
          H_X, H_Y      : marginal Rényi-2 entropies (for reference)
        """
        if self.total_samples == 0:
            raise ValueError("No data accumulated yet. Call update() first.")

        X = torch.cat(self.all_x, dim=0).to(self.device)
        Y = torch.cat(self.all_y, dim=0).to(self.device)
        n = X.size(0)
        if self.max_samples is not None and n > self.max_samples:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            idx = torch.randperm(n, generator=g)[:self.max_samples].to(self.device)
            X, Y = X[idx], Y[idx]
            n = self.max_samples

        if self.block_size is not None:
            # tiled path: no N×N Gram in memory; permuting Y's rows == Gram(Y[perm])
            sx = self._resolve_sigma(X, self.sigma_x)
            sy = self._resolve_sigma(Y, self.sigma_y)
            raw_t, H_X, H_Y = self._tiled_mi_parts(X, Y, sx, sy)
            raw = raw_t
            g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
            null = torch.empty(self.n_shuffles, device=self.device, dtype=torch.float64)
            for s in range(self.n_shuffles):
                perm = torch.randperm(n, generator=g).to(self.device)
                mi_s, _, _ = self._tiled_mi_parts(X, Y[perm], sx, sy)
                null[s] = mi_s
            raw = raw.to(torch.float64)
        else:
            A_x, _ = self._gram_norm(X, self.sigma_x)
            A_y, _ = self._gram_norm(Y, self.sigma_y)
            H_X, H_Y = self._S2(A_x), self._S2(A_y)
            raw = self._mi(A_x, A_y)
            g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
            null = torch.empty(self.n_shuffles, device=self.device)
            for s in range(self.n_shuffles):
                perm = torch.randperm(n, generator=g).to(self.device)
                null[s] = self._mi(A_x, A_y[perm][:, perm])

        floor_mean = null.mean()
        floor_std = null.std(unbiased=True)
        z = (raw - floor_mean) / (floor_std + 1e-12)
        p = (1.0 + (null >= raw).sum().float()) / (self.n_shuffles + 1)

        out = {
            "raw_mi": raw.item(),
            "floor_mean": floor_mean.item(),
            "floor_std": floor_std.item(),
            "debiased_mi": (raw - floor_mean).item(),
            "z": z.item(),
            "p": p.item(),
            "H_X": H_X.item(),
            "H_Y": H_Y.item(),
            "n_samples": n,
        }
        if reset_after:
            self.reset()
        return out