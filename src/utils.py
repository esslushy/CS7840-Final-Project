import json
import os
import random
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


def cycle_pairs(feats, step):
    """
    Build the CYCLE pairing for one equivariance score.

    feats[j] holds one layer's activations for the batch rotated by the j-th
    group element (feats[0] = unrotated), for ALL N elements of a cyclic group
    C_N, rows in the same image order. Returns (left, right) where every
    orientation j is paired with orientation j + step (mod N):

        left  = [feats[0],    feats[1],      ..., feats[N-1]]
        right = [feats[step], feats[step+1], ..., feats[step-1]]

    Why not pair feats[0] with feats[step] (the "star" pairing)? That only
    checks that SOME map sends upright features to rotated ones, separately per
    angle, and never constrains that map on rotated inputs -- so CKA = 1 at every
    angle is consistent with unrelated maps that are not a group representation.
    Stacking every orientation on the left makes the evaluation set closed under
    the group, so a single score of 1 forces ONE map for this step from every
    orientation, and the maps compose into a homomorphism (for linear CKA: an
    orthogonal one). For C_N, step 1 (the generator) already implies the rest;
    the other steps are reported for consistency.

    This is the REFERENCE definition. The training scripts call update_cycle,
    which produces the same scores without materializing these pairs.
    """
    n = len(feats)
    left = torch.cat(feats, dim=0)
    right = torch.cat([feats[(j + step) % n] for j in range(n)], dim=0)
    return left, right


def rotate_2d(vecs, theta):
    """Rotate 2D vectors by angle theta (radians). vecs: (..., 2)."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    x, y = vecs[..., 0], vecs[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def set_seed(seed: int):
    """Seed every RNG a training run touches (python, numpy, torch CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Random90Rotation:
    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=(-2, -1))

def unbiased_hsic(K, L):
    """Unbiased HSIC estimator (Song et al. 2012), kernel-agnostic. Zeros the
    diagonal. trace(K @ L) is computed as sum(K * L^T), which is O(n^2) rather
    than O(n^3) -- it matters for the C17 particle cycle, where n = 17 * batch."""
    n = K.size(0)
    if n < 4:
        raise ValueError("Batch size must be >= 4 to compute unbiased HSIC.")

    K = K.clone()
    L = L.clone()
    K.fill_diagonal_(0.0)
    L.fill_diagonal_(0.0)

    kl_trace = (K * L.t()).sum()
    k_sum = torch.sum(K)
    l_sum = torch.sum(L)

    row_sum_k = torch.sum(K, dim=1)
    row_sum_l = torch.sum(L, dim=1)
    row_vectors_dot = torch.dot(row_sum_k, row_sum_l)

    hsic = (kl_trace + (k_sum * l_sum) / ((n - 1) * (n - 2))
            - 2.0 * row_vectors_dot / (n - 2)) / (n * (n - 3))
    return hsic


def update_cycle(trackers, feats):
    """
    Feed every step of the cycle pairing into its tracker, from ONE Gram matrix.

    feats[j] is one layer's activations for the batch rotated by the j-th of N
    group elements (feats[0] unrotated, same image order in every entry);
    trackers[s - 1] receives step s, for s = 1 .. N-1. The scores are those of
    tracker.update(*cycle_pairs(feats, s)), computed without materializing the
    pairs:

      * The right-hand side of every step is a REORDERING of the same stacked
        rows, so all steps share one centered Gram G = F F^T over the N*B rows,
        and step s's right Gram is G with rows and columns permuted by
        r -> (r + s*B) mod N*B. RBF distances follow from G as well.
      * The denominators HSIC(K, K) and HSIC(L, L) are then equal, and the same
        for every step, so they are computed once.

    The only O(n^2 d) work is one fp32 matmul per layer per batch, instead of
    six Gram/distance computations per step on 2x the rows. Features are centered
    first (HSIC and pairwise distances are translation invariant) and everything
    after the matmul is float64, which removes the float32 cancellation that put
    exactly-equivariant inputs at 0.9998 instead of 1.

    RBF bandwidth: the median pairwise distance among the N*B stacked rows. The
    star pairing pooled the clean and rotated sets; under the cycle they are the
    same set, so pooling would only add zero distances between duplicates.
    """
    nb = feats[0].shape[0]
    F = torch.cat([f.detach().flatten(start_dim=1) for f in feats]).float()
    F = F - F.mean(dim=0, keepdim=True)
    K = (F @ F.t()).double()
    del F
    n = K.size(0)

    sq = K.diagonal()
    D2 = (sq[:, None] + sq[None, :] - 2.0 * K).clamp_min_(0.0)
    iu = torch.triu_indices(n, n, offset=1, device=K.device)
    sigma = torch.median(D2[iu[0], iu[1]].sqrt()).item() or 1e-6
    R = torch.exp(-D2 / (2.0 * sigma ** 2))
    del D2, iu

    lin_den = unbiased_hsic(K, K).item()
    rbf_den = unbiased_hsic(R, R).item()
    rows = torch.arange(n, device=K.device)
    for step, tracker in enumerate(trackers, start=1):
        p = (rows + step * nb) % n
        lin_num = unbiased_hsic(K, K[p][:, p]).item()
        rbf_num = unbiased_hsic(R, R[p][:, p]).item()
        tracker.accumulate(lin_num, lin_den, lin_den, rbf_num, rbf_den, rbf_den, sigma)


def loader_kwargs(device):
    """DataLoader settings for the cluster jobs. Uses the CPUs SLURM granted
    (train.sbatch asks for 8) instead of a fixed 2 workers; train and test
    loaders are both persistent, so each gets half. pin_memory speeds the
    host-to-GPU copy. None of this changes what is computed -- but the worker
    count does change which worker draws each sample's augmentation, so runs
    are not bit-identical to ones made with num_workers=2."""
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 2))
    workers = max(1, min(4, cpus // 2))
    return dict(num_workers=workers, pin_memory=str(device).startswith("cuda"),
                persistent_workers=True)


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
        return unbiased_hsic(K, L)

    def accumulate(self, lin_num, lin_den_X, lin_den_Y, rbf_num, rbf_den_X, rbf_den_Y, sigma):
        """Add one batch's HSIC terms (Python floats). update() and update_cycle()
        both end here, so the two paths resolve identically."""
        self.lin_num += lin_num
        self.lin_den_X += lin_den_X
        self.lin_den_Y += lin_den_Y
        self.rbf_num += rbf_num
        self.rbf_den_X += rbf_den_X
        self.rbf_den_Y += rbf_den_Y
        self.last_sigma = sigma
        self.total_batches += 1

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

        h = lambda A, B: unbiased_hsic(A, B).item()
        self.accumulate(h(K_lin, L_lin), h(K_lin, K_lin), h(L_lin, L_lin),
                        h(K_rbf, L_rbf), h(K_rbf, K_rbf), h(L_rbf, L_rbf), sigma)

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