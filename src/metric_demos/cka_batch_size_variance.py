"""
Does it matter that minibatch CKA is non-deterministic?

The training sweep never computes CKA on a whole dataset at once. It feeds
EquivarianceTracker (src/utils.py) one minibatch at a time, accumulating the
unbiased-HSIC numerator and denominators across batches and resolving
num / sqrt(den_X * den_Y) at the end. The result depends on how the samples
happen to be grouped, so two shuffles of the same data give two different
numbers. This script measures how much they differ, for every power-of-2 batch
size from 4 (the smallest the unbiased HSIC allows) up to the full sample.

No model is involved. X is Gaussian and Y = f(X) for a few fixed synthetic
functions chosen to cover the range of CKA values. For each batch size b, the
same N pairs are reshuffled --trials times and split into N/b minibatches.
Per function and kernel it reports:

  per-batch sd    spread of CKA computed from one minibatch alone
  accum sd        spread, across shuffles, of the tracker's final value after a
                  full pass -- the number training actually records
  max |dev|       worst observed |accumulated - reference| over all shuffles

The reference is b = N: a single batch, hence deterministic.

Run (from anywhere):
    python cka_batch_size_variance.py
    python cka_batch_size_variance.py --n-samples 8192 --trials 100 --tol 1e-3
"""
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from utils import EquivarianceTracker, set_seed  # noqa: E402

MIN_BATCH = 4  # unbiased HSIC needs n >= 4
KERNELS = ("linear", "rbf")


def make_functions(dim, gen):
    """Fixed synthetic maps X -> Y, spanning high to near-zero similarity."""
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
    W = torch.randn(dim, dim, generator=gen) / math.sqrt(dim)
    noise = lambda X, s: s * torch.randn(X.shape, generator=gen)
    return {
        "orthogonal (CKA=1)": lambda X: X @ Q,
        "tanh(XW)":           lambda X: torch.tanh(X @ W),
        "tanh(XW) + noise":   lambda X: torch.tanh(X @ W) + noise(X, 1.0),
        "half-dims kept":     lambda X: torch.cat([X[:, :dim // 2],
                                                   noise(X[:, dim // 2:], 1.0)], 1),
        "independent":        lambda X: noise(X, 1.0),
    }


def run_pass(X, Y, batch_size, perm):
    """One pass through the data in the order given by perm, mirroring training.
    Returns (per-batch CKA per kernel, accumulated CKA per kernel)."""
    tracker = EquivarianceTracker("cpu")
    per_batch = {k: [] for k in KERNELS}
    for start in range(0, len(perm), batch_size):
        idx = perm[start:start + batch_size]
        prev = {"linear": (tracker.lin_num, tracker.lin_den_X, tracker.lin_den_Y),
                "rbf": (tracker.rbf_num, tracker.rbf_den_X, tracker.rbf_den_Y)}
        tracker.update(X[idx], Y[idx])
        now = {"linear": (tracker.lin_num, tracker.lin_den_X, tracker.lin_den_Y),
               "rbf": (tracker.rbf_num, tracker.rbf_den_X, tracker.rbf_den_Y)}
        for k in KERNELS:
            num, dx, dy = (a - b for a, b in zip(now[k], prev[k]))
            # unclamped: a single small batch can legitimately leave [0, 1]
            per_batch[k].append(num / math.sqrt(dx * dy) if dx > 0 and dy > 0 else float("nan"))
    stats = tracker.compute_stats()
    return per_batch, {"linear": stats["linear_cka"], "rbf": stats["rbf_cka"]}


def sd(xs):
    xs = torch.tensor([x for x in xs if not math.isnan(x)], dtype=torch.float64)
    return xs.std(unbiased=True).item() if len(xs) > 1 else 0.0


def main():
    p = ArgumentParser(description="Variance of minibatch CKA vs. batch size, on synthetic data.")
    p.add_argument("--n-samples", type=int, default=2048, help="rounded down to a power of 2")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--trials", type=int, default=20, help="reshuffles per batch size")
    p.add_argument("--tol", type=float, default=1e-2,
                   help="report the smallest batch size whose accum sd is below this")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    n = 2 ** int(math.log2(args.n_samples))
    batch_sizes = [2 ** e for e in range(int(math.log2(MIN_BATCH)), int(math.log2(n)) + 1)]

    X = torch.randn(n, args.dim, generator=gen)
    print(f"N = {n} samples, X ~ N(0, I_{args.dim}), {args.trials} shuffles per batch size\n")

    summary = []
    for name, f in make_functions(args.dim, gen).items():
        Y = f(X)
        print(f"=== Y = {name} ===")
        rows = {}
        for b in batch_sizes:
            trials = 1 if b == n else args.trials  # one full batch is deterministic
            pb = {k: [] for k in KERNELS}
            acc = {k: [] for k in KERNELS}
            for _ in range(trials):
                per_batch, accumulated = run_pass(X, Y, b, torch.randperm(n, generator=gen))
                for k in KERNELS:
                    pb[k].extend(per_batch[k])
                    acc[k].append(accumulated[k])
            rows[b] = (pb, acc)

        for k in KERNELS:
            ref = rows[n][1][k][0]
            print(f"  {k} CKA, reference (b = N): {ref:.5f}")
            print(f"  {'batch':>7}{'per-batch sd':>14}{'accum mean':>12}{'accum sd':>11}{'max |dev|':>11}")
            first_ok = None
            for b in batch_sizes:
                pb, acc = rows[b]
                a_sd = sd(acc[k])
                print(f"  {b:>7}{sd(pb[k]):>14.5f}{sum(acc[k]) / len(acc[k]):>12.5f}"
                      f"{a_sd:>11.5f}{max(abs(v - ref) for v in acc[k]):>11.5f}")
                if first_ok is None and a_sd < args.tol:
                    first_ok = b
            summary.append((name, k, ref, first_ok))
            print()

    print(f"=== smallest batch size with accumulated sd < {args.tol:g} ===")
    for name, k, ref, b in summary:
        print(f"  {name:<20}{k:<8}ref {ref:.4f}   b = {b}")


if __name__ == "__main__":
    main()
