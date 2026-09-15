# An Information-Theoretic Approach to Learned Equivariance

CS7840 final project. Two questions, pursued in two threads that share one codebase:

1. **Does rotation augmentation actually buy equivariance, and what does it cost?**
   A 66-config training sweep over six tasks, measuring layer-by-layer CKA between
   features of clean and rotated inputs at every epoch. → [`src/figures/`](src/figures/)
2. **Does the metric you measure equivariance *with* actually measure equivariance?**
   A set of adversarial demos scored against models whose equivariance is known
   exactly, rather than estimated. → [`src/metric_demos/`](src/metric_demos/),
   [`src/escnn_experiments/`](src/escnn_experiments/)

The second thread is where the title comes from. Every geometric metric in common use
— Local Equivariance Error under an assumed ρ(g), fitted linear maps, Procrustes,
linear and RBF CKA, SVCCA, distance correlation, RSA — turns out to be a statement
about the coordinate system a representation happens to be written in. Mutual
information is a statement about the representation itself, and that difference is
measurable: on a discrete space where MI can be computed rather than estimated, a
bijective relabeling of the readout takes linear CKA from 1.0000 to 0.0376 while MI
does not move by a single bit.

## Layout

| folder | what is in it |
|---|---|
| [`src/`](src/) | Everything. Start with its README for the working-directory contract. |
| [`src/figures/`](src/figures/) | The sweep's figure pipeline: 660 result JSONs → 9 claim-driven figures. |
| [`src/metric_demos/`](src/metric_demos/) | Metric-critique demos that need no escnn: exact-MI on an exhaustive discrete space, plus the two Lie-derivative demos that use pretrained checkpoints. |
| [`src/escnn_experiments/`](src/escnn_experiments/) | The half of that thread needing an exactly-equivariant escnn model as ground truth. Has its own venv — see below. |
| [`src/Models/`](src/Models/) | Network architectures, one module per task. |
| [`src/print_digit/`](src/print_digit/) | The printed-digit (`mnist_font`) dataset and its loader. |
| [`slurm/`](slurm/) | Cluster launchers for the sweep. |

Three folders under `src/` hold generated output rather than code and are documented in
their parents' READMEs: `results/` (660 per-seed JSONs, tracked), `models/` (`.pth`
checkpoints, gitignored), `data/` (downloaded datasets, gitignored).

## Environments

Two, deliberately:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # numpy 2.2.6
cd src/escnn_experiments && ./setup_venv.sh                         # numpy 1.26.4
```

`escnn` needs `lie_learn`, a Cython extension built against the numpy 1.x C ABI, so it
cannot run under the root pin. The sub-venv shares torch with the root one instead of
duplicating 6.3 GB of CUDA wheels; [`src/escnn_experiments/README.md`](src/escnn_experiments/README.md)
explains the mechanism. Everything outside that folder uses the root venv.

## Running things

```bash
cd src && python classification.py --model cnn --dataset cifar --rotation --seed 0
./slurm/launch_full_sweep.sh --dry-run        # from the repo root
cd src/metric_demos && ../../.venv/bin/python3 exact_mi_demo.py
cd src/escnn_experiments && .venv/bin/python3 equivariance_metrics_comparison.py
```

Training scripts must be run from `src/`: they write to relative `results/` and
`models/` paths. The demo scripts resolve everything from `__file__` and run from
anywhere.

## Reading the results

Two caveats change headline numbers, and both are stated wherever the numbers are:

- **Isotropic configs are degenerate.** Isotropic data is rotation-symmetric by
  construction, so a rotation-equivariance error measured on it does not mean what it
  means elsewhere. Four of the 33 pairs are affected; every figure script takes
  `--no-isotropic`. Excluding them moves the grid-task correlation between ΔCKA and
  upright-task cost from r = −0.04 to r = +0.48.
- **Which test loss you average over flips the conclusion.** Over all probe angles,
  augmentation helps in 33/33 configs — near-tautological, since the baseline is scored
  on rotations it never trained on. On the upright test set it *hurts* in 31/33.

See [`src/figures/README.md`](src/figures/README.md) for both in full.
