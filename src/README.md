# src

Two research threads share this folder. They are unrelated in method and answer
different questions, so it is worth knowing which one you are looking at.

**Thread 1 — the training sweep.** Do networks trained with rotation augmentation
become equivariant, and what does it cost? Nine task scripts live here (six of them in
the sweep proper), each training a model on clean data while measuring, every epoch and
every layer, the CKA between features of clean and rotated inputs. Output lands in
`results/` and is turned into per-config posters by [`figures/`](figures/).

**Thread 2 — the metric critique.** Is the *metric* trustworthy? These demos construct
models whose equivariance is known exactly and then check whether standard metrics
report it. They are self-contained, fast, and print ASCII tables rather than writing
artifacts. See [`metric_demos/`](metric_demos/) and
[`escnn_experiments/`](escnn_experiments/).

## Working-directory contract

**Training scripts must be run from `src/`.** They are not path-independent:

- `utils.save_all()` writes to relative `results/` and `models/`
- `print_digit.load_mnist_font_dataset()` reads relative `print_digit/dataset/assets/`
- datasets download to relative `./data`

```bash
cd src
python classification.py --model cnn --dataset cifar --rotation --seed 0
```

Everything in `metric_demos/`, `escnn_experiments/` and `figures/` resolves paths from
`__file__` instead and runs from anywhere.

## The task scripts

All nine share a CLI (`--model`, `--dataset`, `--rotation`, `--seed`, plus
`--thicker` / `--finetune` / `--resume` / `--holdout` where applicable) and a common
shape: build the task, train, and on every epoch record task performance at each probe
angle alongside per-layer CKA. `--rotation` is the whole independent variable — on, it
applies `Random90Rotation` to the training data and the run is tagged
`learned_equivariant`; off, `non_equivariant`.

| script | task | models | datasets | in sweep |
|---|---|---|---|---|
| `classification.py` | 10-class labels | cnn, naive, vit | cifar, mnist_font | yes |
| `colorization.py` | grayscale → RGB | cnn, unet, naive, vit | cifar, stl10 | yes |
| `fluid_flow.py` | next-step grid advection | cnn, unet, naive, vit | isotropic, buoyant | yes |
| `fluid_flow_particles.py` | next-step particle state | pointnet, mlp, naive, transformer | isotropic, buoyant | yes |
| `stress_prediction.py` | Airy → stress/force field | cnn, unet, naive, vit | isotropic, anisotropic | yes |
| `stress_prediction_particles.py` | per-particle force | pointnet, mlp, naive, transformer | isotropic, anisotropic | yes |
| `gradient_field.py` | Sobel (dI/dx, dI/dy) field | cnn, unet, naive, vit | cifar, mnist | no |
| `image_flow.py` | wave-surface flow field | cnn, unet, naive, vit | isotropic, directional | no |
| `image_inversion.py` | pixel inversion | cnn, naive, vit | cifar, mnist | no |

The last three are not in the sweep and have no per-seed results. `gradient_field.py`
matters anyway: it is a genuine rotation-*covariant* vector-field task (rotating the
image rotates the gradient vectors), and its two checkpoints are what
`metric_demos/lie_vs_cka_vector_field_demo.py` uses to show a scalar-per-channel
equivariance assumption failing on a vector field.

### What the sweep found

Linear CKA at the deepest layer, final epoch, averaged over 10 seeds and probe angles;
loss columns are augmented / baseline ratios, so >1 means augmentation hurt. Isotropic
pairs excluded (see the caveats below).

| task | pairs | CKA base | CKA aug. | ΔCKA | upright loss | all-angle loss |
|---|---|---|---|---|---|---|
| `classification` | 5 | 0.4411 | 0.9501 | **+0.5090** | ×1.19 | ×0.80 |
| `colorization` | 8 | 0.9633 | 0.9936 | +0.0303 | ×1.02 | ×0.99 |
| `fluid_flow` | 4 | 0.6927 | 0.8229 | +0.1302 | ×2.31 | ×0.11 |
| `fluid_flow_particles` | 4 | 0.9596 | 0.9861 | +0.0265 | ×62.28 | ×0.11 |
| `stress_prediction` | 4 | 0.7627 | 0.9450 | +0.1823 | ×1.70 | ×0.21 |
| `stress_prediction_particles` | 4 | 0.7434 | 0.8232 | +0.0798 | ×1.04 | ×0.91 |

Across the 29 non-isotropic pairs: augmentation raises measured equivariance in 26/29
(median ΔCKA +0.0442), makes the **upright** task worse in 27/29 (median ×1.186), and
makes the **all-angle** task better in 27/29 (median ×0.804). The same runs, scored two
ways, give opposite answers — which is why the caveat below is not a footnote.

How much a task gains is mostly a matter of headroom: `classification` starts at 0.4411
and gains +0.51, while `colorization` starts at 0.9633 and has nothing left to gain.
Per-config numbers for all 33 pairs are in [`figures/README.md`](figures/README.md).

The grid tasks probe the three non-identity C4 rotations (90/180/270, exact
`torch.rot90` permutations, no interpolation). The two particle tasks rotate by an exact
2×2 matmul instead, so they probe 16 evenly spaced SO(2) angles.

## `utils.py`

Everything shared by the task scripts, and the only place CKA is implemented for the
sweep:

- **`EquivarianceTracker`** — accumulates CKA between clean and rotated features across
  batches using the **unbiased HSIC** estimator (diagonal zeroed), and reports
  `linear_cka`, `rbf_cka`, `cka_gap` (linear − RBF) and `calibrated_sigma`. The RBF
  bandwidth is the median pairwise distance of the pooled batch, recomputed per batch.
  Prefer this over the short `linear_cka` helpers in the demo scripts, which are the
  biased variant.
- **`Random90Rotation`** — the augmentation. A uniform random `k·90°` per sample.
- **`save_all`** — atomic write-temp-then-rename of stats and weights, every epoch, so
  an interrupted run cannot leave a corrupt file. This is what makes `--resume` work.
- **`set_seed`**, **`so2_eval_angles`**, **`rotate_2d`** — seeding of all four RNGs;
  n evenly spaced SO(2) elements excluding the identity, returned with integer-degree
  labels for use as JSON keys; and 2D vector rotation.

A note on the RBF bandwidth: `unlearnable_rho_demo.py` documents a trap where a
too-small bandwidth makes RBF CKA score a spurious 1.0000 — and its shuffled control
too. The median heuristic used here is safely away from that regime, but it is also why
the final paper reports **linear** CKA and keeps RBF only as a robustness check.

## Generated output

Three folders here hold output, not code.

**`results/`** — 660 JSONs, `<task>_<regime>_<model>_dataset_<dataset>_seed_<n>_statistics.json`,
one per config per seed (66 configs = 33 matched `learned_equivariant` /
`non_equivariant` pairs × 10 seeds). Included in the repository; they are the sweep's actual data.
Each file is a dict of parallel lists, index = epoch, length `NUM_EPOCHS + 1` (index 0
is the pre-training eval):

```
equivariant_loss[epoch][layer][angle] -> {rbf_cka, linear_cka, cka_gap, calibrated_sigma}
train_loss[epoch]                     -> float
train_accuracy[epoch]                 -> float      # classification only
test_loss[epoch][angle]               -> float      # angle "0" is the upright set
test_accuracy[epoch][angle]           -> float      # classification only
```

Angles are string keys (`"0"`, `"90"`, …). The regression tasks record only
`equivariant_loss`, `train_loss` and `test_loss` — no accuracy keys at all.
Despite the name, `equivariant_loss` holds CKA *similarity*: higher is more equivariant.

**`models/`** — `<tag>_model.pth` state dicts, last epoch only, written every epoch by
`save_all`. Created at runtime and not included, so a fresh copy has none; the two
`lie_vs_cka*` demos need the `classification_*_cnn_dataset_cifar` and
`gradient_field_*_unet_dataset_mnist` checkpoints and will fail without them.

**`data/`** — torchvision download target (MNIST, CIFAR-10, STL-10, CelebA).
Created at runtime and not included, shared by every thread including
`escnn_experiments/`, which resolves it from `__file__`.

## Two caveats before citing any sweep number

- **Isotropic configs measure something degenerate.** Isotropic data is
  rotation-symmetric by construction, so rotation-equivariance error on it does not mean
  what it means elsewhere. Four pairs are affected (`fluid_flow/unet`,
  `fluid_flow_particles/pointnet`, `stress_prediction/unet`,
  `stress_prediction_particles/pointnet`). `figures/fig_posters.py` takes
  `--no-isotropic`; excluding them moves the grid-task correlation between ΔCKA and
  upright-task cost from r = −0.04 to r = +0.48.
- **Which test loss you average over flips the headline.** Over all probe angles,
  augmentation helps in 33/33 configs — but the baseline is being scored on rotations it
  never saw, so that is close to tautological. On the upright (`"0"`) set it hurts in
  31/33, median +18.6% relative loss.

[`figures/README.md`](figures/README.md) carries both in full, with the numbers that
change.
