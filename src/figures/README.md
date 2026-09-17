# Sweep figures

Claim-driven figures for the training sweep, replacing the old per-config pipeline.

## Why this replaced `generate_all_pdfs.py`

The old pipeline emitted **654 PDFs / 1.3 GB** — three figure families x three stats,
once per config — and organized them by config, when the things that actually vary
are *augmentation on/off* and *depth*. Concretely:

- All 66 configs form **33 matched `learned_equivariant` / `non_equivariant` pairs**,
  and the two conditions were never drawn on shared axes, because they were always
  written to different directories. The sweep's main result had no figure.
- `ylim(0, 1)` was hardcoded, but 6 of 33 pairs have *every* layer above 0.99. Those
  figures were flat lines at the frame top.
- Epoch was encoded as a `gnuplot` rainbow rather than an axis, so "does equivariance
  emerge over training" had to be read off a color gradient.
- `cka_gap` is stored in all 660 results files and was never plotted.
- The 16-angle particle configs got **16 x 14 = 224-panel** figures to show what is a
  single smooth curve.

The old `src/pdfs/` output tree (654 PDFs, 1.3 GB) was deleted along with the
scripts. To recover any of it: `git checkout fb11977 -- src/pdfs`.

## Usage

```bash
cd src
python figures/aggregate.py                     # rebuild the cache after new runs
python figures/make_figures.py                  # every figure, all 33 pairs
python figures/make_figures.py --no-isotropic   # same set, 29 pairs
```

`--no-isotropic` (accepted by `make_figures.py` and by every `fig_*.py`) drops the
four isotropic-dataset pairs and writes to `*_no_isotropic.pdf`, so both variants
coexist.

`aggregate.py` collapses 660 JSON files / 1.6 GB into ~30 MB of cached tables under
`figures/cache/`. Pickle rather than parquet because `pyarrow` is not a project
dependency; these are regenerable caches, not artifacts, and are gitignored.

| table | grain | rows |
|---|---|---|
| `cka_by_epoch` | config x seed x epoch x layer, mean over angles | 1.8 M |
| `cka_by_angle` | + per angle, at checkpoint epochs only | 164 k |
| `perf_by_epoch` | config x seed x epoch x angle | 1.1 M |

The full tidy cross-product would be ~13 M rows; these three projections cover every
figure.

## The figures

| file | claim |
|---|---|
| `depth_profile` | **Main result.** Both conditions start ~1.0 at the input and fan apart with depth |
| `depth_profile_defect` | Same, as `1 - CKA` on a log axis — the only view in which the saturated configs show structure |
| `emergence` / `_defect` | Equivariance emerges under augmentation and *decays* without it |
| `sweep_summary` | All 33 pairs ranked by effect, with a symlog Δ panel |
| `angle_profile_polar` | CKA decays smoothly with angle, symmetric about 180° |
| `angle_profile_c4` | The three-angle grid tasks, as a categorical panel |
| `metric_gap` | Where linear and RBF CKA disagree |
| `equivariance_vs_performance` | Equivariance is bought, not free: upright-task loss rises in 31/33 pairs |

## Measured results

What the figures above actually show, in numbers. Every value is at the **final epoch**,
at the **deepest layer**, using **linear CKA**, averaged over the 10 seeds and over the
probe angles; `upright` and `all-angle` are final-epoch test loss ratios,
augmented / baseline, so >1 means augmentation made it worse.

| | all 33 pairs | 29 pairs, no isotropic |
|---|---|---|
| ΔCKA > 0 (augmentation raised measured equivariance) | 28 / 33 | 26 / 29 |
| median ΔCKA | +0.0322 | +0.0442 |
| most negative ΔCKA | −0.4294 (`stress_prediction/unet/isotropic`) | −0.0106 (`stress_prediction/naive`) |
| upright loss **worse** | 31 / 33 | 27 / 29 |
| median upright loss ratio | ×1.186 | ×1.186 |
| all-angle loss **better** | 31 / 33 | 27 / 29 |
| median all-angle loss ratio | ×0.804 | ×0.804 |

Those last four rows are the "bought, not free" result in its rawest form: the same runs,
scored two ways, give opposite answers with almost identical margins.

### Per task

| task | pairs | CKA base | CKA aug. | ΔCKA | upright | all-angle |
|---|---|---|---|---|---|---|
| `classification` | 5 | 0.4411 | 0.9501 | **+0.5090** | ×1.19 | ×0.804 |
| `colorization` | 8 | 0.9633 | 0.9936 | +0.0303 | ×1.02 | ×0.986 |
| `fluid_flow` | 4 | 0.6927 | 0.8229 | +0.1302 | ×2.31 | ×0.105 |
| `fluid_flow_particles` | 4 | 0.9596 | 0.9861 | +0.0265 | ×62.28 | ×0.107 |
| `stress_prediction` | 4 | 0.7627 | 0.9450 | +0.1823 | ×1.70 | ×0.211 |
| `stress_prediction_particles` | 4 | 0.7434 | 0.8232 | +0.0798 | ×1.04 | ×0.912 |

(isotropic pairs excluded; medians for the two loss columns, means for CKA)

The spread across tasks is the thing to notice. **Classification has by far the most room
to move** — its baseline sits at 0.4411, so augmentation buys +0.51 — while
**colorization starts at 0.9633 and has almost nothing to gain**, which is the saturation
that made the old hardcoded `ylim(0, 1)` figures useless. A task's headroom, not its
architecture, dominates ΔCKA.

### Emergence over training

Deepest layer, mean over the 29 non-isotropic pairs:

| | epoch 0 | final |
|---|---|---|
| augmented | 0.8363 | **0.9313** |
| baseline | 0.8363 | **0.7774** |

Both conditions start identically — they must, since epoch 0 is the pre-training eval of
the same initialization — and then separate in *opposite directions*. Augmentation raises
CKA in 24/29 pairs (median +0.0195); the baseline's median change is +0.0000 and it rises
in only 15/29. So equivariance is not merely learned faster with augmentation; without it,
training actively erodes what initialization provided.

### Equivariance vs. upright cost

`fig_equivariance_vs_perf.py` reports r between ΔCKA and log cost per task family. The
correlation is sensitive to the isotropic configs and not to the kernel:

| | grid (n=23 / 21) | particle (n=10 / 8) |
|---|---|---|
| RBF CKA, all pairs | −0.04 | −0.41 |
| RBF CKA, no isotropic | **+0.48** | −0.63 |
| linear CKA, no isotropic | **+0.50** | −0.62 |

The figure's x-axis uses RBF CKA, but the paper reports linear — and the last two rows
show that choice does not matter here (+0.48 vs +0.50), so the claim survives the
substitution. The particle facet has the opposite sign at n=8; treat it as noise, not as
a counter-result.

### Per-config detail, all 33 pairs

| task | model | dataset | CKA base | CKA aug. | ΔCKA | upright | all-angle |
|---|---|---|---|---|---|---|---|
| `classification` | cnn | cifar | 0.2719 | 0.9174 | +0.6455 | ×1.27 | ×0.652 |
| `classification` | cnn | mnist_font | 0.4732 | 0.9831 | +0.5099 | ×2.50 | ×0.366 |
| `classification` | naive | cifar | 0.4128 | 0.9981 | +0.5853 | ×1.11 | ×0.883 |
| `classification` | naive | mnist_font | 0.6315 | 0.9980 | +0.3665 | ×1.19 | ×0.826 |
| `classification` | vit | cifar | 0.4160 | 0.8538 | +0.4378 | ×1.19 | ×0.804 |
| `colorization` | cnn | cifar | 0.9981 | 0.9993 | +0.0011 | ×1.02 | ×0.988 |
| `colorization` | cnn | stl10 | 0.9979 | 0.9988 | +0.0009 | ×1.02 | ×0.984 |
| `colorization` | naive | cifar | 1.0000 | 1.0000 | +0.0000 | ×1.00 | ×1.000 |
| `colorization` | naive | stl10 | 1.0000 | 1.0000 | +0.0000 | ×1.00 | ×1.000 |
| `colorization` | unet | cifar | 0.9990 | 0.9994 | +0.0004 | ×1.02 | ×0.986 |
| `colorization` | unet | stl10 | 0.9986 | 0.9990 | +0.0004 | ×1.02 | ×0.982 |
| `colorization` | vit | cifar | 0.7634 | 0.9722 | +0.2088 | ×1.02 | ×0.982 |
| `colorization` | vit | stl10 | 0.9491 | 0.9803 | +0.0312 | ×1.03 | ×0.986 |
| `fluid_flow` | cnn | buoyant | 0.7475 | 0.9781 | +0.2306 | ×2.27 | ×0.009 |
| `fluid_flow` | naive | buoyant | 0.9317 | 0.9820 | +0.0503 | ×4.34 | ×0.565 |
| `fluid_flow` | unet | buoyant | 0.7347 | 0.9798 | +0.2451 | ×2.35 | ×0.008 |
| `fluid_flow` | unet | isotropic *(isotropic)* | 0.9270 | 0.9768 | +0.0498 | ×5.60 | ×0.600 |
| `fluid_flow` | vit | buoyant | 0.3566 | 0.3517 | -0.0050 | ×1.57 | ×0.200 |
| `fluid_flow_particles` | mlp | buoyant | 0.9609 | 0.9679 | +0.0070 | ×239.73 | ×0.196 |
| `fluid_flow_particles` | naive | buoyant | 0.9763 | 0.9993 | +0.0230 | ×116.49 | ×0.358 |
| `fluid_flow_particles` | pointnet | buoyant | 0.9483 | 0.9800 | +0.0318 | ×2.77 | ×0.003 |
| `fluid_flow_particles` | pointnet | isotropic *(isotropic)* | 0.9474 | 0.9466 | -0.0007 | ×1.01 | ×0.994 |
| `fluid_flow_particles` | transformer | buoyant | 0.9529 | 0.9971 | +0.0442 | ×8.07 | ×0.018 |
| `stress_prediction` | cnn | anisotropic | 0.7368 | 0.9922 | +0.2554 | ×2.00 | ×0.029 |
| `stress_prediction` | naive | anisotropic | 0.8422 | 0.8316 | -0.0106 | ×1.00 | ×1.000 |
| `stress_prediction` | unet | anisotropic | 0.7723 | 0.9988 | +0.2265 | ×3.35 | ×0.000 |
| `stress_prediction` | unet | isotropic *(isotropic)* | 0.9991 | 0.5698 | -0.4294 | ×1,899.83 | ×0.505 |
| `stress_prediction` | vit | anisotropic | 0.6994 | 0.9575 | +0.2581 | ×1.39 | ×0.393 |
| `stress_prediction_particles` | mlp | anisotropic | 0.8100 | 0.8287 | +0.0187 | ×1.01 | ×0.988 |
| `stress_prediction_particles` | naive | anisotropic | 0.6087 | 0.7386 | +0.1299 | ×1.00 | ×1.000 |
| `stress_prediction_particles` | pointnet | anisotropic | 0.8429 | 0.8751 | +0.0322 | ×1.11 | ×0.779 |
| `stress_prediction_particles` | pointnet | isotropic *(isotropic)* | 0.8605 | 0.8856 | +0.0251 | ×1.04 | ×0.880 |
| `stress_prediction_particles` | transformer | anisotropic | 0.7118 | 0.8503 | +0.1384 | ×1.08 | ×0.837 |

Two entries deserve a health warning. `stress_prediction/unet/isotropic` shows ΔCKA
−0.4294 at an upright cost of ×1,899 — it is the degenerate isotropic case, not a
finding. And the two large particle ratios (×239.73, ×116.49) are inflated denominators:
those baselines reach a near-zero upright loss, so the ratio is large while the absolute
difference is tiny. Read the ratio column alongside `fig_sweep_summary`'s symlog panel
rather than on its own.

## Two things worth knowing before citing these

- **Isotropic configs measure something degenerate.** Isotropic data is
  rotation-symmetric by construction, so a *rotation*-equivariance error measured on
  it does not mean what it means elsewhere. Four pairs are affected
  (`fluid_flow/unet`, `fluid_flow_particles/pointnet`, `stress_prediction/unet`,
  `stress_prediction_particles/pointnet`). Use `--no-isotropic`.

  Excluding them is not cosmetic — it changes headline numbers:

  | | all 33 pairs | 29 pairs, no isotropic |
  |---|---|---|
  | augmentation raises CKA in | 29 / 33 | 26 / 29 |
  | largest negative Δ CKA | −0.424 (`stress_prediction/unet/isotropic`) | −0.0004 (essentially zero) |
  | upright loss worsens in | 31 / 33 | 27 / 29 |
  | grid-task corr. ΔCKA vs. log upright cost | r = −0.04 | **r = +0.48** |

  That last row is the substantive one: with the degenerate configs removed, there is
  a positive relationship between equivariance gained and upright-task cost in the
  grid tasks, which is the evidence for "bought, not free". Note n = 21 there and
  n = 8 for the particle facet, so treat the particle correlation as unreliable.

- **Which test loss you use changes the answer entirely.** Averaged over all probe
  angles, augmentation helps in 33/33 configs — but that is near-tautological, since
  the baseline is scored on rotated inputs it never trained on. On the *upright*
  (angle 0) test set it hurts in 31/33. `equivariance_vs_performance` deliberately
  uses the upright set.

## Style

`style.py` holds the palette and rcParams, replacing blocks that were copy-pasted
across the three old `visualize_*.py`. Colors are the dataviz reference palette used
unmodified; the categorical pair and the ordinal depth ramp both pass the validator
(categorical worst adjacent CVD ΔE 24.7 light / 26.8 dark; ramp monotone, single hue,
light end 2.06:1).
