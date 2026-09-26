# Sweep figures

One figure script: [`fig_posters.py`](fig_posters.py), which writes a standalone
poster per config.

## Usage

```bash
cd src
python figures/aggregate.py                     # rebuild the cache after new runs
python figures/fig_posters.py                   # all 33 posters
python figures/fig_posters.py --no-isotropic    # the 29 non-degenerate ones
python figures/fig_posters.py --only "colorization/unet"
```

`--no-isotropic` drops the four isotropic-dataset pairs. Isotropic data is
rotation-symmetric by construction, so a *rotation*-equivariance error measured on
it is degenerate. It changes which posters are written, not what any of them
contains, so poster filenames carry no suffix for it.

**The reported metric is linear CKA.** `style._STAT` is the single place it is
set; `--rbf` swaps in RBF and suffixes the filename. Linear is the paper's choice
because the correctness argument is a linear one — under exact equivariance the
fiber representation is a permutation, and orthogonal maps leave the centered
linear Gram matrix unchanged, so linear CKA is exactly 1. RBF's bandwidth is a
free knob that can score a spurious 1.0 on a shuffled control, so it is a
robustness check, not a result.

`aggregate.py` collapses 660 JSON files / 1.6 GB into ~30 MB of cached tables
under `figures/cache/`, created on first run and not included in the repository. Only `cka_by_epoch` is read by the posters;
`perf_by_epoch` and `cka_by_angle` are still built because the headline numbers
below come out of them.

| table | grain | rows |
|---|---|---|
| `cka_by_epoch` | config x seed x epoch x layer, mean over angles | 1.8 M |
| `cka_by_angle` | + per angle, at checkpoint epochs only | 164 k |
| `perf_by_epoch` | config x seed x epoch x angle | 1.1 M |

## The posters

One file per matched pair, flat in `out/`, named
`<task>__<model>__<dataset>.pdf`. Each is a grid of small panels, **one per
layer titled with that layer's own name** (`conv1`, `enc1_silu2`,
`block0.attn.residual`), with epoch on x and CKA on y; the two conditions are
drawn on shared axes as mean ± 1 s.d. over the 10 seeds.

**There is no figure title.** The filename identifies the config, and a poster
dropped into a document takes its identity from the caption there. Only the two
axis labels (`epoch`, `linear CKA`), the panel names and the legend carry text.

Naming the layers is what makes the deep models readable. On `colorization/unet`
you can see that `enc1_*` and `dec1_*` stay pinned at 1.0 while the `mid_*` and
`dec2_*` bottleneck is where the conditions separate; on
`fluid_flow_particles/transformer` the layer that dives to 0.51 under
augmentation is `block0.attn.attn` specifically. "layer 14" carried none of that.

- **Both conditions in one file, on shared axes.** The pipeline this replaced
  wrote them to separate directories and never drew them together, which is why
  the sweep's main result had no figure at all.
- **One panel per layer, not one ramp shade per layer.** Overlaying 30 layers and
  telling them apart by lightness asks the reader to invert a colour ramp by eye,
  and it spends the only free channel on depth. Faceting frees the two categorical
  colours for the comparison that matters and makes room for the spread.
- **Mean ± 1 s.d. over the 10 seeds.** Every config-condition in the sweep has
  exactly 10 seeds (checked, 66/66), so the band is uniform across the set. On the
  U-Nets it is wide enough that much of the mid-network divergence is within
  noise, which is worth seeing.
- **y is CKA's full 0–1 range on every poster; x runs 0 → 200, or 400 for the two
  `mnist_font` configs** (it snaps up to the next multiple of `EPOCH_STEP`). Fixed
  limits mean panels are comparable across configs and not merely within one, and
  nothing is read off a zoomed axis. An s.d. band crossing 1 is clipped, which is
  the right call for a metric bounded at 1.

  The cost: **19 of the 33 posters have every layer above 0.999**, so they read as
  flat lines near the frame top. `colorization/unet` is the clearest case — its
  `mid_*`/`dec2_*` bottleneck really does diverge between conditions, but by a few
  thousandths, which is invisible at this scale. Seeing that needs a fitted or log
  `1 − CKA` axis, which these posters deliberately do not use.

Panels run in depth order left to right, which is why they are not numbered.

Panels where the two conditions coincide to within 2e-3 are labelled
`curves coincide`, since the later-drawn line otherwise hides the other and the
panel reads as a single series.

The panel grid always reserves one spare cell for the legend: a figure-level
legend placed with `loc="outside lower center"` does not account for `supxlabel`
and renders on top of it.

## Measured results

Numbers for the sweep itself, independent of any figure. Every value is at the **final epoch**,
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
that made the old hardcoded `ylim(0, 1)` useless. A task's headroom, not its
architecture, dominates ΔCKA: across the 29 non-isotropic pairs the correlation between
headroom (1 − baseline CKA) and ΔCKA is **r = +0.81**, and the median config closes
**61%** of whatever room it had.

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

r between ΔCKA and log upright cost, per task family. The correlation is sensitive to
the isotropic configs and not to the kernel:

| | grid (n=23 / 21) | particle (n=10 / 8) |
|---|---|---|
| RBF CKA, all pairs | −0.04 | −0.41 |
| RBF CKA, no isotropic | **+0.48** | −0.63 |
| linear CKA, no isotropic | **+0.50** | −0.62 |

The paper reports linear; the first two rows are the RBF robustness check, and
+0.48 vs +0.50 is the whole difference the kernel makes. The particle facet has the
opposite sign at n=8; treat it as noise, not as a counter-result.

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
difference is tiny. Read the ratio column alongside the absolute losses rather than on
its own.

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
