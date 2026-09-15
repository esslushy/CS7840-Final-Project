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
