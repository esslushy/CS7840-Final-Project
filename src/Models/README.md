# Models — network architectures

One module per task, each exporting the same family of architectures so that
"equivariance emerges under augmentation" can be compared across inductive biases
rather than across implementations.

**Not to be confused with `../models/`** (lowercase), which holds trained `.pth`
checkpoints and is created at runtime. The two differ only in case, so they collide on
case-insensitive filesystems (macOS, Windows) — if you unpack there, expect trouble.

## The shared architecture family

Grid tasks export four; particle tasks export the same four ideas over sets:

| grid | particle | idea |
|---|---|---|
| `NaiveNet` | `NaiveNet` | Flatten and use an MLP. No spatial structure at all — the floor. |
| `CNN` | `MLP` | Local weight sharing, the standard baseline. |
| `UNet` | `PointNet` | Multi-scale / permutation-invariant pooling, the strongest task model. |
| `ViT` | `SetTransformer` | Attention, no built-in locality. Built from the `FeedForward` / `Attention` / `Transformer` blocks in the same file. |

None of these is architecturally equivariant. That is the point: the sweep asks whether
augmentation *teaches* equivariance, so escnn's steerable layers are deliberately absent
here and confined to [`../escnn_experiments/`](../escnn_experiments/), where an exact
guarantee is needed as ground truth instead.

`--thicker` on the training scripts widens the channel dimension; `NaiveNet` rejects it.

## Files

| file | used by | exports |
|---|---|---|
| `Classification/` | `classification.py` | See [its README](Classification/README.md) — split by input size. |
| `ColorizationNets.py` | `colorization.py` | NaiveNet, CNN, UNet, ViT |
| `FluidFlowNets.py` | `fluid_flow.py` | NaiveNet, CNN, UNet, ViT |
| `StressPredictionNets.py` | `stress_prediction.py` | NaiveNet, CNN, UNet, ViT |
| `GradientFieldNets.py` | `gradient_field.py`, `../metric_demos/lie_vs_cka_vector_field_demo.py` | NaiveNet, CNN, UNet, ViT |
| `ImageFlowNets.py` | `image_flow.py` | NaiveNet, CNN, UNet, ViT |
| `FluidFlowParticlesNets.py` | `fluid_flow_particles.py` | NaiveNet, MLP, PointNet, SetTransformer |
| `StressPredictionParticlesNets.py` | `stress_prediction_particles.py` | NaiveNet, MLP, PointNet, SetTransformer |
| `InvertNets.py` | `image_inversion.py` | CNN only |

The five grid files are near-identical by design — same four classes, differing in input
and output channel counts and the head. They were copied rather than parameterized, so a
change to the shared `Transformer` block has to be made in each.

## Layer naming matters

`EquivarianceTracker` is keyed on layer name, so `results/` files hold one CKA series
per name (`conv1`, `relu1`, `pool1`, …, `fc1`, `softmax`) — and two downstream things
depend on that:

- `figures/aggregate.py` takes the layer list as `list(el[0].keys())`, i.e. **the order
  the training script recorded them in**, and turns the position into
  `depth_frac = i / (n_layers − 1)`. That shared 0→1 axis is what lets architectures with
  2 and 30 layers appear on one plot. So reordering the `tracker.update()` calls silently
  rescales the depth axis, and adding a layer shifts every existing one.
- `../metric_demos/lie_vs_cka_demo.py` hardcodes the CIFAR CNN's layer names.

Renaming a layer orphans its series in results already on disk, since old and new files
no longer share a key.
