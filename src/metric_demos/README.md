# Metric-critique demos

Does the metric you measure learned equivariance *with* actually measure equivariance?

Every script here builds a case where the right answer is known in advance — by
construction, not by estimate — and then checks what standard metrics report. They are
the half of the critique thread that needs **no escnn**; the half that needs an
exactly-equivariant steerable model lives in
[`../escnn_experiments/`](../escnn_experiments/) and shares the same argument.

Run from anywhere with the root venv:

```bash
cd src/metric_demos
../../.venv/bin/python3 exact_mi_demo.py
```

Output is ASCII tables on stdout. Nothing here writes artifacts, and the numbers each
script produces are already embedded in its docstring, so you can read the result
without running it.

## The metric families

The critique organizes every metric by **what it assumes about ρ(g)**, the map the
features transform by when the input is rotated:

| family | assumption | members |
|---|---|---|
| A | ρ(g) **hardcoded** | LEE / Lie derivative, EQ-R |
| B | ρ(g) **fitted** | Lenc & Vedaldi affine fit, Procrustes |
| C | **no** ρ(g) | linear CKA, RBF CKA, SVCCA, distance correlation, RSA |
| — | no ρ(g), no geometry | mutual information |

That axis predicts the failures exactly. Each family is blind to a different group of
readout reparameterizations, and the groups are nested:

```
family A (hardcoded ρ)   exactly one coordinate system
CKA, linear and RBF      isometries + isotropic scale
CCA / SVCCA              all invertible linear maps
mutual information       ALL bijections          <- maximal
```

So a readout that leaves one group but not the next separates the families cleanly. The
demos are constructions that do exactly that.

## The scripts

| script | what it shows |
|---|---|
| `exact_mi_demo.py` | The foundation. Enumerates **all 2^16** binary 4×4 images through a circular convolution, so the joint distribution is exact and MI is *computed*, not estimated — no KSG, no bandwidth, no sampling error. Walks a ladder of readouts and pins each metric to its invariance group: an invertible *linear* map with condition number 1e3 already takes both CKA variants from 1.0000 to ~0.04 while SVCCA stays at 1.0000; an alphabet permutation takes SVCCA down too while MI does not move. Also documents the **plug-in MI trap** — at full 16-bit granularity the shuffled null is 14.37 against a signal of 15.18 and the kernel ranking *inverts* — and the equivariant coarse-graining that fixes it. |
| `unlearnable_rho_exact_demo.py` | The three-way comparison: **LEE vs linear CKA vs exact MI**, with a graded knob. A bijective scramble of the alphabet (lossless at every setting) walks CKA out of its invariance group, 1.0000 → 0.5170 → 0.1735 → 0.0376, while MI holds 7.3205 bits to the last bit and prints the invariance as `+0.00e+00`. LEE, handed the *exact* true ρ(g), scores 0.0000 in natural coordinates and 1.2554 after merely permuting four coordinates — its own shuffled null is 1.3993, so it survives nothing. Then the headline: a lossless scramble (linCKA 0.0376) and a readout destroying 90% of the alphabet (linCKA 0.0431) are indistinguishable to CKA, which ranks the lossy one *higher*, while MI separates them by 4.16 bits. |
| `energy_readout_demo.py` | The complement case, where **A and B fail but C and MI both succeed** — so the critique is not "MI always wins". The readout emits squared magnitudes next to a linear block, `σ(v) = [Mv | (Nv)²]`, which is what fiber norms, attention scores and second-moment features all do in real networks. Energy features transform *quadratically*, so there is no ρ(g) matrix to assume and an affine fit cannot express the cross terms (residual 0.0000 → 0.6723). CKA barely notices (0.7379) and σ is injective, so MI/H = 1.000 proves nothing was lost and the A/B failure is theirs. |
| `lie_vs_cka_demo.py` | Faithful replication of the Gruver et al. ([arXiv:2210.02984](https://arxiv.org/abs/2210.02984)) Lie-derivative test against linear CKA, on the pretrained CIFAR CNN. The target is their `img_like = (len(z.shape) == 4)` convention: once the CNN flattens into `fc1`, the reference code silently switches from "must rotate" to "must not change at all" — assuming the trivial representation, a far stronger claim. |
| `lie_vs_cka_vector_field_demo.py` | The same convention failing *before* any flatten. `img_like == True` rotates the (H,W) grid but leaves channels independent, which is correct for a scalar activation map and wrong for a genuine 2D vector field, where rotation must also mix the channel pair. `gradient_field.py` trains exactly such a task, so this compares the naive scalar assumption against the channel-mixing one on its pretrained checkpoints. |
| `discretized.ipynb` | The scratch notebook the exhaustive 4×4 construction came from. `KERNELS` and the enumeration in `exact_mi_demo.py` are lifted from it. |

`energy_readout_demo.py` and `unlearnable_rho_exact_demo.py` both import
`exact_mi_demo` for the shared space, kernels and exact-MI helpers, so it is the file to
read first.

## What the demos require

The three discrete-space demos are pure numpy/scipy and self-contained.

The two `lie_vs_cka*` demos are not: they need `src/Models/` on the path plus pretrained
checkpoints from `src/models/` (`classification_*_cnn_dataset_cifar` and
`gradient_field_*_unet_dataset_mnist`) and a dataset under `src/data/`. All three are
resolved from `__file__` via `SRC_ROOT`, so the scripts still run from any working
directory — but `models/` is gitignored, so a fresh clone must train those four
checkpoints first. That dependency is why these two sit here rather than in
`escnn_experiments/`: they critique the same convention, but with trained models
instead of an architectural guarantee.

`lie_vs_cka_demo.py` uses a central finite difference rather than
`torch.autograd.functional.jvp`, which raises `NotImplementedError` here because
`grid_sampler_2d` has no forward-mode AD support in the installed PyTorch. It was
verified stable to ~15% across eps ∈ [0.005, 0.1] rad, so it measures the same local
derivative.

## Two disciplines worth copying

- **Every detector reports a shuffled null.** Row-shuffling the rotated branch destroys
  the pairing, so an honest metric must score ~0. This is not ceremony: it is what
  caught the RBF bandwidth trap (CKA 1.0000 *and* control 1.0000) and the plug-in MI
  granularity trap (signal 15.18, null 14.37).
- **Every claim has a negative control.** A non-equivariant kernel (Sobel-x) runs
  through every table. Without it, "MI is invariant to the readout" cannot be
  distinguished from "MI is blind" — with it, MI is exactly invariant to a *low* value,
  MI/H 0.195 against 1.000 for the equivariant kernel.
