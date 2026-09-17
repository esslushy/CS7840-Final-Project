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
| `energy_readout_demo.py` | The complement case, where **A and B fail but C and MI both succeed** — so the critique is not "MI always wins". The readout emits squared magnitudes next to a linear block, `σ(v) = [Mv \| (Nv)²]`, which is what fiber norms, attention scores and second-moment features all do in real networks. Energy features transform *quadratically*, so there is no ρ(g) matrix to assume and an affine fit cannot express the cross terms (residual 0.0000 → 0.6723). CKA barely notices (0.7379) and σ is injective, so MI/H = 1.000 proves nothing was lost and the A/B failure is theirs. |
| `lie_vs_cka_demo.py` | Faithful replication of the Gruver et al. ([arXiv:2210.02984](https://arxiv.org/abs/2210.02984)) Lie-derivative test against linear CKA, on the pretrained CIFAR CNN. The target is their `img_like = (len(z.shape) == 4)` convention: once the CNN flattens into `fc1`, the reference code silently switches from "must rotate" to "must not change at all" — assuming the trivial representation, a far stronger claim. |
| `lie_vs_cka_vector_field_demo.py` | The same convention failing *before* any flatten. `img_like == True` rotates the (H,W) grid but leaves channels independent, which is correct for a scalar activation map and wrong for a genuine 2D vector field, where rotation must also mix the channel pair. `gradient_field.py` trains exactly such a task, so this compares the naive scalar assumption against the channel-mixing one on its pretrained checkpoints. |
| `discretized.ipynb` | The scratch notebook the exhaustive 4×4 construction came from. `KERNELS` and the enumeration in `exact_mi_demo.py` are lifted from it. |

`energy_readout_demo.py` and `unlearnable_rho_exact_demo.py` both import
`exact_mi_demo` for the shared space, kernels and exact-MI helpers, so it is the file to
read first.

## Measured results

Every number below is from a run of the script as it stands in this folder, on the
exhaustive 65,536-image space. They are deterministic (`SEED = 0`), so re-running
reproduces them exactly.

### `exact_mi_demo.py` — each metric pinned to its invariance group

Laplacian kernel (exactly C4-equivariant), at the valid quadrant coarse-grained
granularity (486 symbols, 134.8 samples/symbol, H = 7.3205 bits):

| readout | nature | MI | MI null | MI/H | linCKA | rbfCKA | SVCCA |
|---|---|---|---|---|---|---|---|
| identity | reference | 7.3205 | 0.8113 | 1.000 | 1.0000 | 1.0000 | 1.0000 |
| bit-permute | bijective, isometry | 7.3205 | 0.8113 | 1.000 | 1.0000 | 1.0000 | 1.0000 |
| ill-cond. linear | bijective, linear, **not** isometry | 7.3205 | 0.8113 | 1.000 | 0.3584 | 0.2280 | 1.0000 |
| alphabet-permute | bijective, **not** linear | 7.3205 | 0.8113 | 1.000 | 0.1113 | 0.1543 | 0.2597 |
| lossy-half | **many-to-one**, info destroyed | 1.8643 | 0.0065 | 0.505 | 0.4835 | 0.4528 | 0.5042 |

Read the table top to bottom: each row leaves one more invariance group, and exactly the
metrics whose group it left collapse. `MI(identity) − MI(alphabet-permute) = +0.00e+00`.
The last row is the one that matters for interpretation — when information is genuinely
destroyed, MI drops (MI/H 0.505) and CKA *rises* relative to the row above it (0.1113 →
0.4835), because a lossy readout leaves more geometry intact than a scrambled one does.

The same table at the **full 16-bit** granularity is the documented trap: 45,707 symbols
for 65,536 samples, so the MI null sits at 14.3668 against a signal of 15.1834 — a 5%
margin, not a detection — and the bias-corrected ordering inverts, ranking
non-equivariant Sobel-x (margin 2.0031) above the equivariant Laplacian (0.8166). Check
samples-per-symbol before reading any plug-in MI number.

Control (Sobel-x, not equivariant, quadrant granularity): MI 1.2676 / MI/H 0.195,
linCKA 0.0018 against a 0.0009 null. Both metrics correctly report almost nothing.

### `unlearnable_rho_exact_demo.py` — LEE vs linear CKA vs exact MI

Laplacian kernel; ρ(g) is the cyclic quadrant shift, verified at LEE = 0.00e+00.
`LEE_ρ` is handed the exact true ρ(g); `mass` is the fraction of samples the readout
relabels:

| readout | lossless | mass | LEE_ρ | LEE null | linCKA | null | MI | MI/H |
|---|---|---|---|---|---|---|---|---|
| identity | yes | 0.00 | **0.0000** | 1.3998 | 1.0000 | 0.0004 | 7.3205 | 1.000 |
| coord-permute (isometry) | yes | 0.68 | 1.2554 | 1.3993 | 1.0000 | 0.0004 | 7.3205 | 1.000 |
| scramble p=0.10 | yes | 0.09 | 0.7710 | 1.3970 | 0.5170 | 0.0005 | 7.3205 | 1.000 |
| scramble p=0.25 | yes | 0.34 | 1.1351 | 1.3862 | 0.1735 | 0.0012 | 7.3205 | 1.000 |
| scramble p=0.50 | yes | 0.53 | 1.3469 | 1.3933 | 0.0453 | 0.0022 | 7.3205 | 1.000 |
| scramble p=1.00 | yes | 1.00 | 1.4175 | 1.3923 | 0.0376 | 0.0009 | 7.3205 | 1.000 |
| merge q=0.50 | **no** | 1.00 | 1.4116 | 1.3802 | 0.0380 | 0.0010 | 5.8681 | 0.890 |
| merge q=0.25 | **no** | 1.00 | 1.3694 | 1.3794 | 0.0303 | 0.0010 | 5.0613 | 0.818 |
| merge q=0.10 | **no** | 1.00 | 1.3843 | 1.3925 | 0.0431 | 0.0006 | 3.1574 | 0.606 |

Three results, in order of how much they cost each metric:

1. **LEE survives nothing.** Exact 0.0000 in natural coordinates, 1.2554 after merely
   permuting four coordinates — indistinguishable from its own 1.3993 shuffled null.
   Its ρ=I fallback is 1.4786 even in the natural coordinates, wrong before any attack.
2. **CKA survives isometries only**, and the scramble sweep walks it out: 1.0000 →
   0.5170 → 0.1735 → 0.0453 → 0.0376. Relabeling 9% of the probability mass halves it.
   MI is 7.3205 on every one of those rows, and the invariance prints as `+0.00e+00`.
3. **Only MI separates hidden from lost.** Lossless scramble p=1.00 scores linCKA
   0.0376; a readout destroying 90% of the alphabet scores 0.0431 — CKA ranks the
   **lossy** readout higher, both on a ~0.0009 null. MI separates them by 4.16 bits
   (7.3205 vs 3.1574).

Control (Sobel-x): MI is still exactly invariant to every bijective readout, but
invariant to a *low* value — 1.2676 bits, MI/H 0.195 against 1.000 — so the constancy
above is invariance, not blindness.

### `energy_readout_demo.py` — the case where A and B fail but C and MI succeed

Laplacian kernel, MI/H = 1.000 under **both** readouts, so σ destroys nothing:

| metric | identity readout | ENERGY readout `[Mv \| (Nv)²]` |
|---|---|---|
| A1 LEE, assumed ρ=I | 1.4933 FAILS | 1.3640 FAILS |
| B1 affine fit (Lenc+15) | 0.0000 detects | 0.6723 **FAILS** |
| B2 Procrustes fit | 0.0000 detects | 0.7225 **FAILS** |
| C1 linear CKA | 1.0000 (null 0.0020) | 0.7379 (null 0.0029) detects |
| C2 RBF CKA | 1.0000 (null 0.0030) | 0.8634 (null 0.0042) detects |
| C3 SVCCA | 1.0000 (null 0.0387) | 0.7802 (null 0.0459) detects |
| MI exact (bits) | 7.3205 (null 0.8100) | 7.3205 (null 0.8100) detects |

The identity column is the reference and shows family B working *perfectly* when ρ(g)
happens to be linear. Changing only the readout takes B from 0.0000 to 0.67/0.72 while
every family-C metric and MI keep detecting — and MI/H stays exactly 1.000, so the
failure is B's hypothesis class, not the representation.

Control (Sobel-x, energy readout): linCKA 0.0071, RBF 0.0091, SVCCA 0.0876 — all at
their nulls, so family C was responding to equivariance and not to the readout's
structure. MI reports 1.2676 bits (margin 1.06), which the crude threshold calls a
detection; a non-equivariant convolution genuinely does share some information between
x and rot(x). Read MI/H (0.195 vs 1.000) rather than thresholding the margin.

### `lie_vs_cka_demo.py` — the trivial-representation assumption past a flatten

Pretrained CIFAR CNN, rel_LEE (lower = more equivariant) against linear CKA:

| layer | rel_LEE aug. | linCKA aug. | rel_LEE base | linCKA base |
|---|---|---|---|---|
| conv1 | 1.1852 | 0.9989 | 1.2206 | 0.9993 |
| pool1 | 0.7852 | 0.9988 | 0.9080 | 0.9991 |
| pool2 | 0.8105 | 0.9881 | 0.9823 | 0.9111 |
| fc1 | 0.6745 | 0.9823 | 0.9837 | 0.8555 |
| fc2 | 0.5280 | 0.9886 | 0.9481 | 0.6756 |
| fc3 | 1.1975 | 0.9589 | 1.7073 | 0.2693 |
| softmax | 1.0594 | 0.9379 | 1.7539 | 0.2560 |

CKA separates the two models exactly as expected and the separation *grows* with depth:
0.9379 vs 0.2560 at the output, from an identical 0.999 at `conv1`. rel_LEE does rank
the augmented model better at every layer, but its scale is uninformative — it exceeds
1.0 at `conv1` for **both** models, and rises again at `fc3`/`softmax` where the
representation is nominally most invariant. Past the flatten the reference convention has
switched to assuming ρ = I, so those last rows are scoring against invariance the model
was never trained to have.

### `lie_vs_cka_vector_field_demo.py` — scalar vs vector-field ρ(g)

Pretrained gradient-field UNets:

| | task MSE clean / rot90 | naive scalar rel_LEE | correct vector-field rel_LEE | linCKA |
|---|---|---|---|---|
| augmented | 0.00027 / 0.00027 | 1.0280 | **0.6084** | 0.9982 |
| baseline | 0.00027 / 0.00032 | 1.0411 | **0.6289** | 0.9977 |

Choosing the correct ρ(g) for a 2-channel vector field cuts the measured "equivariance
error" by ~40% (1.03 → 0.61) on the *same* model and the same forward pass. That gap is
pure representation mismatch: the naive assumption rotates the (H,W) grid but leaves the
channel pair unmixed, which is the wrong transformation law for a gradient field.

Note what this pair of rows does *not* show: both models score nearly identically on all
three metrics, and behaviour agrees (MSE 0.00027 vs 0.00032 under rotation). This task is
easy enough that augmentation barely matters, so the demo is evidence about the metric's
convention, not about learned equivariance.

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
