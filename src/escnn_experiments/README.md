# escnn equivariance-metric experiments

Experiments that need an **exactly equivariant** model as ground truth, built with
[escnn](https://github.com/QUVA-Lab/escnn) (E(2)-steerable CNNs). They exist to test a
claim from Gruver, Finzi, Goldblum & Wilson, *"The Lie Derivative for Measuring Learned
Equivariance"* ([arXiv:2210.02984](https://arxiv.org/abs/2210.02984)), namely that
architectures *without* an equivariance prior can end up more equivariant than ones with
it.

The value of escnn here is that its guarantee is architectural, not learned: a steerable
network is exactly equivariant whether or not it is trained, and each layer's `FieldType`
declares the exact representation ρ(g) its channels transform by. That gives a known
ground truth to score equivariance metrics *against*, rather than comparing one estimate
to another.

## Why this folder has its own environment

`escnn` imports its SO(3) machinery unconditionally, so `import escnn` requires
`lie_learn` — and `lie_learn` 0.0.2 is a Cython extension built against the **numpy 1.x
C ABI**. Under the `numpy==2.2.6` that the repo root pins, importing it fails outright:

```
ImportError: numpy.core.multiarray failed to import
```

So this is a real ABI incompatibility, not a conservative pin, and it cannot be waived.
Rather than downgrade numpy for the whole project, these experiments get their own venv:

```bash
./setup_venv.sh          # from src/escnn_experiments/
.venv/bin/python3 equivariance_metrics_comparison.py
```

`setup_venv.sh` builds a **thin** venv: torch plus the nvidia CUDA wheels are ~6.3 GB, and
reinstalling them purely to change numpy's version is wasteful, so it installs only the
numpy-ABI-sensitive packages locally and shares `torch`/`torchvision` from the repo-root
venv via a `.pth` file. Local site-packages takes precedence, so numpy 1.26.4 wins while
torch resolves to the shared copy — ~285 MB instead of ~6.6 GB. If you have disk to spare
and want a fully standalone environment, ignore the script and
`pip install -r requirements.txt` instead (that file is the full canonical list).

Datasets are shared with the rest of the repo (`src/data`, resolved from `__file__`, so
scripts run from any working directory). `equivariance_metrics_comparison.py` also reuses
`src/utils.py`'s `EquivarianceTracker` for RBF CKA via a `sys.path` insert.

## The scripts

Roughly in order of how much they build on each other.

| script | what it shows |
|---|---|
| `lie_vs_cka_escnn_exact_demo.py` | Against an exactly-equivariant model, the naive "each channel is an independent scalar field" assumption is off by 80-100% at grid-aligned angles, where the correct representation gives exactly 0.00000. |
| `lie_vs_cka_aliasing_demo.py` | Aliasing from **spatial** downsampling at export genuinely destroys equivariance (error 0.13 → 0.22, recovered to 0.15 by blur-then-subsample). CKA does *not* detect this — the counterpoint to the demos below. |
| `lie_vs_cka_group_aliasing_demo.py` | Aliasing of the **rotation group**. A C_N model honors only 4 of its N elements exactly (100% / 50% / 25% for C4 / C8 / C16), because only quarter turns are lattice symmetries of a square grid. Confirmed to be aliasing: error halves per doubling of resolution. |
| `lee_bandwidth_confound_demo.py` | LEE is confounded by feature bandwidth. Six models, *all* exactly equivariant, span 25× in LEE. A blatantly non-equivariant anisotropic filter beats a provably equivariant model by 2.4×, and the ranking inverts the moment the probe angle stops being lattice-exact. |
| `learned_vs_architectural_equivariance.py` | The head-to-head. A plain CNN trained with rotation augmentation vs. an escnn C8 network. At lattice-exact angles LEE ranks the plain CNN higher while behaviour and CKA say the opposite — LEE is wrong. Off-lattice the architectural guarantee genuinely evaporates and the plain CNN really does win, so there LEE is right and **CKA is wrong**. Both metrics fail somewhere. |
| `equivariance_metrics_comparison.py` | Nine metrics from the literature on that same head-to-head. The split is total and falls on one axis: metrics that hardcode ρ(g) get it wrong 0/2, metrics that fit ρ(g) 2/2, metrics that assume no ρ(g) 5/5. |
| `so2_exact_lee_reversal_demo.py` | The aliasing defense removed. Data is **2D point clouds**, so rotation is an exact orthogonal matmul — no grid, no resampling, no lattice, every angle exact to ~1e-6 (asserted, not assumed). Against a provably SO(2)-equivariant model (drift and CKA error *identically* 0), LEE ranks last at 0.9999 while preferring a deeper, wider MLP that fits the task better but drifts 9.9% under rotation (0.370) — and preferring most of all an **untrained** network that predicts nothing (0.252, test MSE 1.00). Linear CKA orders all three correctly. Needs no escnn. |
| `liegg_demo.py` | The strongest form of the rebuttal to all of the above: **LieGG** (Moskalev+22) *discovers* the group generator from the network instead of assuming one, and on the same exact-SO(2) point clouds it works — null space of dimension exactly 1, alignment 1.0000 with the true generator, and it ranks the untrained network **last** where LEE ranks it first. So "ρ-dependent metrics fail" is too strong. What it costs: a linear action on the input, an **invariant** output (the same trivial-representation assumption that breaks LEE, here granted by the task), a hand-set null-space threshold that flips its verdict, and it measures input-space symmetry rather than hidden-layer ρ(g). Needs no escnn. |
| `cn_angle_sweep_demo.py` | The one thing **only** CKA can do: sweep equivariance error continuously over the whole group. LEE needs ρ(g), so it can only be evaluated at group elements; CKA needs none, so θ can run 0→360 in half-degree steps. C_N's error then touches **exactly 0** at every multiple of 360/N and nowhere else — N lobes of width 360/N, shrinking monotonically (peak 0.51 → 0.45 → 0.27 → 0.11 → 0.04 for C1 → C16) toward a 1e-4 SO(2) floor. The figure plots that error **in units of the SO(2) control's peak** on a log axis (417× → 29× → 1×), because raw the six curves span 417× and only C1/C2/C4 are readable on one linear axis; `--norm none` restores the raw version. The group is recovered from feature geometry alone, ρ(g) never constructed. Writes a PDF to `out/`. `--lee` adds the head-to-head against the ρ-dependent family: over the same sweep the finite-θ error under the naive ρ is **flat at ~1.0 for every group** (and ranks SO(2) *worst*, at 1.18), while LEE — Local Equivariance Error, the Lie-derivative norm ‖Lₓf‖/‖f‖ at the identity — is **~6 for every group**, one scalar with no θ argument at all. Neither resolves the group order CKA recovers exactly. In that second figure each metric is a validated ordinal ramp over C4 → C8 → C16 → SO(2) (blue for CKA, orange for the finite-θ error) rather than one hue at three alphas, which was not separable where the curves overlap; metric identity is carried by line style as well as hue. |
| `discrete_group_demo.py` | The blind spot that is a **theorem**, not a wrong assumption. LEE and LieGG are first-order, so they see only the **identity component** of the symmetry group; `det exp(A) = e^tr(A) > 0` puts a reflection outside every one-parameter subgroup. Against a model exactly equivariant to the Klein four-group, LieGG's `sym_var` spans **1.01x** across it, an augmented MLP and an MLP with no augmentation, while CKA reads **exactly 0** for the first. A Procrustes fit recovers ro(g) = diag(I, chi(g)I) and it **obeys the group law** (defect 0.000144), so what CKA found is a representation, not a per-element coincidence. Needs no escnn. |
| `unlearnable_rho_demo.py` | Removes family B's crutch: the same exactly-equivariant model observed through a fixed **nonlinear** readout, so ρ(g) can no longer be fitted linearly. Documents the **bandwidth trap** (small-bandwidth RBF CKA scores a spurious 1.0000 — its shuffled control does too), shows a bijective change of coordinates hides exact equivariance from every CKA variant, and then shows **mutual information recovers it** — because `I(σ(X);σ(Y)) = I(X;Y)` for bijective σ, an invariance CKA lacks. MI also correctly *fails* on the lossy readout, separating "coordinates scrambled" from "information destroyed", which CKA conflates. |

Two related scripts live in [`../metric_demos/`](../metric_demos/) instead, because they
need `src/Models/` and the pretrained checkpoints in `src/models/` rather than escnn:
`lie_vs_cka_demo.py` (the trivial-representation assumption past a `flatten`) and
`lie_vs_cka_vector_field_demo.py` (the scalar-vs-vector-field channel assumption on the
gradient-field task). That folder also holds the discrete-space demos where mutual
information is computed exactly rather than estimated.

## Measured results

All numbers below are from a run of each script as it stands in this folder, in the
sub-venv. Timings are one pass each on this machine: 40 s to 155 s, so the whole folder
reproduces in about 10 minutes.

### The ρ(g) mismatch, against an exactly-equivariant model

`lie_vs_cka_escnn_exact_demo.py`. Hidden `FieldType` is
`[trivial, trivial, irrep1, irrep1, irrep2]`, 8 channels; linear CKA reads 1.0000 at the
discrete 90° rotation.

| θ (rad) | correct ρ(g) | naive scalar ρ(g) |
|---|---|---|
| 1.5708 (grid-aligned) | **0.00000** | 0.80582 |
| 3.1416 (grid-aligned) | **0.00000** | 1.02328 |
| 0.1000 | 0.11989 | 0.13448 |
| 1.0000 | 0.18457 | 0.59194 |
| 2.0000 | 0.16951 | 0.92194 |

At the lattice rotations the correct representation gives *exactly* zero while the naive
one reports 81–102% error on a provably equivariant model. The whole of that gap is
representation mismatch.

### LEE is confounded by feature bandwidth

`lee_bandwidth_confound_demo.py`, three parts, and the second is the damaging one.

**Part A** — six models, *all* exactly equivariant, so the correct answer is identical in
every row:

| feature freq | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| LEE | 0.01157 | 0.10368 | 0.16462 | 0.17394 | 0.22404 | 0.28770 |

A **25× spread** where the measured property does not change at all.

**Part B** — exactly-equivariant model vs. a blatantly non-equivariant anisotropic filter:

| export group | equivariant (EXACT) | anisotropic (NOT) | verdict |
|---|---|---|---|
| C4 (quarter turns, lattice-exact) | 0.00000 | 0.06380 | correct |
| C8 (adds 45°, interpolated) | 0.22190 | 0.10175 | **INVERTED** |
| C16 (adds 22.5°) | 0.22175 | 0.08419 | **INVERTED** |
| SO(2) (generic angles) | 0.22404 | 0.09279 | **INVERTED** |

Sample the group any finer than the lattice and the provably equivariant model measures
~2.4× *worse* than one with a hard-coded directional bias.

**Part C** — the mechanism. Anisotropy ratio held at ~8:3, only the blur scale grows:

| anisotropic Gaussian | 3.0/1.0 | 4.0/1.5 | 6.0/2.0 | 8.0/3.0 | 12.0/4.5 |
|---|---|---|---|---|---|
| LEE | 0.22298 | 0.16352 | 0.12966 | 0.09279 | 0.09092 |

Same directional bias, same (absent) equivariance, monotonically falling LEE. Smoothness
is what the metric rewards — which matters because architectures being compared in the
literature differ in exactly that.

### Nine metrics on one head-to-head

`equivariance_metrics_comparison.py`. Sanity check first: the architectural model's
residual under the **correct** 96×96 permutation ρ(g) is 3.73e-07, so it is exactly
equivariant by construction. Learned acc 0.8417 / 0.8405 rotated; architectural
0.8022 / 0.8022.

| family | metric | learned | architectural | picks | vs ground truth |
|---|---|---|---|---|---|
| A1 | LEE relative error (Gruver+23) | 0.20973 | 0.43561 | learned | **WRONG** |
| A2 | EQ-R PSNR dB (Karras+21) | 22.59 | 17.28 | learned | **WRONG** |
| B1 | learned linear map (Lenc+15) | 0.08055 | **0.00000** | architectural | correct |
| B2 | Procrustes orthogonal map | 0.36653 | **0.00000** | architectural | correct |
| C1 | linear CKA (Kornblith+19) | 0.96618 | **1.00000** | architectural | correct |
| C2 | RBF CKA | 0.96737 | **1.00000** | architectural | correct |
| C3 | SVCCA (Raghu+17) | 0.90654 | **1.00000** | architectural | correct |
| C4 | distance correlation | 0.97025 | **1.00000** | architectural | correct |
| C5 | RSA Spearman | 0.95402 | **1.00000** | architectural | correct |
| D1 | prediction consistency (ground truth) | 0.87890 | 1.00000 | architectural | — |
| D2 | accuracy drop under rotation (ground truth) | 0.00120 | 0.00000 | architectural | — |

**Family A 0/2, family B 2/2, family C 5/5.** The split is total and falls on exactly one
axis: whether the metric commits to a ρ(g) in advance.

### The bijective-readout attack, and the bandwidth trap

`unlearnable_rho_demo.py`. Ground truth: residual under the true permutation 3.92e-07.
Under the lossless TWIST readout at ω=2.0, every geometric metric is gone and MI is not
(margins are signal − shuffled control):

| metric | true | shuffled | margin | verdict |
|---|---|---|---|---|
| linear fit residual | 1.0074 | — | — | ρ(g) unlearnable |
| linear CKA | 0.0288 | 0.0064 | 0.0224 | fails |
| RBF CKA @1.0× median | 0.0625 | 0.0117 | 0.0508 | fails |
| RBF CKA @0.3× median | 0.4993 | 0.3535 | 0.1458 | fails |
| RBF CKA @0.05× median | 1.0000 | **1.0000** | 0.0000 | **VACUOUS** |
| KSG MI (nats) | 1.4167 | 0.0173 | **1.3995** | detects |

That `RBF CKA @0.05× = 1.0000` with a control of `1.0000` is the **bandwidth trap** in one
line: as σ shrinks both Gram matrices approach the identity and CKA(I, I) = 1 regardless
of the data. Anything at or below ~0.2× median is vacuous here, and without the shuffled
control it would read as a perfect score.

MI also fails in the right place. Under the **lossy** COSINE readout at ω=5.0 the margin
collapses to 0.0384 — information genuinely destroyed, correctly reported — while under
the lossless TWIST at ω=2.0 it holds 1.3995. CKA fails identically on both (0.0288 and
0.0514) and so cannot tell the two situations apart.

(These magnitudes are at the script's `SUB = 2000`; the docstring's larger table is at
n = 4000, and KSG is sample-size dependent. The pattern is the same.)

### CKA's own blind spot: aliasing

`lie_vs_cka_aliasing_demo.py` — spatial downsampling at export, generic angles:

| θ = 2.0 | full | subsample | blur-then-subsample |
|---|---|---|---|
| equivariance error | 0.13429 | **0.21735** | 0.15250 |
| linear CKA | 0.9994 | **0.9995** | 0.9989 |

The error rises 62% and recovers with proper prefiltering; CKA moves in the **fourth
decimal** and, if anything, slightly *up*. So CKA is not a drop-in replacement — it is
blind to a real, mechanism-identified loss of equivariance.

`lie_vs_cka_group_aliasing_demo.py` shows the same blindness on group aliasing. A C_N
model honors only its lattice elements exactly — 4/4 for C4 (100%), 4/8 for C8 (50%),
4/16 for C16 (25%) — with the non-lattice elements at 0.130–0.162 error while CKA sits at
0.9995–0.9997 throughout. That it is aliasing is confirmed by refinement: C16's 22.5°
error falls 0.12964 → 0.05145 → 0.01825 → 0.00744 as the grid goes 29² → 57² → 113² →
225², roughly halving per doubling, while the 90° column stays exactly 0.00000 at every
rate.

### LEE with no lattice left to blame

`so2_exact_lee_reversal_demo.py`. Data is 2D point clouds, so rotation is an exact
orthogonal matmul — verified equivariant to ~1e-6 at generic angles, no grid, no
interpolation. Three seeds, 300 epochs:

| model | LEE naive | LEE correct | 1 − CKA | prediction drift | test MSE |
|---|---|---|---|---|---|
| EquivariantSO2 | 0.9999 ± 0.000 | 0.0000 | **0.000000** | **0.000000** | 0.04994 |
| LearnedMLP | **0.3702** ± 0.036 | n/a | 0.008651 | 0.098904 | 0.01222 |
| LearnedMLP (untrained) | **0.2518** ± 0.015 | n/a | 0.460116 | 0.028638 | 1.00273 |

Read the LEE column against the drift column. LEE ranks the provably SO(2)-equivariant
model **last** and an **untrained** network that predicts nothing (test MSE 1.00) **first**,
while that model's predictions are bit-identical under rotation and the MLP's move by
9.9%. Linear CKA and drift both order all three correctly. The baseline is deeper, wider
*and* fits better (0.0122 vs 0.0499 MSE), so it was not crippled to produce this.

This is the cleanest counterexample in the folder, because every escape route is closed:
no lattice, no interpolation, exact group action, and the correct ρ(g) available for the
one model that has one.

### LieGG: the one ρ-dependent method that gets it right

`liegg_demo.py`, 3 seeds, 300 epochs, n=2048, null tolerance 0.05. Same models as the
SO(2) demo plus a fourth control the host demo lacked — the same MLP trained with **no**
augmentation, because a symmetry-*discovery* method has to be shown finding nothing when
there is nothing.

| model | LieGG `sym_var` | `null_dim` | `align` | ‖grad‖ | LEE naive | 1 − CKA |
|---|---|---|---|---|---|---|
| EquivariantSO2 | **0.00000** | **1.0** | **1.0000** | 5.9228 | 0.9999 | 0.000000 |
| LearnedMLP (aug) | 0.05760 | 0.0 | 0.0000 | 5.9187 | 0.3702 | 0.008651 |
| LearnedMLP (NO aug) | 0.06845 | 0.0 | 0.0000 | 6.1034 | 0.6413 | 0.042468 |
| LearnedMLP (untrained) | 0.16950 | 0.0 | 0.0000 | **0.0083** | **0.2518** | 0.460116 |

**It recovers SO(2) from the network.** For the equivariant model the polarization
matrix's smallest normalized singular value is 0.00000, the numerical null space has
dimension exactly 1 (SO(2) *is* one-dimensional), and it aligns with the true I₈⊗J at
1.0000. Nothing was supplied.

**It is not fooled by the degeneracy that inverts LEE**, and the ‖grad‖ column is why:

```
LieGG   equivariant 0.00000 < aug 0.05760 < no-aug 0.06845 < untrained 0.16950
LEE     untrained   0.2518  < aug 0.3702  < no-aug 0.6413  < equivariant 0.9999
```

The untrained model's polarization matrix is 0.0083 against ~5.9 — a 713× smaller
signal. LEE divides by ‖f(x)‖, so a network whose features barely respond gets a small
numerator and looks equivariant. LieGG normalizes the spectrum by its own largest
singular value, so scaling every gradient down cancels exactly.

**Two limitations, both measured.** Its binary verdict rides on the threshold: the
augmented MLP lands at 0.05760 against a tolerance of 0.05, so LieGG reports "no
symmetry" for a model whose CKA error is 0.0087 and whose predictions drift 9.9%. At 60
epochs the same model measured 0.04163 — just *below* — and LieGG returned `null_dim 1`,
`align 0.9498`. The qualitative answer flips on a 15% change in a hand-set constant, which
is the same species of knob as RBF CKA's bandwidth and plug-in MI's granularity. And it
is weakly discriminative among *approximate* symmetries, which is what the `src/` sweep
is made of:

| augmented vs. non-augmented MLP | separation |
|---|---|
| LieGG `sym_var` (0.05760 vs 0.06845) | 1.19× |
| prediction drift (ground truth) | 2.31× |
| 1 − linear CKA (0.00865 vs 0.04247) | **4.91×** |

So LieGG certifies an *exact* symmetry superbly and barely separates two approximate
ones, where linear CKA separates more sharply than behaviour does.

**This narrows the folder's claim.** "Metrics that depend on ρ(g) fail" is too strong —
LieGG depends on a group action, derives it correctly, and beats LEE on every row here.
The defensible version is about *where* the assumption sits:

- **LEE** needs ρ(g) **specified in feature space**, and ships a wrong default.
- **LieGG** needs the action **linear on the input** and the output **invariant** — that
  second condition is the trivial-representation assumption itself, granted here by the
  task rather than by the method — and derives the rest.
- **Linear CKA** needs ρ(g) only to **be** an isometry, which a group representation on
  feature channels is by construction.

Caveat on transfer: LieGG's derivation needs the group to act linearly on the input, which
is true of a point cloud and false of image rotation acting on pixel values. None of this
result carries over to the grid experiments above.

### A discrete symmetry: blind by theorem, not by assumption

`discrete_group_demo.py`, 3 seeds, 300 epochs. Every other result in this folder is about a
metric assuming the *wrong* representation, and `liegg_demo.py` closes that escape route by
deriving the right one. This opens a different one, and it cannot be closed the same way.

LEE differentiates along a one-parameter flow; LieGG searches for a generator `A` with
`grad f(x)^T A x = 0`. Both see only the tangent space at the identity, so only the
**identity component** of the symmetry group. For reflections that is provable rather than
empirical: `det exp(A) = e^tr(A) > 0`, so no one-parameter subgroup ever reaches `det = -1`.
The answer is not in LieGG's hypothesis class, and no threshold or sample size changes it.

The setting is LieGG's best case in every respect but the one under test — point clouds, so
the action is linear on the input; exact orthogonal group elements, no grid; and LieGG is
handed an **invariant** readout `f(x)^2` so its trivial-representation condition holds by
construction. Only the group changes, from SO(2) to the Klein four-group
`V = {e, r, mx, my}` with `r` = rotation by pi, under a sign character
`chi = (+1, -1, +1, -1)`. The character matters: it makes the hidden representation
`ro(g) = diag(I, chi(g) I)`, which is an isometry (so CKA reads 1) and nontrivial (so the
assumed-trivial fallback does not).

| model | LieGG `sym_var` | `null_dim` | 1 − CKA (my) | naive ρ (my) | drift | test MSE |
|---|---|---|---|---|---|---|
| ExactD2 | 0.02523 | 82.0 | **0.000000** | 0.9676 | **0.0000** | 0.019 |
| LearnedMLP (aug) | 0.02481 | 82.0 | 0.121733 | 1.4046 | 0.1192 | 0.026 |
| LearnedMLP (NO aug) | 0.02543 | 84.3 | 0.090516 | 1.4033 | 0.1837 | 0.024 |
| LearnedMLP (untrained) | 0.16907 | 0.0 | 0.665082 | 0.2163 | 1.3453 | 0.996 |

**LieGG is uninformative here, not wrong.** Asked "does this network have a continuous
symmetry?" it answers no, which is true — `dim V = 0`. But the answer does not *move*:
`sym_var` spans **1.01×** across a provably V-equivariant model, an augmented MLP and one
trained with no augmentation at all, where 1 − CKA separates the first from the rest
infinitely. It cannot locate the symmetry because none of these models has the kind of
symmetry it is built to see. This is a weaker claim than the LEE inversions above, and it is
stated weakly on purpose.

At `NULL_TOL = 0.05` it does not return an empty null space either — ~83 directions for
every trained model. Applying LieGG's own `exp(tA)` identifies them: the prediction moves by
0.0074, against 0.0470 for a random direction. About 10× better than chance and not zero,
i.e. near-symmetry directions of a badly conditioned polarization matrix. Same threshold
fragility `liegg_demo.py` documents, in a setting where the true answer is 0.

**The rotation generator is not recovered even though `r` is a symmetry.** Every model is
exactly equivariant to rotation by pi, an element of SO(2). `I_8 ⊗ J` still scores 0.00255
against 0.00373 for a *random* generator — no better. A discrete element of a continuous
group carries no information about the flow through it.

**The trivial-ρ fallback fails at exactly the elements with character −1**, and the `mx`
column is the internal control: `chi(mx) = +1`, so there the trivial representation *is*
correct and reads 0.0000, while `r` and `my` read 0.9676 on a model whose predictions are
exact. The gap is the sign irrep and nothing else.

**What CKA found is a representation, not a coincidence.** This is the standing objection to
family C — `CKA = 1` says only that *some* isometry relates two feature sets at *one*
element. Fitting `Q(g)` per element and checking the group law closes it:

| model | Procrustes residual | ‖Φ Q(mx)Q(my) − Φ Q(r)‖ / ‖Φ Q(r)‖ |
|---|---|---|
| ExactD2 | **0.000096** | **0.000144** |
| LearnedMLP (aug) | 0.248673 | 0.031122 |
| LearnedMLP (NO aug) | 0.259743 | 0.085657 |
| LearnedMLP (untrained) | 0.150450 | 0.093816 |

`Q(my)` comes out diagonal ±1 with exactly 64 entries at +1 and 64 at −1 — `ro(g) =
diag(I, chi(g) I)`, read off the features with the group never supplied. Both columns are
measured on Φ rather than on the bare matrices: `Q` is identified only on the row space of
the features, ReLU leaves near-dead channels, and off that subspace the Procrustes SVD
returns an arbitrary rotation. The bare-matrix defect for ExactD2 is ~20× larger and
measures that arbitrary part, not the group law.

The θ-sweep confirms the group is genuinely discrete rather than a continuous one in
disguise — 1 − CKA for ExactD2 is 0.0000 at 0° and 180° and 0.54–0.99 at 45°, 90°, 135°:
`C_2`, recovered from feature geometry alone.

**And one result against CKA, left in.** On the two *approximate* models CKA disagrees with
behaviour: drift prefers aug (0.1192 vs 0.1837), 1 − CKA prefers no-aug (0.1217 vs 0.0905).
The direction is not stable across epoch budgets — at 15 epochs it points the other way — so
the honest reading is that CKA is unreliable at *ranking* two approximately-symmetric
models. The clean result here is confined to the exact model. Needing no ρ(g) buys
correctness about **which** group a network has, not about how much of it survives, which is
the same boundary `lie_vs_cka_aliasing_demo.py` draws.

Read this against `unlearnable_rho_demo.py`, which is its mirror image: a nonlinear change
of coordinates on the *output* hides exact equivariance from every CKA variant while MI
still finds it. So the summary is not that CKA is stronger but that the detectable sets are
**incomparable**, each metric's blind spot following from the quantifier in its definition —
LEE asks "under *this* ρ?", LieGG "does there exist a *linear input generator*?", CKA "does
there exist an *output isometry*, at this element?"

### One result that no longer reproduces as written

`learned_vs_architectural_equivariance.py` reproduces the LEE inversion at all three
probes — LEE naive prefers the plain CNN every time (0.64628 vs 1.13192 at 90°, 0.75036
vs 0.99635 at 45°, 0.68861 vs 0.95313 generic) while CKA and behaviour prefer the escnn
model. But it **no longer reproduces the off-lattice reversal** described under *Caveat on
scope* below. Behaviourally the architectural model now wins at every probe:

| probe | learned acc drop | architectural acc drop | behaviour prefers |
|---|---|---|---|
| 90° (lattice-exact) | +0.0235 | +0.0000 | architectural |
| 45° (interpolated) | +0.0675 | +0.0239 | architectural |
| generic (not in C8) | +0.0454 | +0.0166 | architectural |

The script prints `VERDICT REVERSED` in all three sections. So the claim below that
"off-lattice the plain CNN really does win" is not what this run shows — the architectural
guarantee degrades off-lattice (0.0000 → 0.0239 → 0.0166) but never far enough to lose.
The caveat section has not been edited, because which of the two is right depends on
training details (seed, epochs, the escnn model is 6,586 params against the CNN's
117,802) and that is worth resolving deliberately rather than by overwriting. Treat the
off-lattice claim as unverified pending a multi-seed rerun.

## Caveat on scope

A counterexample refutes the inference rule "lower LEE ⇒ more equivariant" as a general
principle, and these are counterexamples against exact ground truth. That does not by
itself prove the paper's specific ViT-vs-CNN measurements are wrong. What it establishes
is that they are **confounded** — by feature bandwidth
(`lee_bandwidth_confound_demo.py`) and by representation mismatch
(`equivariance_metrics_comparison.py`) — so a cross-architecture LEE comparison needs
those held fixed before it can support a claim about equivariance.

Two things worth stating plainly, because they cut against a simple "LEE bad, CKA good"
reading:

1. **Off-lattice, the paper's claim holds on its merits.** A pixel grid cannot represent
   a non-lattice rotation, so an architectural equivariance guarantee genuinely
   evaporates off its group while augmentation-trained robustness does not. A
   learned-equivariant model really can be more rotation-robust there. What the
   measurements do not support is the *explanation* — this is discretization defeating
   the guarantee, not learning outperforming architecture.
2. **CKA is not a drop-in replacement.** It needs no ρ(g), so it cannot be wrong about
   one, but it measures hidden-layer representational consistency, which is not
   end-task robustness. At off-lattice angles it prefers the model that behaviour says
   is worse, and `lie_vs_cka_aliasing_demo.py` shows it is blind to aliasing-induced
   equivariance loss entirely. The reliable ground truth in these experiments is
   behaviour (prediction consistency and accuracy under rotation); the only
   equivariance-error variant correct at every probe is LEE computed under the *correct*
   ρ(g), which requires knowing the model's representation.

   `cn_angle_sweep_demo.py` is the counterweight to that second point, and it cuts the
   other way: needing no ρ(g) is not only a way of avoiding a mistake, it buys measurements
   LEE cannot make at all. Sweeping θ continuously is one. It also separates two things LEE
   conflates — at C16's 22.5° group element, LEE under the exact ρ(g) reads 0.354, much of
   it the cost of *interpolating* a sampled feature map to apply ρ(g), while the
   interpolation-free ring readout puts 1−CKA at 0.020 and correctly reports a group element.
   At the *lattice* rotations, where ρ(g) is a pure index permutation and no interpolation is
   needed, the two agree exactly at 0.
   Neither metric dominates: LEE is right about what the grid destroys, CKA about what the
   group preserves.

   `so2_exact_lee_reversal_demo.py` then closes the grid escape route. With an exact group
   action and no lattice at all, LEE still ranks a provably SO(2)-equivariant model below an
   untrained MLP. So "LEE under the correct ρ(g)" being the one always-right variant is true
   but thin: the models it is used to *rank* are the ones whose ρ(g) nobody knows, and the
   conventional fallback (assume the trivial representation) is what produces the inversion.
