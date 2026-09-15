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
| `cn_angle_sweep_demo.py` | The one thing **only** CKA can do: sweep equivariance error continuously over the whole group. LEE needs ρ(g), so it can only be evaluated at group elements; CKA needs none, so θ can run 0→360 in half-degree steps. C_N's error then touches **exactly 0** at every multiple of 360/N and nowhere else — N lobes of width 360/N, shrinking monotonically (peak 0.51 → 0.45 → 0.27 → 0.11 → 0.04 for C1 → C16) toward a 1e-4 SO(2) floor. The group is recovered from feature geometry alone, ρ(g) never constructed. Writes a PDF to `out/`. `--lee` adds the head-to-head against the ρ-dependent family: over the same sweep the finite-θ error under the naive ρ is **flat at ~1.0 for every group** (and ranks SO(2) *worst*, at 1.18), while LEE — Local Equivariance Error, the Lie-derivative norm ‖Lₓf‖/‖f‖ at the identity — is **~6 for every group**, one scalar with no θ argument at all. Neither resolves the group order CKA recovers exactly. |
| `unlearnable_rho_demo.py` | Removes family B's crutch: the same exactly-equivariant model observed through a fixed **nonlinear** readout, so ρ(g) can no longer be fitted linearly. Documents the **bandwidth trap** (small-bandwidth RBF CKA scores a spurious 1.0000 — its shuffled control does too), shows a bijective change of coordinates hides exact equivariance from every CKA variant, and then shows **mutual information recovers it** — because `I(σ(X);σ(Y)) = I(X;Y)` for bijective σ, an invariance CKA lacks. MI also correctly *fails* on the lossy readout, separating "coordinates scrambled" from "information destroyed", which CKA conflates. |

Two related scripts live in [`../metric_demos/`](../metric_demos/) instead, because they
need `src/Models/` and the pretrained checkpoints in `src/models/` rather than escnn:
`lie_vs_cka_demo.py` (the trivial-representation assumption past a `flatten`) and
`lie_vs_cka_vector_field_demo.py` (the scalar-vs-vector-field channel assumption on the
gradient-field task). That folder also holds the discrete-space demos where mutual
information is computed exactly rather than estimated.

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
