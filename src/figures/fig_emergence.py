"""
Does equivariance emerge over training? Epoch on the x-axis, depth as color.

The old visualize_equivariant_losses_over_time.py put layer on x and encoded epoch
as a 'gnuplot' rainbow, so the central question had to be read off a
non-perceptually-uniform color gradient. Here epoch is an axis and depth is a
single-hue ordinal ramp, which is the right assignment for two ordered variables.

Run from src/:
    python figures/fig_emergence.py [--defect]
"""
import matplotlib.pyplot as plt
import numpy as np

import style as st

# Representative configs: a large-effect case, two mid-range, one saturated.
CONFIGS = [
    ("classification", "cnn", "cifar"),
    ("stress_prediction", "unet", "anisotropic"),
    ("fluid_flow", "cnn", "buoyant"),
    ("colorization", "unet", "cifar"),
]


def main(args):
    use_defect = args.defect
    st.apply()
    df = st.apply_filters(st.load("cka_by_epoch"), args)
    configs = [c for c in CONFIGS if ((df.task == c[0]) & (df.dataset == c[2])).any()]

    fig, axes = plt.subplots(len(configs), 2, figsize=(8.4, 2.3 * len(configs)),
                             sharex="row", squeeze=False, constrained_layout=True)

    for r, (task, model, dataset) in enumerate(configs):
        sub = df[(df.task == task) & (df.model == model) & (df.dataset == dataset)]
        layers = (sub[["layer_idx", "depth_frac"]].drop_duplicates()
                    .sort_values("layer_idx"))
        colors = st.depth_colors(len(layers))
        # Shared y across the two conditions so the pair is comparable. Set the
        # range explicitly from both panels' data: calling ax.sharey() after
        # plotting inherits the first panel's limits without re-autoscaling, which
        # silently clips the other condition when the two differ by orders of
        # magnitude (they do, on the log defect axis).
        row_axes = axes[r]
        lo, hi = np.inf, -np.inf
        for c, augmented in enumerate((True, False)):
            ax = row_axes[c]
            d = sub[sub.augmented == augmented]
            for (li, _), color in zip(layers.itertuples(index=False), colors):
                s_ = (d[d.layer_idx == li].groupby("epoch", observed=True)["rbf_cka"]
                        .mean().sort_index())
                y = np.clip(1 - s_.to_numpy(), 1e-5, None) if use_defect else s_.to_numpy()
                ax.plot(s_.index.to_numpy(), y, color=color, lw=1.4, alpha=0.95)
                lo, hi = min(lo, y.min()), max(hi, y.max())
            ax.set_title("Rotation-augmented" if augmented else "Baseline",
                         fontsize=9, color=st.AUG if augmented else st.BASE)
            if c == 1:
                ax.tick_params(labelleft=False)
        for ax in row_axes:
            if use_defect:
                ax.set_yscale("log")
                ax.set_ylim(lo * 0.6, hi * 1.6)
            else:
                ax.set_ylim(0, 1.03)
        row_axes[0].set_ylabel(f"{st.TASK_LABEL[task]}\n{model} / {dataset}",
                               fontsize=8.5, color=st.INK)

    for ax in axes[-1]:
        ax.set_xlabel("Training epoch", fontsize=9)

    st.depth_colorbar(fig, axes.ravel().tolist())
    ylab = ("equivariance defect  1 - CKA  (log)" if use_defect else "RBF CKA")
    fig.suptitle("Equivariance emerges under augmentation and decays without it\n"
                 f"y: {ylab}   |   one line per layer, mean over 10 seeds{st.note(args)}",
                 fontsize=11.5, color=st.INK)
    st.save(fig, ("emergence_defect" if use_defect else "emergence") + st.suffix(args))


if __name__ == "__main__":
    ap = st.common_parser(__doc__)
    ap.add_argument("--defect", action="store_true")
    main(ap.parse_args())
