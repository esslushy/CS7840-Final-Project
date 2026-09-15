"""
Where the two CKA variants disagree.

cka_gap (linear CKA - RBF CKA) is computed by EquivarianceTracker and stored in all
660 results files, and was never plotted: the old scripts' --stat choices were
rbf_cka / linear_cka / calibrated_sigma only. The gap is the sweep's own evidence
that the choice of similarity kernel changes the answer -- directly on-thesis for a
project arguing that these metrics are confounded.

Gap values live in roughly +/-0.05, so this needs a diverging scale centered on zero,
not the [0,1] the old scripts hardcoded.

Run from src/:  python figures/fig_metric_gap.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import style as st


def main(args):
    st.apply()
    df = st.at_final_epoch(st.apply_filters(st.load("cka_by_epoch"), args))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # --- left: gap vs depth, one line per config, colored by condition ---
    # Not colored by gap magnitude: the diverging ramp's midpoint is near-white by
    # design, which would make the many near-zero configs invisible on the surface.
    ax = axes[0]
    lim = float(np.nanpercentile(np.abs(df.cka_gap), 99.5))
    for (task, model, dataset, aug), sub in df.groupby(st.ID, observed=True):
        g = sub.groupby("depth_frac", observed=True)["cka_gap"].mean().sort_index()
        ax.plot(g.index, g.to_numpy(), color=st.AUG if aug else st.BASE,
                lw=1.2, alpha=0.55)
    ax.axhline(0, color=st.AXIS, lw=1.2, zorder=1)
    ax.set_xlabel("Normalized layer depth  (0 = input, 1 = output)", fontsize=9)
    ax.set_ylabel("linear CKA $-$ RBF CKA", fontsize=9)
    ax.set_ylim(-lim * 1.15, lim * 1.15)
    st.condition_legend(ax, loc="lower left", fontsize=8)
    ax.set_title("The two kernels agree at the input and diverge with depth",
                 fontsize=10, color=st.INK, loc="left")

    # --- right: does the disagreement depend on the condition? ---
    axr = axes[1]
    deep = df[df.layer_idx == df.groupby(st.ID, observed=True)["layer_idx"].transform("max")]
    tasks = [t for t in st.TASK_LABEL if t in set(df.task)]
    data, pos, colors = [], [], []
    for i, task in enumerate(tasks):
        for off, aug in ((0.19, True), (-0.19, False)):
            v = deep[(deep.task == task) & (deep.augmented == aug)]["cka_gap"].dropna()
            if len(v):
                data.append(v.to_numpy())
                pos.append(i + off)
                colors.append(st.AUG if aug else st.BASE)
    bp = axr.boxplot(data, positions=pos, vert=False, patch_artist=True, widths=0.32,
                     showfliers=False, medianprops=dict(color=st.INK, lw=1.1))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor(c)
    for key in ("whiskers", "caps"):
        for el in bp[key]:
            el.set_color(st.MUTED)
    axr.axvline(0, color=st.AXIS, lw=1.2)
    axr.set_yticks(range(len(tasks)))
    axr.set_yticklabels([st.TASK_LABEL[t] for t in tasks], fontsize=8)
    axr.set_ylim(-0.6, len(tasks) - 0.4)
    axr.set_xlabel("linear CKA $-$ RBF CKA  (deepest layer)", fontsize=9)
    axr.grid(axis="y", visible=False)
    st.condition_legend(axr, loc="upper right", fontsize=8)
    axr.set_title("Disagreement is largest where the conditions differ most",
                  fontsize=10, color=st.INK, loc="left")

    fig.suptitle("Metric disagreement: linear vs. RBF CKA on the same activations\n"
                 "stored in all 660 results files and never previously plotted",
                 fontsize=11.5, color=st.INK)
    st.save(fig, "metric_gap" + st.suffix(args))
    print(f"  |gap| 99.5th pct = {lim:.4f}; "
          f"max |gap| = {np.nanmax(np.abs(df.cka_gap)):.4f}")


if __name__ == "__main__":
    main(st.common_parser(__doc__).parse_args())
