"""
Main result: measured equivariance vs. network depth, augmented vs. baseline.

Both conditions start near 1.0 at the input (a rotated image is still an image,
and early conv features are nearly rotation-covariant) and fan apart with depth.
This comparison appears nowhere in the old per-config figure set, because the two
conditions were always written to different directories.

Run from src/:
    python figures/fig_depth_profile.py            # CKA on a linear axis
    python figures/fig_depth_profile.py --defect   # 1-CKA on a log axis
"""
import matplotlib.pyplot as plt
import numpy as np

import style as st


def panel(ax, sub, use_defect):
    for augmented, color, label in ((True, st.AUG, "Augmented"), (False, st.BASE, "Baseline")):
        d = sub[sub.augmented == augmented]
        if d.empty:
            continue
        # mean +/- std across the 10 seeds, per layer
        g = d.groupby("depth_frac", observed=True)["rbf_cka"]
        x = np.array(sorted(d.depth_frac.unique()))
        mean = g.mean().reindex(x).to_numpy()
        sd = g.std().reindex(x).to_numpy()
        lo, hi = mean - sd, mean + sd
        if use_defect:
            mean, lo, hi = (np.clip(1 - v, 1e-5, None) for v in (mean, hi, lo))
        ax.plot(x, mean, color=color, marker="o", markersize=3.5, zorder=3, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
    if use_defect:
        ax.set_yscale("log")


def main(args):
    use_defect = args.defect
    st.apply()
    df = st.at_final_epoch(st.apply_filters(st.load("cka_by_epoch"), args))

    tasks = [t for t in st.TASK_LABEL if t in set(df.task)]
    # one column per model within a task -> grid of task x model, but models differ
    # per task, so lay out as one row per task with a panel per (model, dataset).
    combos = (df[st.ID].drop_duplicates()
                .assign(md=lambda d: d.model.astype(str) + " / " + d.dataset.astype(str))
                .groupby("task", observed=True)["md"].unique().to_dict())
    ncols = max(len(v) for v in combos.values())
    nrows = len(tasks)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.1 * nrows),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)
    for r, task in enumerate(tasks):
        mds = sorted(combos[task])
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(mds):
                ax.set_visible(False)
                continue
            model, dataset = mds[c].split(" / ")
            sub = df[(df.task == task) & (df.model == model) & (df.dataset == dataset)]
            panel(ax, sub, use_defect)
            ax.set_title(mds[c], fontsize=8, color=st.INK)
            if c == 0:
                # Row label carries the task; no supylabel, which would collide.
                ax.set_ylabel(st.TASK_LABEL[task], fontsize=8.5, color=st.INK)

    if not use_defect:
        axes[0][0].set_ylim(0, 1.03)

    # Figure-level legend: a per-axes legend would land on a hidden panel,
    # since tasks have different numbers of model/dataset combos.
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=st.AUG, lw=2.0, marker="o", ms=4,
                               label="Rotation-augmented"),
                        Line2D([], [], color=st.BASE, lw=2.0, marker="o", ms=4,
                               label="Baseline (no augmentation)")],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), ncols=2, fontsize=9)

    ylab = ("equivariance defect  1 - CKA  (log axis)" if use_defect
            else "RBF CKA between clean and rotated activations")
    fig.suptitle("Measured equivariance across depth, at the final epoch\n"
                 f"mean $\\pm$ 1 s.d. over 10 seeds   |   y: {ylab}{st.note(args)}",
                 fontsize=11.5, color=st.INK)
    fig.supxlabel("Normalized layer depth  (0 = input, 1 = output)",
                  fontsize=9.5, color=st.INK_2)
    st.save(fig, ("depth_profile_defect" if use_defect else "depth_profile") + st.suffix(args))


if __name__ == "__main__":
    ap = st.common_parser(__doc__)
    ap.add_argument("--defect", action="store_true",
                    help="plot 1-CKA on a log axis (spreads the saturated configs)")
    main(ap.parse_args())
