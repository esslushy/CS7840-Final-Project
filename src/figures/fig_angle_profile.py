"""
How equivariance varies with rotation angle.

Replaces the old per-angle grids, which gave the 16-angle particle configs a
16 x 14 = 224-panel figure to display what is a single smooth curve: CKA decays
monotonically with angular distance from the identity and is symmetric about 180
degrees. Angle is a circular variable, so a linear axis invents a discontinuity at
the wraparound; a polar axis does not.

The 46 three-angle (C4) configs get a plain categorical panel instead -- a polar
plot of three points is not worth the ink.

Run from src/:  python figures/fig_angle_profile.py
"""
import matplotlib.pyplot as plt
import numpy as np

import style as st


def main(args):
    st.apply()
    df = st.apply_filters(st.load("cka_by_angle"), args)
    df = df[df.epoch == st.final_epochs(df)]

    n_ang = df.groupby(st.ID, observed=True)["angle"].transform("nunique")
    so2 = df[n_ang == 16]
    configs = sorted(so2[["task", "model", "dataset"]].drop_duplicates()
                        .itertuples(index=False, name=None))

    ncols = 5
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3.3 * nrows),
                             subplot_kw={"projection": "polar"}, squeeze=False,
                             constrained_layout=True)

    for i, (task, model, dataset) in enumerate(configs):
        ax = axes[i // ncols][i % ncols]
        sub = so2[(so2.task == task) & (so2.model == model) & (so2.dataset == dataset)
                  & (so2.augmented)]
        layers = sorted(sub.layer_idx.unique())
        colors = st.depth_colors(len(layers))
        for li, color in zip(layers, colors):
            s = (sub[sub.layer_idx == li].groupby("angle", observed=True)["rbf_cka"]
                   .mean().sort_index())
            theta = np.deg2rad(s.index.to_numpy())
            r = s.to_numpy()
            # close the curve so the circular structure reads as continuous
            theta = np.append(theta, theta[0] + 2 * np.pi)
            r = np.append(r, r[0])
            ax.plot(theta, r, color=color, lw=1.3)
        # Radial axis zoomed to the data: CKA spans only ~0.85-1.0 here, so a
        # 0->1 radial axis renders every panel as a featureless circle.
        lo, hi = sub.rbf_cka.min(), sub.rbf_cka.max()
        pad = max((hi - lo) * 0.12, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks(np.round(np.linspace(lo, hi, 3), 3))
        ax.set_title(f"{st.TASK_LABEL[task]}\n{model} / {dataset}\n"
                     f"CKA {lo:.3f}-{hi:.3f}", fontsize=7.5,
                     color=st.INK, pad=16)
        ax.set_theta_zero_location("E")
        ax.set_rlabel_position(135)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=7)
        ax.grid(color=st.GRID, lw=0.6)

    for j in range(len(configs), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    st.depth_colorbar(fig, axes.ravel().tolist())
    fig.suptitle("Equivariance decays smoothly with rotation angle, symmetric about 180°\n"
                 "radius = RBF CKA, one trace per layer · rotation-augmented models, "
                 "final epoch, mean over 10 seeds\n"
                 "radial axis is zoomed per panel to the range printed in each title",
                 fontsize=10.5, color=st.INK)
    st.save(fig, "angle_profile_polar" + st.suffix(args))

    # --- C4 configs: categorical panel ---
    c4 = df[n_ang == 3]
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for augmented, color, lbl in ((True, st.AUG, "Rotation-augmented"),
                                  (False, st.BASE, "Baseline")):
        d = c4[c4.augmented == augmented]
        # deepest layer only, one line per task
        deep = d[d.layer_idx == d.groupby(st.ID, observed=True)["layer_idx"].transform("max")]
        g = deep.groupby("angle", observed=True)["rbf_cka"].agg(["mean", "std"])
        ax2.errorbar(g.index, g["mean"], yerr=g["std"], color=color, marker="o",
                     capsize=3, lw=2, label=lbl)
    ax2.set_xticks([90, 180, 270])
    ax2.set_xticklabels(["90°", "180°", "270°"])
    ax2.set_xlabel("Rotation angle", fontsize=9)
    ax2.set_ylabel("RBF CKA at deepest layer", fontsize=9)
    ax2.set_ylim(0, 1.03)
    ax2.legend(fontsize=9)
    # 90 and 270 come out equal (0.732 / 0.731 baseline, 0.918 / 0.916 augmented)
    # and 180 sits above both -- a half-turn is the easiest non-identity C4 element,
    # and the gap is ~4x larger for the baseline than for the augmented models.
    ax2.set_title("C4 (grid) tasks: 90° and 270° are equivalent, 180° is easier\n"
                  "deepest layer, pooled over 23 configs per condition x 10 seeds, "
                  "mean $\\pm$ 1 s.d.",
                  fontsize=11, color=st.INK, loc="left")
    st.save(fig2, "angle_profile_c4" + st.suffix(args))


if __name__ == "__main__":
    main(st.common_parser(__doc__).parse_args())
