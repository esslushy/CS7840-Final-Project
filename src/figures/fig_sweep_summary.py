"""
The whole sweep in one figure: all 33 matched pairs, ranked by augmentation effect.

Replaces browsing 654 per-config PDFs. Each row is one (task, model, dataset)
config; the dot pair shows deepest-layer CKA at the final epoch without and with
rotation augmentation, and the connector length is the effect.

Run from src/:  python figures/fig_sweep_summary.py
"""
import matplotlib.pyplot as plt
import numpy as np

import style as st


def main(args):
    st.apply()
    df = st.at_final_epoch(st.apply_filters(st.load("cka_by_epoch"), args))
    # Deepest layer = the config's readout; where the conditions differ most.
    deep = df[df.layer_idx == df.groupby(st.ID, observed=True)["layer_idx"].transform("max")]
    g = (deep.groupby(st.ID, observed=True)["rbf_cka"]
             .agg(["mean", "std"]).reset_index())

    piv = g.pivot_table(index=["task", "model", "dataset"], columns="augmented",
                        values=["mean", "std"], observed=True).dropna()
    piv["delta"] = piv[("mean", True)] - piv[("mean", False)]
    piv = piv.sort_values("delta")

    labels = [f"{t.replace('_',' ')} · {m} · {d}" for t, m, d in piv.index]
    y = np.arange(len(piv))
    a = piv[("mean", True)].to_numpy()
    b = piv[("mean", False)].to_numpy()

    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(10.6, 0.30 * len(piv) + 2.0), sharey=True,
        gridspec_kw={"width_ratios": [2.4, 1]}, constrained_layout=True)

    # --- left: dumbbell, absolute CKA ---
    for yi, bi, ai in zip(y, b, a):
        ax.plot([bi, ai], [yi, yi], color=(st.AUG if ai >= bi else "#d03b3b"),
                lw=2.0, alpha=0.55, zorder=2, solid_capstyle="round")
    ax.scatter(b, y, s=42, color=st.BASE, zorder=3, label="Baseline",
               edgecolor=st.SURFACE, lw=1.2)
    ax.scatter(a, y, s=42, color=st.AUG, zorder=4, label="Rotation-augmented",
               edgecolor=st.SURFACE, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.8, len(piv) - 0.2)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("RBF CKA at deepest layer, final epoch", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.035), fontsize=8.5, ncols=2)

    # --- right: the effect itself ---
    # Many pairs saturate near 1.0, where the two dots coincide and the dumbbell
    # degenerates to a single mark. A symlog delta axis keeps those rows readable:
    # it is linear inside +/-0.01 and logarithmic outside, so +0.0004 and +0.65
    # can share one axis without either vanishing.
    delta = piv["delta"].to_numpy()
    err = np.sqrt(piv[("std", True)].to_numpy() ** 2 + piv[("std", False)].to_numpy() ** 2)
    axd.barh(y, delta, height=0.6, color=[st.AUG if v >= 0 else "#d03b3b" for v in delta],
             zorder=2)
    axd.errorbar(delta, y, xerr=err, fmt="none", ecolor=st.MUTED, elinewidth=0.8,
                 capsize=1.5, zorder=3)
    axd.axvline(0, color=st.AXIS, lw=1.0, zorder=1)
    axd.set_xscale("symlog", linthresh=0.01, linscale=0.5)
    axd.set_xlim(-1.2, 1.2)
    axd.set_xticks([-1, -0.1, 0, 0.1, 1])
    axd.set_xticklabels(["-1", "-0.1", "0", "+0.1", "+1"], fontsize=8)
    axd.set_xlabel("$\\Delta$ CKA  (augmented $-$ baseline, symlog)", fontsize=9)
    axd.grid(axis="y", visible=False)

    n_pos = int((piv["delta"] > 0).sum())
    fig.suptitle(f"Rotation augmentation raises measured equivariance in "
                 f"{n_pos} of {len(piv)} matched pairs\n"
                 f"sorted by effect size; red marks the exceptions, "
                 f"error bars are 1 s.d. over 10 seeds",
                 fontsize=11, color=st.INK)
    st.save(fig, "sweep_summary" + st.suffix(args))

    print(f"\n  most negative: {labels[0]}  delta={piv['delta'].iloc[0]:+.3f}")
    print(f"  most positive: {labels[-1]}  delta={piv['delta'].iloc[-1]:+.3f}")


if __name__ == "__main__":
    main(st.common_parser(__doc__).parse_args())
