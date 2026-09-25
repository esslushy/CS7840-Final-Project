"""
Baseline and augmented linear CKA for every configuration, on one axis.

The per-config posters show every layer and every epoch for one configuration.
This is the complement: two numbers per configuration, so all 29 matched pairs
are visible at once. Nothing is pooled ACROSS configurations -- each column is one
configuration, and the only aggregation is over the 10 seeds inside it, which
are genuine replicates. It is therefore not the sweep_summary sheet removed on
2026-09-18, which averaged across configurations and so mixed architectures
with different layer counts and feature dimensionalities.

WHAT EXACTLY IS PLOTTED, since a single CKA number could mean several things:

  layer    the DEEPEST layer only (layer_idx == n_layers - 1). Not an average
           over layers -- layers differ enormously within a network and an
           average over them has no referent.
  epoch    the FINAL epoch.
  angles   averaged over the probe angles, which is what cka_by_epoch caches:
           the three non-identity C4 rotations for grid tasks, 16 evenly
           spaced SO(2) elements for particle tasks.
  seeds    median over the 10 seeds, with the interquartile range as the bar.

Columns are sorted within each task by the BASELINE value, so the ceiling is
visible as structure: configurations that start near 1.0 have nowhere to go,
and their short connectors are headroom rather than absence of effect.

Run: cd src && python figures/fig_sweep_effects.py [--no-isotropic]
"""
import pandas as pd
import matplotlib.pyplot as plt

import style


def summarise(df):
    """Per-config baseline and augmented CKA at the deepest layer, final epoch,
    summarised over seeds."""
    df = df[df.layer_idx == df.n_layers - 1]
    last = df.groupby(["task", "model", "dataset"], observed=True)["epoch"].transform("max")
    df = df[df.epoch == last]

    piv = df.pivot_table(index=["task", "model", "dataset", "seed"],
                         columns="augmented", values=style.stat(),
                         observed=True).dropna()

    rows = []
    for (t, m, d), g in piv.groupby(level=[0, 1, 2], observed=True):
        rows.append(dict(
            task=t, label=f"{m}/{d}", n=len(g),
            base=g[False].median(),
            base_lo=g[False].quantile(.25), base_hi=g[False].quantile(.75),
            aug=g[True].median(),
            aug_lo=g[True].quantile(.25), aug_hi=g[True].quantile(.75)))
    return pd.DataFrame(rows)


def main():
    ap = style.common_parser(__doc__)
    args = ap.parse_args()
    style.apply(args)

    r = summarise(style.apply_filters(style.load("cka_by_epoch"), args))
    r = r.sort_values(["task", "base"], ascending=[True, True])

    xpos, labels, seps, heads = [], [], [], []
    x = 0.0
    groups = list(r.groupby("task", observed=True, sort=False))
    for gi, (task, g) in enumerate(groups):
        heads.append((x + (len(g) - 1) / 2, style.TASK_LABEL.get(task, task)))
        for _ in range(len(g)):
            xpos.append(x); x += 1
        labels.extend(g.label.tolist())
        if gi < len(groups) - 1:
            seps.append(x - 0.5 + 0.8)
        x += 1.6
    x -= 1.6
    r = r.assign(x=xpos)

    fig, ax = plt.subplots(figsize=(0.30 * x + 2.0, 5.2))
    for s in seps:
        ax.axvline(s, color=style.GRID, lw=0.6, zorder=0)

    for _, p in r.iterrows():
        ax.plot([p.x, p.x], [p.base, p.aug], color=style.MUTED, lw=1.2,
                alpha=0.8, zorder=1, solid_capstyle="round")
        for lo, hi, val, c in [(p.base_lo, p.base_hi, p.base, style.BASE),
                               (p.aug_lo, p.aug_hi, p.aug, style.AUG)]:
            ax.plot([p.x, p.x], [lo, hi], color=c, lw=3.2, alpha=0.40,
                    solid_capstyle="butt", zorder=2)
            ax.plot([p.x], [val], "o", ms=6, color=c, zorder=3)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8, rotation=60, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.8, x + 0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"{style.stat_label()}\nat the deepest layer, final epoch")

    for xtxt, name in heads:
        ax.annotate(name, xy=(xtxt, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5,
                    color=style.INK, fontweight="bold")

    handles = [plt.Line2D([], [], color=style.BASE, marker="o", ls="none",
                          ms=6, label="baseline (no augmentation)"),
               plt.Line2D([], [], color=style.AUG, marker="o", ls="none",
                          ms=6, label="rotation-augmented")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.5,
               frameon=False)

    fig.subplots_adjust(bottom=0.30, top=0.86)
    path = style.save(fig, "sweep_effects")
    up = int((r.aug > r.base).sum())
    print(f"wrote {path}   {len(r)} configs, augmented higher in {up}/{len(r)}")


if __name__ == "__main__":
    main()
