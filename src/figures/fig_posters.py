"""
One self-contained poster per config: a small panel per layer, epochs on x.

The sheets (`scissors`, `depth_profile`) give one panel per config and so can only
show the deepest layer or the final epoch. This is the full per-config view --
every layer's whole trajectory -- at one file per matched pair.

Layer gets its own panel rather than its own shade of one ramp. Overlaying 30
layers and distinguishing them by lightness asks the reader to invert a colour
ramp by eye, and it leaves no channel free for the thing the panel is actually
comparing. Faceting frees the two categorical colours for the comparison that
matters -- augmented vs. baseline -- and makes room for the spread: every panel
is the mean over the 10 seeds with a +/-1 s.d. band, which an overlay of 30
banded curves could not show at all.

y is CKA's full 0-1 range on every poster and x runs 0 to 200 or 400 depending on
the run, so panels are comparable across configs and not merely within one, and
nothing is read off a zoomed axis. The cost is that saturated configs are flat
lines near the frame top; their structure is real but only a few thousandths
tall, so it is invisible at this scale.

Run from src/:
    python figures/fig_posters.py                    # all 33
    python figures/fig_posters.py --no-isotropic     # 29
    python figures/fig_posters.py --only "colorization/unet"
"""
import matplotlib.pyplot as plt
import numpy as np

import style as st

MAX_COLS = 6
EPOCH_STEP = 200
COINCIDE_TOL = 2e-3


def grid_shape(n):
    """Rows and columns for n layer panels, always leaving one cell spare.

    The spare cell holds the legend. A figure-level legend is the obvious
    alternative but `loc="outside lower center"` does not account for
    `supxlabel`, so the two render on top of each other; widening the grid by one
    column is cheaper than fighting the layout engine.
    """
    ncols = min(MAX_COLS, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    if nrows * ncols == n:
        ncols += 1                      # one wider beats one taller
        nrows = int(np.ceil(n / ncols))
        if nrows * ncols == n:
            nrows += 1
    return nrows, ncols


def main(args):
    st.apply(args)
    df = st.apply_filters(st.load("cka_by_epoch"), args)

    # Mean and s.d. over the 10 seeds, per config x condition x layer x epoch.
    g = (df.groupby(st.ID + ["layer_idx", "layer", "epoch"], observed=True)[st.stat()]
           .agg(["mean", "std", "count"]).reset_index())

    configs = sorted(g[["task", "model", "dataset"]].drop_duplicates()
                      .itertuples(index=False, name=None))
    if args.only:
        configs = [c for c in configs if args.only in "/".join(c)]

    written = []
    for task, model, dataset in configs:
        sub = g[(g.task == task) & (g.model == model) & (g.dataset == dataset)]
        if sub.empty:
            continue
        layers = [(int(r.layer_idx), str(r.layer)) for r in
                  sub[["layer_idx", "layer"]].drop_duplicates()
                     .sort_values("layer_idx").itertuples(index=False)]
        n_layers = len(layers)
        # Long names (block0.attn.residual) need a smaller face than short ones
        # (conv1) to fit a ~1.75in panel without being clipped.
        longest = max(len(nm) for _, nm in layers)
        name_fs = 6.0 if longest > 14 else 6.7 if longest > 10 else 7.4

        # y is CKA's full range, the same on every poster, so panels are
        # comparable across configs and not just within one. An s.d. band that
        # crosses 1 is clipped, which is the right call for a metric bounded at 1.
        #
        # x snaps up to the next multiple of EPOCH_STEP. Runs are 200 epochs
        # except the two mnist_font configs at 400, so this gives every poster
        # one of two widths rather than a per-config axis.
        xmax = EPOCH_STEP * int(np.ceil(float(sub.epoch.max()) / EPOCH_STEP))

        nrows, ncols = grid_shape(n_layers)
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                 squeeze=False, constrained_layout=True,
                                 figsize=(max(1.75, 4.6 - 0.7 * ncols) * ncols + 0.9
                                          if ncols <= 3 else 1.75 * ncols + 0.9,
                                          1.45 * nrows + 0.5))

        for i, (li, lname) in enumerate(layers):
            ax = axes[i // ncols][i % ncols]
            means = {}
            for augmented, color in ((False, st.BASE), (True, st.AUG)):
                d = (sub[(sub.augmented == augmented) & (sub.layer_idx == li)]
                     .sort_values("epoch"))
                if d.empty:
                    continue
                x = d.epoch.to_numpy()
                m = d["mean"].to_numpy()
                sd = np.nan_to_num(d["std"].to_numpy())
                ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.20,
                                linewidth=0, zorder=2)
                ax.plot(x, m, color=color, lw=1.3, zorder=3)
                means[augmented] = m
            # The module's own layer name, nothing else. Panels run in depth
            # order left to right, so "layer 7 (readout)" spent a title on
            # information the reading order already carries.
            ax.set_title(lname, fontsize=name_fs, color=st.INK, pad=2.5)
            ax.tick_params(labelsize=6.5)
            # Where the conditions coincide the later-drawn line hides the other
            # and the panel reads as a single series. Say so rather than let the
            # reader conclude the baseline is missing.
            if (len(means) == 2 and len(means[True]) == len(means[False])
                    and np.max(np.abs(means[True] - means[False])) < COINCIDE_TOL):
                ax.annotate("curves coincide", (0.5, 0.06), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=6.0, color=st.MUTED)

        spare = list(range(n_layers, nrows * ncols))
        for j in spare:
            axes[j // ncols][j % ncols].set_axis_off()

        # sharex hides tick labels on every panel that is not in the bottom row,
        # but the last row is partly empty, so the columns that end one row early
        # would carry no x scale at all. Re-enable them on each column's lowest
        # visible panel.
        for c in range(ncols):
            last = max((i for i in range(n_layers) if i % ncols == c), default=None)
            if last is not None:
                axes[last // ncols][c].tick_params(labelbottom=True)

        axes[0][0].set_ylim(0, 1)
        axes[0][0].set_xlim(0, xmax)

        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=st.AUG, lw=2.0, label="Rotation-augmented"),
                   Line2D([], [], color=st.BASE, lw=2.0,
                          label="Baseline (no augmentation)")]
        # grid_shape guarantees at least one spare cell.
        lax = axes[spare[0] // ncols][spare[0] % ncols]
        lax.legend(handles=handles, loc="center", fontsize=7.5, frameon=False)

        # No figure title. The filename identifies the config, and a poster
        # dropped into a document gets its identity from the caption there.
        fig.supxlabel("epoch", fontsize=8.5, color=st.INK_2)
        fig.supylabel(st.stat_label(), fontsize=8.5, color=st.INK_2)

        # No `--no-isotropic` suffix: a poster is built from one config's data
        # alone, so the filter changes which posters exist, not what any of them
        # contains. `--rbf` is kept, since that does change the content.
        rbf = "_rbf" if getattr(args, "rbf", False) else ""
        written.append(st.save(fig, f"{task}__{model}__{dataset}{rbf}"))

    if len(written) <= 3:
        for path in written:
            print(f"  wrote {path}")
    print(f"\n  {len(written)} poster{'s' if len(written) != 1 else ''} in {st.OUT}")


if __name__ == "__main__":
    ap = st.common_parser(__doc__)
    ap.add_argument("--only", default=None,
                    help="substring match on task/model/dataset, for one poster")
    main(ap.parse_args())
