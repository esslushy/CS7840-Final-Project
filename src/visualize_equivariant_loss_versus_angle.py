import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from utils import strip_seed_suffix


def stat_value(entry, angle):
    """
    Reduce one epoch's statistic entry to a scalar x-value for a given angle.

    If the statistic is per-angle (e.g. test_loss = {angle: value}), use the
    matching angle when present (falling back to the mean over angles if that
    angle key is absent, e.g. the '0' unrotated entry has no match). If it is
    already scalar (e.g. train_loss), return it unchanged.
    """
    if isinstance(entry, dict):
        if angle in entry:
            return float(entry[angle])
        return float(np.mean([float(v) for v in entry.values()]))
    return float(entry)


def main(statistics_pths: list, statistic: str, field: str):
    seeds_statistics = []
    for pth in statistics_pths:
        with pth.open() as f:
            seeds_statistics.append(json.load(f))

    num_epochs = min(len(s["equivariant_loss"]) for s in seeds_statistics)
    if any(len(s["equivariant_loss"]) != num_epochs for s in seeds_statistics):
        print(f"Warning: seeded runs have differing epoch counts "
              f"({[len(s['equivariant_loss']) for s in seeds_statistics]}); truncating to {num_epochs}.")

    equivariant_loss0 = seeds_statistics[0]["equivariant_loss"]
    layer_names = list(equivariant_loss0[0].keys())
    num_layers = len(layer_names)
    angle_keys = sorted(next(iter(equivariant_loss0[0].values())).keys(), key=lambda a: float(a))
    num_angles = len(angle_keys)

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]

    # rows = angles (down the page), columns = layers (across the page)
    fig, axes = plt.subplots(num_angles, num_layers,
                             figsize=(max(4, 3 * num_layers), max(3, 2.8 * num_angles)),
                             dpi=150, sharey=True, squeeze=False, constrained_layout=True)

    stat_title = statistic.replace("_", " ").title()
    multi_seed = len(statistics_pths) > 1

    for i, angle in enumerate(angle_keys):
        for j, layer in enumerate(layer_names):
            ax = axes[i][j]
            for ep in range(num_epochs):
                xs = np.array([stat_value(s[statistic][ep], angle) for s in seeds_statistics])
                ys = np.array([float(s["equivariant_loss"][ep][layer][angle][field]) for s in seeds_statistics])
                if multi_seed:
                    ax.errorbar(xs.mean(), ys.mean(), xerr=xs.std(), yerr=ys.std(),
                               marker='o', c=colors[ep], ecolor=colors[ep], alpha=0.7, capsize=2)
                else:
                    ax.plot(xs[0], ys[0], marker='o', c=colors[ep])
            if field != "calibrated_sigma":
                ax.set_ylim(bottom=0, top=1)
            if i == 0:
                ax.set_title(layer, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{float(angle):g}°\n{field}")
            if i == num_angles - 1:
                ax.set_xlabel(stat_title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = fig.colorbar(sm, ticks=[], ax=axes.ravel().tolist())
    cbar.set_label('Number Training Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(num_epochs - 1), transform=cbar.ax.transAxes, va='bottom', ha='center')

    base_stem = strip_seed_suffix(statistics_pths[0].stem)
    title_suffix = f"  (mean ± std over {len(statistics_pths)} seeds)" if multi_seed else f"  ({base_stem})"
    fig.suptitle(f"{field} vs {stat_title}, per layer (cols) and angle (rows){title_suffix}")

    seed_suffix = f"_n{len(statistics_pths)}seeds" if multi_seed else ""
    Path(f"pdfs/{base_stem}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"pdfs/{base_stem}/equivariant_vs_{statistic}_per_angle_{field}_{base_stem}{seed_suffix}.pdf")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pths", help="The path(s) where the statistics are stored. Pass multiple "
                                               "seeded runs of the same experiment to plot mean +/- std across seeds.",
                       type=Path, nargs="+")
    args.add_argument("train_statistic", help="The statistic to compare the equivariance field to.", type=str)
    args.add_argument("--stat", help="The statistic to show against train statistic", choices=["rbf_cka", "linear_cka", "calibrated_sigma"])
    args = args.parse_args()
    main(args.statistics_pths, args.train_statistic, args.stat)