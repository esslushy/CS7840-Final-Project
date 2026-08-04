import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from utils import strip_seed_suffix, mean_cka_per_layer


def scalarize(value):
    """
    Reduce a saved statistic entry to a single scalar.

    test_loss / test_accuracy are now per-angle dicts {angle: value}; average
    over angles. Anything already scalar (e.g. train_loss) is returned as-is.
    """
    if isinstance(value, dict):
        return float(np.mean([float(v) for v in value.values()]))
    return float(value)


def main(statistics_pths: list, train_statistic: str, stat: str):
    seeds_statistics = []
    for pth in statistics_pths:
        with pth.open() as f:
            seeds_statistics.append(json.load(f))

    num_epochs = min(len(s["equivariant_loss"]) for s in seeds_statistics)
    if any(len(s["equivariant_loss"]) != num_epochs for s in seeds_statistics):
        print(f"Warning: seeded runs have differing epoch counts "
              f"({[len(s['equivariant_loss']) for s in seeds_statistics]}); truncating to {num_epochs}.")

    layer_names = list(seeds_statistics[0]["equivariant_loss"][0].keys())
    num_layers = len(layer_names)

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5), dpi=150, constrained_layout=True)
    if num_layers == 1:
        axes = [axes]

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]
    multi_seed = len(statistics_pths) > 1

    # Precompute mean-over-angle cka per epoch, per seed (num_epochs, num_seeds, num_layers)
    cka_per_epoch = [[mean_cka_per_layer(s["equivariant_loss"][jdx], stat) for s in seeds_statistics]
                      for jdx in range(num_epochs)]
    # Reduce the comparison statistic to a scalar per epoch, per seed (num_epochs, num_seeds)
    stat_per_epoch = [[scalarize(s[train_statistic][jdx]) for s in seeds_statistics]
                       for jdx in range(num_epochs)]

    for idx, (layer_name, ax) in enumerate(zip(layer_names, axes)):
        ax.set_title(layer_name)
        ax.set_xlabel(train_statistic.replace("_", " ").title())
        ax.set_ylim(bottom=0, top=1)
        if idx == 0:
            ax.set_ylabel(stat.replace("_", " "))

        for jdx in range(num_epochs):
            xs = np.array(stat_per_epoch[jdx])
            ys = np.array([per_layer[idx] for per_layer in cka_per_epoch[jdx]])
            if multi_seed:
                ax.errorbar(xs.mean(), ys.mean(), xerr=xs.std(), yerr=ys.std(),
                           marker='o', c=colors[jdx], ecolor=colors[jdx], alpha=0.7, capsize=2)
            else:
                ax.plot(xs[0], ys[0], marker='o', c=colors[jdx])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[], ax=axes)
    cbar.set_label('Number Training Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(num_epochs - 1), transform=cbar.ax.transAxes, va='bottom', ha='center')

    if multi_seed:
        fig.suptitle(f"Mean ± std over {len(statistics_pths)} seeds")

    base_stem = strip_seed_suffix(statistics_pths[0].stem)
    seed_suffix = f"_n{len(statistics_pths)}seeds" if multi_seed else ""
    Path(f"pdfs/{base_stem}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"pdfs/{base_stem}/equivariant_vs_{train_statistic}_{base_stem}{seed_suffix}.pdf")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pths", help="The path(s) where the statistics are stored. Pass multiple "
                                               "seeded runs of the same experiment to plot mean +/- std across seeds.",
                       type=Path, nargs="+")
    args.add_argument("train_statistic", help="The statistic to compare equivariance cka to.", type=str)
    args.add_argument("--stat", help="The statistic to show against train statistic", choices=["rbf_cka", "linear_cka", "calibrated_sigma"])
    args = args.parse_args()
    main(args.statistics_pths, args.train_statistic, args.stat)