import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from utils import strip_seed_suffix, mean_cka_per_layer


def main(statistics_pths: list, stat: str):
    seeds_equivariant_loss = []
    for pth in statistics_pths:
        with pth.open() as f:
            seeds_equivariant_loss.append(json.load(f)["equivariant_loss"])

    n_epochs = min(len(el) for el in seeds_equivariant_loss)
    if any(len(el) != n_epochs for el in seeds_equivariant_loss):
        print(f"Warning: seeded runs have differing epoch counts "
              f"({[len(el) for el in seeds_equivariant_loss]}); truncating to {n_epochs}.")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Use keys from the first seed's first epoch dict as layer names
    layer_names = list(seeds_equivariant_loss[0][0].keys())
    ax.set_xlabel("Layer")
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel(stat.replace("_", " "))

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, n_epochs)]

    # Plot mean-over-seed cka-score per layer for each epoch, shaded by std across seeds
    for i in range(n_epochs):
        # (num_seeds, num_layers)
        per_seed_values = np.array([mean_cka_per_layer(el[i], stat) for el in seeds_equivariant_loss])
        means = per_seed_values.mean(axis=0)
        stds = per_seed_values.std(axis=0)
        layers = range(len(means))
        ax.plot(layers, means, marker='o', c=colors[i], alpha=0.7)
        if len(statistics_pths) > 1:
            ax.fill_between(layers, means - stds, means + stds, color=colors[i], alpha=0.15, linewidth=0)

    # Colorbar for epochs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[], ax=ax)
    cbar.set_label('Number Training Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(n_epochs - 1), transform=cbar.ax.transAxes, va='bottom', ha='center')

    if len(statistics_pths) > 1:
        fig.suptitle(f"Mean ± std across {len(statistics_pths)} seeds")

    fig.tight_layout()

    base_stem = strip_seed_suffix(statistics_pths[0].stem)
    seed_suffix = f"_n{len(statistics_pths)}seeds" if len(statistics_pths) > 1 else ""
    Path(f"pdfs/{base_stem}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"pdfs/{base_stem}/equivariant_loss_{base_stem}{seed_suffix}.pdf")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pths", help="The path(s) where the statistics are stored. Pass multiple "
                                               "seeded runs of the same experiment to plot mean +/- std across seeds.",
                       type=Path, nargs="+")
    args.add_argument("--stat", help="The statistic to show", choices=["rbf_cka", "linear_cka", "calibrated_sigma"])
    args = args.parse_args()
    main(args.statistics_pths, args.stat)