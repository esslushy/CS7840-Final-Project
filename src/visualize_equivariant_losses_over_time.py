import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np


def mean_cka_per_layer(epoch_dict, stat):
    """
    epoch_dict maps layer -> {angle: compute_stats_dict}.
    Return a list of mean cka-scores (averaged over angles) per layer,
    in the layer order given by the dict keys.
    """
    means = []
    for layer, per_angle in epoch_dict.items():
        cka_vals = [stats[stat] for stats in per_angle.values()]
        means.append(float(np.mean(cka_vals)))
    return means


def main(statistics_pth: Path, stat: str):
    with statistics_pth.open() as f:
        data = json.load(f)

    equivariant_loss = data["equivariant_loss"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Use keys from the first epoch's dict as layer names
    layer_names = list(equivariant_loss[0].keys())
    ax.set_xlabel("Layer")
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel(stat.replace("_", " "))

    n_epochs = len(equivariant_loss)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, n_epochs)]

    # Plot mean-over-angle cka-score per layer for each epoch
    for i in range(n_epochs):
        values = mean_cka_per_layer(equivariant_loss[i], stat)
        layers = range(len(values))
        ax.plot(layers, values, marker='o', c=colors[i], alpha=0.7)

    # Colorbar for epochs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[], ax=ax)
    cbar.set_label('Number Training Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(n_epochs - 1), transform=cbar.ax.transAxes, va='bottom', ha='center')

    fig.tight_layout()

    Path(f"pdfs/{statistics_pth.stem}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"pdfs/{statistics_pth.stem}/equivariant_loss_{statistics_pth.stem}.pdf")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pth", help="The path where the statistics are stored.", type=Path)
    args.add_argument("--stat", help="The statistic to show", choices=["rbf_cka", "linear_cka", "calibrated_sigma"])
    args = args.parse_args()
    main(args.statistics_pth, args.stat)