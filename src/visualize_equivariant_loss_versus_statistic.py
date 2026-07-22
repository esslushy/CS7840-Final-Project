import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np


def mean_cka_per_layer(epoch_dict):
    """
    epoch_dict maps layer -> {angle: compute_stats_dict}.
    Return a list of mean cka-scores (averaged over angles) per layer,
    in the layer order given by the dict keys.
    """
    means = []
    for layer, per_angle in epoch_dict.items():
        cka_vals = [stats["cka"] for stats in per_angle.values()]
        means.append(float(np.mean(cka_vals)))
    return means


def scalarize(value):
    """
    Reduce a saved statistic entry to a single scalar.

    test_loss / test_accuracy are now per-angle dicts {angle: value}; average
    over angles. Anything already scalar (e.g. train_loss) is returned as-is.
    """
    if isinstance(value, dict):
        return float(np.mean([float(v) for v in value.values()]))
    return float(value)


def main(statistics_pth: Path, statistic: str):
    with statistics_pth.open() as f:
        statistics = json.load(f)

    layer_names = list(statistics["equivariant_loss"][0].keys())
    num_layers = len(layer_names)
    num_epochs = len(statistics["equivariant_loss"])

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5), dpi=150, constrained_layout=True)
    if num_layers == 1:
        axes = [axes]

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]

    # Precompute mean-over-angle cka per epoch (list of per-layer lists)
    cka_per_epoch = [mean_cka_per_layer(statistics["equivariant_loss"][jdx]) for jdx in range(num_epochs)]
    # Reduce the comparison statistic to a scalar per epoch
    stat_per_epoch = [scalarize(statistics[statistic][jdx]) for jdx in range(num_epochs)]

    for idx, (layer_name, ax) in enumerate(zip(layer_names, axes)):
        ax.set_title(layer_name)
        ax.set_xlabel(statistic.replace("_", " ").title())
        ax.set_ylim(bottom=0, top=1)
        if idx == 0:
            ax.set_ylabel("Renyi2 MI CKA (mean over angles)")

        for jdx in range(num_epochs):
            ax.plot(stat_per_epoch[jdx], cka_per_epoch[jdx][idx],
                    marker='o', c=colors[jdx])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[], ax=axes)
    cbar.set_label('Number Training Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(num_epochs - 1), transform=cbar.ax.transAxes, va='bottom', ha='center')

    Path(f"pdfs/{statistics_pth.stem}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"pdfs/{statistics_pth.stem}/equivariant_vs_{statistic}_{statistics_pth.stem}.pdf")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pth", help="The path where the statistics are stored.", type=Path)
    args.add_argument("statistic", help="The statistic to compare equivariance cka to.", type=str)
    args = args.parse_args()
    main(args.statistics_pth, args.statistic)