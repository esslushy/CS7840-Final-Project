import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np


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


def main(statistics_pth: Path, statistic: str, field: str):
    with statistics_pth.open() as f:
        statistics = json.load(f)

    equivariant_loss = statistics["equivariant_loss"]
    num_epochs = len(equivariant_loss)

    layer_names = list(equivariant_loss[0].keys())
    num_layers = len(layer_names)
    angle_keys = sorted(next(iter(equivariant_loss[0].values())).keys(), key=lambda a: float(a))
    num_angles = len(angle_keys)

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]

    # rows = angles (down the page), columns = layers (across the page)
    fig, axes = plt.subplots(num_angles, num_layers,
                             figsize=(max(4, 3 * num_layers), max(3, 2.8 * num_angles)),
                             dpi=150, sharey=True, squeeze=False, constrained_layout=True)

    stat_title = statistic.replace("_", " ").title()

    for i, angle in enumerate(angle_keys):
        for j, layer in enumerate(layer_names):
            ax = axes[i][j]
            for ep in range(num_epochs):
                x = stat_value(statistics[statistic][ep], angle)
                y = float(equivariant_loss[ep][layer][angle][field])
                ax.plot(x, y, marker='o', c=colors[ep])
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

    fig.suptitle(f"{field} vs {stat_title}, per layer (cols) and angle (rows)  ({statistics_pth.stem})")

    Path(f"pdfs/{statistics_pth.stem}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"pdfs/{statistics_pth.stem}/equivariant_vs_{statistic}_per_angle_{statistics_pth.stem}.pdf")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pth", help="The path where the statistics are stored.", type=Path)
    args.add_argument("train_statistic", help="The statistic to compare the equivariance field to.", type=str)
    args.add_argument("--stat", help="The statistic to show against train statistic", choices=["rbf_cka", "linear_cka", "calibrated_sigma"])
    args = args.parse_args()
    main(args.statistics_pth, args.train_statistic, args.stat)