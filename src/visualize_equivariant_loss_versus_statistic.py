import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

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

    for idx, (layer_name, ax) in enumerate(zip(layer_names, axes)):
        ax.set_title(layer_name)
        ax.set_xlabel(statistic.replace("_", " ").title())
        ax.set_ylim(bottom=0, top=1)
        if idx == 0:
            ax.set_ylabel("CKA (Higher is More Equivariant)")

        for jdx in range(num_epochs):
            eq_values = list(statistics["equivariant_loss"][jdx].values())
            bl_values = list(statistics["baseline_cka"][jdx].values())
            ax.plot(statistics[statistic][jdx], eq_values[idx],
                    marker='o', c=colors[jdx])
            ax.plot(statistics[statistic][jdx], bl_values[idx],
                    marker='x', c=colors[jdx], linestyle='none', alpha=0.5)

    # Legend entries for marker styles
    axes[-1].plot([], [], 'ko', label='Rotation CKA')
    axes[-1].plot([], [], 'kx', alpha=0.5, label='Unrelated Baseline')
    axes[-1].legend(loc='lower right')

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
    args.add_argument("statistic", help="The statistic to compare HSIC to.", type=str)
    args = args.parse_args()
    main(args.statistics_pth, args.statistic)