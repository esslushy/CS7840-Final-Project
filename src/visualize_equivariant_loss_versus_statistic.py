import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

def main(statistics_pth: Path, statistic: str):
    with statistics_pth.open() as f:
        statistics = json.load(f)

    num_layers = len(statistics["equivariant_losses"][0])
    num_epochs = len(statistics["equivariant_losses"])

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5), dpi=150, constrained_layout=True)
    if num_layers == 1:
        axes = [axes]

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]

    for idx, ax in enumerate(axes):
        ax.set_title(f"Layer {idx}")
        ax.set_xlabel(statistic.replace("_", " ").title())
        ax.set_ylim(bottom=0, top=1)
        if idx == 0:
            ax.set_ylabel("CKA (Higher is More Equivariant)")

        for jdx in range(num_epochs):
            ax.plot(statistics[statistic][jdx], statistics["equivariant_losses"][jdx][idx],
                    marker='o', c=colors[jdx])
            ax.plot(statistics[statistic][jdx], statistics["baseline_cka"][jdx][idx],
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
    plt.savefig(f"pdfs/{statistics_pth.stem}/equivariant_vs_test.pdf")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pth", help="The path where the statistics are stored.", type=Path)
    args.add_argument("statistic", help="The statistic to compare HSIC to.", type=str)
    args = args.parse_args()
    main(args.statistics_pth, args.statistic)