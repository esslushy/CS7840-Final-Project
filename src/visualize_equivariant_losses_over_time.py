import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

def main(statistics_pth: Path):
    with statistics_pth.open() as f:
        data = json.load(f)

    equivariant_losses = data["equivariant_losses"]
    cka_baseline = data["cka_baseline"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(len(equivariant_losses[0])))
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel("CKA (Higher is More Equivariant)")

    n_epochs = len(equivariant_losses)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, n_epochs)]

    # Plot equivariant CKA for each epoch
    for i in range(n_epochs):
        layers = range(len(equivariant_losses[i]))
        ax.plot(layers, equivariant_losses[i], marker='o', c=colors[i], alpha=0.7)

    # Plot baseline for each epoch with dashed lines, same color scheme
    for i in range(n_epochs):
        layers = range(len(cka_baseline[i]))
        ax.plot(layers, cka_baseline[i], marker='x', c=colors[i],
                linestyle='--', alpha=0.5, linewidth=1)

    # Colorbar for epochs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[], ax=ax)
    cbar.set_label('Number Training Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(n_epochs - 1), transform=cbar.ax.transAxes, va='bottom', ha='center')

    # Legend entries for line styles
    ax.plot([], [], 'k-', marker='o', label='Rotation CKA')
    ax.plot([], [], 'k--', marker='x', label='Unrelated Baseline')
    ax.legend(loc='lower right')

    Path(f"pdfs/{statistics_pth.stem}").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"pdfs/{statistics_pth.stem}/equivariant_losses.pdf")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("statistics_pth", help="The path where the statistics are stored.", type=Path)
    args = args.parse_args()
    main(args.statistics_pth)