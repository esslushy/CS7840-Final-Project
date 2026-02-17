import json
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

def main(equivariant_losses_pth: Path):
    with equivariant_losses_pth.open() as f:
        equivariant_losses = json.load(f)

    fig = plt.figure(figsize=(12, 6),dpi=150)
    plt.xlabel("Layer")
    plt.xticks(range(len(equivariant_losses[0])))
    plt.ylim(bottom=0, top=1)
    plt.ylabel("Average Delta InfoNCE Loss")

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(equivariant_losses))]

    for i in range(0, len(equivariant_losses)):
        plt.plot(range(len(equivariant_losses[i])),equivariant_losses[i],marker='o',c=colors[i])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[], ax=fig.axes)
    cbar.set_label('Number Training Epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(len(equivariant_losses)-1), transform=cbar.ax.transAxes, va='bottom', ha='center')
    plt.savefig(f"{equivariant_losses_pth.stem}.pdf")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("equivariant_losses_pth", help="The path where the equivariant losses are stored.", type=Path)
    args = args.parse_args()
    main(args.equivariant_losses_pth)