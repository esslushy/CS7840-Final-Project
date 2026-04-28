from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np

def main(losses_pth: Path):
    with losses_pth.open() as f:
        losses = json.load(f)

    print(np.argmin(losses, axis=0))


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("equivariant_losses_pth", help="The path where the equivariant losses are stored.", type=Path)
    args = args.parse_args()
    main(args.equivariant_losses_pth)