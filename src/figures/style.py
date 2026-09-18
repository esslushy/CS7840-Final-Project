"""
Shared plotting style for the per-config posters.

Palette is the dataviz reference instance, used unmodified. Only the categorical
pair is left: augmented vs. baseline is the one comparison the posters make, and
the ordinal depth ramp went with the figures that encoded layer depth as colour
(each layer now gets its own panel instead).
"""
from pathlib import Path

import matplotlib as mpl
import pandas as pd

# --- palette -------------------------------------------------------------
AUG = "#2a78d6"       # categorical slot 1 (blue)   -- rotation-augmented
BASE = "#eb6834"      # categorical slot 2 (orange) -- baseline
# Validated as a categorical pair on the light surface: worst adjacent CVD
# dE 24.7 light / 26.8 dark, normal-vision dE well clear of the floor.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

OUT = Path(__file__).resolve().parent / "out"
CACHE = Path(__file__).resolve().parent / "cache"

TASK_LABEL = {
    "classification": "Classification",
    "colorization": "Colorization",
    "fluid_flow": "Fluid flow (grid)",
    "fluid_flow_particles": "Fluid flow (particles)",
    "stress_prediction": "Stress (grid)",
    "stress_prediction_particles": "Stress (particles)",
}
ID = ["task", "model", "dataset", "augmented"]

# --- which CKA variant the figures report --------------------------------
# The paper reports *linear* CKA. Under exact equivariance the fiber
# representation is a permutation, and orthogonal maps leave the centered
# linear Gram matrix unchanged, so linear CKA is exactly 1 -- the metric
# matches the correctness argument. RBF is kept only as a robustness check
# (`--rbf`), because its bandwidth is a free knob that can score a spurious
# 1.0 on a shuffled control. Both columns are in the cache, so switching
# costs nothing but the flag.
_STAT = {"col": "linear_cka", "label": "linear CKA"}


def stat():
    return _STAT["col"]


def stat_label():
    return _STAT["label"]


def apply(args=None):
    if getattr(args, "rbf", False):
        _STAT.update(col="rbf_cka", label="RBF CKA")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.dpi": 150,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
    })


def load(name):
    path = CACHE / f"{name}.pkl"
    if not path.exists():
        raise SystemExit(f"cache missing: {path}\n"
                         f"run `python figures/aggregate.py` from src/ first")
    return pd.read_pickle(path, compression="gzip")


def save(fig, name):
    import matplotlib.pyplot as plt
    path = OUT / f"{name}.pdf"
    # parents=True on the resolved path, so `name` may contain a subdirectory
    # (the per-config posters write into out/posters/).
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- config filtering ----------------------------------------------------
def common_parser(description=None):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--no-isotropic", action="store_true",
                    help="drop the isotropic-dataset configs. Isotropic data is "
                         "rotation-symmetric by construction, so a rotation-"
                         "equivariance error measured on it is degenerate.")
    ap.add_argument("--rbf", action="store_true",
                    help="report RBF CKA instead of linear. Robustness check "
                         "only -- the paper reports linear (see style._STAT).")
    return ap


def apply_filters(df, args):
    if getattr(args, "no_isotropic", False):
        df = df[df.dataset != "isotropic"].copy()
        df["dataset"] = df["dataset"].cat.remove_unused_categories()
        df["task"] = df["task"].cat.remove_unused_categories()
    return df
