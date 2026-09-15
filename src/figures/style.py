"""
Shared plotting style for the sweep figures.

Replaces the blocks that were copy-pasted verbatim across the three old
visualize_*.py scripts (the 'gnuplot' colormap, the `sm._A = []` colorbar hack,
the ylim guard). Palette is the dataviz reference instance, used unmodified; the
categorical pair and the ordinal depth ramp both pass the validator
(categorical: worst adjacent CVD dE 24.7 light / 26.8 dark; ramp: monotone L,
single hue, light end 2.06:1).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

# --- palette -------------------------------------------------------------
AUG = "#2a78d6"       # categorical slot 1 (blue)  -- rotation-augmented
BASE = "#eb6834"      # categorical slot 2 (orange) -- baseline
DEPTH_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
DIVERGING = ["#1c5cab", "#2a78d6", "#f0efec", "#d03b3b", "#8f2020"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

depth_cmap = LinearSegmentedColormap.from_list("depth", DEPTH_RAMP)
div_cmap = LinearSegmentedColormap.from_list("div", DIVERGING)

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


def apply():
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
    return pd.read_pickle(CACHE / f"{name}.pkl", compression="gzip")


def final_epochs(df):
    """Max epoch per config -- runs are not all the same length (mnist_font = 400)."""
    return df.groupby(ID, observed=True)["epoch"].transform("max")


def at_final_epoch(df):
    return df[df.epoch == final_epochs(df)]


def defect(series):
    """1 - CKA, floored so it can go on a log axis.

    CKA saturates at 1.0 for much of the sweep; on a linear [0,1] axis those
    configs are a flat line at the frame top. Plotting the defect on a log axis
    spreads the saturated regime and the low-CKA regime with one transform.
    """
    return np.clip(1.0 - series.to_numpy(dtype=float), 1e-5, None)


def depth_colors(n):
    if n == 1:
        return [DEPTH_RAMP[2]]
    return [depth_cmap(i) for i in np.linspace(0, 1, n)]


def depth_colorbar(fig, axes, label="Layer depth (input to output)"):
    sm = plt.cm.ScalarMappable(cmap=depth_cmap, norm=Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, ticks=[0, 1], pad=0.015, aspect=30)
    cbar.ax.set_yticklabels(["input", "output"])
    cbar.set_label(label, color=INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    return cbar


def condition_legend(ax, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=AUG, lw=2.0, label="Rotation-augmented"),
               Line2D([], [], color=BASE, lw=2.0, label="Baseline")]
    return ax.legend(handles=handles, **kw)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path.cwd()) if str(path).startswith(str(Path.cwd())) else path}")
    return path


# --- config filtering ----------------------------------------------------
def common_parser(description=None):
    """Argument parser shared by every figure script."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--no-isotropic", action="store_true",
                    help="drop the isotropic-dataset configs. Isotropic data is "
                         "rotation-symmetric by construction, so a rotation-"
                         "equivariance error measured on it is degenerate.")
    return ap


def apply_filters(df, args):
    if getattr(args, "no_isotropic", False):
        df = df[df.dataset != "isotropic"].copy()
        df["dataset"] = df["dataset"].cat.remove_unused_categories()
        df["task"] = df["task"].cat.remove_unused_categories()
    return df


def suffix(args):
    return "_no_isotropic" if getattr(args, "no_isotropic", False) else ""


def note(args):
    return ("\nisotropic configs excluded (rotation-symmetric data)"
            if getattr(args, "no_isotropic", False) else "")
