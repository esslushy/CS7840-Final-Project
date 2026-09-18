"""
Collapse results/*_statistics.json into a few small cached tables.

The raw sweep is 660 JSON files / 1.6 GB, and the figures would otherwise re-parse
all of it. The full tidy cross-product (seed x epoch x layer x angle) would be
~13M rows, so instead we cache three projections:

  cka_by_epoch      mean over angles, per (config, seed, epoch, layer)   ~2.1M rows
  cka_by_angle      per angle, at a few checkpoint epochs only           ~0.5M rows
  perf_by_epoch     train/test scalars per (config, seed, epoch, angle)  ~0.1M rows

Only `cka_by_epoch` is read by a figure now -- `fig_posters.py` is the only one
left. The other two are still built because the project's headline performance
numbers come out of `perf_by_epoch` (the upright vs. all-angle test-loss ratios,
which is the "bought, not free" result) and the angle-decay numbers out of
`cka_by_angle`. Dropping them here would mean those are no longer recomputable.

Stored as pickle rather than parquet because pyarrow is not a project dependency;
these are regenerable caches, not artifacts.

Run from src/:  python figures/aggregate.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent / "cache"
RESULTS = Path(__file__).resolve().parent.parent / "results"

STATS = ("rbf_cka", "linear_cka", "cka_gap", "calibrated_sigma")

# Tasks whose names contain an underscore, longest-first so the greedy match wins.
TASKS = (
    "fluid_flow_particles",
    "stress_prediction_particles",
    "fluid_flow",
    "stress_prediction",
    "classification",
    "colorization",
)


def parse_config(stem: str) -> dict:
    """Split '<task>_<mode>_<model>_dataset_<dataset>_seed_<n>_statistics' into fields.

    Both task and dataset names contain underscores (fluid_flow_particles,
    mnist_font), so this anchors on the two fixed markers -- the mode string and
    '_dataset_' -- rather than splitting on '_'.
    """
    m = re.match(r"^(?P<rest>.+)_seed_(?P<seed>\d+)_statistics$", stem)
    if not m:
        raise ValueError(f"unparseable stem: {stem}")
    seed = int(m.group("seed"))
    rest = m.group("rest")

    for mode, augmented in (("learned_equivariant", True), ("non_equivariant", False)):
        marker = f"_{mode}_"
        if marker in rest:
            task, tail = rest.split(marker, 1)
            break
    else:
        raise ValueError(f"no mode marker in: {stem}")

    if task not in TASKS:
        raise ValueError(f"unknown task {task!r} in {stem}")
    model, dataset = tail.split("_dataset_", 1)
    return dict(task=task, model=model, dataset=dataset, augmented=augmented, seed=seed)


def load_one(path: Path):
    """Return (cka_epoch_rows, cka_angle_rows, perf_rows) for a single results file."""
    cfg = parse_config(path.stem)
    with path.open() as f:
        d = json.load(f)

    el = d["equivariant_loss"]
    n_epochs = len(el)
    layers = list(el[0].keys())
    n_layers = len(layers)
    # Keep full per-angle detail only at these epochs; the angle figures are
    # final-epoch views and the endpoints are useful for sanity checks.
    checkpoints = sorted({0, n_epochs // 2, n_epochs - 1})

    base = (cfg["task"], cfg["model"], cfg["dataset"], cfg["augmented"], cfg["seed"])

    cka_epoch, cka_angle = [], []
    for e, epoch_dict in enumerate(el):
        keep_angles = e in checkpoints
        for li, layer in enumerate(layers):
            per_angle = epoch_dict[layer]
            # depth_frac lets architectures with 2..30 layers share one x-axis.
            depth = li / (n_layers - 1) if n_layers > 1 else 0.0
            vals = {s: [] for s in STATS}
            for angle, rec in per_angle.items():
                for s in STATS:
                    v = rec.get(s)
                    if v is not None:
                        vals[s].append(v)
                if keep_angles:
                    cka_angle.append(
                        base + (e, layer, li, depth, float(angle),
                                rec["rbf_cka"], rec["linear_cka"])
                    )
            cka_epoch.append(
                base + (e, layer, li, depth, n_layers)
                + tuple(float(np.mean(vals[s])) if vals[s] else np.nan for s in STATS)
            )

    perf = []
    train_loss = d.get("train_loss", [])
    train_acc = d.get("train_accuracy", [])
    for e in range(n_epochs):
        tl = d["test_loss"][e] if e < len(d.get("test_loss", [])) else {}
        ta = d["test_accuracy"][e] if e < len(d.get("test_accuracy", [])) else {}
        for angle in tl:
            perf.append(
                base + (e, float(angle),
                        float(tl[angle]),
                        float(ta[angle]) if angle in ta else np.nan,
                        float(train_loss[e]) if e < len(train_loss) else np.nan,
                        float(train_acc[e]) if e < len(train_acc) else np.nan)
            )
    return cka_epoch, cka_angle, perf


ID = ["task", "model", "dataset", "augmented", "seed"]


def main():
    paths = sorted(RESULTS.glob("*_statistics.json"))
    if not paths:
        raise SystemExit(f"no results found under {RESULTS}")
    print(f"reading {len(paths)} results files from {RESULTS} ...")

    A, B, C = [], [], []
    for i, p in enumerate(paths, 1):
        a, b, c = load_one(p)
        A.extend(a); B.extend(b); C.extend(c)
        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  (rows: {len(A):,} epoch / {len(B):,} angle)")

    cka_by_epoch = pd.DataFrame(
        A, columns=ID + ["epoch", "layer", "layer_idx", "depth_frac", "n_layers"] + list(STATS))
    cka_by_angle = pd.DataFrame(
        B, columns=ID + ["epoch", "layer", "layer_idx", "depth_frac", "angle",
                         "rbf_cka", "linear_cka"])
    perf_by_epoch = pd.DataFrame(
        C, columns=ID + ["epoch", "angle", "test_loss", "test_accuracy",
                         "train_loss", "train_accuracy"])

    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [("cka_by_epoch", cka_by_epoch),
                     ("cka_by_angle", cka_by_angle),
                     ("perf_by_epoch", perf_by_epoch)]:
        for col in ("task", "model", "dataset", "layer"):
            if col in df:
                df[col] = df[col].astype("category")
        for col in df.select_dtypes("float64").columns:
            df[col] = df[col].astype("float32")
        path = OUT / f"{name}.pkl"
        df.to_pickle(path, compression="gzip")
        print(f"wrote {path}  {len(df):,} rows  {path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
