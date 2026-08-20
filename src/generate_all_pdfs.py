"""
Generate every equivariance PDF plot for every experiment config found in results/.

For each group of seeded statistics files sharing the same config (task, regime,
model, dataset -- see utils.strip_seed_suffix), runs all three visualize_* scripts
with all seeds combined (mean +/- std across seeds), across every --stat choice
(rbf_cka, linear_cka, calibrated_sigma) and, for the two "versus" plots, against
both test_loss and test_accuracy as the x-axis statistic.

Usage: python generate_all_pdfs.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from utils import strip_seed_suffix
import visualize_equivariant_losses_over_time as over_time
import visualize_equivariant_loss_versus_angle as versus_angle
import visualize_equivariant_loss_versus_statistic as versus_statistic

STATS = ["rbf_cka", "linear_cka", "calibrated_sigma"]
CANDIDATE_TRAIN_STATISTICS = ["test_loss", "test_accuracy"]


def group_result_files():
    groups = defaultdict(list)
    for f in glob.glob("results/*_statistics.json"):
        base = strip_seed_suffix(Path(f).stem)
        groups[base].append(Path(f))
    return {base: sorted(pths) for base, pths in sorted(groups.items())}


def available_train_statistics(pths):
    with pths[0].open() as f:
        keys = json.load(f).keys()
    return [s for s in CANDIDATE_TRAIN_STATISTICS if s in keys]


def run(label, fn, *args):
    try:
        fn(*args)
    except Exception as e:
        print(f"FAILED: {label}: {e}")
    finally:
        plt.close("all")


def main():
    groups = group_result_files()
    print(f"Found {len(groups)} experiment configs, {sum(len(v) for v in groups.values())} seed files total.")

    for base, pths in groups.items():
        for stat in STATS:
            run(f"[{base}] over_time stat={stat}", over_time.main, pths, stat)

        train_stats = available_train_statistics(pths)
        for train_stat in train_stats:
            for stat in STATS:
                run(f"[{base}] versus_angle {train_stat} stat={stat}",
                    versus_angle.main, pths, train_stat, stat)
                run(f"[{base}] versus_statistic {train_stat} stat={stat}",
                    versus_statistic.main, pths, train_stat, stat)

    print("Done.")


if __name__ == "__main__":
    main()
