"""
Regenerate the whole figure set.

Replaces generate_all_pdfs.py, which produced 654 PDFs / 1.3 GB (one figure per
config, three families x three stats x 66 configs). This produces ~10 figures
organized by claim rather than by config.

Run from src/:
    python figures/aggregate.py                     # once, after new training runs
    python figures/make_figures.py
    python figures/make_figures.py --no-isotropic   # same set, isotropic pairs dropped
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

JOBS = [
    ("fig_depth_profile.py", []),
    ("fig_depth_profile.py", ["--defect"]),
    ("fig_emergence.py", []),
    ("fig_emergence.py", ["--defect"]),
    ("fig_sweep_summary.py", []),
    ("fig_angle_profile.py", []),
    ("fig_metric_gap.py", []),
    ("fig_equivariance_vs_perf.py", []),
]


def main():
    if not (HERE / "cache" / "cka_by_epoch.pkl").exists():
        sys.exit("cache missing -- run `python figures/aggregate.py` first")
    passthrough = [a for a in sys.argv[1:] if a == "--no-isotropic"]
    failed = []
    for script, args in JOBS:
        label = " ".join([script, *args, *passthrough])
        print(f"\n=== {label} ===")
        r = subprocess.run([sys.executable, str(HERE / script), *args, *passthrough], cwd=HERE)
        if r.returncode != 0:
            failed.append(label)
    pdfs = sorted((HERE / "out").glob("*.pdf"))
    print(f"\n{len(pdfs)} figures in {HERE / 'out'}")
    for p in pdfs:
        print(f"  {p.name}  ({p.stat().st_size/1e3:.0f} kB)")
    if failed:
        sys.exit("FAILED: " + ", ".join(failed))


if __name__ == "__main__":
    main()
