#!/bin/bash
#SBATCH --partition=robot_learning
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Downloads and preprocesses every published flow/elasticity dataset once
# (src/published_data.py), so the training jobs that depend on it start from a
# warm cache instead of all downloading at once. CPU only, no GPU.
# Downloads: PDEArena ~21.5 GB (all train/valid/test files), LagrangeBench
# ~1.9 GB, Geo-FNO <1 GB. With the isotropic controls, PDEBench adds a 55 GB
# remote read that can take hours; it resumes block by block if the job is
# killed, so resubmitting continues where it stopped. Extra arguments are
# passed to published_data.py (launch_published.sh passes --no-isotropic).
#
# Not meant to be submitted directly - launch_published.sh submits it and makes
# the training jobs depend on it. To run it by hand from the repo root:
#   sbatch --job-name=prefetch_published slurm/prefetch.sbatch
#
# Respects EQUIV_DATA_DIR (sbatch exports the submitting shell's environment),
# so set it before launching if the data should live on shared scratch.

set -euo pipefail

module load python

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT/src"

"$PROJECT_ROOT/.venv/bin/python" published_data.py "$@"
