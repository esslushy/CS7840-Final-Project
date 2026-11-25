#!/bin/bash
#SBATCH --partition=robot_learning
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Downloads and preprocesses every published flow/elasticity dataset once
# (src/published_data.py), so the training jobs that depend on it start from a
# warm cache instead of all downloading at once. CPU only, no GPU.
# About 7 GB of downloads: PDEBench is read remotely (~2.4 GB) and is the slow
# part, typically 10-30 minutes.
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

"$PROJECT_ROOT/.venv/bin/python" published_data.py
