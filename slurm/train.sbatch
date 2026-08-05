#!/bin/bash
#SBATCH --partition=robot_learning
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-9
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#
# Generic training job: one array task per seed (SLURM_ARRAY_TASK_ID 0-9).
# Not meant to be submitted directly - use launch_all.sh, which sets
# --job-name and passes the training script + extra flags:
#   sbatch --job-name=<name> slurm/train.sbatch <script.py> [extra args...]

set -euo pipefail

module load cuda python

PROJECT_ROOT=/projects/gllab/claire/CS7840-Final-Project
cd "$PROJECT_ROOT/src"

TRAIN_SCRIPT="$1"
shift

"$PROJECT_ROOT/.venv/bin/python" "$TRAIN_SCRIPT" --seed "$SLURM_ARRAY_TASK_ID" "$@"
