#!/bin/bash
# Submits every training job (baseline + learned-equivariant variant, default
# --model/--dataset) as a 10-seed SLURM job array (seeds 0-9) on the
# robot_learning partition with one L40S GPU each, 14 day time limit.
#
# Usage: ./slurm/launch_all.sh

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

JOBS=(
  classification.py
  colorization.py
  fluid_flow.py
  fluid_flow_particles.py
  stress_prediction.py
  stress_prediction_particles.py
)

for job in "${JOBS[@]}"; do
  tag="${job%.py}"
  for variant in baseline equivariant; do
    name="${tag}_${variant}"
    extra_args=()
    if [[ "$variant" == "equivariant" ]]; then
      extra_args+=(--rotation)
    fi
    sbatch --job-name="$name" slurm/train.sbatch "$job" "${extra_args[@]}"
  done
done
