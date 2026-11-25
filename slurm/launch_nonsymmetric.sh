#!/bin/bash
# Submits fluid-flow and stress-prediction training (plus their particle
# counterparts) on the non-symmetric variant of each dataset -- pdearena_smoke
# and lagrangebench_dam for the fluid-flow tasks, anisotropic and
# geofno_elasticity for the stress-prediction tasks --
# as 10-seed SLURM job arrays (seeds 0-9) on the robot_learning partition
# with one L40S GPU each, 14 day time limit. Submits both the baseline and
# learned-equivariant variant of each job.
#
# Usage: ./slurm/launch_nonsymmetric.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

# job script -> non-symmetric dataset choice for that job
JOBS=(
  "fluid_flow.py:pdearena_smoke"
  "fluid_flow_particles.py:lagrangebench_dam"
  "stress_prediction.py:anisotropic"
  "stress_prediction_particles.py:geofno_elasticity"
)

for entry in "${JOBS[@]}"; do
  job="${entry%%:*}"
  dataset="${entry##*:}"
  tag="${job%.py}"
  for variant in baseline equivariant; do
    name="${tag}_${variant}_${dataset}"
    extra_args=(--dataset "$dataset")
    if [[ "$variant" == "equivariant" ]]; then
      extra_args+=(--rotation)
    fi
    sbatch --job-name="$name" slurm/train.sbatch "$job" "${extra_args[@]}"
  done
done
