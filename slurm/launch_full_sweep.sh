#!/bin/bash
# Submits the full sweep: every model choice x both augmentation regimes
# (non-augmented "baseline" and rotation-augmented "equivariant", i.e. the
# --rotation flag off/on) x seeds 0-9, for classification, colorization,
# fluid_flow, fluid_flow_particles, stress_prediction and
# stress_prediction_particles -- restricted to each script's non-isotropic
# dataset(s) (classification/colorization have none named "isotropic" so all
# of their datasets run; the flow/stress scripts skip "isotropic" and use
# their symmetry-broken dataset instead).
#
# Before submitting each (job, model, dataset, regime) job array, per-seed
# completion is checked two ways and already-done seeds are dropped from the
# --array spec (or the whole submission is skipped if all 10 are covered):
#   1. results/<tag>_seed_<n>_statistics.json already has a finished run
#      (slurm/check_done.py).
#   2. A job with the same --job-name and that seed's array index is already
#      queued/running (squeue), so re-running this script is idempotent.
#
# Usage: ./slurm/launch_full_sweep.sh [--dry-run]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

PYTHON="$PROJECT_ROOT/.venv/bin/python"
CHECK_DONE="$PROJECT_ROOT/slurm/check_done.py"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# job script -> space-separated --model choices to sweep
declare -A JOB_MODELS=(
  [classification.py]="cnn naive vit"
  [colorization.py]="cnn unet naive vit"
  [fluid_flow.py]="cnn unet naive vit"
  [fluid_flow_particles.py]="pointnet mlp naive transformer"
  [stress_prediction.py]="cnn unet naive vit"
  [stress_prediction_particles.py]="pointnet mlp naive transformer"
)
# job script -> space-separated --dataset choices, isotropic excluded
declare -A JOB_DATASETS=(
  [classification.py]="cifar mnist_font"
  [colorization.py]="cifar stl10"
  [fluid_flow.py]="buoyant"
  [fluid_flow_particles.py]="buoyant"
  [stress_prediction.py]="anisotropic"
  [stress_prediction_particles.py]="anisotropic"
)

# classification trains 400 epochs on mnist_font, 200 otherwise; every other
# job trains 200 epochs regardless of dataset (see NUM_EPOCHS in each script).
num_epochs_for() {
  local job="$1" dataset="$2"
  if [[ "$job" == "classification.py" && "$dataset" == "mnist_font" ]]; then
    echo 400
  else
    echo 200
  fi
}

# Currently queued/running array task ids for a given job-name, one per line.
queued_seeds_for() {
  local name="$1"
  command -v squeue >/dev/null 2>&1 || return 0
  squeue -h -u "$USER" -n "$name" -o "%K" 2>/dev/null || true
}

for job in "${!JOB_MODELS[@]}"; do
  tag="${job%.py}"
  read -ra models <<< "${JOB_MODELS[$job]}"
  read -ra datasets <<< "${JOB_DATASETS[$job]}"

  for dataset in "${datasets[@]}"; do
    num_epochs="$(num_epochs_for "$job" "$dataset")"

    for model in "${models[@]}"; do
      for regime in baseline equivariant; do
        extra_args=(--model "$model" --dataset "$dataset")
        regime_tag="non_equivariant"
        if [[ "$regime" == "equivariant" ]]; then
          extra_args+=(--rotation)
          regime_tag="learned_equivariant"
        fi

        name="${tag}_${regime}_${model}_${dataset}"
        result_prefix="results/${tag}_${regime_tag}_${model}_dataset_${dataset}"

        mapfile -t queued_seeds < <(queued_seeds_for "$name")

        missing_seeds=()
        for seed in $(seq 0 9); do
          if "$PYTHON" "$CHECK_DONE" "${result_prefix}_seed_${seed}_statistics.json" "$num_epochs"; then
            continue
          fi
          already_queued=false
          for q in "${queued_seeds[@]:-}"; do
            [[ "$seed" == "$q" ]] && already_queued=true && break
          done
          $already_queued || missing_seeds+=("$seed")
        done

        if [[ ${#missing_seeds[@]} -eq 0 ]]; then
          echo "skip:   $name (all 10 seeds already complete or queued)"
          continue
        fi

        array_spec="$(IFS=,; echo "${missing_seeds[*]}")"
        echo "submit: $name --array=$array_spec"
        if ! $DRY_RUN; then
          sbatch --job-name="$name" --array="$array_spec" slurm/train.sbatch "$job" "${extra_args[@]}"
        fi
      done
    done
  done
done
