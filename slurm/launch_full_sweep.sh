#!/bin/bash
# Submits the full sweep: every model choice x both augmentation regimes
# (non-augmented "baseline" and rotation-augmented "equivariant", i.e. the
# --rotation flag off/on) x seeds 0-9, for classification, colorization,
# fluid_flow, fluid_flow_particles, stress_prediction and
# stress_prediction_particles -- restricted to each script's non-isotropic
# dataset(s) (classification/colorization have no isotropic dataset so all
# of their datasets run; the flow/stress scripts skip their isotropic control
# -- pdebench_cfd, lagrangebench_tgv, isotropic -- and use their symmetry-broken
# dataset instead).
#
# The flow/stress datasets are published simulations that download on first
# use (src/published_data.py). If compute nodes have no internet, prefetch them
# once on a login node first:  cd src && python published_data.py
#
# Before submitting each (job, model, dataset, regime) job array, per-seed
# completion is checked two ways and already-done seeds are dropped from the
# --array spec (or the whole submission is skipped if all 10 are covered):
#   1. src/results/<tag>_seed_<n>_statistics.json already has a finished run
#      (slurm/check_done.py).
#   2. A job with the same --job-name and that seed's array index is already
#      queued/running (squeue), so re-running this script is idempotent.
#
# Usage: ./slurm/launch_full_sweep.sh [--dry-run] [--regime=baseline|equivariant|both]
#                                     [--with-isotropic]
#   --regime=baseline     only the non-augmented runs (no --rotation, tagged non_equivariant)
#   --regime=equivariant  only the rotation-augmented runs (tagged learned_equivariant)
#   --regime=both         both (default)
#   --with-isotropic      also the isotropic controls, one model each as in the
#                         original sweep: fluid_flow/unet/pdebench_cfd,
#                         fluid_flow_particles/pointnet/lagrangebench_tgv,
#                         stress_prediction/unet/isotropic. With both regimes this
#                         is the complete sweep: 33 pairs, 66 job arrays.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

PYTHON="$PROJECT_ROOT/.venv/bin/python"
CHECK_DONE="$PROJECT_ROOT/slurm/check_done.py"
DRY_RUN=false
ISOTROPIC=false
REGIMES=(baseline equivariant)
for arg in "$@"; do
  case "$arg" in
    --dry-run)              DRY_RUN=true ;;
    --with-isotropic)       ISOTROPIC=true ;;
    --regime=baseline)      REGIMES=(baseline) ;;
    --regime=equivariant)   REGIMES=(equivariant) ;;
    --regime=both)          REGIMES=(baseline equivariant) ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

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
  [fluid_flow.py]="pdearena_smoke"
  [fluid_flow_particles.py]="lagrangebench_dam"
  [stress_prediction.py]="anisotropic"
  [stress_prediction_particles.py]="geofno_elasticity"
)

# Epochs per run, matching NUM_EPOCHS in each script: classification trains 400
# on mnist_font and 200 otherwise; the three published-data tasks train 400;
# colorization and stress_prediction train 200.
num_epochs_for() {
  local job="$1" dataset="$2"
  case "$job" in
    fluid_flow.py|fluid_flow_particles.py|stress_prediction_particles.py) echo 400 ;;
    classification.py) [[ "$dataset" == "mnist_font" ]] && echo 400 || echo 200 ;;
    *) echo 200 ;;
  esac
}

# Currently queued/running array task ids for a given job-name, one per line.
queued_seeds_for() {
  local name="$1"
  command -v squeue >/dev/null 2>&1 || return 0
  squeue -h -r -u "$USER" -n "$name" -o "%K" 2>/dev/null || true
}

# "script dataset model model ..." -- one line per (task, dataset)
CONFIGS=()
for job in "${!JOB_MODELS[@]}"; do
  for dataset in ${JOB_DATASETS[$job]}; do
    CONFIGS+=("$job $dataset ${JOB_MODELS[$job]}")
  done
done
if $ISOTROPIC; then
  CONFIGS+=(
    "fluid_flow.py pdebench_cfd unet"
    "fluid_flow_particles.py lagrangebench_tgv pointnet"
    "stress_prediction.py isotropic unet"
  )
fi

for config in "${CONFIGS[@]}"; do
  read -ra parts <<< "$config"
  job="${parts[0]}"; dataset="${parts[1]}"; models=("${parts[@]:2}")
  tag="${job%.py}"
  {
    num_epochs="$(num_epochs_for "$job" "$dataset")"

    for model in "${models[@]}"; do
      for regime in "${REGIMES[@]}"; do
        extra_args=(--model "$model" --dataset "$dataset")
        regime_tag="non_equivariant"
        if [[ "$regime" == "equivariant" ]]; then
          extra_args+=(--rotation)
          regime_tag="learned_equivariant"
        fi

        name="${tag}_${regime}_${model}_${dataset}"
        result_prefix="src/results/${tag}_${regime_tag}_${model}_dataset_${dataset}"

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
  }
done
