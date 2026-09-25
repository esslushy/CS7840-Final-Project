#!/bin/bash
# Submits the classification vit/mnist_font runs, which are missing from the
# full sweep: the MNIST ViT crashed on its first step (it expected 3 input
# channels for grayscale digits, and Transformer.forward did not unpack the
# (out, acts) tuple returned by Attention). Both are fixed, so this submits
# the 2 regimes x 10 seeds = 20 runs.
#
# Resources differ from train.sbatch's defaults, so they are overridden on the
# sbatch command line (which takes precedence over its #SBATCH lines):
# partition gpu, 8 hours, 8 CPUs, 8G memory, any single GPU.
#
# Seeds that already have a finished run (check_done.py) or are already
# queued/running under the same job name are skipped, so re-running this is
# idempotent.
#
# Usage:
#   ./slurm/launch_vit_mnist_font.sh --dry-run    # print the plan, submit nothing
#   ./slurm/launch_vit_mnist_font.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

PYTHON="$PROJECT_ROOT/.venv/bin/python"
CHECK_DONE="$PROJECT_ROOT/slurm/check_done.py"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

JOB=classification.py
MODEL=vit
DATASET=mnist_font
NUM_EPOCHS=400   # classification on mnist_font (see NUM_EPOCHS in classification.py)

SBATCH_RESOURCES=(
  --partition=gpu
  --gres=gpu:1
  --time=08:00:00
  --cpus-per-task=8
  --mem=8G
)

queued_seeds_for() {
  local name="$1"
  command -v squeue >/dev/null 2>&1 || return 0
  squeue -h -u "$USER" -n "$name" -o "%K" 2>/dev/null || true
}

tag="${JOB%.py}"
for regime in baseline equivariant; do
  extra_args=(--model "$MODEL" --dataset "$DATASET")
  regime_tag="non_equivariant"
  if [[ "$regime" == "equivariant" ]]; then
    extra_args+=(--rotation)
    regime_tag="learned_equivariant"
  fi

  name="${tag}_${regime}_${MODEL}_${DATASET}"
  result_prefix="src/results/${tag}_${regime_tag}_${MODEL}_dataset_${DATASET}"

  mapfile -t queued_seeds < <(queued_seeds_for "$name")

  missing_seeds=()
  for seed in $(seq 0 9); do
    if "$PYTHON" "$CHECK_DONE" "${result_prefix}_seed_${seed}_statistics.json" "$NUM_EPOCHS"; then
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
  $DRY_RUN || sbatch "${SBATCH_RESOURCES[@]}" --job-name="$name" \
      --array="$array_spec" slurm/train.sbatch "$JOB" "${extra_args[@]}"
done

echo
echo "After these finish, rebuild the cache and figures:"
echo "  cd src && python figures/aggregate.py && python figures/fig_sweep_effects.py"
