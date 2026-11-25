#!/bin/bash
# Submits the sweep for the three tasks that now train on published datasets
# (src/published_data.py):
#
#   fluid_flow.py                  pdearena_smoke     x cnn unet naive vit
#                                  pdebench_cfd       x unet              (isotropic control)
#   fluid_flow_particles.py        lagrangebench_dam  x pointnet mlp naive transformer
#                                  lagrangebench_tgv  x pointnet          (isotropic control)
#   stress_prediction_particles.py geofno_elasticity  x pointnet mlp naive transformer
#
# Each config gets both regimes (baseline / --rotation) x seeds 0-9, as a SLURM
# job array on slurm/train.sbatch. The isotropic controls run on one model
# each, as the original sweep did; pass --no-isotropic to skip them.
#
# Data: by default a CPU-only prefetch job (slurm/prefetch.sbatch) is submitted
# first, and every training job waits for it (afterok). If the prefetch fails,
# the dependent jobs are cancelled rather than left pending. Options:
#   --prefetch=job    (default) prefetch as a SLURM job, training depends on it
#   --prefetch=here   prefetch now, on this node, before submitting anything.
#                     Use this if compute nodes have no internet.
#   --regime=baseline|equivariant|both   which training regime(s) to submit
#                     (default both). baseline = no --rotation (non_equivariant).
#   --prefetch=skip   no prefetch. The first training job of each dataset
#                     downloads it and the rest wait on its lock file.
#
# Idempotent, like launch_full_sweep.sh: seeds whose results JSON already holds
# a finished run (slurm/check_done.py), or that are already queued under the
# same job name (squeue), are dropped from --array. A config with nothing left
# is skipped. The prefetch job is only submitted if some training is.
#
# Usage (from anywhere; it cds to the repo root):
#   ./slurm/launch_published.sh --dry-run          # print what would be submitted
#   ./slurm/launch_published.sh                    # submit
#   ./slurm/launch_published.sh --prefetch=here --no-isotropic
#
# Set EQUIV_DATA_DIR before launching to keep the data on shared scratch
# instead of src/data. sbatch passes it through to every job.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
CHECK_DONE="$PROJECT_ROOT/slurm/check_done.py"
NUM_EPOCHS=200            # NUM_EPOCHS in all three task scripts
DRY_RUN=false
PREFETCH=job
ISOTROPIC=true
REGIMES=(baseline equivariant)

for arg in "$@"; do
  case "$arg" in
    --dry-run)        DRY_RUN=true ;;
    --no-isotropic)   ISOTROPIC=false ;;
    --regime=baseline)    REGIMES=(baseline) ;;
    --regime=equivariant) REGIMES=(equivariant) ;;
    --regime=both)        REGIMES=(baseline equivariant) ;;
    --prefetch=job|--prefetch=here|--prefetch=skip) PREFETCH="${arg#--prefetch=}" ;;
    -h|--help)        sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
  esac
done

# "script dataset model model ..." -- one line per (task, dataset)
CONFIGS=(
  "fluid_flow.py pdearena_smoke cnn unet naive vit"
  "fluid_flow_particles.py lagrangebench_dam pointnet mlp naive transformer"
  "stress_prediction_particles.py geofno_elasticity pointnet mlp naive transformer"
)
if $ISOTROPIC; then
  CONFIGS+=(
    "fluid_flow.py pdebench_cfd unet"
    "fluid_flow_particles.py lagrangebench_tgv pointnet"
  )
fi

# Currently queued/running array task ids for a given job-name, one per line.
queued_seeds_for() {
  local name="$1"
  command -v squeue >/dev/null 2>&1 || return 0
  squeue -h -r -u "$USER" -n "$name" -o "%K" 2>/dev/null || true
}

# --- work out what needs submitting before touching the queue --------------
SUBMISSIONS=()   # "name|array_spec|script|extra args..."
for config in "${CONFIGS[@]}"; do
  read -ra parts <<< "$config"
  job="${parts[0]}"; dataset="${parts[1]}"; models=("${parts[@]:2}")
  tag="${job%.py}"

  for model in "${models[@]}"; do
    for regime in "${REGIMES[@]}"; do
      extra="--model $model --dataset $dataset"
      regime_tag="non_equivariant"
      if [[ "$regime" == "equivariant" ]]; then
        extra+=" --rotation"
        regime_tag="learned_equivariant"
      fi

      name="${tag}_${regime}_${model}_${dataset}"
      result_prefix="src/results/${tag}_${regime_tag}_${model}_dataset_${dataset}"

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
      SUBMISSIONS+=("$name|$(IFS=,; echo "${missing_seeds[*]}")|$job|$extra")
    done
  done
done

if [[ ${#SUBMISSIONS[@]} -eq 0 ]]; then
  echo "nothing to submit"
  exit 0
fi

# --- data ----------------------------------------------------------------
DEPENDENCY=()
case "$PREFETCH" in
  here)
    echo "prefetch: running src/published_data.py on this node"
    $DRY_RUN || (cd src && "$PYTHON" published_data.py)
    ;;
  job)
    if $DRY_RUN; then
      echo "submit: prefetch_published (CPU), training jobs depend on it"
    else
      prefetch_id="$(sbatch --parsable --job-name=prefetch_published slurm/prefetch.sbatch)"
      prefetch_id="${prefetch_id%%;*}"
      echo "submit: prefetch_published -> job $prefetch_id"
      DEPENDENCY=(--dependency="afterok:$prefetch_id" --kill-on-invalid-dep=yes)
    fi
    ;;
  skip)
    echo "prefetch: skipped (first job per dataset downloads it)"
    ;;
esac

# --- training ------------------------------------------------------------
for entry in "${SUBMISSIONS[@]}"; do
  IFS='|' read -r name array_spec job extra <<< "$entry"
  read -ra extra_args <<< "$extra"
  echo "submit: $name --array=$array_spec"
  if ! $DRY_RUN; then
    sbatch ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} --job-name="$name" --array="$array_spec" \
      slurm/train.sbatch "$job" "${extra_args[@]}"
  fi
done

if $DRY_RUN; then
  echo "${#SUBMISSIONS[@]} job arrays would be submitted (dry run)"
else
  echo "${#SUBMISSIONS[@]} job arrays submitted"
fi
