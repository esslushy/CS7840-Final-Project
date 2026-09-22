#!/bin/bash
# Re-runs the configs invalidated by the vector-rotation handedness fix.
#
# THE BUG. rotate_force_field (stress_prediction.py) and rotate_vector_field /
# rotate_buoyant_field (fluid_flow.py) mixed vector components with the opposite
# handedness to the spatial torch.rot90, so at k=1 and k=3 they returned exactly
# the NEGATIVE of the correctly rotated field. Measured against the generative
# ground truth -- phi is a scalar field, so rot90(phi) unambiguously is the
# rotated configuration -- the relative defect was exactly 2.00000 at 90 and 270
# degrees and 0.00005 at 180, which is handedness-blind. The tensor law in
# rotate_stress_field could not detect it: swap-sigma_xx/sigma_yy-and-negate-shear is
# identical for +90 and -90, which is why only the vector fields were affected.
#
# WHAT IT BROKE, in the two grid tasks that rotate a vector field:
#   * --rotation augmentation paired an input with the target of the OPPOSITE
#     configuration for k in {1,3}, i.e. half of all augmented samples.
#   * the CKA probe at 90 and 270 degrees compared features of x against
#     features of -Rx rather than Rx. The networks are nonlinear, so those
#     differ. The 180 degree probe was correct throughout.
#   * test loss at those same two angles.
#
# The particle tasks are unaffected: utils.rotate_2d applies one matrix to
# positions and velocities with no grid and no rot90, so no conflict arises.
# classification and colorization are unaffected: their data is scalar/RGB, with
# no vector channels to mix.
#
# SCOPE. 20 configurations x 10 seeds = 200 runs (10 matched pairs, of which 8
# are non-isotropic and enter the headline numbers). The isotropic pairs are
# excluded from the paper as degenerate but are re-run anyway so results/ stays
# internally consistent.
#
# Existing results for these configs are INVALID. They are moved aside rather
# than deleted, both because results/ is tracked in git and so the old and new
# numbers can be compared. launch_full_sweep.sh skips seeds that already have a
# finished run, so archiving is also what makes the resubmission actually fire.
#
# Usage:
#   ./slurm/relaunch_handedness_fix.sh --dry-run    # print the plan, touch nothing
#   ./slurm/relaunch_handedness_fix.sh              # archive, then submit

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

ARCHIVE="src/results_pre_handedness_fix"

# Only the configurations that exist on disk. The grid tasks were run on all
# four models for their symmetry-broken dataset but on unet only for isotropic,
# so this is not a full cross product.
CONFIGS=(
  "fluid_flow.py cnn buoyant"
  "fluid_flow.py naive buoyant"
  "fluid_flow.py unet buoyant"
  "fluid_flow.py vit buoyant"
  "fluid_flow.py unet isotropic"
  "stress_prediction.py cnn anisotropic"
  "stress_prediction.py naive anisotropic"
  "stress_prediction.py unet anisotropic"
  "stress_prediction.py vit anisotropic"
  "stress_prediction.py unet isotropic"
)

NUM_EPOCHS=200   # both grid tasks, every dataset (see NUM_EPOCHS in each script)

queued_seeds_for() {
  local name="$1"
  command -v squeue >/dev/null 2>&1 || return 0
  squeue -h -u "$USER" -n "$name" -o "%K" 2>/dev/null || true
}

echo "=== stale results ==="
moved=0
for entry in "${CONFIGS[@]}"; do
  read -r job model dataset <<< "$entry"
  tag="${job%.py}"
  for regime_tag in non_equivariant learned_equivariant; do
    prefix="src/results/${tag}_${regime_tag}_${model}_dataset_${dataset}"
    for seed in $(seq 0 9); do
      f="${prefix}_seed_${seed}_statistics.json"
      [[ -f "$f" ]] || continue
      moved=$((moved + 1))
      if ! $DRY_RUN; then
        mkdir -p "$ARCHIVE"
        mv "$f" "$ARCHIVE/"
      fi
    done
  done
done
if $DRY_RUN; then
  echo "would move $moved result files to $ARCHIVE/"
else
  echo "moved $moved result files to $ARCHIVE/"
fi

echo
echo "=== submissions ==="
for entry in "${CONFIGS[@]}"; do
  read -r job model dataset <<< "$entry"
  tag="${job%.py}"

  for regime in baseline equivariant; do
    extra_args=(--model "$model" --dataset "$dataset")
    regime_tag="non_equivariant"
    if [[ "$regime" == "equivariant" ]]; then
      extra_args+=(--rotation)
      regime_tag="learned_equivariant"
    fi

    name="${tag}_${regime}_${model}_${dataset}"
    mapfile -t queued_seeds < <(queued_seeds_for "$name")

    missing_seeds=()
    for seed in $(seq 0 9); do
      already_queued=false
      for q in "${queued_seeds[@]:-}"; do
        [[ "$seed" == "$q" ]] && already_queued=true && break
      done
      $already_queued || missing_seeds+=("$seed")
    done

    if [[ ${#missing_seeds[@]} -eq 0 ]]; then
      echo "skip:   $name (all 10 seeds already queued)"
      continue
    fi

    array_spec="$(IFS=,; echo "${missing_seeds[*]}")"
    echo "submit: $name --array=$array_spec"
    $DRY_RUN || sbatch --job-name="$name" --array="$array_spec" \
        slurm/train.sbatch "$job" "${extra_args[@]}"
  done
done

echo
echo "After these finish: rebuild the cache and the posters, since aggregate.py"
echo "reads src/results/ and both affected tasks appear in the headline numbers."
echo "  cd src && python figures/aggregate.py && python figures/fig_posters.py --no-isotropic"
