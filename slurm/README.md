# Cluster launchers

SLURM submission for the training sweep. Nothing here is specific to the science; it
exists so that 660 runs (66 configs × 10 seeds) can be submitted, resumed, and checked
without hand-editing job specs.

Run every script from the **repo root**, not from this folder:

```bash
./slurm/launch_full_sweep.sh --dry-run    # print what would be submitted
./slurm/launch_full_sweep.sh              # submit the missing seeds
```

## The scripts

| file | what it does |
|---|---|
| `train.sbatch` | The one job template. Not submitted directly — the launchers pass it a training script plus flags. One array task per seed, `--array=0-9`, mapping `SLURM_ARRAY_TASK_ID` to `--seed`. Requests one GPU, 8 CPUs, 16 GB, 14-day limit, and `cd`s to `src/` first because the training scripts write to relative `results/` and `models/` paths. |
| `launch_full_sweep.sh` | **The one to use.** Every `--model` × `--dataset` × augmentation regime for the six sweep tasks, isotropic datasets excluded. Idempotent — see below. |
| `launch_all.sh` | The earlier, simpler launcher: the six tasks at their *default* model and dataset only, both regimes, 10 seeds. Submits unconditionally with no completion check. Kept because it is the minimal reproduction path. |
| `launch_nonsymmetric.sh` | Just the four flow/stress tasks on their symmetry-broken datasets (`buoyant`, `anisotropic`), default model. A subset of the full sweep, useful when only those need rerunning. |
| `check_done.py` | Exit 0 if a `results/*_statistics.json` holds a finished run — at least `NUM_EPOCHS + 1` recorded entries, the pre-training eval plus one per epoch. Used by `launch_full_sweep.sh`; also handy by hand. |

## Why the full sweep is safe to re-run

`launch_full_sweep.sh` drops already-covered seeds from the `--array` spec, checking two
independent sources before submitting anything:

1. `check_done.py` on that seed's results JSON — did it already finish?
2. `squeue -n <job-name>` — is that array index already queued or running?

If all ten seeds are covered it prints `skip:` and submits nothing. So re-running after
a partial failure resubmits exactly the gaps. Epoch counts differ by config
(`classification.py` trains 400 epochs on `mnist_font`, everything else 200), and the
script encodes that in `num_epochs_for()`; the completion check is wrong if it drifts
from `NUM_EPOCHS` in the training scripts.

The regime naming is worth knowing when reading filenames: the `--rotation` flag on
produces the `learned_equivariant` tag, off produces `non_equivariant`. Job names use
`equivariant` / `baseline` for the same distinction.

## Logs

`logs/` collects `%x_%A_%a.out` / `.err` per array task and is not included in the
repository. The launchers `mkdir -p` it, so a fresh copy needs no setup.
