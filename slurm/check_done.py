#!/usr/bin/env python3
"""Exit 0 if a results/*_statistics.json file shows a finished training run
(at least NUM_EPOCHS + 1 recorded entries: the initial pre-training eval plus
one per epoch -- see save_all()/update_statistics() in the training scripts)
scored with the cycle pairing, exit 1 otherwise (missing, unreadable, still
in progress, or an old star-pairing file).

Usage: check_done.py <path/to/*_statistics.json> <num_epochs>
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
num_epochs = int(sys.argv[2])

if not path.exists():
    sys.exit(1)

try:
    statistics = json.loads(path.read_text())
except (json.JSONDecodeError, OSError):
    sys.exit(1)

train_loss = statistics.get("train_loss", [])
# Files from before the switch to the cycle pairing (no "equivariance_pairing"
# key, see utils.cycle_pairs) measured a different quantity, so they never
# count as done: a launcher reruns them rather than keeping star-pairing
# results that figures/aggregate.py refuses to pool with cycle ones.
is_cycle = statistics.get("equivariance_pairing") == "cycle"
sys.exit(0 if is_cycle and len(train_loss) >= num_epochs + 1 else 1)
