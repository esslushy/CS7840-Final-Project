#!/usr/bin/env python3
"""Exit 0 if a results/*_statistics.json file shows a finished training run
(at least NUM_EPOCHS + 1 recorded entries: the initial pre-training eval plus
one per epoch -- see save_all()/update_statistics() in the training scripts),
exit 1 otherwise (missing, unreadable, or still in progress).

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
sys.exit(0 if len(train_loss) >= num_epochs + 1 else 1)
