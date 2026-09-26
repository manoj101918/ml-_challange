"""Phase 12 experiments — thin wrapper, the code lives in src/train.py.

    python experiments/phase12_run.py E0 E1 ...      (same as: python -m src.train E0 E1 ...)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import EXPERIMENTS, run

if __name__ == "__main__":
    for experiment in sys.argv[1:]:
        run(experiment, **EXPERIMENTS[experiment])
