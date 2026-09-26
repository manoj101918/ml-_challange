"""Phase 6: test each normalization step ON ITS OWN against the baseline, then combinations.

Run from the project root:   python experiments/phase6_normalization.py [single|combined]
Results are appended to experiments/phase6_results.tsv (one row per experiment).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rule_experiment import load_query_sets, run

RESULTS = Path(__file__).resolve().parent / "phase6_results.tsv"

SINGLE_STEPS = {
    # experiment name: (name steps, address steps)
    "baseline": ((), ()),
    "accents": (("accents",), ("accents",)),
    "invisible": (("invisible",), ("invisible",)),
    "trade_name": (("trade_name",), ()),
    "web": (("web",), ()),
    "dots": (("dots",), ()),
    "ampersand": (("ampersand",), ()),
    "legal_canonical": (("legal_canonical",), ()),
    "legal_remove": (("legal_remove",), ()),
    "address_abbrev": ((), ("address_abbrev",)),
    "null": ((), ("null",)),
    "zeros": ((), ("zeros",)),
}


def already_done():
    if not RESULTS.exists():
        return set()
    with open(RESULTS, encoding="utf-8") as f:
        return {row["experiment"] for row in csv.DictReader(f, delimiter="\t")}


def save(row):
    new_file = not RESULTS.exists()
    with open(RESULTS, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_all(experiments):
    s1, truth = load_query_sets()
    done = already_done()
    for name, (name_steps, address_steps) in experiments.items():
        if name in done:
            print("skip (already done):", name)
            continue
        print("running:", name, name_steps, address_steps, flush=True)
        result, _ = run(name_steps, address_steps, s1=s1, truth=truth)
        save({
            "experiment": name,
            "name_steps": "+".join(name_steps),
            "address_steps": "+".join(address_steps),
            **{k: (round(float(v), 4) if isinstance(v, float) or hasattr(v, "dtype") else v) for k, v in result.items()},
        })


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "single":
        run_all(SINGLE_STEPS)
    else:
        # filled in after the single-step results are known
        from experiments.phase6_combined import COMBINED
        run_all(COMBINED)
