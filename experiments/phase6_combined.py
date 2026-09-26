"""Phase 6, step 2: combinations of the single steps, plus per-entity scores for a paired significance test.

Chosen from experiments/phase6_results.tsv (single steps):
  helped          : address_abbrev (+0.0029), accents (+0.0004)
  neutral (±0.0005): invisible, web, trade_name, null, zeros, ampersand
  hurt            : dots, legal_canonical, legal_remove  -> never used

Run from the project root:   python experiments/phase6_combined.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import score_per_entity
from src.rule_experiment import load_query_sets, run
from experiments.phase6_normalization import already_done, save

PER_ENTITY_DIR = Path(__file__).resolve().parent.parent / "output" / "experiments" / "phase6"

COMBINED = {
    "combo_0_baseline": ((), ()),
    "combo_1_helped": (("accents",), ("accents", "address_abbrev")),
    "combo_2_helped_plus_neutral": (
        ("trade_name", "invisible", "accents", "web"),
        ("invisible", "accents", "address_abbrev", "null", "zeros"),
    ),
}


def main():
    PER_ENTITY_DIR.mkdir(parents=True, exist_ok=True)
    s1, truth = load_query_sets()
    done = already_done()
    for name, (name_steps, address_steps) in COMBINED.items():
        if name in done:
            print("skip (already done):", name)
            continue
        print("running:", name, name_steps, address_steps, flush=True)
        result, pairs = run(name_steps, address_steps, s1=s1, truth=truth)

        chosen = pairs[(pairs["query_set"] == "validation") & (pairs["score"] >= result["threshold"])]
        predictions = chosen.groupby("source1_entity_id")["candidate_id"].agg(",".join).rename("matched_entity_ids").reset_index()
        per_entity = score_per_entity(predictions, truth["validation"])
        per_entity[["source1_entity_id", "n_true", "n_pred", "n_correct", "f05"]].to_csv(
            PER_ENTITY_DIR / f"{name}.tsv", sep="\t", index=False, lineterminator="\n")

        save({
            "experiment": name,
            "name_steps": "+".join(name_steps),
            "address_steps": "+".join(address_steps),
            **{k: (round(float(v), 4) if isinstance(v, float) or hasattr(v, "dtype") else v) for k, v in result.items()},
        })


if __name__ == "__main__":
    main()
