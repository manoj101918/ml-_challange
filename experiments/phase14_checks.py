"""Phase 14: sanity checks of the test submission (memory-friendly for an 8 GB laptop).

The full rule check of both files was done by the official validator on Kaggle (PASS), and locally
`validate_submission.py --check-ids` passed for matching_results.tsv. This script adds:
  1. a byte scan of candidate_pairs.tsv (row / ID counts, empty rows, line endings) — seconds, no parsing
  2. what the predictions look like (per country, per source, one-owner rule)
  3. a comparison with the validation predictions of the same model (E4)

    python experiments/phase14_checks.py        (from the project root; writes experiments/phase14_checks.txt)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
out_lines = []


def report(text=""):
    print(text, flush=True)
    out_lines.append(str(text))


s1 = pd.read_parquet(ROOT / "cache" / "test_source1.parquet", columns=["entity_id", "country"])

# ---------------------------------------------------------------- 1. candidate file, byte scan
report("=== 1. candidate_pairs.tsv (byte scan) ===")
commas = newlines = empty_rows = 0
carriage_returns = False
tail = b""
with open(ROOT / "output" / "candidate_pairs.tsv", "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        commas += block.count(b",")
        newlines += block.count(b"\n")
        empty_rows += (tail + block).count(b"\t\n") - tail.count(b"\t\n")
        carriage_returns |= b"\r" in block
        tail = block[-1:]
rows = newlines - 1                                  # minus the header line
n_ids = commas + (rows - empty_rows)                 # a non-empty row with k commas holds k + 1 IDs
pool_country = pd.read_parquet(ROOT / "cache" / "test_pool_norm_8b33bbf7.parquet", columns=["country"])["country"].value_counts()
possible = int((s1["country"].value_counts() * pool_country.reindex(s1["country"].value_counts().index)).sum())
report(f"rows: {rows:,} (expected {len(s1):,}) | empty rows: {empty_rows} | Windows line endings: {carriage_returns}")
report(f"candidate pairs: {n_ids:,} (avg {n_ids / (rows - empty_rows):.1f} per entity with candidates)")
report(f"possible same-country pairs: {possible:,} -> reduction ratio {1 - n_ids / possible:.6%}")

# ---------------------------------------------------------------- 2. predictions
report("\n=== 2. matching_results.tsv — what the predictions look like ===")
matching = pd.read_csv(ROOT / "output" / "matching_results.tsv", sep="\t", keep_default_na=False)
matching["n_pred"] = matching["matched_entity_ids"].map(lambda s: len(s.split(",")) if s else 0)
matching = matching.merge(s1, left_on="source1_entity_id", right_on="entity_id")
all_ids = matching.loc[matching["n_pred"] > 0, "matched_entity_ids"].str.split(",").explode()
report(f"predicted pairs: {len(all_ids):,} | S2/S3 IDs used by two S1 entities (one-owner rule): {int(all_ids.duplicated().sum())}")
report(f"share of predicted IDs from S2 / S3: {all_ids.str.startswith('S2').mean():.3f} / {all_ids.str.startswith('S3').mean():.3f}")
by_country = matching.groupby("country").agg(entities=("n_pred", "size"),
                                             predicted_no_match_pct=("n_pred", lambda x: round((x == 0).mean() * 100, 1)),
                                             avg_matches=("n_pred", "mean"))
report(by_country.round(2).to_string())

# ---------------------------------------------------------------- 3. comparison with validation
report("\n=== 3. comparison with validation (model E4 on 20k held-out training entities) ===")
val = pd.read_csv(ROOT / "output" / "experiments" / "phase12" / "E4.tsv", sep="\t")
comparison = pd.DataFrame({
    "test predicted %": matching["n_pred"].clip(upper=8).value_counts(normalize=True).sort_index(),
    "validation predicted %": val["n_pred"].clip(upper=8).value_counts(normalize=True).sort_index(),
    "validation TRUE %": val["n_true"].clip(upper=8).value_counts(normalize=True).sort_index(),
}).fillna(0).mul(100).round(1)
comparison.index.name = "matches per entity (8 = 8+)"
report(comparison.to_string())
report(f"avg matches per entity: test {matching['n_pred'].mean():.2f} | validation predicted {val['n_pred'].mean():.2f} "
       f"| validation true {val['n_true'].mean():.2f}")

(ROOT / "experiments" / "phase14_checks.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
