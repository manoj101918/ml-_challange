"""Phase 7, step 2: chosen blocking (max_df=1000, ranking="all", K=50) + the Phase 6 rule scorer.

Saves the candidate table to cache/phase7_candidates.parquet so Phase 8 can build features without re-blocking,
and appends the result to experiments/phase7_end_to_end.tsv.
Run from the project root:   python experiments/phase7_end_to_end.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.blocking import build_index, generate_candidates
from src.preprocessing import ADDRESS_STEPS, NAME_STEPS, normalize
from src.rule_experiment import CACHE_DIR, load_query_sets, score_candidates

MAX_DF, RANKING, TOP_K = 1000, "all", 50

s1, truth = load_query_sets()
s1["name_norm"] = normalize(s1["business_name"], NAME_STEPS)
s1["addr_norm"] = normalize(s1["business_address"], ADDRESS_STEPS)

index = build_index(s1, CACHE_DIR / "train_pool_norm.parquet", max_df=MAX_DF)
candidates = generate_candidates(s1, index, top_k=TOP_K, ranking=RANKING)
candidates.to_parquet(CACHE_DIR / "phase7_candidates.parquet", index=False)
s1[["entity_id", "query_set", "country", "business_name", "business_address", "name_norm", "addr_norm"]].to_parquet(
    CACHE_DIR / "phase7_queries.parquet", index=False)
print(f"candidates: {len(candidates):,}", flush=True)

result, _ = score_candidates(candidates, s1, truth)
row = {"blocking": f"5 key types, max_df={MAX_DF}, ranking={RANKING}, K={TOP_K}", "scorer": "Jaccard mean (Phase 6 rule)",
       **{k: round(float(v), 4) if not isinstance(v, (int, str)) else v for k, v in result.items()}}
out = Path(__file__).resolve().parent / "phase7_end_to_end.tsv"
pd.DataFrame([row]).to_csv(out, sep="\t", index=False, lineterminator="\n")
print(row)
