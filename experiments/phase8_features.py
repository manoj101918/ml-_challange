"""Phase 8: compute pair features for the Phase 7 candidates (tuning + small-validation entities).

Output:
  cache/phase8_features.parquet          features + ids + query_set + label (label only for analysis / training)
  experiments/phase8_feature_auc.tsv     how well each feature ALONE separates true from false pairs (tuning set)
Run from the project root:   python experiments/phase8_features.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from src.features import FEATURE_COLUMNS, add_features, fit_tfidf
from src.rule_experiment import CACHE_DIR, load_query_sets

start = time.time()
log = lambda msg: print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)

candidates = pd.read_parquet(CACHE_DIR / "phase7_candidates.parquet")
queries = pd.read_parquet(CACHE_DIR / "phase7_queries.parquet")
pool_norm = pq.read_table(CACHE_DIR / "train_pool_norm.parquet", columns=["entity_id", "name_norm", "addr_norm"])

# TF-IDF vocabularies / idf from 1M random UNLABELED pool records
rng = np.random.default_rng(0)
sample = pool_norm.take(np.sort(rng.choice(pool_norm.num_rows, 1_000_000, replace=False))).to_pandas()
name_vectorizer = fit_tfidf(sample["name_norm"])
addr_vectorizer = fit_tfidf(sample["addr_norm"])
del sample
log("tf-idf fitted")

rows = np.sort(candidates["pool_row"].unique())
text = pool_norm.take(rows).to_pandas()
del pool_norm
pairs = (
    candidates[["source1_entity_id", "candidate_id", "s1_row", "score_all", "score_name", "score_addr"]]
    .merge(queries[["entity_id", "query_set", "name_norm", "addr_norm"]].rename(columns={"name_norm": "name_s1", "addr_norm": "addr_s1"}),
           left_on="source1_entity_id", right_on="entity_id").drop(columns="entity_id")
    .merge(text.rename(columns={"name_norm": "name_c", "addr_norm": "addr_c"}), left_on="candidate_id", right_on="entity_id")
    .drop(columns="entity_id")
    .reset_index(drop=True)
)
log(f"{len(pairs):,} pairs with text")

pairs = add_features(pairs, name_vectorizer, addr_vectorizer)
log("features computed")

_, truth = load_query_sets()
true_keys = set()
for t in truth.values():
    for s1_id, ids in zip(t["source1_entity_id"], t["matched_entity_ids"]):
        true_keys.update((s1_id, m) for m in ids.split(",") if m)
pairs["label"] = np.array([(a, b) in true_keys for a, b in zip(pairs["source1_entity_id"], pairs["candidate_id"])], dtype="int8")

keep = ["source1_entity_id", "candidate_id", "query_set", "name_s1", "addr_s1", "name_c", "addr_c", "label"] + FEATURE_COLUMNS
pairs[keep].to_parquet(CACHE_DIR / "phase8_features.parquet", index=False)
log(f"saved; positives {pairs['label'].sum():,} of {len(pairs):,}")

tuning = pairs[pairs["query_set"] == "tuning"]
auc = []
for col in FEATURE_COLUMNS:
    values = tuning[col].astype("float64")
    known = values.notna()
    score = roc_auc_score(tuning.loc[known, "label"], values[known]) if known.any() else np.nan
    auc.append({
        "feature": col,
        "AUC": round(score, 4),
        "direction": "higher = match" if score >= 0.5 else "lower = match",
        "separation": round(abs(score - 0.5) * 2, 4),
        "missing_%": round((~known).mean() * 100, 1),
        "mean_true": round(values[tuning["label"] == 1].mean(), 3),
        "mean_false": round(values[tuning["label"] == 0].mean(), 3),
    })
pd.DataFrame(auc).sort_values("separation", ascending=False).to_csv(
    Path(__file__).resolve().parent / "phase8_feature_auc.tsv", sep="\t", index=False, lineterminator="\n")
log("feature AUC saved")
