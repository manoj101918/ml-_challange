"""Round 2: how much do REVERSE competition features help the classifier?

Phase 8 pairs (20k tuning + 20k small-validation entities, K = 50). For every candidate record we look up its top 10 S1
among ALL train S1 of its country (reverse blocking, full density) and add per pair:
    r_rank        rank of this S1 among the record's reverse candidates (11 = not in its top 10)
    r_score       this S1's reverse idf score (0 if absent)
    r_best_other  best reverse score of ANOTHER S1 for the same record
    r_gap         r_score - r_best_other          (> 0: this S1 is the record's favourite)
Then HGB with FAST features vs FAST + reverse features (same OOF threshold procedure as experiments/xgb_check.py).

    python experiments/competition_check.py        (reverse table cached in cache/phase8_reverse.parquet)
"""

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

from src.blocking import build_index_hashed, iter_candidates
from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_report
from src.features import FAST_FEATURE_COLUMNS
from src.pipeline import CACHE_DIR, normalized_pool, prepare_queries
from src.rule_experiment import load_query_sets
from src.train import THRESHOLDS, TRANSLIT_ADDRESS, TRANSLIT_NAME, make_model

K_REV, MAX_DF, QUERY_CHUNK = 10, 1000, 400_000
REVERSE_COLUMNS = ["r_rank", "r_score", "r_best_other", "r_gap"]
start = time.time()
log = lambda msg: print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)

pairs = pd.read_parquet(CACHE_DIR / "phase8_features.parquet",
                        columns=["source1_entity_id", "candidate_id", "query_set", "label"] + FAST_FEATURE_COLUMNS)
reverse_path = CACHE_DIR / "phase8_reverse.parquet"
if not reverse_path.exists():
    pool_path = normalized_pool("train", TRANSLIT_NAME, TRANSLIT_ADDRESS, log=log)
    records = pq.read_table(pool_path, columns=["entity_id", "country", "name_norm", "addr_norm"],
                            filters=[("entity_id", "in", pairs["candidate_id"].unique().tolist())]).to_pandas()
    s1_all = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
    n_s1 = len(s1_all)
    log(f"{len(records):,} candidate records, {n_s1:,} train S1")
    parts = []
    for country in sorted(records["country"].unique()):
        s1_c = prepare_queries(s1_all[s1_all["country"] == country].reset_index(drop=True), TRANSLIT_NAME, TRANSLIT_ADDRESS)
        s1_c = s1_c[["entity_id", "country", "name_norm", "addr_norm"]]
        rec_c = records[records["country"] == country].reset_index(drop=True)
        for first in range(0, len(rec_c), QUERY_CHUNK):
            q = rec_c.iloc[first:first + QUERY_CHUNK].reset_index(drop=True)
            index = build_index_hashed(q, (s1_c.iloc[i:i + 250_000] for i in range(0, len(s1_c), 250_000)),
                                       n_pool=n_s1, max_df=MAX_DF, log=log)
            for c in iter_candidates(q, index, top_k=K_REV, batch_size=2_000, yield_every=20_000):
                parts.append(c[["source1_entity_id", "candidate_id", "score_all"]].rename(
                    columns={"source1_entity_id": "record_id", "candidate_id": "s1_id", "score_all": "r_score"}))
            del index
            gc.collect()
            log(f"  {country}: {min(first + QUERY_CHUNK, len(rec_c)):,}/{len(rec_c):,} records")
        del s1_c
        gc.collect()
    pd.concat(parts, ignore_index=True).to_parquet(reverse_path, index=False)
    del records, s1_all, parts
    gc.collect()

rev = pd.read_parquet(reverse_path)
rev["r_rank"] = rev.groupby("record_id")["r_score"].rank(method="first", ascending=False).astype("float32")
best = rev[rev["r_rank"] == 1].set_index("record_id")
second = rev[rev["r_rank"] == 2].set_index("record_id")["r_score"]
pairs = pairs.merge(rev.rename(columns={"record_id": "candidate_id", "s1_id": "source1_entity_id"}),
                    on=["source1_entity_id", "candidate_id"], how="left")
best_score = pairs["candidate_id"].map(best["r_score"]).fillna(0).to_numpy()
best_id = pairs["candidate_id"].map(best["s1_id"]).to_numpy()
second_score = pairs["candidate_id"].map(second).fillna(0).to_numpy()
pairs["r_best_other"] = np.where(best_id == pairs["source1_entity_id"].to_numpy(), second_score, best_score).astype("float32")
pairs["r_rank"] = pairs["r_rank"].fillna(K_REV + 1)
pairs["r_score"] = pairs["r_score"].fillna(0).astype("float32")
pairs["r_gap"] = (pairs["r_score"] - pairs["r_best_other"]).astype("float32")
log(f"pairs with this S1 in the record's reverse top {K_REV}: {(pairs['r_rank'] <= K_REV).mean():.3f} "
    f"(true pairs {(pairs.loc[pairs['label'] == 1, 'r_rank'] <= K_REV).mean():.3f}); "
    f"true pairs where this S1 is the record's favourite: {(pairs.loc[pairs['label'] == 1, 'r_rank'] == 1).mean():.3f}")

_, truth = load_query_sets()
train = pairs[pairs["query_set"] == "tuning"].reset_index(drop=True)
valid = pairs[pairs["query_set"] == "validation"].reset_index(drop=True)


def macro(table, p, t, truth_table, per_entity=False):
    chosen = to_submission(resolve_conflicts(global_threshold(table.assign(probability=p), t)))
    return score_report(chosen, truth_table)


results = {}
for label, columns in [("fast features", FAST_FEATURE_COLUMNS), ("fast + reverse", FAST_FEATURE_COLUMNS + REVERSE_COLUMNS)]:
    X, y = train[columns].astype("float32"), train["label"].to_numpy()
    oof = np.zeros(len(train))
    for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=train["source1_entity_id"]):
        oof[pred_idx] = make_model("hgb").fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[pred_idx])[:, 1]
    threshold = max(THRESHOLDS, key=lambda t: macro(train, oof, t, truth["tuning"])["macro_F0.5"])
    p = make_model("hgb").fit(X, y).predict_proba(valid[columns].astype("float32"))[:, 1]
    report = macro(valid, p, threshold, truth["validation"])
    results[label] = report
    log(f"{label:15s}: threshold {threshold} | validation F0.5 {report['macro_F0.5']:.4f} | precision "
        f"{report['pair precision']:.4f} | recall {report['pair recall']:.4f} | singletons {report['F0.5 on singletons']:.4f}")
log(f"gain from reverse features: {results['fast + reverse']['macro_F0.5'] - results['fast features']['macro_F0.5']:+.4f}")
