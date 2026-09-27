"""Do the teammate's 10 extra features (src/features.EXTRA_FEATURE_COLUMNS) add anything ON TOP of the reverse
competition features? Phase 8 pairs (20k tuning / 20k validation), same HGB + OOF-threshold procedure as
experiments/competition_check.py (needs its cache/phase8_reverse.parquet).

    python experiments/extra_features_check.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_report
from src.features import EXTRA_FEATURE_COLUMNS, FAST_FEATURE_COLUMNS, add_features
from src.pipeline import CACHE_DIR
from src.reverse import REVERSE_COLUMNS, ReverseLookup, add_reverse_features
from src.rule_experiment import load_query_sets
from src.train import THRESHOLDS, make_model

start = time.time()
log = lambda msg: print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)

pairs = pd.read_parquet(CACHE_DIR / "phase8_features.parquet",
                        columns=["source1_entity_id", "candidate_id", "query_set", "label", "name_s1", "addr_s1", "name_c",
                                 "addr_c", "name_c_foreign_script", "block_score", "block_score_name", "block_score_addr"])
pairs = pairs.rename(columns={"block_score": "score_all", "block_score_name": "score_name", "block_score_addr": "score_addr",
                              "name_c_foreign_script": "name_c_foreign"})
pairs["s1_row"] = pd.factorize(pairs["source1_entity_id"])[0]
parts = [add_features(pairs.iloc[i:i + 250_000]) for i in range(0, len(pairs), 250_000)]
pairs = pd.concat(parts, ignore_index=True)
log(f"features for {len(pairs):,} pairs")
rev = pd.read_parquet(CACHE_DIR / "phase8_reverse.parquet")
rev["r_rank"] = rev.groupby("record_id")["r_score"].rank(method="first", ascending=False).astype("int8")
pairs = add_reverse_features(pairs, ReverseLookup(rev))

_, truth = load_query_sets()
train = pairs[pairs["query_set"] == "tuning"].reset_index(drop=True)
valid = pairs[pairs["query_set"] == "validation"].reset_index(drop=True)


def macro(table, p, t, truth_table):
    return score_report(to_submission(resolve_conflicts(global_threshold(table.assign(probability=p), t))), truth_table)


per_entity = {}
for label, columns in [("fast + reverse", FAST_FEATURE_COLUMNS + REVERSE_COLUMNS),
                       ("fast + reverse + extra", FAST_FEATURE_COLUMNS + REVERSE_COLUMNS + EXTRA_FEATURE_COLUMNS)]:
    X, y = train[columns].astype("float32"), train["label"].to_numpy()
    oof = np.zeros(len(train))
    for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=train["source1_entity_id"]):
        oof[pred_idx] = make_model("hgb").fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[pred_idx])[:, 1]
    threshold = max(THRESHOLDS, key=lambda t: macro(train, oof, t, truth["tuning"])["macro_F0.5"])
    p = make_model("hgb").fit(X, y).predict_proba(valid[columns].astype("float32"))[:, 1]
    report = macro(valid, p, threshold, truth["validation"])
    log(f"{label:24s}: threshold {threshold} | validation F0.5 {report['macro_F0.5']:.4f} | precision "
        f"{report['pair precision']:.4f} | recall {report['pair recall']:.4f}")
    from src.evaluation import score_per_entity
    chosen = to_submission(resolve_conflicts(global_threshold(valid.assign(probability=p), threshold)))
    per_entity[label] = score_per_entity(chosen, truth["validation"]).set_index("source1_entity_id")["f05"]

a, b = per_entity["fast + reverse"], per_entity["fast + reverse + extra"]
diff = (b - a.reindex(b.index)).to_numpy()
rng = np.random.default_rng(0)
boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
log(f"extra features on top of reverse: {diff.mean():+.4f}  95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")
