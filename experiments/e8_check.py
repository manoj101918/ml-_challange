"""E8 check: do the "noise model" features (src/stage2.NOISE_FEATURES) improve STAGE 2?

Phase 8 pairs (20k tuning / 20k validation) with the E7 stage-1 feature set (fast + reverse + extra), HGB stage 1
(out-of-fold p1 on tuning, final model on validation) -> stage 2 (src/stage2.py) with and without NOISE_FEATURES,
same HGB_STAGE2 / threshold procedure as src/train.train_stage2; paired bootstrap on the validation entities.

    python experiments/e8_check.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_per_entity, score_report
from src.features import EXTRA_FEATURE_COLUMNS, FAST_FEATURE_COLUMNS, add_features
from src.pipeline import CACHE_DIR
from src.preprocessing import ADDRESS_STEPS, NAME_STEPS
from src.reverse import REVERSE_COLUMNS, ReverseLookup, add_reverse_features
from src.rule_experiment import load_query_sets
from src.stage2 import NOISE_FEATURES, feature_columns, noise_features, stage2_features, word_frequencies
from src.train import HGB_STAGE2, THRESHOLDS, THRESHOLDS_STAGE2, make_model

start = time.time()
log = lambda msg: print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)

pairs = pd.read_parquet(CACHE_DIR / "phase8_features.parquet",
                        columns=["source1_entity_id", "candidate_id", "query_set", "label", "name_s1", "addr_s1", "name_c",
                                 "addr_c", "name_c_foreign_script", "block_score", "block_score_name", "block_score_addr"])
pairs = pairs.rename(columns={"block_score": "score_all", "block_score_name": "score_name", "block_score_addr": "score_addr",
                              "name_c_foreign_script": "name_c_foreign"})
pairs["s1_row"] = pd.factorize(pairs["source1_entity_id"])[0]
pairs = pd.concat([add_features(pairs.iloc[i:i + 250_000]) for i in range(0, len(pairs), 250_000)], ignore_index=True)
rev = pd.read_parquet(CACHE_DIR / "phase8_reverse.parquet")
rev["r_rank"] = rev.groupby("record_id")["r_score"].rank(method="first", ascending=False).astype("int8")
pairs = add_reverse_features(pairs, ReverseLookup(rev))
del rev
_, truth = load_query_sets()
columns = FAST_FEATURE_COLUMNS + REVERSE_COLUMNS + EXTRA_FEATURE_COLUMNS
train = pairs[pairs["query_set"] == "tuning"].reset_index(drop=True)
valid = pairs[pairs["query_set"] == "validation"].reset_index(drop=True)
X, y = train[columns].astype("float32"), train["label"].to_numpy()
train["p1"] = 0.0
for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=train["source1_entity_id"]):
    train.loc[pred_idx, "p1"] = make_model("hgb").fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[pred_idx])[:, 1]
valid["p1"] = make_model("hgb").fit(X, y).predict_proba(valid[columns].astype("float32"))[:, 1]
log("stage 1 done")


def macro(table, p, t, truth_table):
    return score_report(to_submission(resolve_conflicts(global_threshold(table.assign(probability=p), t))), truth_table)


# phase 8 texts use the E0 normalization -> word frequencies with the same steps
freq = word_frequencies("train", NAME_STEPS, ADDRESS_STEPS, log=log)
s2_t, s2_v = stage2_features(train), stage2_features(valid)
t0 = time.time()
s2_t, s2_v = noise_features(s2_t, freq), noise_features(s2_v, freq)
log(f"noise features for {len(s2_t) + len(s2_v):,} stage-2 pairs ({time.time() - t0:.0f}s)")
for name, part in [("true", s2_v[s2_v.label == 1]), ("false", s2_v[s2_v.label == 0])]:
    log(f"  {name:5s} pairs: num_conflict_close {part.num_conflict_close.mean():.3f} | rec_unmatched {part.rec_unmatched.mean():.2f} "
        f"| rec_unmatched_freq median {part.rec_unmatched_freq.median():.2f}")

per_entity = {}
for label, cols in [("stage 2 (E7)", [c for c in feature_columns(s2_t) if c not in NOISE_FEATURES]),
                    ("stage 2 + noise (E8)", feature_columns(s2_t))]:
    Xs, ys = s2_t[cols].astype("float32"), s2_t["label"].to_numpy()
    oof = np.zeros(len(s2_t))
    for fit_idx, pred_idx in GroupKFold(n_splits=3).split(Xs, ys, groups=s2_t["source1_entity_id"]):
        oof[pred_idx] = HistGradientBoostingClassifier(**HGB_STAGE2).fit(Xs.iloc[fit_idx], ys[fit_idx]).predict_proba(Xs.iloc[pred_idx])[:, 1]
    t = max(THRESHOLDS_STAGE2, key=lambda t: macro(s2_t, oof, t, truth["tuning"])["macro_F0.5"])
    p = HistGradientBoostingClassifier(**HGB_STAGE2).fit(Xs, ys).predict_proba(s2_v[cols].astype("float32"))[:, 1]
    report = macro(s2_v, p, t, truth["validation"])
    log(f"{label:22s}: {len(cols)} features, threshold {t} | validation F0.5 {report['macro_F0.5']:.4f} | precision "
        f"{report['pair precision']:.4f} | recall {report['pair recall']:.4f}")
    chosen = to_submission(resolve_conflicts(global_threshold(s2_v.assign(probability=p), t)))
    per_entity[label] = score_per_entity(chosen, truth["validation"]).set_index("source1_entity_id")["f05"]

a, b = per_entity["stage 2 (E7)"], per_entity["stage 2 + noise (E8)"]
diff = (b - a.reindex(b.index)).to_numpy()
rng = np.random.default_rng(0)
boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
log(f"noise features in stage 2: {diff.mean():+.4f}  95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")
