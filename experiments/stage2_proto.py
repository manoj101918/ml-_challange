"""Quick test of the stage-2 "collective" model on data already on disk (Phase 8 features + Phase 9 probabilities).

Stage 1 = Phase 9 HGB: out-of-fold p1 on the 20k tuning entities, final-model p1 on the 20k validation entities
(baseline validation F0.5 0.934). Stage 2 is trained on the tuning entities only; validation is only scored.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_per_entity, score_report
from src.rule_experiment import load_query_sets
from src.stage2 import MIN_P1, feature_columns, stage2_features

HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
           early_stopping=False, random_state=0)
THRESHOLDS = np.round(np.arange(0.30, 0.91, 0.025), 3)

features = pd.read_parquet(ROOT / "cache" / "phase8_features.parquet")
probs = pd.read_parquet(ROOT / "cache" / "phase9_predictions.parquet")[["source1_entity_id", "candidate_id", "probability"]]
pairs = features.merge(probs.rename(columns={"probability": "p1"}), on=["source1_entity_id", "candidate_id"])
_, truth = load_query_sets()


def macro(table, p, t, truth_table, per_entity=False):
    chosen = resolve_conflicts(global_threshold(table.assign(probability=p), t))
    if per_entity:
        return score_per_entity(to_submission(chosen), truth_table).set_index("source1_entity_id")["f05"]
    return score_report(to_submission(chosen), truth_table)["macro_F0.5"]


tuning = pairs[pairs["query_set"] == "tuning"].reset_index(drop=True)
valid = pairs[pairs["query_set"] == "validation"].reset_index(drop=True)

base_t = max(THRESHOLDS, key=lambda t: macro(tuning, tuning["p1"], t, truth["tuning"]))
base_scores = macro(valid, valid["p1"], base_t, truth["validation"], per_entity=True)
print(f"baseline stage 1: threshold {base_t} -> validation F0.5 {base_scores.mean():.4f}")
for name, part in [("tuning", tuning), ("validation", valid)]:
    kept = part[part["p1"] >= MIN_P1]
    print(f"{name}: pairs {len(part):,} -> {len(kept):,} with p1 >= {MIN_P1} "
          f"({len(kept) / part['source1_entity_id'].nunique():.1f} per entity); true pairs kept {kept['label'].sum() / part['label'].sum():.4f}")

s2_t, s2_v = stage2_features(tuning), stage2_features(valid)
cols = feature_columns(s2_t)
X, y = s2_t[cols].astype("float32"), s2_t["label"].to_numpy()
oof = np.zeros(len(s2_t))
for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=s2_t["source1_entity_id"]):
    oof[pred_idx] = HistGradientBoostingClassifier(**HGB).fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[pred_idx])[:, 1]
t2 = max(THRESHOLDS, key=lambda t: macro(s2_t, oof, t, truth["tuning"]))
model = HistGradientBoostingClassifier(**HGB).fit(X, y)
p2 = model.predict_proba(s2_v[cols].astype("float32"))[:, 1]
s2_scores = macro(s2_v, p2, t2, truth["validation"], per_entity=True)
print(f"stage 2: threshold {t2} | tuning OOF F0.5 {macro(s2_t, oof, t2, truth['tuning']):.4f} | validation F0.5 {s2_scores.mean():.4f}")

diff = (s2_scores - base_scores.reindex(s2_scores.index)).to_numpy()
rng = np.random.default_rng(0)
boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
print(f"stage 2 vs stage 1: {diff.mean():+.4f}  95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]  "
      f"better {int((diff > 0).sum())} / worse {int((diff < 0).sum())}")
report = score_report(to_submission(resolve_conflicts(global_threshold(s2_v.assign(probability=p2), t2))), truth["validation"])
print(report.round(4).to_string())
