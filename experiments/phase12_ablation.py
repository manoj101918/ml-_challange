"""Phase 12: feature ablation — how much do the slow TF-IDF features contribute?

Uses the saved Phase 8 features (20k tuning + 20k validation entities, no transliteration) and the Phase 9 model settings.
Threshold chosen on 3-fold out-of-fold tuning predictions, validation only scored.
Output: experiments/phase12_ablation.tsv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_report
from src.features import FEATURE_COLUMNS
from src.rule_experiment import load_query_sets

HGB = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
           early_stopping=False, random_state=0)
THRESHOLDS = np.round(np.arange(0.50, 0.91, 0.025), 3)
VARIANTS = {
    "all 34 features": FEATURE_COLUMNS,
    "without TF-IDF (name_tfidf, addr_tfidf)": [c for c in FEATURE_COLUMNS if c not in ("name_tfidf", "addr_tfidf")],
}

pairs = pd.read_parquet(Path(__file__).resolve().parent.parent / "cache" / "phase8_features.parquet")
_, truth = load_query_sets()
train = pairs[pairs["query_set"] == "tuning"].reset_index(drop=True)
valid = pairs[pairs["query_set"] == "validation"].reset_index(drop=True)


def macro(table, probabilities, threshold, truth_table):
    chosen = resolve_conflicts(global_threshold(table.assign(probability=probabilities), threshold))
    return score_report(to_submission(chosen), truth_table)["macro_F0.5"]


rows = []
for name, columns in VARIANTS.items():
    X, y = train[columns].astype("float32"), train["label"].to_numpy()
    oof = np.zeros(len(train))
    for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=train["source1_entity_id"]):
        oof[pred_idx] = HistGradientBoostingClassifier(**HGB).fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[pred_idx])[:, 1]
    scores = [macro(train, oof, t, truth["tuning"]) for t in THRESHOLDS]
    threshold = float(THRESHOLDS[int(np.argmax(scores))])
    p = HistGradientBoostingClassifier(**HGB).fit(X, y).predict_proba(valid[columns].astype("float32"))[:, 1]
    rows.append({"variant": name, "features": len(columns), "threshold": threshold,
                 "tuning_F0.5_oof": round(max(scores), 4), "validation_F0.5": round(macro(valid, p, threshold, truth["validation"]), 4)})
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent / "phase12_ablation.tsv", sep="\t", index=False, lineterminator="\n")
