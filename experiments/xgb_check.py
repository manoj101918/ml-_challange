"""Check of the XGBoost learner (src/train.make_model("xgb")) before the Colab GPU run: same Phase 8 pairs
(20k tuning -> 3-fold OOF threshold, 20k validation scored), fast features, HGB vs XGBoost. Also checks that the saved
XGBoost model (device switched to cpu) loads and predicts like the in-memory one.

    set XGB_DEVICE=cpu && python experiments/xgb_check.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_report
from src.features import FAST_FEATURE_COLUMNS
from src.rule_experiment import load_query_sets
from src.train import THRESHOLDS, make_model

pairs = pd.read_parquet(ROOT / "cache" / "phase8_features.parquet")
_, truth = load_query_sets()
train = pairs[pairs["query_set"] == "tuning"].reset_index(drop=True)
valid = pairs[pairs["query_set"] == "validation"].reset_index(drop=True)
X, y = train[FAST_FEATURE_COLUMNS].astype("float32"), train["label"].to_numpy()
Xv = valid[FAST_FEATURE_COLUMNS].astype("float32")


def macro(table, p, t, truth_table):
    return score_report(to_submission(resolve_conflicts(global_threshold(table.assign(probability=p), t))), truth_table)["macro_F0.5"]


for learner in ["hgb", "xgb"]:
    start = time.time()
    oof = np.zeros(len(train))
    for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=train["source1_entity_id"]):
        oof[pred_idx] = make_model(learner).fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[pred_idx])[:, 1]
    threshold = max(THRESHOLDS, key=lambda t: macro(train, oof, t, truth["tuning"]))
    model = make_model(learner).fit(X, y)
    if learner == "xgb":
        model.set_params(device="cpu")
    p = model.predict_proba(Xv)[:, 1]
    print(f"{learner}: threshold {threshold} | tuning OOF {macro(train, oof, threshold, truth['tuning']):.4f} "
          f"| validation F0.5 {macro(valid, p, threshold, truth['validation']):.4f} | {time.time() - start:.0f}s", flush=True)

path = ROOT / "cache" / "xgb_check.joblib"
joblib.dump({"model": model}, path)
p_loaded = joblib.load(path)["model"].predict_proba(Xv)[:, 1]
print(f"saved model reproduces predictions: {np.allclose(p, p_loaded)}")
path.unlink()
