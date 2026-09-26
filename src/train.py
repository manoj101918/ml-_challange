"""Training the matching model (Phase 12) — the final configuration is "E4".

    python -m src.train E4          (run from the project root; E0..E4 are the Phase 12 experiments)

E0  Phase 9 setup through the pipeline code (reproduces ~0.934)
E1  + transliteration of Indian scripts (names and addresses)
E2  E1 + more training entities (20k -> 60k)
E3  E1 + K = 100 candidates per entity (20k training entities)
E4  FINAL: translit + K = 100 + fast features (no TF-IDF) + 40k training entities -> cache/models/E4.joblib

Model: HistGradientBoosting. Threshold: chosen on 3-fold out-of-fold predictions of the TRAINING entities;
the 20k small-validation entities are only scored. Results -> experiments/phase12_results.tsv
"""

import csv
import gc
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_per_entity, score_report
from src.features import FAST_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.pipeline import CACHE_DIR, label_pairs, prepare_queries, run_blocking_and_features
from src.preprocessing import ADDRESS_STEPS, NAME_STEPS
from src.rule_experiment import TRAIN_DIR, load_query_sets
from src.splits import make_validation_split

TRANSLIT_NAME = ("translit",) + NAME_STEPS
TRANSLIT_ADDRESS = ("translit",) + ADDRESS_STEPS
EXPERIMENTS = {
    "E0": dict(name_steps=NAME_STEPS, address_steps=ADDRESS_STEPS, extra_train=0, top_k=50),
    "E1": dict(name_steps=TRANSLIT_NAME, address_steps=TRANSLIT_ADDRESS, extra_train=0, top_k=50),
    # E2 / E3 sized for ~2-3 GB of free RAM: each changes ONE thing relative to E1
    "E2": dict(name_steps=TRANSLIT_NAME, address_steps=TRANSLIT_ADDRESS, extra_train=40_000, top_k=50),
    "E3": dict(name_steps=TRANSLIT_NAME, address_steps=TRANSLIT_ADDRESS, extra_train=0, top_k=100),
    # E4: the candidate final configuration. Fast features (no TF-IDF, -0.001 but 3.3x faster, see phase12_ablation.tsv),
    # K = 100, 40k training entities (8 GB laptop: 60k x 100 candidates would not fit in memory)
    "E4": dict(name_steps=TRANSLIT_NAME, address_steps=TRANSLIT_ADDRESS, extra_train=20_000, top_k=100, use_tfidf=False),
}
HGB = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
           early_stopping=False, random_state=0)
THRESHOLDS = np.round(np.arange(0.50, 0.91, 0.025), 3)
OUT = Path(__file__).resolve().parent.parent / "experiments" / "phase12_results.tsv"
PER_ENTITY_DIR = Path(__file__).resolve().parent.parent / "output" / "experiments" / "phase12"


def query_sets(extra_train):
    s1, truth = load_query_sets()
    if extra_train:
        gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", keep_default_na=False)
        split = make_validation_split(gt)
        pool = split.loc[(split["split"] == "train") & ~split["source1_entity_id"].isin(s1["entity_id"]), "source1_entity_id"]
        extra_ids = pool.sample(extra_train, random_state=11)
        extra = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
        extra = extra[extra["entity_id"].isin(extra_ids)].assign(query_set="tuning")
        s1 = pd.concat([s1, extra], ignore_index=True)
        truth["tuning"] = pd.concat([truth["tuning"], gt[gt["source1_entity_id"].isin(extra_ids)]], ignore_index=True)
    return s1, truth


def macro(pairs, probabilities, threshold, truth_table):
    chosen = global_threshold(pairs.assign(probability=probabilities), threshold)
    return score_report(to_submission(resolve_conflicts(chosen)), truth_table)


def run(name, name_steps, address_steps, extra_train, top_k, use_tfidf=True):
    start = time.time()
    log = lambda msg: print(f"[{name} {time.time() - start:5.0f}s] {msg}", flush=True)
    s1, truth = query_sets(extra_train)
    queries = prepare_queries(s1, name_steps, address_steps)
    candidates, pairs, vectorizers = run_blocking_and_features(queries, "train", name_steps, address_steps, top_k=top_k,
                                                               use_tfidf=use_tfidf, log=log)
    features = FEATURE_COLUMNS if use_tfidf else FAST_FEATURE_COLUMNS
    pairs = pairs.merge(queries[["entity_id", "query_set"]], left_on="source1_entity_id", right_on="entity_id").drop(columns="entity_id")
    pairs["label"] = label_pairs(pairs, list(truth.values()))

    recall = {}
    for set_name in ["tuning", "validation"]:
        t = truth[set_name]
        n_true = t["matched_entity_ids"].map(lambda s: len(s.split(",")) if s else 0).sum()
        recall[set_name] = pairs.loc[pairs["query_set"] == set_name, "label"].sum() / n_true
    del candidates
    gc.collect()

    train = pairs[pairs["query_set"] == "tuning"].reset_index(drop=True)
    valid = pairs[pairs["query_set"] == "validation"].reset_index(drop=True)
    del pairs
    X, y = train[features].astype("float32"), train["label"].to_numpy()

    oof = np.zeros(len(train))
    for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=train["source1_entity_id"]):
        model = HistGradientBoostingClassifier(**HGB).fit(X.iloc[fit_idx], y[fit_idx])
        oof[pred_idx] = model.predict_proba(X.iloc[pred_idx])[:, 1]
    tuning_scores = [macro(train, oof, t, truth["tuning"])["macro_F0.5"] for t in THRESHOLDS]
    threshold = float(THRESHOLDS[int(np.argmax(tuning_scores))])
    log(f"out-of-fold done, threshold {threshold}")

    model = HistGradientBoostingClassifier(**HGB).fit(X, y)
    (CACHE_DIR / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "vectorizers": vectorizers, "features": features,
                 "name_steps": name_steps, "address_steps": address_steps, "top_k": top_k, "max_df": 1000},
                CACHE_DIR / "models" / f"{name}.joblib")
    p_valid = model.predict_proba(valid[features].astype("float32"))[:, 1]
    report = macro(valid, p_valid, threshold, truth["validation"])

    PER_ENTITY_DIR.mkdir(parents=True, exist_ok=True)
    chosen = resolve_conflicts(global_threshold(valid.assign(probability=p_valid), threshold))
    score_per_entity(to_submission(chosen), truth["validation"])[["source1_entity_id", "n_true", "n_pred", "n_correct", "f05"]].to_csv(
        PER_ENTITY_DIR / f"{name}.tsv", sep="\t", index=False, lineterminator="\n")

    row = {
        "experiment": name, "translit": "translit" in name_steps, "training_entities": train["source1_entity_id"].nunique(),
        "top_k": top_k, "training_pairs": len(train),
        "tuning_candidate_recall": round(recall["tuning"], 4), "validation_candidate_recall": round(recall["validation"], 4),
        "threshold": threshold, "tuning_F0.5_oof": round(max(tuning_scores), 4),
        "validation_F0.5": round(report["macro_F0.5"], 4), "validation_singletons": round(report["F0.5 on singletons"], 4),
        "validation_precision": round(report["pair precision"], 4), "validation_recall": round(report["pair recall"], 4),
        "minutes": round((time.time() - start) / 60, 1),
    }
    new_file = not OUT.exists()
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    log(f"DONE {row}")


if __name__ == "__main__":
    for experiment in sys.argv[1:]:
        run(experiment, **EXPERIMENTS[experiment])
        gc.collect()
