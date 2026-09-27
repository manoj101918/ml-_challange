"""Research for a new approach: which CANDIDATE pairs does the model still decide wrongly?

Phase 8 pairs (20k tuning / 20k validation) with fast + reverse + extra features (the E7 feature set), HGB,
OOF threshold on tuning. For the validation pairs: missed true pairs (FN) and false merges (FP) by category,
with examples. Writes experiments/decision_errors.txt.

    python experiments/decision_errors.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

from src.decision import global_threshold, resolve_conflicts, to_submission
from src.evaluation import score_report
from src.features import EXTRA_FEATURE_COLUMNS, FAST_FEATURE_COLUMNS, add_features
from src.pipeline import CACHE_DIR
from src.reverse import REVERSE_COLUMNS, ReverseLookup, add_reverse_features
from src.rule_experiment import load_query_sets
from src.train import THRESHOLDS, make_model

start = time.time()
lines = []


def log(msg=""):
    print(f"[{time.time() - start:5.0f}s] {msg}" if msg else "", flush=True)
    lines.append(str(msg))


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
oof = np.zeros(len(train))
for fit_idx, pred_idx in GroupKFold(n_splits=3).split(X, y, groups=train["source1_entity_id"]):
    oof[pred_idx] = make_model("hgb").fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[pred_idx])[:, 1]


def chosen_pairs(table, p, t):
    return resolve_conflicts(global_threshold(table.assign(probability=p), t))


threshold = max(THRESHOLDS, key=lambda t: score_report(to_submission(chosen_pairs(train, oof, t)), truth["tuning"])["macro_F0.5"])
valid["p"] = make_model("hgb").fit(X, y).predict_proba(valid[columns].astype("float32"))[:, 1]
chosen = chosen_pairs(valid, valid["p"], threshold)
report = score_report(to_submission(chosen), truth["validation"])
log(f"validation F0.5 {report['macro_F0.5']:.4f} | threshold {threshold}")
key = lambda d: pd.MultiIndex.from_frame(d[["source1_entity_id", "candidate_id"]])
valid["predicted"] = key(valid).isin(key(chosen))
valid["error"] = np.select([(valid.label == 1) & ~valid.predicted, (valid.label == 0) & valid.predicted], ["FN", "FP"], "ok")

# whose record is a false merge? (owned by another S1, or a distractor)
gt = pd.read_csv(ROOT / "dataset" / "train" / "train_ground_truth.tsv", sep="\t", keep_default_na=False)
owner = gt.assign(r=gt["matched_entity_ids"].str.split(",")).explode("r")
owner = owner[owner["r"].notna() & (owner["r"] != "")].set_index("r")["source1_entity_id"]
fp = valid[valid["error"] == "FP"]
fp_owned = fp["candidate_id"].map(owner).notna()
log(f"false merges: {len(fp):,} ({fp_owned.mean():.2f} of them belong to ANOTHER S1, {1 - fp_owned.mean():.2f} are distractors)")
log(f"missed true candidates: {(valid.error == 'FN').sum():,} of {int(valid.label.sum()):,} true candidate pairs")

valid["addr_empty"] = valid["addr_c_empty"] == 1
valid["foreign"] = valid["name_c_foreign_script"] == 1
valid["favourite"] = valid["r_rank"] == 1
valid["name_hi"] = valid["name_token_set"] >= 0.9
valid["addr_hi"] = valid["addr_token_set"] >= 0.9
cats = ["addr_empty", "foreign", "favourite", "name_hi", "addr_hi", "is_source3"]
log("\n=== share of pairs with each property: FN / FP / correct true pairs ===")
tbl = pd.DataFrame({e: valid.loc[m, cats].mean() for e, m in [("FN", valid.error == "FN"), ("FP", valid.error == "FP"),
                                                                ("correct true", (valid.label == 1) & valid.predicted)]})
log(tbl.round(3).to_string())
log(f"median probability: FN {valid.loc[valid.error == 'FN', 'p'].median():.3f} | FP {valid.loc[valid.error == 'FP', 'p'].median():.3f}")

for kind in ["FN", "FP"]:
    log(f"\n=== 25 random {kind} ===")
    for _, r in valid[valid.error == kind].sample(25, random_state=2).iterrows():
        log(f"p {r.p:.2f} | name_ts {r.name_token_set:.2f} addr_ts {r.addr_token_set if pd.notna(r.addr_token_set) else -1:.2f} "
            f"| r_rank {r.r_rank:.0f} r_gap {r.r_gap:.0f} | block_rank {r.block_rank:.0f}")
        log(f"   S1 : {r.name_s1} | {r.addr_s1}")
        log(f"   REC: {r.name_c} | {r.addr_c}")
(ROOT / "experiments" / "decision_errors.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
