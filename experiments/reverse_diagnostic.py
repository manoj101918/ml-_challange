"""Round 2, Step 0: go / no-go for REVERSE matching (every S2/S3 record looks for its S1 owner).

On the 20k small-validation S1 entities (train split), per country:
  * S1-side candidates: today's blocking (top 100 pool records per S1)          -> recall cap now (~0.952)
  * reverse candidates: every true record of these S1 (+ a sample of distractor records) queries an index of ALL
    train S1 of its country and keeps its top K_REV S1 by idf sum                 -> rank of the true owner
  * union cap = S1-side top 100  UNION  reverse top k, and the oracle macro F0.5 (perfect classifier on the candidates)
  * separability: best reverse score of distractors vs the owner score of true records

    python experiments/reverse_diagnostic.py        (from the project root; writes experiments/reverse_diagnostic.txt)
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

from src.blocking import build_index_hashed, iter_candidates
from src.evaluation import score_report
from src.pipeline import CACHE_DIR, normalized_pool, prepare_queries
from src.splits import make_validation_split
from src.train import TRANSLIT_ADDRESS, TRANSLIT_NAME

K_REV, K_S1, MAX_DF = 10, 100, 1000
DISTRACTORS_PER_COUNTRY = 10_000
start = time.time()
lines = []


def log(msg=""):
    print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)
    lines.append(str(msg))


def pool_batches(pool_file, country, columns=("entity_id", "country", "name_norm", "addr_norm")):
    for batch in pool_file.iter_batches(batch_size=250_000, columns=list(columns)):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["country"] == country]
        if len(chunk):
            yield chunk


gt = pd.read_csv(ROOT / "dataset" / "train" / "train_ground_truth.tsv", sep="\t", keep_default_na=False)
split = make_validation_split(gt)
val_ids = set(split.loc[split["small_validation"], "source1_entity_id"])
truth = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
true_pairs = (truth.assign(record_id=truth["matched_entity_ids"].str.split(",")).explode("record_id")
              [["source1_entity_id", "record_id"]])
true_pairs = true_pairs[true_pairs["record_id"].notna() & (true_pairs["record_id"] != "")].reset_index(drop=True)
owned = set(gt["matched_entity_ids"].str.split(",").explode())
del gt
gc.collect()

pool_path = normalized_pool("train", TRANSLIT_NAME, TRANSLIT_ADDRESS, log=log)
pool_file = pq.ParquetFile(pool_path)
n_pool = pool_file.metadata.num_rows
s1_all = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
n_s1 = len(s1_all)
log(f"validation S1 {len(val_ids):,} | true pairs {len(true_pairs):,} | train S1 {n_s1:,} | pool {n_pool:,}")

# a random sample of distractor records (owned by no S1) per country, read once
pool_ids = pool_file.read(columns=["entity_id", "country"]).to_pandas()
pool_ids = pool_ids[~pool_ids["entity_id"].isin(owned)]
distractor_sample = {c: g["entity_id"].sample(min(DISTRACTORS_PER_COUNTRY, len(g)), random_state=0)
                     for c, g in pool_ids.groupby("country")}
del pool_ids, owned
gc.collect()

s1_side, reverse, distractor_best = [], [], []
for country in sorted(s1_all["country"].unique()):
    s1_c = prepare_queries(s1_all[s1_all["country"] == country].reset_index(drop=True), TRANSLIT_NAME, TRANSLIT_ADDRESS)
    s1_c = s1_c[["entity_id", "country", "name_norm", "addr_norm"]]
    val_c = s1_c[s1_c["entity_id"].isin(val_ids)].reset_index(drop=True)

    # --- S1-side (today's blocking) for the validation S1 of this country
    index = build_index_hashed(val_c, pool_batches(pool_file, country), n_pool=n_pool, max_df=MAX_DF, log=log)
    for c in iter_candidates(val_c, index, top_k=K_S1, batch_size=1_000, yield_every=5_000):
        s1_side.append(c[["source1_entity_id", "candidate_id"]].rename(columns={"candidate_id": "record_id"}))
    del index
    gc.collect()

    # --- reverse: true records of these S1 + a sample of distractor records query an index of ALL S1 of the country
    true_c = set(true_pairs.loc[true_pairs["source1_entity_id"].isin(val_c["entity_id"]), "record_id"])
    distractors = distractor_sample[country]
    wanted = list(true_c | set(distractors))
    records = pq.read_table(pool_path, columns=["entity_id", "country", "name_norm", "addr_norm"],
                            filters=[("entity_id", "in", wanted)]).to_pandas().reset_index(drop=True)
    index = build_index_hashed(records, (s1_c.iloc[i:i + 250_000] for i in range(0, len(s1_c), 250_000)),
                               n_pool=n_s1, max_df=MAX_DF, log=log)
    for c in iter_candidates(records, index, top_k=K_REV, batch_size=2_000, yield_every=10_000):
        reverse.append(c[["source1_entity_id", "candidate_id", "score_all"]].rename(
            columns={"source1_entity_id": "record_id", "candidate_id": "s1_id", "score_all": "r_score"}))
    distractor_best.append(pd.Series(list(distractors), name="record_id"))
    del index, records, s1_c, val_c
    gc.collect()
    log(f"{country} done")

s1_side = pd.concat(s1_side, ignore_index=True)
reverse = pd.concat(reverse, ignore_index=True)
reverse["r_rank"] = reverse.groupby("record_id")["r_score"].rank(method="first", ascending=False)
distractor_ids = set(pd.concat(distractor_best))

# --- 1. where does the true owner rank from the record's side?
tp = true_pairs.merge(reverse.rename(columns={"s1_id": "source1_entity_id"}), on=["source1_entity_id", "record_id"], how="left")
tp["in_s1_side"] = pd.MultiIndex.from_frame(tp[["source1_entity_id", "record_id"]]).isin(
    pd.MultiIndex.from_frame(s1_side[["source1_entity_id", "record_id"]]))
log("\n=== 1. rank of the TRUE owner among the record's reverse candidates ===")
log(f"S1-side top-{K_S1} recall (today's cap): {tp['in_s1_side'].mean():.4f}")
rows = []
for k in [1, 2, 3, 5, 10]:
    rev_k = tp["r_rank"] <= k
    rows.append({"k": k, "reverse_recall": round(rev_k.mean(), 4), "union_recall": round((rev_k | tp["in_s1_side"]).mean(), 4),
                 "recovered_of_s1_side_misses": round(rev_k[~tp["in_s1_side"]].mean(), 4)})
log(pd.DataFrame(rows).to_string(index=False))

# --- 2. oracle macro F0.5 (perfect classifier on the candidate set)
log("\n=== 2. oracle macro F0.5 (every true pair among the candidates is found, nothing else) ===")
for label, found in [("S1-side top 100 (today)", tp["in_s1_side"]),
                     ("union with reverse top 3", tp["in_s1_side"] | (tp["r_rank"] <= 3)),
                     ("union with reverse top 5", tp["in_s1_side"] | (tp["r_rank"] <= 5))]:
    lists = tp[found].groupby("source1_entity_id")["record_id"].agg(",".join)
    matching = pd.DataFrame({"source1_entity_id": truth["source1_entity_id"],
                             "matched_entity_ids": truth["source1_entity_id"].map(lists).fillna("").to_numpy()})
    log(f"{label:28s}: {score_report(matching, truth)['macro_F0.5']:.4f}")

# --- 3. separability: owner score of true records vs best score of distractors
log("\n=== 3. true records vs distractor records (reverse side) ===")
best = reverse.groupby("record_id")["r_score"].max()
second = reverse[reverse["r_rank"] == 2].set_index("record_id")["r_score"]
owner_rank1 = tp[tp["r_rank"] == 1]
gap = owner_rank1["r_score"].to_numpy() - owner_rank1["record_id"].map(second).fillna(0).to_numpy()
d_best = best.reindex(list(distractor_ids)).dropna()
log(f"distractors with any reverse candidate: {len(d_best):,} / {len(distractor_ids):,}")
log(f"median best score: true owner {tp['r_score'].median():.1f} | distractor {d_best.median():.1f}")
t_scores, d_scores = tp["r_score"].dropna().to_numpy(), d_best.to_numpy()
auc = (np.searchsorted(np.sort(d_scores), t_scores, side="left").mean()) / len(d_scores)
log(f"AUC (owner score of true records > best score of distractors): {auc:.3f}")
log(f"owner ranked 1st: median score gap to the 2nd S1 {np.median(gap):.1f}; gap > 5 for {(gap > 5).mean():.3f}")
log(f"reverse pairs per record (test-size estimate at k=5): ~{min(5, reverse.groupby('record_id').size().mean()):.1f}")

(ROOT / "experiments" / "reverse_diagnostic.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
