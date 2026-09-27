"""Research for a new approach: what do the true pairs look like that NEITHER search finds?

20k small-validation S1: S1-side top 100 (today's blocking) UNION reverse top 10 (src/reverse.py). For the true pairs
missed by both: raw + normalized texts, similarity numbers, shared blocking keys, and simple categories.
Writes experiments/miss_analysis.txt (category table + 60 random examples).

    python experiments/miss_analysis.py
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
from rapidfuzz import fuzz

from src.blocking import build_index_hashed, iter_candidates
from src.blocking_keys import record_keys
from src.pipeline import CACHE_DIR, normalized_pool, prepare_queries
from src.reverse import TRANSLIT_ADDRESS, TRANSLIT_NAME, _query_keys, s1_index
from src.splits import make_validation_split

start = time.time()
lines = []


def log(msg=""):
    print(f"[{time.time() - start:5.0f}s] {msg}" if msg else "", flush=True)
    lines.append(str(msg))


gt = pd.read_csv(ROOT / "dataset" / "train" / "train_ground_truth.tsv", sep="\t", keep_default_na=False)
split = make_validation_split(gt)
val_ids = set(split.loc[split["small_validation"], "source1_entity_id"])
truth = gt[gt["source1_entity_id"].isin(val_ids)]
true_pairs = truth.assign(record_id=truth["matched_entity_ids"].str.split(",")).explode("record_id")[["source1_entity_id", "record_id"]]
true_pairs = true_pairs[true_pairs["record_id"].notna() & (true_pairs["record_id"] != "")].reset_index(drop=True)
del gt
pool_path = normalized_pool("train", TRANSLIT_NAME, TRANSLIT_ADDRESS, log=log)
pool_file = pq.ParquetFile(pool_path)
n_pool = pool_file.metadata.num_rows
s1_all = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
n_s1 = len(s1_all)

found = []
s1_norm = []
for country in sorted(s1_all["country"].unique()):
    s1_c = prepare_queries(s1_all[s1_all["country"] == country], TRANSLIT_NAME, TRANSLIT_ADDRESS)
    s1_c = s1_c[["entity_id", "country", "name_norm", "addr_norm"]]
    val_c = s1_c[s1_c["entity_id"].isin(val_ids)].reset_index(drop=True)
    s1_norm.append(val_c)

    def batches():
        for b in pool_file.iter_batches(batch_size=250_000, columns=["entity_id", "country", "name_norm", "addr_norm"]):
            c = b.to_pandas()
            c = c[c["country"] == country]
            if len(c):
                yield c
    index = build_index_hashed(val_c, batches(), n_pool=n_pool, max_df=1000, log=log)
    for c in iter_candidates(val_c, index, top_k=100, batch_size=1_000, yield_every=5_000):
        found.append(c[["source1_entity_id", "candidate_id"]].rename(columns={"candidate_id": "record_id"}))
    del index
    gc.collect()

    recs = true_pairs.loc[true_pairs["source1_entity_id"].isin(val_c["entity_id"]), "record_id"].unique().tolist()
    records = pq.read_table(pool_path, columns=["entity_id", "country", "name_norm", "addr_norm"],
                            filters=[("entity_id", "in", recs)]).to_pandas().reset_index(drop=True)
    idx = s1_index(s1_c, n_s1)
    idx["query_keys"] = _query_keys(records, idx)
    for c in iter_candidates(records, idx, top_k=10, batch_size=5_000, yield_every=50_000):
        found.append(c[["candidate_id", "source1_entity_id"]].rename(columns={"candidate_id": "source1_entity_id",
                                                                             "source1_entity_id": "record_id"}))
    del idx, records, s1_c
    gc.collect()
    log(f"{country} done")

found = pd.concat(found, ignore_index=True).drop_duplicates()
tp = true_pairs.merge(found.assign(found=True), on=["source1_entity_id", "record_id"], how="left")
tp["found"] = tp["found"].fillna(False).astype(bool)
missed = tp[~tp["found"]].reset_index(drop=True)
log(f"true pairs {len(tp):,} | found by S1-side top 100 or reverse top 10: {tp['found'].mean():.4f} | missed {len(missed):,}")

# ---- texts of the missed pairs (+ a sample of found pairs for reference)
ref = tp[tp["found"]].sample(3000, random_state=0)
sample = pd.concat([missed.assign(kind="missed"), ref.assign(kind="found")], ignore_index=True)
s1n = pd.concat(s1_norm, ignore_index=True).rename(columns={"entity_id": "source1_entity_id", "name_norm": "s1_name_norm",
                                                            "addr_norm": "s1_addr_norm"})
s1raw = s1_all[s1_all["entity_id"].isin(sample["source1_entity_id"])].rename(
    columns={"entity_id": "source1_entity_id", "business_name": "s1_name", "business_address": "s1_addr"})
rec_ids = sample["record_id"].unique().tolist()
rnorm = pq.read_table(pool_path, columns=["entity_id", "name_norm", "addr_norm", "name_foreign"],
                      filters=[("entity_id", "in", rec_ids)]).to_pandas().rename(columns={"entity_id": "record_id"})
rraw = pq.read_table(CACHE_DIR / "train_pool.parquet", columns=["entity_id", "business_name", "business_address"],
                     filters=[("entity_id", "in", rec_ids)]).to_pandas().rename(
    columns={"entity_id": "record_id", "business_name": "r_name", "business_address": "r_addr"})
d = (sample.merge(s1raw, on="source1_entity_id").merge(s1n.drop(columns="country"), on="source1_entity_id")
     .merge(rraw, on="record_id").merge(rnorm, on="record_id"))

d["name_sim"] = [fuzz.token_set_ratio(a, b) for a, b in zip(d["s1_name_norm"], d["name_norm"])]
d["addr_sim"] = [fuzz.token_set_ratio(a, b) if a and b else np.nan for a, b in zip(d["s1_addr_norm"], d["addr_norm"])]
d["addr_empty"] = d["addr_norm"] == ""
d["foreign_script"] = d["name_foreign"] == 1
d["is_s3"] = d["record_id"].str.startswith("S3")
k1 = record_keys(d["s1_name_norm"], d["s1_addr_norm"], d["country"])
k2 = record_keys(d["name_norm"], d["addr_norm"], d["country"])
g1 = k1.groupby("row")["key"].agg(set)
g2 = k2.groupby("row")["key"].agg(set)
shared = [g1.get(i, set()) & g2.get(i, set()) for i in range(len(d))]
d["n_shared_keys"] = [len(s) for s in shared]
d["shared_name_key"] = [any("|nw|" in k or "|nb|" in k or "|nj|" in k for k in s) for s in shared]
d["shared_addr_key"] = [any("|aw|" in k or "|ab|" in k for k in s) for s in shared]
d["name_digits"] = d["s1_name_norm"].str.contains(r"\d") | d["name_norm"].str.contains(r"\d")

log("\n=== missed vs found true pairs (share of pairs) ===")
table = d.groupby("kind").agg(pairs=("record_id", "size"), no_shared_key=("n_shared_keys", lambda x: (x == 0).mean()),
                              no_shared_name_key=("shared_name_key", lambda x: 1 - x.mean()),
                              no_shared_addr_key=("shared_addr_key", lambda x: 1 - x.mean()),
                              addr_empty=("addr_empty", "mean"), foreign_script=("foreign_script", "mean"),
                              is_s3=("is_s3", "mean"), name_sim_median=("name_sim", "median"),
                              addr_sim_median=("addr_sim", "median")).T
log(table.round(3).to_string())
log(f"missed by country: {d[d.kind == 'missed'].groupby('country').size().to_dict()}")
m = d[d["kind"] == "missed"]
log("\nname_sim buckets of missed pairs: " + str(pd.cut(m["name_sim"], [-1, 30, 50, 70, 90, 100]).value_counts(sort=False).to_dict()))
log("addr_sim buckets of missed pairs: " + str(pd.cut(m["addr_sim"], [-1, 30, 50, 70, 90, 100]).value_counts(sort=False).to_dict()))

log("\n=== 60 random missed pairs (raw S1 | raw record) ===")
for _, r in m.sample(min(60, len(m)), random_state=1).iterrows():
    log(f"[{r.country}] name_sim {r.name_sim:.0f} addr_sim {r.addr_sim if pd.notna(r.addr_sim) else -1:.0f} keys {r.n_shared_keys}")
    log(f"   S1 : {r.s1_name} | {r.s1_addr}")
    log(f"   REC: {r.r_name} | {r.r_addr}")
    log(f"   norm: {r.s1_name_norm} | {r.s1_addr_norm}  <->  {r.name_norm} | {r.addr_norm}")
(ROOT / "experiments" / "miss_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
