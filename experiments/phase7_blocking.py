"""Phase 7: compare blocking designs by candidate recall vs number of candidates.

Builds the key index ONCE (all 5 key types, keys with df <= 1000), then evaluates:
  max_df  : 100 / 300 / 1000      (how common a key may be and still create candidates)
  ranking : "all"    = top K by idf sum over all shared keys
            "split"  = top K/2 by name-key idf sum  UNION  top K/2 by address-key idf sum
  K       : 25 / 50 / 100 / 200   (candidates kept per S1 entity)
The design is CHOSEN on the tuning entities; validation recall is reported for the record.
Output: experiments/phase7_blocking_results.tsv

Run from the project root:   python experiments/phase7_blocking.py
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking_keys import ALL_KEY_TYPES, record_keys
from src.preprocessing import ADDRESS_STEPS, NAME_STEPS, normalize
from src.rule_experiment import CACHE_DIR, load_query_sets

MAX_DF_LIST = [100, 300, 1000]
K_LIST = [25, 50, 100, 200]
BATCH = 1_000
NAME_TYPES = {"nw", "nb", "nj"}
OUT = Path(__file__).resolve().parent / "phase7_blocking_results.tsv"

start = time.time()
log = lambda msg: print(f"[{time.time() - start:5.0f}s] {msg}", flush=True)

# ---------------------------------------------------------------- queries
s1, truth = load_query_sets()
s1["name_norm"] = normalize(s1["business_name"], NAME_STEPS)
s1["addr_norm"] = normalize(s1["business_address"], ADDRESS_STEPS)
q = record_keys(s1["name_norm"], s1["addr_norm"], s1["country"], ALL_KEY_TYPES)
vocabulary = pd.Index(q["key"].unique())
key_is_name = np.array([k.split("|")[1] in NAME_TYPES for k in vocabulary])
q = pd.DataFrame({"s1_row": q["row"].astype("int32"), "code": vocabulary.get_indexer(q["key"]).astype("int32")}).drop_duplicates()
log(f"{len(s1):,} queries, {len(vocabulary):,} distinct keys")

# ---------------------------------------------------------------- pool index (one pass)
df = np.zeros(len(vocabulary), dtype=np.int64)
pool_ids, pool_keys, offset = [], [], 0
for batch in pq.ParquetFile(CACHE_DIR / "train_pool_norm.parquet").iter_batches(batch_size=1_000_000):
    chunk = batch.to_pandas()
    pool_ids.append(chunk["entity_id"])
    k = record_keys(chunk["name_norm"], chunk["addr_norm"], chunk["country"], ALL_KEY_TYPES)
    codes = vocabulary.get_indexer(k["key"])
    keep = codes >= 0
    k = pd.DataFrame({"pool_row": (k["row"].to_numpy()[keep] + offset).astype("int32"), "code": codes[keep].astype("int32")}).drop_duplicates()
    chunk_counts = np.bincount(k["code"], minlength=len(vocabulary))
    df += chunk_counts
    pool_keys.append(k[chunk_counts[k["code"]] <= max(MAX_DF_LIST)])   # certainly too common otherwise
    offset += len(chunk)
    log(f"indexed {offset:,} pool records")
pool_ids = pd.concat(pool_ids, ignore_index=True)
pool_keys = pd.concat(pool_keys, ignore_index=True)
pool_keys = pool_keys[df[pool_keys["code"]] <= max(MAX_DF_LIST)]
idf = np.log(len(pool_ids) / np.maximum(df, 1))
gc.collect()
log(f"pool index: {len(pool_keys):,} (record, key) rows")

# ---------------------------------------------------------------- truth as (s1_row, pool_row) integer pairs
pool_index = pd.Index(pool_ids)
true_rows = []
for name, t in truth.items():
    p = t.assign(match_id=t["matched_entity_ids"].str.split(",")).explode("match_id")
    p = p[p["match_id"].notna() & (p["match_id"] != "")]
    s1_row = pd.Index(s1["entity_id"]).get_indexer(p["source1_entity_id"])
    true_rows.append(pd.DataFrame({"s1_row": s1_row, "pool_row": pool_index.get_indexer(p["match_id"]), "set": name}))
true_rows = pd.concat(true_rows, ignore_index=True)
query_set = s1["query_set"].to_numpy()

# ---------------------------------------------------------------- evaluate
results = []
for max_df in MAX_DF_LIST:
    usable_q = q[df[q["code"]] <= max_df]
    usable_p = pool_keys[df[pool_keys["code"]] <= max_df]
    kept = []
    for b in range(0, len(s1), BATCH):
        part = usable_q[(usable_q["s1_row"] >= b) & (usable_q["s1_row"] < b + BATCH)].merge(usable_p, on="code")
        part["w"] = idf[part["code"]]
        part["w_name"] = np.where(key_is_name[part["code"]], part["w"], 0.0)
        pair = part.groupby(["s1_row", "pool_row"]).agg(score_all=("w", "sum"), score_name=("w_name", "sum")).reset_index()
        pair["score_addr"] = pair["score_all"] - pair["score_name"]
        for col in ["score_all", "score_name", "score_addr"]:
            pair["rank_" + col[6:]] = pair.groupby("s1_row")[col].rank(method="first", ascending=False)
        kept.append(pair[(pair["rank_all"] <= max(K_LIST)) | (pair["rank_name"] <= max(K_LIST) // 2) | (pair["rank_addr"] <= max(K_LIST) // 2)]
                    [["s1_row", "pool_row", "rank_all", "rank_name", "rank_addr"]])
    kept = pd.concat(kept, ignore_index=True)
    log(f"max_df {max_df}: candidate table {len(kept):,} rows")

    for ranking in ["all", "split"]:
        for k in K_LIST:
            if ranking == "all":
                chosen = kept[kept["rank_all"] <= k]
            else:
                chosen = kept[(kept["rank_name"] <= k // 2) | (kept["rank_addr"] <= k // 2)]
            chosen = chosen[["s1_row", "pool_row"]].drop_duplicates()
            hits = true_rows.merge(chosen, on=["s1_row", "pool_row"], how="left", indicator=True)
            hits["found"] = hits["_merge"] == "both"
            per_entity = chosen.groupby("s1_row").size().reindex(range(len(s1)), fill_value=0)
            row = {"max_df": max_df, "ranking": ranking, "K": k}
            for name in ["tuning", "validation"]:
                row[f"{name}_recall"] = round(hits.loc[hits["set"] == name, "found"].mean(), 4)
                row[f"{name}_mean_candidates"] = round(per_entity[query_set == name].mean(), 1)
                row[f"{name}_no_candidates"] = int((per_entity[query_set == name] == 0).sum())
            results.append(row)
            log(str(row))
    del kept
    gc.collect()

pd.DataFrame(results).to_csv(OUT, sep="\t", index=False, lineterminator="\n")
log(f"saved {OUT}")
