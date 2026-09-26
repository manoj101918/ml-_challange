"""Candidate generation (blocking) — Phase 7.

For every Source 1 entity, candidates are pool (S2 + S3) records that share at least one RARE key
(document frequency <= max_df) of the same country. Keys: see src/blocking_keys.py.
Candidates are ranked by the sum of idf = log(N / df) over the shared keys, and the best ones are kept:

  ranking="all"   : top `top_k` by the idf sum over all shared keys
  ranking="split" : top `top_k // 2` by the idf sum of NAME keys  UNION  top `top_k // 2` by ADDRESS keys

Memory design (8 GB laptop): keys are turned into integer codes, the pool is read in 1M-row chunks,
and the queries are processed in batches.
"""

import gc
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking_keys import ALL_KEY_TYPES, record_keys

NAME_KEY_TYPES = {"nw", "nb", "nj"}


def build_index(s1, pool_path, max_df, key_types=ALL_KEY_TYPES, chunk_rows=1_000_000, log=print):
    """Integer index of the rare keys shared between the queries and the pool.

    s1 needs columns country, name_norm, addr_norm. Returns a dict with the query keys, pool keys,
    idf per key code, whether each key is a name key, and the pool entity IDs (position = pool row).
    """
    start = time.time()
    q = record_keys(s1["name_norm"], s1["addr_norm"], s1["country"], key_types)
    vocabulary = pd.Index(q["key"].unique())
    is_name_key = np.asarray(vocabulary.str.split("|").str[1].isin(NAME_KEY_TYPES))
    q = pd.DataFrame({"s1_row": q["row"].astype("int32"), "code": vocabulary.get_indexer(q["key"]).astype("int32")}).drop_duplicates()

    df = np.zeros(len(vocabulary), dtype=np.int64)
    pool_ids, pool_keys, offset = [], [], 0
    for batch in pq.ParquetFile(pool_path).iter_batches(batch_size=chunk_rows, columns=["entity_id", "country", "name_norm", "addr_norm"]):
        chunk = batch.to_pandas()
        pool_ids.append(chunk["entity_id"])
        k = record_keys(chunk["name_norm"], chunk["addr_norm"], chunk["country"], key_types)
        codes = vocabulary.get_indexer(k["key"])
        keep = codes >= 0
        k = pd.DataFrame({"pool_row": (k["row"].to_numpy()[keep] + offset).astype("int32"),
                          "code": codes[keep].astype("int32")}).drop_duplicates()
        chunk_counts = np.bincount(k["code"], minlength=len(vocabulary))
        df += chunk_counts
        pool_keys.append(k[chunk_counts[k["code"]] <= max_df])   # too common in this chunk alone -> too common overall
        offset += len(chunk)
    pool_ids = pd.concat(pool_ids, ignore_index=True)
    pool_keys = pd.concat(pool_keys, ignore_index=True)
    pool_keys = pool_keys[df[pool_keys["code"]] <= max_df]
    q = q[df[q["code"]] <= max_df]
    gc.collect()
    log(f"  index built: {len(q):,} query keys, {len(pool_keys):,} pool keys ({time.time() - start:.0f}s)")
    return {
        "query_keys": q,
        "pool_keys": pool_keys,
        "idf": np.log(len(pool_ids) / np.maximum(df, 1)),
        "is_name_key": is_name_key,
        "pool_ids": pool_ids,
    }


def generate_candidates(s1, index, top_k=50, ranking="all", batch_size=1_000):
    """Candidate table: source1_entity_id, candidate_id, s1_row, pool_row, score_all, score_name, score_addr."""
    q, pool_keys, idf, is_name_key = index["query_keys"], index["pool_keys"], index["idf"], index["is_name_key"]
    kept = []
    for b in range(0, len(s1), batch_size):
        part = q[(q["s1_row"] >= b) & (q["s1_row"] < b + batch_size)].merge(pool_keys, on="code")
        part["w"] = idf[part["code"]]
        part["w_name"] = np.where(is_name_key[part["code"]], part["w"], 0.0)
        pair = part.groupby(["s1_row", "pool_row"]).agg(score_all=("w", "sum"), score_name=("w_name", "sum")).reset_index()
        pair["score_addr"] = pair["score_all"] - pair["score_name"]
        if ranking == "all":
            keep = pair.groupby("s1_row")["score_all"].rank(method="first", ascending=False) <= top_k
        else:
            keep = (pair.groupby("s1_row")["score_name"].rank(method="first", ascending=False) <= top_k // 2) | \
                   (pair.groupby("s1_row")["score_addr"].rank(method="first", ascending=False) <= top_k // 2)
        kept.append(pair[keep])
    candidates = pd.concat(kept, ignore_index=True)
    candidates["source1_entity_id"] = s1["entity_id"].to_numpy()[candidates["s1_row"]]
    candidates["candidate_id"] = index["pool_ids"].to_numpy()[candidates["pool_row"]]
    return candidates
