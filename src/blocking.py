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


# ---------------------------------------------------------------- scalable version (Phase 13: full test set)

def _hash(keys):
    """Text keys -> 64-bit integers (8 bytes each instead of ~60 bytes of text). Collisions are negligible."""
    return pd.util.hash_pandas_object(pd.Series(keys), index=False).to_numpy()


def build_index_hashed(s1, pool_batches, n_pool, max_df, key_types=ALL_KEY_TYPES, query_chunk=100_000, log=print):
    """Same index as build_index, but keys are hashed and the pool is given as an iterable of DataFrames
    (entity_id, country, name_norm, addr_norm) - e.g. the records of ONE country, read in chunks.

    n_pool = size of the WHOLE pool (all countries), so idf = log(n_pool / df) is on the same scale as in training.
    """
    start = time.time()
    q_parts = []
    for first in range(0, len(s1), query_chunk):
        part = s1.iloc[first:first + query_chunk]
        k = record_keys(part["name_norm"], part["addr_norm"], part["country"], key_types)
        q_parts.append(pd.DataFrame({"s1_row": (k["row"].to_numpy() + first).astype("int32"),
                                     "h": _hash(k["key"]), "is_name": k["is_name"].to_numpy()}))
    q = pd.concat(q_parts, ignore_index=True).drop_duplicates(["s1_row", "h"])
    vocabulary, inverse = np.unique(q["h"].to_numpy(), return_inverse=True)
    q["code"] = inverse.astype("int32")
    is_name_key = np.zeros(len(vocabulary), dtype=bool)
    is_name_key[q["code"].to_numpy()] = q["is_name"].to_numpy().astype(bool)
    q = q[["s1_row", "code"]]

    df = np.zeros(len(vocabulary), dtype=np.int64)
    pool_ids, pool_keys, offset = [], [], 0
    for chunk in pool_batches:
        chunk = chunk.reset_index(drop=True)
        pool_ids.append(chunk["entity_id"])
        k = record_keys(chunk["name_norm"], chunk["addr_norm"], chunk["country"], key_types)
        h = _hash(k["key"])
        position = np.searchsorted(vocabulary, h)
        position[position == len(vocabulary)] = 0
        hit = vocabulary[position] == h
        k = pd.DataFrame({"pool_row": (k["row"].to_numpy()[hit] + offset).astype("int32"),
                          "code": position[hit].astype("int32")}).drop_duplicates()
        chunk_counts = np.bincount(k["code"], minlength=len(vocabulary))
        df += chunk_counts
        pool_keys.append(k[chunk_counts[k["code"]] <= max_df])
        offset += len(chunk)
    pool_ids = pd.concat(pool_ids, ignore_index=True)
    pool_keys = pd.concat(pool_keys, ignore_index=True)
    pool_keys = pool_keys[df[pool_keys["code"]] <= max_df]
    q = q[df[q["code"]] <= max_df]
    gc.collect()
    log(f"  hashed index: {len(s1):,} queries, {len(q):,} query keys, {len(pool_keys):,} pool keys ({time.time() - start:.0f}s)")
    return {"query_keys": q, "pool_keys": pool_keys, "idf": np.log(n_pool / np.maximum(df, 1)),
            "is_name_key": is_name_key, "pool_ids": pool_ids}


def iter_candidates(s1, index, top_k=50, batch_size=5_000, yield_every=10_000):
    """Like generate_candidates(ranking="all"), but yields the candidates in pieces of ~`yield_every` S1 entities,
    so the full candidate table (87-173M rows on the test set) never has to be in memory at once.

    Speed: the pool keys are sorted by key code ONCE and an offset array records where each code's pool rows start
    (like the index of a book). A batch then gathers its matching pool rows with NumPy arithmetic only —
    no pandas merge, which would re-hash the whole pool-key table for every batch.
    """
    idf, is_name_key = index["idf"], index["is_name_key"]
    pool_keys = index.pop("pool_keys")            # the sorted arrays below replace it -> free its memory
    pool_codes = pool_keys["code"].to_numpy()
    order = np.argsort(pool_codes, kind="stable")
    pool_rows_sorted = pool_keys["pool_row"].to_numpy()[order]
    offsets = np.searchsorted(pool_codes[order], np.arange(len(idf) + 1))     # rows of code c: offsets[c]:offsets[c+1]
    del pool_keys, pool_codes, order
    gc.collect()

    q = index["query_keys"].sort_values("s1_row")
    q_rows, q_codes = q["s1_row"].to_numpy(), q["code"].to_numpy()
    starts = np.searchsorted(q_rows, np.arange(0, len(s1) + batch_size, batch_size))
    kept = []
    for i, first in enumerate(range(0, len(s1), batch_size)):
        rows, codes = q_rows[starts[i]:starts[i + 1]], q_codes[starts[i]:starts[i + 1]]
        counts = offsets[codes + 1] - offsets[codes]
        total = int(counts.sum())
        if total:
            begin = np.repeat(offsets[codes], counts)
            within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            code_rep = np.repeat(codes, counts)
            weight = idf[code_rep]
            part = pd.DataFrame({"s1_row": np.repeat(rows, counts), "pool_row": pool_rows_sorted[begin + within],
                                 "w": weight, "w_name": np.where(is_name_key[code_rep], weight, 0.0)})
            pair = part.groupby(["s1_row", "pool_row"]).agg(score_all=("w", "sum"), score_name=("w_name", "sum")).reset_index()
            pair["score_addr"] = pair["score_all"] - pair["score_name"]
            kept.append(pair[pair.groupby("s1_row")["score_all"].rank(method="first", ascending=False) <= top_k])
        if (first + batch_size) % yield_every == 0 or first + batch_size >= len(s1):
            if kept:
                candidates = pd.concat(kept, ignore_index=True)
                candidates["source1_entity_id"] = s1["entity_id"].to_numpy()[candidates["s1_row"]]
                candidates["candidate_id"] = index["pool_ids"].to_numpy()[candidates["pool_row"]]
                yield candidates
            kept = []
