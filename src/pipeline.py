"""End-to-end pipeline pieces, shared by the Phase 12 experiments and the final test prediction.

normalize -> blocking (src/blocking.py) -> pair features (src/features.py) -> model -> decision (src/decision.py)
"""

import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking import build_index, generate_candidates
from src.features import EXTRA_FEATURE_COLUMNS, FAST_FEATURE_COLUMNS, FEATURE_COLUMNS, add_features, fit_tfidf
from src.preprocessing import INDIAN_SCRIPTS, normalize
from src.progress import Progress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def normalization_tag(name_steps, address_steps):
    text = "+".join(name_steps) + "|" + "+".join(address_steps)
    return hashlib.md5(text.encode()).hexdigest()[:8]


def normalized_pool(split, name_steps, address_steps, log=print):
    """Parquet file with entity_id, country, name_norm, addr_norm, name_foreign for the S2+S3 pool (built once, cached)."""
    path = CACHE_DIR / f"{split}_pool_norm_{normalization_tag(name_steps, address_steps)}.parquet"
    if path.exists():
        return path
    start = time.time()
    writer = None
    for batch in pq.ParquetFile(CACHE_DIR / f"{split}_pool.parquet").iter_batches(batch_size=1_000_000):
        chunk = batch.to_pandas()
        out = pd.DataFrame({
            "entity_id": chunk["entity_id"],
            "country": chunk["country"],
            "name_norm": normalize(chunk["business_name"], name_steps),
            "addr_norm": normalize(chunk["business_address"], address_steps),
            "name_foreign": chunk["business_name"].str.contains(INDIAN_SCRIPTS).astype("int8"),
        })
        table = pa.Table.from_pandas(out, preserve_index=False)
        writer = writer or pq.ParquetWriter(path, table.schema)
        writer.write_table(table, row_group_size=1_000_000)
    writer.close()
    log(f"  normalized {split} pool cached ({time.time() - start:.0f}s): {path.name}")
    return path


def prepare_queries(s1, name_steps, address_steps):
    s1 = s1.reset_index(drop=True).copy()
    s1["name_norm"] = normalize(s1["business_name"], name_steps)
    s1["addr_norm"] = normalize(s1["business_address"], address_steps)
    return s1


def fit_vectorizers(pool_path, sample_size=1_000_000, seed=0):
    table = pq.read_table(pool_path, columns=["name_norm", "addr_norm"])
    rows = np.sort(np.random.default_rng(seed).choice(table.num_rows, min(sample_size, table.num_rows), replace=False))
    sample = table.take(rows).to_pandas()
    return fit_tfidf(sample["name_norm"]), fit_tfidf(sample["addr_norm"])


def pair_features(candidates, queries, pool_path, vectorizers, chunk_entities=10_000, log=print):
    """Features for all candidate pairs, computed in chunks of S1 entities to limit memory."""
    start = time.time()
    rows = np.sort(candidates["pool_row"].unique())
    text = pq.read_table(pool_path, columns=["entity_id", "name_norm", "addr_norm", "name_foreign"]).take(rows).to_pandas()
    text = text.rename(columns={"entity_id": "candidate_id", "name_norm": "name_c", "addr_norm": "addr_c", "name_foreign": "name_c_foreign"})
    q = queries[["entity_id", "name_norm", "addr_norm"]].rename(
        columns={"entity_id": "source1_entity_id", "name_norm": "name_s1", "addr_norm": "addr_s1"})

    parts = []
    firsts = range(0, candidates["s1_row"].max() + 1, chunk_entities)
    progress = Progress(len(firsts), "features: entity chunks", log)
    for first in firsts:
        chunk = candidates[(candidates["s1_row"] >= first) & (candidates["s1_row"] < first + chunk_entities)]
        if not chunk.empty:
            chunk = (chunk[["source1_entity_id", "candidate_id", "s1_row", "score_all", "score_name", "score_addr"]]
                     .merge(q, on="source1_entity_id").merge(text, on="candidate_id"))
            chunk = add_features(chunk, *(vectorizers or (None, None)))
            columns = (FEATURE_COLUMNS if vectorizers else FAST_FEATURE_COLUMNS) + EXTRA_FEATURE_COLUMNS
            parts.append(chunk[["source1_entity_id", "candidate_id"] + columns])
            gc.collect()
        progress.update()
    log(f"  features for {sum(len(p) for p in parts):,} pairs ({time.time() - start:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def label_pairs(pairs, truth_tables):
    true_keys = set()
    for t in truth_tables:
        for s1_id, ids in zip(t["source1_entity_id"], t["matched_entity_ids"]):
            true_keys.update((s1_id, m) for m in ids.split(",") if m)
    return np.array([(a, b) in true_keys for a, b in zip(pairs["source1_entity_id"], pairs["candidate_id"])], dtype="int8")


def run_blocking_and_features(queries, split, name_steps, address_steps, top_k=50, max_df=1000, use_tfidf=True, log=print):
    pool_path = normalized_pool(split, name_steps, address_steps, log=log)
    index = build_index(queries, pool_path, max_df=max_df, chunk_rows=250_000, log=log)   # small chunks: ~8 GB laptop
    candidates = generate_candidates(queries, index, top_k=top_k, ranking="all", log=log)
    del index
    gc.collect()
    log(f"  {len(candidates):,} candidates")
    vectorizers = fit_vectorizers(pool_path) if use_tfidf else None
    return candidates, pair_features(candidates, queries, pool_path, vectorizers, log=log), vectorizers
