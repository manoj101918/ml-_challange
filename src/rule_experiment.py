"""The Phase 5 rule-based pipeline as one reusable function, so normalization ideas can be compared fairly.

blocking : pool records sharing a rare word (df <= max_df) with the S1 entity, same country; top_k by idf sum
scoring  : mean of name Jaccard and address Jaccard
threshold: tuned on the "tuning" entities, applied unchanged to the "validation" entities

It reads the parquet cache made in notebook 06 (cache/train_pool.parquet, cache/train_source1.parquet).
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation import candidate_recall, score_report
from src.preprocessing import normalize
from src.splits import make_validation_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"


def load_query_sets(tuning_size=20_000, seed=7):
    """20k random train entities (tuning) + the 20k small-validation entities, with their truth."""
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", keep_default_na=False)
    split = make_validation_split(gt)
    tuning_ids = split.loc[split["split"] == "train", "source1_entity_id"].sample(tuning_size, random_state=seed)
    validation_ids = split.loc[split["small_validation"], "source1_entity_id"]
    truth = {
        "tuning": gt[gt["source1_entity_id"].isin(tuning_ids)].reset_index(drop=True),
        "validation": gt[gt["source1_entity_id"].isin(validation_ids)].reset_index(drop=True),
    }
    query_set = pd.concat([
        pd.DataFrame({"entity_id": tuning_ids, "query_set": "tuning"}),
        pd.DataFrame({"entity_id": validation_ids, "query_set": "validation"}),
    ])
    s1 = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
    s1 = s1.merge(query_set, on="entity_id").reset_index(drop=True)
    return s1, truth


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)


def run(name_steps=(), address_steps=(), s1=None, truth=None, max_df=1000, top_k=50,
        batch_size=2_000, pool_batch_rows=1_000_000, verbose=True):
    start = time.time()
    if s1 is None:
        s1, truth = load_query_sets()
    s1 = s1.copy()
    s1["name_clean"] = normalize(s1["business_name"], name_steps)
    s1["addr_clean"] = normalize(s1["business_address"], address_steps)

    # ---- query words -> integer key codes ("country|word")
    s1_words = s1[["country"]].assign(word=(s1["name_clean"] + " " + s1["addr_clean"]).str.split()).explode("word").dropna()
    s1_words["key"] = s1_words["country"] + "|" + s1_words["word"]
    s1_words["s1_row"] = s1_words.index
    s1_words = s1_words[["s1_row", "key"]].drop_duplicates()
    vocabulary = pd.Index(s1_words["key"].unique())
    s1_words["key_code"] = vocabulary.get_indexer(s1_words["key"]).astype("int32")

    # ---- pass 1 over the pool: (pool_row, key_code) for query words, dropping words that are too common
    pool_file = pq.ParquetFile(CACHE_DIR / "train_pool.parquet")
    pool_ids, pool_words, too_common, offset = [], [], set(), 0
    for batch in pool_file.iter_batches(batch_size=pool_batch_rows):
        chunk = batch.to_pandas()
        pool_ids.append(chunk["entity_id"])
        text = normalize(chunk["business_name"], name_steps) + " " + normalize(chunk["business_address"], address_steps)
        words = chunk[["country"]].assign(word=text.str.split()).explode("word").dropna()
        codes = vocabulary.get_indexer(words["country"] + "|" + words["word"])
        words = pd.DataFrame({"pool_row": (words.index.to_numpy() + offset).astype("int32"), "key_code": codes.astype("int32")})
        words = words[words["key_code"] >= 0].drop_duplicates()
        counts = words["key_code"].value_counts()
        too_common |= set(counts[counts > max_df].index)
        pool_words.append(words[~words["key_code"].isin(too_common)])
        offset += len(chunk)
    pool_ids = pd.concat(pool_ids, ignore_index=True)
    pool_words = pd.concat(pool_words, ignore_index=True)
    pool_words = pool_words[~pool_words["key_code"].isin(too_common)]

    df = pool_words["key_code"].value_counts()
    df = df[df <= max_df]
    idf = np.log(len(pool_ids) / df)
    pool_words = pool_words[pool_words["key_code"].isin(df.index)]
    s1_rare = s1_words[s1_words["key_code"].isin(df.index)]
    if verbose:
        print(f"  pool indexed ({time.time() - start:.0f}s)")

    # ---- blocking in batches
    batches = []
    for batch_start in range(0, len(s1), batch_size):
        batch = s1_rare[(s1_rare["s1_row"] >= batch_start) & (s1_rare["s1_row"] < batch_start + batch_size)]
        shared = batch.merge(pool_words, on="key_code")
        shared["idf"] = idf.reindex(shared["key_code"]).to_numpy()
        batches.append(
            shared.groupby(["s1_row", "pool_row"])["idf"].sum().rename("idf_sum").reset_index()
            .sort_values("idf_sum", ascending=False).groupby("s1_row").head(top_k)
        )
    candidates = pd.concat(batches, ignore_index=True)
    candidates["source1_entity_id"] = s1["entity_id"].to_numpy()[candidates["s1_row"]]
    candidates["candidate_id"] = pool_ids.to_numpy()[candidates["pool_row"]]
    del batches, pool_words, s1_words, s1_rare
    gc.collect()

    # ---- pass 2: text of the candidate records only
    wanted = candidates["pool_row"].drop_duplicates().sort_values().to_numpy()
    pool_table = pq.read_table(CACHE_DIR / "train_pool.parquet", columns=["entity_id", "business_name", "business_address"])
    cand = pool_table.take(wanted).to_pandas()
    del pool_table
    cand["name_clean"] = normalize(cand["business_name"], name_steps)
    cand["addr_clean"] = normalize(cand["business_address"], address_steps)

    pairs = (
        candidates[["source1_entity_id", "candidate_id"]]
        .merge(s1[["entity_id", "query_set", "name_clean", "addr_clean"]], left_on="source1_entity_id", right_on="entity_id")
        .drop(columns="entity_id")
        .merge(cand[["entity_id", "name_clean", "addr_clean"]], left_on="candidate_id", right_on="entity_id", suffixes=("_s1", "_c"))
        .drop(columns="entity_id")
    )
    pairs["name_jaccard"] = [jaccard(a, b) for a, b in zip(pairs["name_clean_s1"], pairs["name_clean_c"])]
    pairs["addr_jaccard"] = [jaccard(a, b) for a, b in zip(pairs["addr_clean_s1"], pairs["addr_clean_c"])]
    pairs["score"] = pairs[["name_jaccard", "addr_jaccard"]].mean(axis=1).fillna(0)

    # ---- metrics
    def lists(table, out_column):
        return table.groupby("source1_entity_id")["candidate_id"].agg(",".join).rename(out_column).reset_index()

    result = {}
    for name in ["tuning", "validation"]:
        in_set = pairs[pairs["query_set"] == name]
        result[f"{name}_candidate_recall"] = candidate_recall(lists(in_set, "candidate_entity_ids"), truth[name])

    tuning_pairs = pairs[pairs["query_set"] == "tuning"]
    thresholds = np.round(np.arange(0.20, 0.96, 0.05), 2)
    tuning_scores = [score_report(lists(tuning_pairs[tuning_pairs["score"] >= t], "matched_entity_ids"), truth["tuning"])["macro_F0.5"]
                     for t in thresholds]
    best = float(thresholds[int(np.argmax(tuning_scores))])

    validation_pairs = pairs[pairs["query_set"] == "validation"]
    report = score_report(lists(validation_pairs[validation_pairs["score"] >= best], "matched_entity_ids"), truth["validation"])
    result.update({
        "threshold": best,
        "tuning_F0.5": max(tuning_scores),
        "validation_F0.5": report["macro_F0.5"],
        "validation_precision": report["pair precision"],
        "validation_recall": report["pair recall"],
        "candidate_pairs": len(pairs),
        "seconds": round(time.time() - start),
    })
    if verbose:
        print(f"  done: validation F0.5 {result['validation_F0.5']:.4f} ({result['seconds']}s)")
    return result, pairs


def score_candidates(candidates, s1, truth, pool_norm_path=CACHE_DIR / "train_pool_norm.parquet", verbose=True):
    """Rule-based scoring (mean of name / address Jaccard, threshold tuned on "tuning") for ANY candidate table.

    candidates: columns source1_entity_id, candidate_id, pool_row (row in the normalized pool parquet)
    s1        : columns entity_id, query_set, name_norm, addr_norm
    """
    start = time.time()
    rows = np.sort(candidates["pool_row"].unique())
    text = pq.read_table(pool_norm_path, columns=["entity_id", "name_norm", "addr_norm"]).take(rows).to_pandas()
    pairs = (
        candidates[["source1_entity_id", "candidate_id"]]
        .merge(s1[["entity_id", "query_set", "name_norm", "addr_norm"]], left_on="source1_entity_id", right_on="entity_id")
        .drop(columns="entity_id")
        .merge(text, left_on="candidate_id", right_on="entity_id", suffixes=("_s1", "_c"))
        .drop(columns="entity_id")
    )
    pairs["name_jaccard"] = [jaccard(a, b) for a, b in zip(pairs["name_norm_s1"], pairs["name_norm_c"])]
    pairs["addr_jaccard"] = [jaccard(a, b) for a, b in zip(pairs["addr_norm_s1"], pairs["addr_norm_c"])]
    pairs["score"] = pairs[["name_jaccard", "addr_jaccard"]].mean(axis=1).fillna(0)

    def lists(table, out_column):
        return table.groupby("source1_entity_id")["candidate_id"].agg(",".join).rename(out_column).reset_index()

    result = {}
    for name in ["tuning", "validation"]:
        result[f"{name}_candidate_recall"] = candidate_recall(lists(pairs[pairs["query_set"] == name], "candidate_entity_ids"), truth[name])
    tuning_pairs = pairs[pairs["query_set"] == "tuning"]
    thresholds = np.round(np.arange(0.20, 0.96, 0.05), 2)
    tuning_scores = [score_report(lists(tuning_pairs[tuning_pairs["score"] >= t], "matched_entity_ids"), truth["tuning"])["macro_F0.5"]
                     for t in thresholds]
    best = float(thresholds[int(np.argmax(tuning_scores))])
    validation_pairs = pairs[pairs["query_set"] == "validation"]
    report = score_report(lists(validation_pairs[validation_pairs["score"] >= best], "matched_entity_ids"), truth["validation"])
    result.update({
        "threshold": best,
        "tuning_F0.5": max(tuning_scores),
        "validation_F0.5": report["macro_F0.5"],
        "validation_precision": report["pair precision"],
        "validation_recall": report["pair recall"],
        "candidate_pairs": len(pairs),
        "seconds": round(time.time() - start),
    })
    if verbose:
        print(f"  scored: validation F0.5 {result['validation_F0.5']:.4f} ({result['seconds']}s)")
    return result, pairs
