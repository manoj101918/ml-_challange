"""Final prediction on a full split (the test set, or the full validation split as a rehearsal) — Phase 13.

Memory design for an 8 GB laptop:
  * one country at a time (keys contain the country, so nothing is lost),
  * hashed blocking keys (src/blocking.build_index_hashed),
  * candidates streamed in pieces of ~10k S1 entities: features -> probability -> keep only what is needed.

Usage (from the project root):
    python -m src.predict --model cache/models/E4.joblib --split test
    python -m src.predict --model cache/models/E4.joblib --split validation     # full 441k validation entities, scored
"""

import argparse
import gc
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking import build_index_hashed, iter_candidates
from src.decision import global_threshold, resolve_conflicts
from src.features import add_features
from src.pipeline import CACHE_DIR, PROJECT_ROOT, normalized_pool, prepare_queries
from src.submission import write_id_list_file

KEEP_PROBABILITY = 0.30      # scored pairs below this are dropped right away (far below the ~0.7 threshold)


def score_country(queries, pool_path, n_pool, artifacts, country, candidate_file, chunk_rows=250_000, log=print):
    """Probabilities for the S1 entities of one country. Candidate lists are written straight to `candidate_file`
    (on the test set they are ~2 GB of text, too big to keep in memory). Returns (scored pairs, IDs written)."""
    start = time.time()
    q = queries[queries["country"] == country].reset_index(drop=True)
    pool = pq.read_table(pool_path, columns=["entity_id", "country", "name_norm", "addr_norm", "name_foreign"],
                         filters=[("country", "=", country)])
    index = build_index_hashed(q, (b.to_pandas() for b in pool.to_batches(max_chunksize=chunk_rows)),
                               n_pool=n_pool, max_df=artifacts["max_df"], log=log)
    q_text = q[["entity_id", "name_norm", "addr_norm"]].rename(
        columns={"entity_id": "source1_entity_id", "name_norm": "name_s1", "addr_norm": "addr_s1"})

    scored, written = [], []
    n_done = 0
    for candidates in iter_candidates(q, index, top_k=artifacts["top_k"]):
        text = pool.take(candidates["pool_row"].to_numpy()).to_pandas()
        pairs = candidates[["source1_entity_id", "s1_row", "score_all", "score_name", "score_addr"]].reset_index(drop=True)
        pairs["candidate_id"] = text["entity_id"].to_numpy()
        pairs["name_c"] = text["name_norm"].to_numpy()
        pairs["addr_c"] = text["addr_norm"].to_numpy()
        pairs["name_c_foreign"] = text["name_foreign"].to_numpy()
        pairs = pairs.merge(q_text, on="source1_entity_id")
        pairs = add_features(pairs, *(artifacts["vectorizers"] or (None, None)))
        pairs["probability"] = artifacts["model"].predict_proba(pairs[artifacts["features"]].astype("float32"))[:, 1]

        lists = pairs.groupby("source1_entity_id")["candidate_id"].agg(",".join)
        candidate_file.write("".join(f"{s1_id}\t{ids}\n" for s1_id, ids in lists.items()))
        written.append(lists.index.to_numpy())
        scored.append(pairs.loc[pairs["probability"] >= KEEP_PROBABILITY, ["source1_entity_id", "candidate_id", "probability"]])
        n_done += pairs["source1_entity_id"].nunique()
        log(f"  {country}: {n_done:,}/{len(q):,} entities scored ({time.time() - start:.0f}s)")
        del pairs, text
        gc.collect()
    del index, pool
    gc.collect()
    return pd.concat(scored, ignore_index=True), np.concatenate(written) if written else np.array([], dtype=object)


def predict(split, s1, artifacts, candidate_path, log=print):
    """Writes candidate_pairs.tsv to `candidate_path` (one row per S1 entity in `s1`);
    returns (matching_results with one row per S1 entity, scored pairs)."""
    queries = prepare_queries(s1, artifacts["name_steps"], artifacts["address_steps"])
    pool_path = normalized_pool(split, artifacts["name_steps"], artifacts["address_steps"], log=log)
    n_pool = pq.ParquetFile(pool_path).metadata.num_rows

    scored, written = [], []
    with open(candidate_path, "w", encoding="utf-8", newline="\n") as candidate_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in sorted(queries["country"].unique()):          # open set of countries - nothing hard-coded
            pairs, ids = score_country(queries, pool_path, n_pool, artifacts, country, candidate_file, log=log)
            scored.append(pairs)
            written.append(ids)
        without_candidates = s1.loc[~s1["entity_id"].isin(np.concatenate(written)), "entity_id"]
        candidate_file.write("".join(f"{s1_id}\t\n" for s1_id in without_candidates))
    log(f"candidate file written ({len(without_candidates):,} entities without candidates)")
    scored = pd.concat(scored, ignore_index=True)

    matches = resolve_conflicts(global_threshold(scored, artifacts["threshold"]))
    match_lists = matches.groupby("source1_entity_id")["candidate_id"].agg(",".join)

    all_ids = s1["entity_id"]
    matching = pd.DataFrame({"source1_entity_id": all_ids,
                             "matched_entity_ids": all_ids.map(match_lists).fillna("").to_numpy()})
    return matching, scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["test", "validation", "small_validation"], default="test")
    args = parser.parse_args()
    start = time.time()
    log = lambda msg: print(f"[{time.time() - start:6.0f}s] {msg}", flush=True)
    artifacts = joblib.load(args.model)

    if args.split == "test":
        s1 = pd.read_parquet(CACHE_DIR / "test_source1.parquet")
        out = PROJECT_ROOT / "output"
        matching, scored = predict("test", s1, artifacts, out / "candidate_pairs.tsv", log=log)
        write_id_list_file(matching, out / "matching_results.tsv", "matched_entity_ids")
        scored.to_parquet(CACHE_DIR / "test_scored_pairs.parquet", index=False)
        log(f"wrote output/matching_results.tsv and output/candidate_pairs.tsv "
            f"({(matching['matched_entity_ids'] != '').sum():,} entities with matches)")
    else:
        from src.evaluation import score_report
        from src.splits import make_validation_split

        gt = pd.read_csv(PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv", sep="\t", keep_default_na=False)
        split = make_validation_split(gt)
        column = "small_validation" if args.split == "small_validation" else "split"
        validation_ids = split.loc[split[column] if column == "small_validation" else split["split"] == "validation", "source1_entity_id"]
        s1 = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
        s1 = s1[s1["entity_id"].isin(validation_ids)].reset_index(drop=True)
        candidate_path = PROJECT_ROOT / "output" / f"{args.split}_candidate_pairs.tsv"
        matching, scored = predict("train", s1, artifacts, candidate_path, log=log)
        candidates = pd.read_csv(candidate_path, sep="	", keep_default_na=False)
        truth = gt[gt["source1_entity_id"].isin(validation_ids)]
        report = score_report(matching, truth)
        recall = score_report(candidates.rename(columns={"candidate_entity_ids": "matched_entity_ids"}), truth)["pair recall"]
        log(f"{args.split.upper()} ({len(s1):,} entities): macro F0.5 {report['macro_F0.5']:.4f} | candidate recall {recall:.4f}")
        log(report.round(4).to_string())
        scored.to_parquet(CACHE_DIR / f"{args.split}_scored_pairs.parquet", index=False)


if __name__ == "__main__":
    main()
