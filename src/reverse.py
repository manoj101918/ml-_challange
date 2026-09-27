"""Reverse matching — round 2: every S2/S3 record looks for its S1 owner.

Data fact: every S2/S3 record is a noisy copy of at most ONE S1. For each pool record we find its top K S1 of the same
country (idf sum over shared rare keys; the index always holds ALL S1 of the split, so training — which uses only a
sample of S1 entities — and the test set see the same density). Every candidate pair then gets 4 "competition" features:

    r_rank        rank of this S1 among the record's top K (K + 1 = not in its top K)
    r_score       this S1's reverse score (0 if not in the record's top K)
    r_best_other  best reverse score of ANOTHER S1 for the same record
    r_gap         r_score - r_best_other          (> 0: this S1 is the record's favourite)

Measured on the Phase 8 pairs: +0.0102 F0.5 (experiments/competition_check.py); experiments/reverse_diagnostic.py:
for 97% of the true pairs the S1 is the record's favourite, usually by a wide margin.

Tables are built once per split and country (countries can run in parallel Colab sessions) and cached:
    python -m src.reverse --split train                     (all countries)
    python -m src.reverse --split test --countries India    (one country)
    -> cache/reverse/{split}_{tag}_{country}.parquet  (record_id, s1_id, r_score, r_rank)
"""

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking import _hash, iter_candidates
from src.blocking_keys import record_keys
from src.pipeline import CACHE_DIR, normalization_tag, normalized_pool, prepare_queries
from src.preprocessing import ADDRESS_STEPS, NAME_STEPS
from src.progress import Progress

K_REVERSE = 10
MAX_DF = 1000
REVERSE_COLUMNS = ["r_rank", "r_score", "r_best_other", "r_gap"]
REVERSE_DIR = Path(os.environ.get("ER_REVERSE_DIR", CACHE_DIR / "reverse"))
TRANSLIT_NAME = ("translit",) + NAME_STEPS            # the normalization of E4/E5/E6 (same as src/train.py)
TRANSLIT_ADDRESS = ("translit",) + ADDRESS_STEPS


def reverse_path(split, tag, country):
    return REVERSE_DIR / f"{split}_{tag}_{''.join(ch if ch.isalnum() else '_' for ch in country)}.parquet"


# ---------------------------------------------------------------- building the tables

def s1_index(s1_c, n_s1, max_df=MAX_DF):
    """Key index of the S1 records of ONE country (built once, reused for every chunk of pool records).
    idf = log(n_s1 / df) with df counted over the S1 records, n_s1 = all S1 of the split."""
    k = record_keys(s1_c["name_norm"], s1_c["addr_norm"], s1_c["country"])
    keys = pd.DataFrame({"pool_row": k["row"].to_numpy().astype("int32"), "h": _hash(k["key"]),
                         "is_name": k["is_name"].to_numpy()}).drop_duplicates(["pool_row", "h"])
    vocabulary, code, df = np.unique(keys["h"].to_numpy(), return_inverse=True, return_counts=True)
    is_name = np.zeros(len(vocabulary), dtype=bool)
    is_name[code] = keys["is_name"].to_numpy().astype(bool)
    keys = pd.DataFrame({"pool_row": keys["pool_row"].to_numpy(), "code": code.astype("int32")})
    keys = keys[df[keys["code"]] <= max_df].reset_index(drop=True)
    return {"vocabulary": vocabulary, "df": df, "idf": np.log(n_s1 / np.maximum(df, 1)), "is_name_key": is_name,
            "pool_keys": keys, "pool_ids": s1_c["entity_id"].reset_index(drop=True), "max_df": max_df}


def _query_keys(records, index):
    k = record_keys(records["name_norm"], records["addr_norm"], records["country"])
    h = _hash(k["key"])
    vocabulary = index["vocabulary"]
    position = np.searchsorted(vocabulary, h)
    position[position == len(vocabulary)] = 0
    hit = (vocabulary[position] == h) & (index["df"][position] <= index["max_df"])
    return pd.DataFrame({"s1_row": k["row"].to_numpy()[hit].astype("int32"),
                         "code": position[hit].astype("int32")}).drop_duplicates()


def build_country(records, s1_c, n_s1, out_path, k=K_REVERSE, chunk=500_000, log=print):
    """Top-k S1 for every record of ONE country -> parquet (record_id, s1_id, r_score, r_rank)."""
    index = s1_index(s1_c, n_s1)
    progress = Progress(len(records), f"reverse {out_path.stem}: records", log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".{os.getpid()}.tmp")          # one temp file per process: parallel runs never share one
    writer = None
    for first in range(0, len(records), chunk):
        part = records.iloc[first:first + chunk].reset_index(drop=True)
        chunk_index = dict(index, query_keys=_query_keys(part, index))
        for c in iter_candidates(part, chunk_index, top_k=k, batch_size=5_000, yield_every=100_000):
            rank = c.groupby("s1_row")["score_all"].rank(method="first", ascending=False)
            table = pa.Table.from_pandas(pd.DataFrame({
                "record_id": c["source1_entity_id"].to_numpy(), "s1_id": c["candidate_id"].to_numpy(),
                "r_score": c["score_all"].to_numpy().astype("float32"), "r_rank": rank.to_numpy().astype("int8")}),
                preserve_index=False)
            writer = writer or pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
        progress.update(len(part))
        gc.collect()
    if writer is None:                                   # no record found any S1 (tiny test inputs)
        writer = pq.ParquetWriter(tmp, pa.schema([("record_id", pa.string()), ("s1_id", pa.string()),
                                                  ("r_score", pa.float32()), ("r_rank", pa.int8())]))
    writer.close()
    tmp.replace(out_path)


def build(split, name_steps=TRANSLIT_NAME, address_steps=TRANSLIT_ADDRESS, countries=None, k=K_REVERSE,
          chunk=500_000, part=None, parts=None, log=print):
    """part/parts: build only slice `part` (0-based) of `parts` equal slices of the records -> a file
    {split}_{tag}_{country}_p{part}of{parts}.parquet, so several processes can build one country in parallel
    (each record's top-k does not depend on the other records, so the parts together equal the whole table)."""
    tag = normalization_tag(name_steps, address_steps)
    pool_path = normalized_pool(split, name_steps, address_steps, log=log)
    s1_all = pd.read_parquet(CACHE_DIR / f"{split}_source1.parquet")
    for country in countries or sorted(s1_all["country"].unique()):
        out = reverse_path(split, tag, country)
        if parts:
            out = out.with_name(f"{out.stem}_p{part}of{parts}.parquet")
        if out.exists():
            log(f"{out.name} exists - skipped")
            continue
        start = time.time()
        s1_c = prepare_queries(s1_all[s1_all["country"] == country], name_steps, address_steps)
        s1_c = s1_c[["entity_id", "country", "name_norm", "addr_norm"]]
        records = pq.read_table(pool_path, columns=["entity_id", "country", "name_norm", "addr_norm"],
                                filters=[("country", "=", country)]).to_pandas()
        if parts:
            records = records.iloc[np.array_split(np.arange(len(records)), parts)[part]].reset_index(drop=True)
        log(f"reverse {split} {country}: {len(records):,} records vs {len(s1_c):,} S1")
        build_country(records, s1_c, len(s1_all), out, k=k, chunk=chunk, log=log)
        log(f"wrote {out.name} ({time.time() - start:.0f}s)")
        del s1_c, records
        gc.collect()


# ---------------------------------------------------------------- using the tables

def _pair_hash(s1_ids, record_ids):
    """One 64-bit key per (S1, record) pair."""
    a, b = _hash(np.asarray(s1_ids, dtype=object)), _hash(np.asarray(record_ids, dtype=object))
    return a * np.uint64(0x9E3779B97F4A7C15) ^ b


class ReverseLookup:
    """Sorted 64-bit keys -> the 4 competition features for any list of (S1, record) pairs, with NumPy only."""

    def __init__(self, table, k=K_REVERSE):
        self.k = k
        keys = _pair_hash(table["s1_id"], table["record_id"])
        order = np.argsort(keys)
        self.pair_keys = keys[order]
        self.pair_score = table["r_score"].to_numpy()[order]
        self.pair_rank = table["r_rank"].to_numpy()[order]
        rank = table["r_rank"].to_numpy()
        record_h = _hash(np.asarray(table["record_id"], dtype=object))
        best = pd.DataFrame({"r": record_h[rank == 1], "best": table["r_score"].to_numpy()[rank == 1],
                             "best_s1": _hash(np.asarray(table["s1_id"], dtype=object))[rank == 1]}).sort_values("r")
        second = pd.Series(table["r_score"].to_numpy()[rank == 2], index=record_h[rank == 2])
        self.record_keys = best["r"].to_numpy()
        self.best = best["best"].to_numpy()
        self.best_s1 = best["best_s1"].to_numpy()
        self.second = second.reindex(self.record_keys).fillna(0).to_numpy().astype("float32")

    @staticmethod
    def _find(sorted_keys, keys):
        position = np.searchsorted(sorted_keys, keys)
        position[position == len(sorted_keys)] = 0
        return position, (sorted_keys[position] == keys) if len(sorted_keys) else np.zeros(len(keys), dtype=bool)

    def features(self, s1_ids, record_ids):
        pos, found = self._find(self.pair_keys, _pair_hash(s1_ids, record_ids))
        r_score = np.where(found, self.pair_score[pos] if len(self.pair_keys) else 0, 0).astype("float32")
        r_rank = np.where(found, self.pair_rank[pos] if len(self.pair_keys) else 0, self.k + 1).astype("float32")
        rpos, rfound = self._find(self.record_keys, _hash(np.asarray(record_ids, dtype=object)))
        if len(self.record_keys):
            best = np.where(rfound, self.best[rpos], 0)
            second = np.where(rfound, self.second[rpos], 0)
            is_best = rfound & (self.best_s1[rpos] == _hash(np.asarray(s1_ids, dtype=object)))
        else:
            best = second = np.zeros(len(r_score))
            is_best = np.zeros(len(r_score), dtype=bool)
        r_best_other = np.where(is_best, second, best).astype("float32")
        return pd.DataFrame({"r_rank": r_rank, "r_score": r_score, "r_best_other": r_best_other,
                             "r_gap": (r_score - r_best_other).astype("float32")})


def load_lookup(split, tag, countries=None, log=print):
    if countries:                                  # the whole-country file, or its _pXofN parts
        paths, missing = [], []
        for c in countries:
            whole = reverse_path(split, tag, c)
            found = [whole] if whole.exists() else sorted(REVERSE_DIR.glob(f"{whole.stem}_p*of*.parquet"))
            paths += found
            missing += [] if found else [whole.name]
    else:
        paths, missing = sorted(REVERSE_DIR.glob(f"{split}_{tag}_*.parquet")), []
    if missing or not paths:
        raise FileNotFoundError(f"reverse tables missing: {missing or split + '_' + tag + '_*'} - run python -m src.reverse --split {split}")
    table = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    log(f"  reverse lookup: {len(table):,} (record, S1) rows from {len(paths)} file(s)")
    return ReverseLookup(table)


def add_reverse_features(pairs, lookup):
    values = lookup.features(pairs["source1_entity_id"].to_numpy(), pairs["candidate_id"].to_numpy())
    for column in REVERSE_COLUMNS:
        pairs[column] = values[column].to_numpy()
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--countries", nargs="*")
    parser.add_argument("--chunk", type=int, default=500_000)
    parser.add_argument("--part", type=int)
    parser.add_argument("--parts", type=int)
    args = parser.parse_args()
    start = time.time()
    build(args.split, countries=args.countries, chunk=args.chunk, part=args.part, parts=args.parts,
          log=lambda msg: print(f"[{time.time() - start:6.0f}s] {msg}", flush=True))


if __name__ == "__main__":
    main()
