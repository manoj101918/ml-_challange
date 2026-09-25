"""Loading helpers for the competition TSV files.

Every function here only READS data. Nothing on disk is changed.
"""

import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path, usecols=None):
    """Read a competition TSV file with safe settings.

    sep="\\t"                 -> columns are separated by TAB characters, not commas
                                (addresses contain commas, so a CSV reader would split them wrongly)
    dtype="string[pyarrow]"  -> compact string storage; uses far less RAM than Python objects
    keep_default_na=False    -> an empty field stays "" instead of becoming NaN,
                                and text like "null" or "NA" stays text
    quoting=csv.QUOTE_NONE   -> quote characters inside names are treated as normal text
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype="string[pyarrow]",
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        usecols=usecols,
    )


def load_source(split, source_number, usecols=None):
    """Load one source file, e.g. load_source("train", 1) -> train_source1.tsv."""
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    return read_tsv(folder / f"{split}_source{source_number}.tsv", usecols=usecols)


def load_ground_truth():
    """Load train_ground_truth.tsv in its original 'wide' form.

    One row per Source 1 entity; matched_entity_ids is a comma-separated string
    ("" when the entity has no match).
    """
    return read_tsv(TRAIN_DIR / "train_ground_truth.tsv")


def ground_truth_to_pairs(ground_truth):
    """Turn the wide ground truth into a 'long' table: one row per (S1, matched record) pair.

    Example:
        S1-1   "S2-5,S3-9"      ->   S1-1  S2-5
                                     S1-1  S3-9
    Source 1 entities with no match produce no rows here.
    """
    has_match = ground_truth[ground_truth["matched_entity_ids"] != ""]
    pairs = has_match.assign(
        match_id=has_match["matched_entity_ids"].str.split(",")
    ).explode("match_id")
    pairs = pairs[["source1_entity_id", "match_id"]].reset_index(drop=True)
    pairs["match_id"] = pairs["match_id"].astype("string[pyarrow]")
    pairs["match_source"] = pairs["match_id"].str[:2]  # "S2" or "S3"
    return pairs
