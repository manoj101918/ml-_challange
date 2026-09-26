"""Pair features for the matching model (Phase 8).

Input: a candidate-pair table with normalized text of both sides
    name_s1, addr_s1   (Source 1 entity)
    name_c,  addr_c    (candidate S2/S3 record)
    score_all, score_name, score_addr, s1_row   (from src/blocking.py)
    candidate_id       (S2-... / S3-...)
Output: the same table with one numeric column per feature (FEATURE_COLUMNS).

Every feature uses only the two records' TEXT (plus blocking scores computed from unlabeled text),
never the ground truth — so exactly the same code runs on the test set.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import LEGAL_WORDS

LEGAL_FORM = {  # only used to COMPARE legal forms, the text itself is not changed (Phase 6 showed that hurts)
    "pvt": "private", "ltd": "limited", "inc": "incorporated", "corp": "corporation", "co": "company",
}
LEGAL_SET = set(LEGAL_WORDS)


# ---------------------------------------------------------------- helpers

def pairwise(scorer, left, right):
    """Score left[i] vs right[i] for every i, in C++ on all CPU cores. Result 0..100 -> 0..1."""
    return process.cpdist(left.tolist(), right.tolist(), scorer=scorer, workers=-1) / 100.0


def jaccard_lists(left, right):
    out = np.full(len(left), np.nan, dtype="float32")
    for i, (a, b) in enumerate(zip(left, right)):
        a, b = set(a), set(b)
        if a and b:
            out[i] = len(a & b) / len(a | b)
    return out


def legal_forms(name_words):
    return [frozenset(LEGAL_FORM.get(w, w) for w in words if w in LEGAL_SET) for words in name_words]


def fit_tfidf(texts, max_features=200_000):
    """Character 3-gram TF-IDF, fitted on UNLABELED pool text (a sample is enough)."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_features=max_features, dtype=np.float32)
    vectorizer.fit(texts)
    return vectorizer


def tfidf_cosine(vectorizer, left, right):
    """Cosine similarity of left[i] and right[i]. Unique strings are transformed once."""
    uniq_left, inv_left = np.unique(left.to_numpy(dtype=object), return_inverse=True)
    uniq_right, inv_right = np.unique(right.to_numpy(dtype=object), return_inverse=True)
    a = vectorizer.transform(uniq_left)[inv_left]
    b = vectorizer.transform(uniq_right)[inv_right]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype("float32")


# ---------------------------------------------------------------- the features

def _core(word_lists):
    """Name without legal words, built from the already-split words (one pass instead of 48 text replacements)."""
    return pd.Series([" ".join(w for w in words if w not in LEGAL_SET) for words in word_lists], dtype="str")


def add_features(pairs, name_vectorizer=None, addr_vectorizer=None):
    """Adds the feature columns. Without vectorizers the two slow TF-IDF features are skipped (FAST_FEATURE_COLUMNS)."""
    f = pairs.reset_index(drop=True)
    name_s1_words = f["name_s1"].str.split()
    name_c_words = f["name_c"].str.split()
    core_s1 = _core(name_s1_words)
    core_c = _core(name_c_words)

    # --- name
    f["name_exact"] = (f["name_s1"] == f["name_c"]).astype("int8")
    f["name_jaccard"] = jaccard_lists(name_s1_words, name_c_words)
    f["name_ratio"] = pairwise(fuzz.ratio, f["name_s1"], f["name_c"])
    f["name_token_sort"] = pairwise(fuzz.token_sort_ratio, f["name_s1"], f["name_c"])
    f["name_token_set"] = pairwise(fuzz.token_set_ratio, f["name_s1"], f["name_c"])
    f["core_ratio"] = pairwise(fuzz.ratio, core_s1, core_c)
    f["core_token_set"] = pairwise(fuzz.token_set_ratio, core_s1, core_c)
    f["core_partial"] = pairwise(fuzz.partial_ratio, core_s1, core_c)
    f["core_joined_ratio"] = pairwise(fuzz.ratio, core_s1.str.replace(" ", "", regex=False), core_c.str.replace(" ", "", regex=False))
    if name_vectorizer is not None:
        f["name_tfidf"] = tfidf_cosine(name_vectorizer, f["name_s1"], f["name_c"])
    f["name_len_diff"] = (f["name_s1"].str.len() - f["name_c"].str.len()).abs().astype("float32")
    if "name_c_foreign" in f:      # flag computed from the RAW name (after transliteration the normalized name is Latin)
        f["name_c_foreign_script"] = f["name_c_foreign"].astype("int8")
    else:
        f["name_c_foreign_script"] = f["name_c"].str.contains("[ऀ-ൿ]").astype("int8")

    # --- legal form: same / conflicting / missing on one side
    forms_s1, forms_c = legal_forms(name_s1_words), legal_forms(name_c_words)
    f["legal_same"] = np.array([bool(a) and a == b for a, b in zip(forms_s1, forms_c)], dtype="int8")
    f["legal_conflict"] = np.array([bool(a) and bool(b) and not (a & b) for a, b in zip(forms_s1, forms_c)], dtype="int8")
    f["legal_one_missing"] = np.array([bool(a) != bool(b) for a, b in zip(forms_s1, forms_c)], dtype="int8")

    # --- address
    f["addr_c_empty"] = (f["addr_c"] == "").astype("int8")
    f["addr_jaccard"] = jaccard_lists(f["addr_s1"].str.split(), f["addr_c"].str.split())
    f["addr_ratio"] = pairwise(fuzz.ratio, f["addr_s1"], f["addr_c"])
    f["addr_token_set"] = pairwise(fuzz.token_set_ratio, f["addr_s1"], f["addr_c"])
    f["addr_token_sort"] = pairwise(fuzz.token_sort_ratio, f["addr_s1"], f["addr_c"])
    if addr_vectorizer is not None:
        f["addr_tfidf"] = tfidf_cosine(addr_vectorizer, f["addr_s1"], f["addr_c"])
    for col in [c for c in ["addr_ratio", "addr_token_set", "addr_token_sort", "addr_tfidf"] if c in f]:
        f.loc[f["addr_c_empty"] == 1, col] = np.nan          # "unknown", not "different"

    # --- numbers in the address (house / plot / door numbers)
    nums_s1 = [set(x) for x in f["addr_s1"].str.findall(r"\d+")]
    nums_c = [set(x) for x in f["addr_c"].str.findall(r"\d+")]
    f["num_shared"] = np.array([len(a & b) for a, b in zip(nums_s1, nums_c)], dtype="float32")
    f["num_jaccard"] = np.array([len(a & b) / len(a | b) if a and b else np.nan for a, b in zip(nums_s1, nums_c)], dtype="float32")
    f["num_conflict"] = np.array([bool(a) and bool(b) and not (a & b) for a, b in zip(nums_s1, nums_c)], dtype="int8")
    f["num_c_subset"] = np.array([bool(b) and b <= a for a, b in zip(nums_s1, nums_c)], dtype="int8")

    # --- blocking evidence and the candidate's position among this entity's candidates
    group = f.groupby("s1_row")
    f["block_score"] = f["score_all"].astype("float32")
    f["block_score_name"] = f["score_name"].astype("float32")
    f["block_score_addr"] = f["score_addr"].astype("float32")
    f["block_rank"] = group["score_all"].rank(method="first", ascending=False).astype("float32")
    f["block_score_rel"] = (f["score_all"] / group["score_all"].transform("max")).astype("float32")
    f["n_candidates"] = group["score_all"].transform("size").astype("float32")
    f["name_token_set_rel"] = (f["name_token_set"] - group["name_token_set"].transform("max")).astype("float32")
    f["addr_token_set_rel"] = (f["addr_token_set"].fillna(0) - group["addr_token_set"].transform("max").fillna(0)).astype("float32")

    f["is_source3"] = f["candidate_id"].str.startswith("S3").astype("int8")
    return f


FEATURE_COLUMNS = [
    "name_exact", "name_jaccard", "name_ratio", "name_token_sort", "name_token_set",
    "core_ratio", "core_token_set", "core_partial", "core_joined_ratio", "name_tfidf", "name_len_diff", "name_c_foreign_script",
    "legal_same", "legal_conflict", "legal_one_missing",
    "addr_c_empty", "addr_jaccard", "addr_ratio", "addr_token_set", "addr_token_sort", "addr_tfidf",
    "num_shared", "num_jaccard", "num_conflict", "num_c_subset",
    "block_score", "block_score_name", "block_score_addr", "block_rank", "block_score_rel", "n_candidates",
    "name_token_set_rel", "addr_token_set_rel",
    "is_source3",
]

# Phase 12: the two TF-IDF features cost 64% of the feature time but only 0.001 F0.5 (experiments/phase12_ablation.tsv)
FAST_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("name_tfidf", "addr_tfidf")]
