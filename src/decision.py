"""Turning pair probabilities into final match lists (Phase 10).

Input: a table with one row per candidate pair: source1_entity_id, candidate_id, probability.
Output of every rule: the same table filtered to the predicted matches.
"""

import numpy as np
import pandas as pd


def global_threshold(pairs, threshold):
    """Predict every pair with probability >= threshold."""
    return pairs[pairs["probability"] >= threshold]


def relative_threshold(pairs, threshold, ratio):
    """Predict pairs with probability >= threshold AND >= ratio x the entity's best probability."""
    best = pairs.groupby("source1_entity_id")["probability"].transform("max")
    return pairs[(pairs["probability"] >= threshold) & (pairs["probability"] >= ratio * best)]


def expected_f05(pairs, missing_rate=0.0):
    """Per entity, choose the top-k candidates (k = 0, 1, 2, ...) that maximise the EXPECTED F0.5.

    With probabilities p1 >= p2 >= ... of the entity's candidates:
        expected number of true matches      N  = sum(p) / (1 - missing_rate)   (blocking misses some matches)
        expected correct among the top k     TP = p1 + ... + pk
        F0.5(k) ~ 1.25 * TP / (k + 0.25 * N)                       for k >= 1
        F0.5(0) = P(entity has no match) ~ prod(1 - p) * (1 - missing_rate)   (all candidates wrong, nothing missed)
    This is the "plug-in" approximation (expectations inside the formula); it needs roughly calibrated probabilities.
    """
    p = pairs.sort_values(["source1_entity_id", "probability"], ascending=[True, False]).copy()
    group = p.groupby("source1_entity_id")["probability"]
    p["k"] = group.cumcount() + 1
    p["tp"] = group.cumsum()
    expected_true = group.transform("sum") / (1.0 - missing_rate)
    p["f_k"] = 1.25 * p["tp"] / (p["k"] + 0.25 * expected_true)

    log_none = np.log(np.clip(1.0 - p["probability"], 1e-12, 1.0)).groupby(p["source1_entity_id"]).transform("sum")
    f_zero = np.exp(log_none) * (1.0 - missing_rate)

    best_k = p.loc[p.groupby("source1_entity_id")["f_k"].idxmax(), ["source1_entity_id", "k", "f_k"]]
    best_k = best_k.set_index("source1_entity_id")
    best_k["f_zero"] = f_zero.groupby(p["source1_entity_id"]).first()
    best_k.loc[best_k["f_zero"] >= best_k["f_k"], "k"] = 0
    return p[p["k"] <= p["source1_entity_id"].map(best_k["k"])].drop(columns=["k", "tp", "f_k"])


def resolve_conflicts(matches):
    """Each S2/S3 record belongs to at most one S1 entity (true in the whole training ground truth).
    If several entities claim the same record, keep only the claim with the highest probability."""
    best = matches.groupby("candidate_id")["probability"].transform("max")
    kept = matches[matches["probability"] == best]
    return kept.drop_duplicates("candidate_id")


def to_submission(matches, column="matched_entity_ids"):
    return matches.groupby("source1_entity_id")["candidate_id"].agg(",".join).rename(column).reset_index()
