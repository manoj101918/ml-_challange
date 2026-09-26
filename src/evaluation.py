"""Scoring exactly like the official leaderboard: macro-averaged F0.5 per Source 1 entity.

Both `predictions` and `truth` are DataFrames in the submission format:
    source1_entity_id | matched_entity_ids   ("S2-1,S3-7" or "" for no match)
"""

import pandas as pd


def to_id_sets(id_strings):
    """'S2-1,S3-7' -> {'S2-1', 'S3-7'};  '' or missing -> empty set."""
    return id_strings.fillna("").map(lambda text: set(text.split(",")) if text else set())


def entity_f05(true_ids, predicted_ids):
    """F0.5 for ONE Source 1 entity (rules from the official problem statement)."""
    if not true_ids and not predicted_ids:
        return 1.0  # correctly predicted "no match"
    correct = len(true_ids & predicted_ids)
    if correct == 0:
        return 0.0
    precision = correct / len(predicted_ids)
    recall = correct / len(true_ids)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def score_per_entity(predictions, truth):
    """One row per Source 1 entity in `truth`, with precision / recall / F0.5.

    Entities missing from `predictions` are treated as "predicted no match".
    """
    table = truth[["source1_entity_id", "matched_entity_ids"]].merge(
        predictions[["source1_entity_id", "matched_entity_ids"]],
        on="source1_entity_id",
        how="left",
        suffixes=("_true", "_pred"),
    )
    true_sets = to_id_sets(table["matched_entity_ids_true"])
    pred_sets = to_id_sets(table["matched_entity_ids_pred"])

    table["n_true"] = true_sets.map(len)
    table["n_pred"] = pred_sets.map(len)
    table["n_correct"] = [len(t & p) for t, p in zip(true_sets, pred_sets)]
    table["f05"] = [entity_f05(t, p) for t, p in zip(true_sets, pred_sets)]
    return table


def macro_f05(predictions, truth):
    """The leaderboard number: average F0.5 over all Source 1 entities in `truth`."""
    return score_per_entity(predictions, truth)["f05"].mean()


def score_report(predictions, truth):
    """Macro F0.5 plus a few numbers that help explain it."""
    table = score_per_entity(predictions, truth)
    singletons = table["n_true"] == 0
    total_pred = table["n_pred"].sum()
    total_true = table["n_true"].sum()
    return pd.Series({
        "macro_F0.5": table["f05"].mean(),
        "F0.5 on entities with matches": table.loc[~singletons, "f05"].mean(),
        "F0.5 on singletons": table.loc[singletons, "f05"].mean(),
        "pair precision": table["n_correct"].sum() / total_pred if total_pred else float("nan"),
        "pair recall": table["n_correct"].sum() / total_true if total_true else float("nan"),
        "predicted pairs": total_pred,
        "true pairs": total_true,
    })


def candidate_recall(candidates, truth):
    """Share of true (S1, match) pairs that survive blocking.

    `candidates` uses the candidate_pairs format (column candidate_entity_ids).
    This is the ceiling for recall: a match that is not a candidate can never be predicted.
    """
    renamed = candidates.rename(columns={"candidate_entity_ids": "matched_entity_ids"})
    table = score_per_entity(renamed, truth)
    return table["n_correct"].sum() / table["n_true"].sum()
