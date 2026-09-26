"""Second-stage ("collective") model — score-improvement round.

Idea: the true matches of one S1 entity are variants of EACH OTHER (same address style, same invented or Indian-script
name, same numbers). A candidate that looks weak against the S1 record can look strong against the entity's other
CONFIDENT matches. Stage 1 (the E4-type model) gives p1 for every candidate; stage 2 re-scores the candidates with
p1 >= MIN_P1 using p1, the entity context and the similarity to the entity's confident matches.

Input table (one row per candidate pair): source1_entity_id, candidate_id, p1, name_c, addr_c
(+ optional stage-1 feature columns listed in STAGE1_CONTEXT, used if present).
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

MIN_P1 = 0.01          # pairs below this are "no" without further work (they are ~0 in practice)
CONFIDENT = 0.90       # a candidate at or above this counts as a confident match of its entity
TOP_CONFIDENT = 3      # compare each candidate with at most this many confident matches

STAGE1_CONTEXT = ["name_token_set", "core_joined_ratio", "addr_token_set", "num_shared", "num_conflict",
                  "legal_conflict", "block_score_rel", "block_rank", "addr_c_empty", "name_c_foreign_script", "is_source3"]
CONTEXT_FEATURES = ["p1", "p1_rank", "p1_gap_to_best", "p1_best_other", "n_confident", "n_confident_same_source",
                    "n_confident_other_source", "n_above_half", "has_confident_other",
                    "conf_name_token_set", "conf_name_ratio", "conf_joined_ratio", "conf_addr_token_set",
                    "conf_addr_ratio", "conf_num_shared", "conf_addr_both_empty"]


def _pairwise(scorer, left, right):
    return process.cpdist(list(left), list(right), scorer=scorer, workers=-1) / 100.0


def stage2_features(pairs):
    """Adds CONTEXT_FEATURES to the pairs with p1 >= MIN_P1 and returns only those pairs."""
    f = pairs[pairs["p1"] >= MIN_P1].reset_index(drop=True).copy()
    f["source"] = f["candidate_id"].str[:2]
    group = f.groupby("source1_entity_id")["p1"]
    f["p1_rank"] = group.rank(method="first", ascending=False).astype("float32")
    f["p1_gap_to_best"] = (group.transform("max") - f["p1"]).astype("float32")
    f["n_above_half"] = group.transform(lambda p: (p >= 0.5).sum()).astype("float32")

    # best p1 among the OTHER candidates of the entity
    top2 = f.sort_values("p1", ascending=False).groupby("source1_entity_id")["p1"].agg(lambda p: list(p.iloc[:2]))
    best, second = top2.map(lambda l: l[0]), top2.map(lambda l: l[1] if len(l) > 1 else 0.0)
    b, s = f["source1_entity_id"].map(best), f["source1_entity_id"].map(second)
    f["p1_best_other"] = np.where(f["p1"] >= b, s, b).astype("float32")

    confident = f[f["p1"] >= CONFIDENT]
    f["n_confident"] = f["source1_entity_id"].map(confident.groupby("source1_entity_id").size()).fillna(0).astype("float32")
    per_source = confident.groupby(["source1_entity_id", "source"]).size()
    same = pd.Series(list(zip(f["source1_entity_id"], f["source"]))).map(per_source).fillna(0).to_numpy()
    self_conf = (f["p1"] >= CONFIDENT).to_numpy()
    f["n_confident_same_source"] = (same - self_conf).astype("float32")
    f["n_confident_other_source"] = (f["n_confident"] - same).astype("float32")
    f["has_confident_other"] = ((f["n_confident"] - self_conf) > 0).astype("int8")

    # similarity to the entity's (other) confident matches
    conf = (confident.sort_values("p1", ascending=False).groupby("source1_entity_id").head(TOP_CONFIDENT)
            [["source1_entity_id", "candidate_id", "name_c", "addr_c"]]
            .rename(columns={"candidate_id": "conf_id", "name_c": "conf_name", "addr_c": "conf_addr"}))
    cross = f[["source1_entity_id", "candidate_id", "name_c", "addr_c"]].reset_index().merge(conf, on="source1_entity_id")
    cross = cross[cross["candidate_id"] != cross["conf_id"]]
    sims = pd.DataFrame({"index": cross["index"].to_numpy()})
    if len(cross):
        sims["conf_name_token_set"] = _pairwise(fuzz.token_set_ratio, cross["name_c"], cross["conf_name"])
        sims["conf_name_ratio"] = _pairwise(fuzz.ratio, cross["name_c"], cross["conf_name"])
        sims["conf_joined_ratio"] = _pairwise(fuzz.ratio, cross["name_c"].str.replace(" ", "", regex=False),
                                              cross["conf_name"].str.replace(" ", "", regex=False))
        both = (cross["addr_c"] != "") & (cross["conf_addr"] != "")
        sims["conf_addr_token_set"] = np.where(both, _pairwise(fuzz.token_set_ratio, cross["addr_c"], cross["conf_addr"]), np.nan)
        sims["conf_addr_ratio"] = np.where(both, _pairwise(fuzz.ratio, cross["addr_c"], cross["conf_addr"]), np.nan)
        sims["conf_addr_both_empty"] = ((cross["addr_c"] == "") & (cross["conf_addr"] == "")).astype("float32").to_numpy()
        n1 = cross["addr_c"].str.findall(r"\d+").map(set)
        n2 = cross["conf_addr"].str.findall(r"\d+").map(set)
        sims["conf_num_shared"] = np.array([len(a & b) for a, b in zip(n1, n2)], dtype="float32")
        best_sims = sims.groupby("index").max()
    else:
        best_sims = pd.DataFrame(columns=["conf_name_token_set"])
    for col in ["conf_name_token_set", "conf_name_ratio", "conf_joined_ratio", "conf_addr_token_set",
                "conf_addr_ratio", "conf_num_shared", "conf_addr_both_empty"]:
        f[col] = f.index.map(best_sims[col]) if col in best_sims else np.nan
        f[col] = f[col].astype("float32")
    return f


def feature_columns(pairs):
    return CONTEXT_FEATURES + [c for c in STAGE1_CONTEXT if c in pairs.columns]
