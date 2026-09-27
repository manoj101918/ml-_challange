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
from rapidfuzz.distance import Levenshtein

from src.reverse import REVERSE_COLUMNS

MIN_P1 = 0.01          # pairs below this are "no" without further work (they are ~0 in practice)
CONFIDENT = 0.90       # a candidate at or above this counts as a confident match of its entity
TOP_CONFIDENT = 3      # compare each candidate with at most this many confident matches

STAGE1_CONTEXT = ["name_token_set", "core_joined_ratio", "addr_token_set", "num_shared", "num_conflict",
                  "legal_conflict", "block_score_rel", "block_rank", "addr_c_empty", "name_c_foreign_script", "is_source3"
                  ] + REVERSE_COLUMNS                    # used only when present (models trained with reverse=True)
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
    return CONTEXT_FEATURES + [c for c in STAGE1_CONTEXT + NOISE_FEATURES if c in pairs.columns]


# ---------------------------------------------------------------- "noise model" features (E8)
# Error analysis (experiments/decision_errors.py): true copies get NOISE (typos -> rare tokens, a house number with one
# digit changed / dropped: 1314 -> 1315, 1655 -> 655), look-alike distractors get SUBSTITUTIONS (a real, common word
# replaced by another real word: "kelly cleaning" -> "moeller cleaning"). These features measure which kind of edit
# separates the two records. Computed only on the stage-2 pairs (~4.5 per entity), so the word loops stay cheap.

NOISE_FEATURES = ["num_close_shared", "num_close_only", "num_conflict_close",
                  "s1_unmatched", "rec_unmatched", "s1_unmatched_freq", "rec_unmatched_freq", "min_word_ratio"]


def _close(x, y):
    """Two house/plot numbers a noisy copy can turn into each other: equal, or one digit substituted/added/dropped."""
    return x == y or (max(len(x), len(y)) >= 2 and abs(len(x) - len(y)) <= 1 and Levenshtein.distance(x, y) <= 1)


def word_frequencies(split, name_steps, address_steps, log=print):
    """How many S1 names of the split contain each core word (unlabeled data only; cached)."""
    from src.features import LEGAL_SET
    from src.pipeline import CACHE_DIR, normalization_tag, prepare_queries

    path = CACHE_DIR / f"{split}_s1_words_{normalization_tag(name_steps, address_steps)}.parquet"
    if not path.exists():
        s1 = prepare_queries(pd.read_parquet(CACHE_DIR / f"{split}_source1.parquet", columns=["entity_id", "business_name",
                             "business_address", "country"]), name_steps, address_steps)
        words = s1["name_norm"].str.split().map(lambda ws: [w for w in set(ws) if w not in LEGAL_SET]).explode().dropna()
        counts = words.value_counts()
        pd.DataFrame({"word": counts.index.astype(str), "count": counts.to_numpy()}).to_parquet(path, index=False)
        log(f"  word frequencies of {split} S1 names cached: {len(counts):,} words")
    table = pd.read_parquet(path)
    return dict(zip(table["word"], table["count"]))


def noise_features(f, word_freq):
    """Adds NOISE_FEATURES; f needs name_s1, addr_s1, name_c, addr_c (normalized texts)."""
    from src.features import LEGAL_SET

    close_shared, close_only, conflict = [], [], []
    for a, b in zip(f["addr_s1"].str.findall(r"\d+"), f["addr_c"].str.findall(r"\d+")):
        sa, sb = set(a), set(b)
        n_close = sum(1 for y in sb if y in sa or any(_close(x, y) for x in sa))
        close_shared.append(n_close)
        close_only.append(n_close - len(sa & sb))
        conflict.append(bool(sa) and bool(sb) and n_close == 0)
    f["num_close_shared"] = np.array(close_shared, dtype="float32")
    f["num_close_only"] = np.array(close_only, dtype="float32")
    f["num_conflict_close"] = np.array(conflict, dtype="int8")

    rows = []
    for n1, n2 in zip(f["name_s1"].str.split(), f["name_c"].str.split()):
        w1 = [w for w in dict.fromkeys(n1) if w not in LEGAL_SET]
        w2 = [w for w in dict.fromkeys(n2) if w not in LEGAL_SET]
        best1 = [max((fuzz.ratio(w, v) for v in w2), default=0) for w in w1]
        best2 = [max((fuzz.ratio(w, v) for v in w1), default=0) for w in w2]
        un1 = [w for w, s in zip(w1, best1) if s < 80]
        un2 = [w for w, s in zip(w2, best2) if s < 80]
        rows.append((len(un1), len(un2),
                     max((np.log1p(word_freq.get(w, 0)) for w in un1), default=np.nan),
                     max((np.log1p(word_freq.get(w, 0)) for w in un2), default=np.nan),
                     min(best1 + best2, default=np.nan) / 100.0))
    values = np.array(rows, dtype="float32").reshape(-1, 5)
    for i, col in enumerate(["s1_unmatched", "rec_unmatched", "s1_unmatched_freq", "rec_unmatched_freq", "min_word_ratio"]):
        f[col] = values[:, i]
    return f
