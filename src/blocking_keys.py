"""Blocking keys: short strings that two records must SHARE to become candidates.

All keys are built from normalized text (src/preprocessing.py) and prefixed with the country and the key type:

    nw  name word           "US|nw|colombier"
    aw  address word        "US|aw|wayne"
    nb  name bigram         "US|nb|williams_colombier"      (two consecutive words)
    ab  address bigram      "US|ab|85_wayne"
    nj  joined name         "US|nj|maurewilliamscolombier"  (legal words removed, no spaces -> matches website names)

Because the country is inside the key, records of different countries can never share a key,
and new countries (France) work without any change.
Everything is vectorized pandas (no per-row Python loops) so it runs on millions of records.
"""

import pandas as pd

from src.preprocessing import LEGAL_WORDS, replace_words

ALL_KEY_TYPES = ["nw", "aw", "nb", "ab", "nj"]
_REMOVE_LEGAL = {word: "" for word in LEGAL_WORDS}


def _word_rows(text):
    """One row per word; the index is the record's position."""
    words = text.reset_index(drop=True).str.split().explode()
    return words[words.notna() & (words != "")]


def _bigram_rows(text):
    words = _word_rows(text)
    following = words.groupby(level=0).shift(-1)
    keep = following.notna()
    return words[keep] + "_" + following[keep]


def _joined_rows(name):
    joined = replace_words(name.reset_index(drop=True), _REMOVE_LEGAL).str.replace(" ", "", regex=False)
    return joined[joined.str.len() >= 4]            # very short joined names are meaningless keys


def record_keys(names, addresses, countries, key_types=ALL_KEY_TYPES):
    """Long table with columns row (record position) and key."""
    makers = {
        "nw": lambda: _word_rows(names),
        "aw": lambda: _word_rows(addresses),
        "nb": lambda: _bigram_rows(names),
        "ab": lambda: _bigram_rows(addresses),
        "nj": lambda: _joined_rows(names),
    }
    countries = countries.reset_index(drop=True)
    parts = []
    for key_type in key_types:
        values = makers[key_type]()
        if values.empty:              # e.g. a batch where every address is empty
            continue
        keys = countries.reindex(values.index) + f"|{key_type}|" + values.astype("str")
        parts.append(pd.DataFrame({"row": values.index.to_numpy(), "key": keys.to_numpy()}))
    return pd.concat(parts, ignore_index=True)
