"""Phase 7 diagnosis (tuning entities only): which blocking keys could find each true pair?

For every true (S1, match) pair we list the keys both records share and the document frequency (df) of the
rarest one. A pair is only reachable by a blocking rule that allows keys at least that common.
Output: experiments/phase7_diagnosis.tsv
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking_keys import KEY_TYPES, record_keys
from src.preprocessing import ADDRESS_STEPS, NAME_STEPS, normalize
from src.rule_experiment import CACHE_DIR, load_query_sets

start = time.time()
s1, truth = load_query_sets()
s1 = s1[s1["query_set"] == "tuning"].reset_index(drop=True)
s1["name_norm"] = normalize(s1["business_name"], NAME_STEPS)
s1["addr_norm"] = normalize(s1["business_address"], ADDRESS_STEPS)

# true pairs of the tuning entities, with both normalized texts
pairs = truth["tuning"].assign(match_id=lambda t: t["matched_entity_ids"].str.split(",")).explode("match_id")
pairs = pairs[pairs["match_id"].notna() & (pairs["match_id"] != "")][["source1_entity_id", "match_id"]]
pool = pd.read_parquet(CACHE_DIR / "train_pool_norm.parquet", filters=[("entity_id", "in", list(pairs["match_id"]))])
matched = pool.set_index("entity_id").loc[pairs["match_id"]].reset_index()
del pool

types = list(KEY_TYPES)
s1_keys = record_keys(s1["name_norm"], s1["addr_norm"], s1["country"], types)
match_keys = record_keys(matched["name_norm"], matched["addr_norm"], matched["country"], types)
vocabulary = pd.Index(pd.concat([s1_keys["key"], match_keys["key"]]).unique())
print(f"query + match keys: {len(vocabulary):,} ({time.time() - start:.0f}s)", flush=True)

# document frequency of every vocabulary key over the full pool (counts only)
df = np.zeros(len(vocabulary), dtype=np.int64)
for batch in pq.ParquetFile(CACHE_DIR / "train_pool_norm.parquet").iter_batches(batch_size=500_000):
    chunk = batch.to_pandas()
    keys = record_keys(chunk["name_norm"], chunk["addr_norm"], chunk["country"], types)
    codes = vocabulary.get_indexer(keys["key"])
    codes = codes[codes >= 0]
    np.add.at(df, codes, 1)
print(f"df counted ({time.time() - start:.0f}s)", flush=True)

# shared keys per true pair
s1_row = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
pairs = pairs.reset_index(drop=True)
pairs["s1_row"] = s1_row[pairs["source1_entity_id"]].to_numpy()
pairs["m_row"] = np.arange(len(pairs))
sk = s1_keys.rename(columns={"row": "s1_row"})
mk = match_keys.rename(columns={"row": "m_row"})
shared = pairs[["s1_row", "m_row"]].merge(sk, on="s1_row").merge(mk, on=["m_row", "key"])
shared["df"] = df[vocabulary.get_indexer(shared["key"])]
shared["type"] = shared["key"].str.split("|").str[1]

result = pairs[["source1_entity_id", "match_id", "m_row"]].copy()
result["match_source"] = result["match_id"].str[:2]
result["country"] = s1["country"].to_numpy()[pairs["s1_row"]]
for key_type in types:
    best = shared[shared["type"] == key_type].groupby("m_row")["df"].min()
    result[f"min_df_{key_type}"] = result["m_row"].map(best)
result["name_indian_script"] = matched["name_norm"].str.contains("[ऀ-ൿ]").to_numpy()
result["addr_empty"] = (matched["addr_norm"] == "").to_numpy()
result.drop(columns="m_row").to_csv(Path(__file__).resolve().parent / "phase7_diagnosis.tsv", sep="\t", index=False, lineterminator="\n")
print(f"saved {len(result):,} true pairs ({time.time() - start:.0f}s)")
