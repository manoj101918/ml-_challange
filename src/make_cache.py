"""Build the Parquet cache used by the experiments (run once from the project root):

    python -m src.make_cache

cache/{split}_pool.parquet     = Source 2 + Source 3 records (in that order), row groups of 1M rows
cache/{split}_source1.parquet  = Source 1 records
cache/{split}_pool_norm.parquet = pool with Phase 6 normalized name / address
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    cache = PROJECT_ROOT / "cache"
    cache.mkdir(exist_ok=True)
    for split in ["train", "test"]:
        folder = PROJECT_ROOT / "dataset" / split
        read = lambda name: pd.read_csv(folder / name, sep="\t", keep_default_na=False)
        pool = pd.concat([read(f"{split}_source2.tsv"), read(f"{split}_source3.tsv")], ignore_index=True)
        pool.to_parquet(cache / f"{split}_pool.parquet", row_group_size=1_000_000, index=False)
        del pool
        read(f"{split}_source1.tsv").to_parquet(cache / f"{split}_source1.parquet", index=False)
        print("cached", split)
        build_normalized_pool(split)



def build_normalized_pool(split):
    """cache/{split}_pool_norm.parquet: entity_id, country, name_norm, addr_norm (Phase 6 normalization)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.preprocessing import ADDRESS_STEPS, NAME_STEPS, normalize

    source = pq.ParquetFile(PROJECT_ROOT / "cache" / f"{split}_pool.parquet")
    writer = None
    for batch in source.iter_batches(batch_size=1_000_000):
        chunk = batch.to_pandas()
        out = pd.DataFrame({
            "entity_id": chunk["entity_id"],
            "country": chunk["country"],
            "name_norm": normalize(chunk["business_name"], NAME_STEPS),
            "addr_norm": normalize(chunk["business_address"], ADDRESS_STEPS),
        })
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(PROJECT_ROOT / "cache" / f"{split}_pool_norm.parquet", table.schema)
        writer.write_table(table, row_group_size=1_000_000)
    writer.close()
    print("normalized pool cached:", split)


if __name__ == "__main__":
    main()
