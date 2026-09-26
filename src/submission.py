"""Writing the two output files in the exact format the scorer expects."""


def write_id_list_file(table, path, id_column):
    """Write `source1_entity_id <TAB> <id_column>` as UTF-8 with Unix line endings.

    id_column is "matched_entity_ids" (matching_results.tsv)
    or "candidate_entity_ids" (candidate_pairs.tsv).
    lineterminator="\\n": on Windows pandas would otherwise write "\\r\\n", and a
    strict scorer could read the last ID of each row as "S2-123\\r".
    """
    table[["source1_entity_id", id_column]].to_csv(
        path, sep="\t", index=False, encoding="utf-8", lineterminator="\n"
    )
