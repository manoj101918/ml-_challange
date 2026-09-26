"""Train / validation split of the Source 1 training entities.

We split Source 1 ENTITIES (never individual pairs), so all matches of one entity
stay together — exactly like the test set, where each S1 entity is scored as a whole.
Source 2 / Source 3 are NOT split: every S1 entity searches the full S2/S3 pool.
"""

import pandas as pd

VALIDATION_FRACTION = 0.20
SMALL_VALIDATION_SIZE = 20_000
SEED = 42


def make_validation_split(ground_truth):
    """Return one row per S1 entity with columns: source1_entity_id, split, small_validation.

    split             "train" or "validation"
    small_validation  True for a fixed 20,000-entity subset of validation, used for fast experiments
    """
    # sort first so the result does not depend on the row order of the file
    all_ids = ground_truth["source1_entity_id"].sort_values().reset_index(drop=True)

    validation_ids = all_ids.sample(frac=VALIDATION_FRACTION, random_state=SEED)
    small_validation_ids = validation_ids.sample(n=SMALL_VALIDATION_SIZE, random_state=SEED)

    split = pd.DataFrame({"source1_entity_id": all_ids})
    split["split"] = "train"
    split.loc[split["source1_entity_id"].isin(validation_ids), "split"] = "validation"
    split["small_validation"] = split["source1_entity_id"].isin(small_validation_ids)
    return split
