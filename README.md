# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source 1 business, find all Source 2 / Source 3 records that describe the same
real-world business. The score is F0.5, which weights precision more than recall.

> Work in progress: this README is filled in phase by phase.

## Project layout

| Folder / file | Purpose |
|---|---|
| `dataset/train/` | Training sources 1–3 + `train_ground_truth.tsv` (labels). Never modified. |
| `dataset/test/` | Test sources 1–3 (no labels). Never modified. |
| `notebooks/` | Step-by-step exploration and experiments, numbered in order. |
| `src/` | Reusable Python code the notebooks import (loading, normalization, blocking, features, ...). |
| `output/` | Generated files: `matching_results.tsv`, `candidate_pairs.tsv`. |
| `requirements.txt` | Python packages needed to run everything. |

## Setup

```
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Progress

- [x] Phase 1: project and dataset inspection (`notebooks/01_data_exploration.ipynb`)
- [x] Phase 2: exploratory data analysis of names, addresses, country (`notebooks/02_eda.ipynb`)
- [x] Phase 3: ground truth, scoring metric (macro F0.5), evidence in true pairs, same-name traps (`notebooks/03_ground_truth.ipynb`)
- [x] Phase 4: train / validation split, local scorer, official-validator dry run (`notebooks/04_validation_split.ipynb`)
- [x] Phase 5: rule-based baseline — rare-word blocking + Jaccard + threshold, **small-validation F0.5 = 0.528** (`notebooks/05_baseline.ipynb`)
- [x] Phase 6: normalization experiments, each step tested alone + paired bootstrap, **F0.5 = 0.538** (`notebooks/06_normalization.ipynb`, `experiments/phase6_*.py`)
- [x] Phase 7: blocking with 5 key types (`src/blocking_keys.py`, `src/blocking.py`), candidate recall 0.59 → 0.94, **F0.5 = 0.706** (`notebooks/07_blocking.ipynb`, `experiments/phase7_*.py`)
- [x] Phase 8: 34 pair features (rapidfuzz, char TF-IDF, numbers, legal forms, blocking rank) for 2M candidate pairs (`src/features.py`, `notebooks/08_features.ipynb`)
- [x] Phase 9: logistic regression 0.889 / random forest 0.917 / **gradient boosting 0.934** validation F0.5 (`notebooks/09_model.ipynb`)
- [x] Phase 10: decision rules — global threshold 0.70 stays best; one-owner conflict rule kept (`src/decision.py`, `notebooks/10_decision.ipynb`)
- [x] Phase 11: error analysis on tuning entities — false merges / missed candidates / blocking misses, Indian-script names are the common cause (`notebooks/11_error_analysis.ipynb`)

## Caches

`cache/*.parquet` (gitignored) are Parquet copies of the TSV files that load ~20× faster.
Build them once with `python -m src.make_cache` before running the experiments.

## Validation setup

- `src/splits.py`: 80 / 20 split of train Source 1 **entities** (seed 42) + a fixed 20,000-entity "small validation" subset.
  Source 2 / 3 are not split: every entity searches the full pool of its country.
- `src/evaluation.py`: local copy of the official metric (macro F0.5 per S1 entity) and candidate recall.
- `src/submission.py`: writes the output TSVs (UTF-8, `\n` line endings).
- Experiments are logged in `EXPERIMENTS.md`.

## Checking a submission

```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Scoring (from the official problem statement)

F0.5 is computed **per Source 1 entity** and then averaged over all of them (macro average).
A singleton scores 1.0 if we predict nothing and 0.0 if we predict anything.
