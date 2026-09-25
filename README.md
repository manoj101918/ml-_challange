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
