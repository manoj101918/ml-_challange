# Business Entity Resolution — reproduction guide

Pipeline: normalization (incl. transliteration of Indian scripts) → multi-key blocking (top 100 candidates per Source 1 entity)
→ 32 similarity features → `HistGradientBoostingClassifier` → global threshold + "one owner per S2/S3 record" rule.
Validation macro F0.5 (20,000 held-out training entities): **0.9421**. Methodology: `Documentation_template.md`.

## 1. Environment
Python 3.14 (3.11+ should work). Install the pinned packages:
```
pip install -r requirements.txt
```
All libraries are open source (pandas, NumPy, scikit-learn, pyarrow, rapidfuzz — MIT, anyascii — ISC). No internet access,
external data or APIs are used at any point.

## 2. Data layout (run everything from this folder)
```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## 3. Steps
```
python -m src.make_cache                                   # Parquet copies of the TSV files (+ normalized pools)      ~10 min
python -m src.train E4                                     # blocking + features + model on 40k training entities    ~40 min
python -m src.predict --model cache/models/E4.joblib --split test                                              # ~4 h on 8 GB
```
Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv` (the exact candidate set the model scored).

Optional checks:
```
python -m src.predict --model cache/models/E4.joblib --split small_validation   # reproduces the 0.9421 validation score
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## 4. Code map (`src/`)
| file | role |
|---|---|
| `make_cache.py` | TSV → Parquet caches |
| `preprocessing.py` | text normalization steps (`NAME_STEPS`, `ADDRESS_STEPS`, `transliterate`) |
| `blocking_keys.py` | the five blocking-key types (name/address words, word pairs, joined name), country-prefixed |
| `blocking.py` | rare-key index, idf ranking, top-K candidates; hashed + streaming version for the full test set |
| `features.py` | pair features (rapidfuzz, numbers, legal forms, blocking rank) |
| `pipeline.py` | normalized-pool cache, candidate + feature generation for training |
| `train.py` | training experiments E0–E4; E4 = final model (`cache/models/E4.joblib`) |
| `predict.py` | final prediction per country, writes both output files |
| `decision.py` | threshold and one-owner rule |
| `evaluation.py` | local implementation of the official macro F0.5 |
| `splits.py` | 80/20 split of training Source 1 entities (seed 42) + 20k validation sample |
| `submission.py`, `rule_experiment.py` | output writer; historical rule-based baseline used by early experiments |

Determinism: all random choices use fixed seeds (split seed 42, sampling seeds 7 / 11 / 0, model `random_state=0`).
