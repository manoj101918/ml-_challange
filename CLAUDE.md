# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

Read this first. It is the hand-over from the previous machine (8 GB laptop) to continue the work.

## How the user wants to work
- The user is an **ML beginner** (knows Python / pandas / NumPy basics). Act as a **patient mentor**: explain every new
  concept simply (concept → why → tiny example → how it applies here → code → what to observe → common mistakes).
- Work in **phases**; after each phase give a summary (what we did, what was learned, what the outputs mean, problems found,
  what comes next) and **wait for "CONTINUE"** before the next phase. The user sometimes says "go ahead / continue" to proceed.
- Notebooks: **plain, visible pandas** code with markdown between cells (the user asked for this). Reusable logic lives in `src/`.
- Commit / push **only when the user asks**. End commit messages with the Co-Authored-By line from the system prompt.
- Change **one thing at a time**, measure it, log it in `EXPERIMENTS.md`. Choose settings on *tuning* (train) entities, confirm on validation.

## The task (official problem statement: `*_problem_statement.pdf`, kept locally, not in git)
- 3 sources of business records (`entity_id, business_name, business_address, country`). Source 1 is deduplicated;
  for every S1 entity find all matching S2/S3 records (0, 1 or many).
- **Metric: macro F0.5 per S1 entity**, averaged over all S1 entities incl. singletons: singleton + empty prediction = 1.0,
  singleton + any prediction = 0.0, non-singleton + empty prediction = 0.0. Precision counts double.
- Outputs (TSV, UTF-8, **`\n` line endings**): `output/matching_results.tsv` (`source1_entity_id, matched_entity_ids`) and
  `output/candidate_pairs.tsv` (`source1_entity_id, candidate_entity_ids`) = the exact candidate set the final model scores;
  matches ⊆ candidates; one row per test S1 entity; no duplicate IDs; only S2/S3 IDs.
- Validator: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
- Final zip: `output/` (2 TSVs), `code/business_entity_resolution/{src/, README.md, requirements.txt}`, filled `Documentation_template.md`.
- Rules: **no external data / APIs / geocoding / web lookups**. Country is an open set (test adds **France**, train has only US + India) —
  never hard-code or one-hot countries. Final model MIT/Apache-2.0-compatible, ≤ 8B parameters (ours: scikit-learn HGB).

## Files that are NOT in git (copy them to the new machine manually)
- `dataset/train/*.tsv`, `dataset/test/*.tsv` (≈ 2.5 GB, competition data)
- `utils/validate_submission.py` and `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (organizer materials, gitignored)
- `cache/` is regenerable (see setup); `output/` is regenerable.

## Setup on a new machine
```
pip install -r requirements.txt          # pandas 3.0.3, numpy, scikit-learn 1.9, pyarrow, rapidfuzz 3.14.6, anyascii 0.3.3 (Python 3.14 was used)
python -m src.make_cache                 # Parquet copies of the TSVs + normalized pool (Phase 6 steps); ~5 min
```
`src/pipeline.normalized_pool()` builds its own cache per normalization config (`cache/{split}_pool_norm_<hash>.parquet`) on first use.

## Data facts that shape the design
- Train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603; ground truth one row per S1 (comma-separated IDs).
  Test: S1 1,732,544 (India 810k, US 663k, France 259k) / S2 4,887,273 / S3 5,082,316.
- 5.6% of S1 are singletons; typical 2–5 matches, max 11. **Every S2/S3 ID belongs to at most one S1.** 26% of S2/S3 match nothing.
- **All true pairs share the same country.** No postal codes. Only 10.8% of true pairs have the same lowercased name;
  ~93% of same-name look-alikes are different businesses → the address is the strongest evidence.
- S2/S3 noise: CAPS, fake accents (`Límited`), `[brackets]`, junk prefixes, `null` in addresses, abbreviations (`st`/`street`),
  state name vs code, reordered address parts, website names (`x.com`), S3 trade names (`X dba Y`, `née`, `formerly known as`),
  **names in 9 Indian scripts** (20–28% of Indian S2/S3 names) — transliterations of the English name.

## Pipeline (current best) and code map
```
normalize (src/preprocessing.py: NAME_STEPS / ADDRESS_STEPS + "translit")
 -> blocking (src/blocking_keys.py keys nw/aw/nb/ab/nj; src/blocking.py: df <= 1000, rank by idf sum, top K=50)
 -> 34 pair features (src/features.py: rapidfuzz, char 3-gram TF-IDF, numbers, legal forms, blocking rank)
 -> HistGradientBoostingClassifier (max_iter=400, lr=0.1, max_leaf_nodes=63, min_samples_leaf=50, l2=1.0)
 -> decision (src/decision.py): global threshold (~0.70, chosen on out-of-fold tuning predictions) + resolve_conflicts (one owner per S2/S3)
```
- `src/evaluation.py` = local copy of the official metric; `src/splits.py` = 80/20 split of train S1 entities (seed 42) + fixed 20k
  "small validation"; S2/S3 are never split (every entity searches the full pool of its country).
- `src/pipeline.py` = reusable end-to-end pieces; `experiments/phase12_run.py E0..E3` runs one full experiment per config.
- `src/submission.py` writes the output TSVs; `src/rule_experiment.py` = Phase 5–7 rule-based pipeline (historical).
- Notebooks `notebooks/01…11` document Phases 1–11 with outputs.

## Results so far (small validation, 20k entities; full log in EXPERIMENTS.md)
| step | F0.5 |
|---|---|
| predict nothing | 0.058 |
| rule baseline (rare-word blocking + Jaccard) | 0.528 |
| + normalization | 0.538 |
| + 5-key blocking (candidate recall 0.59 → 0.94) | 0.706 |
| + gradient boosting on 34 features | 0.934 |
| + transliteration of Indian scripts (E1) | 0.938 |
| + 60k training entities instead of 20k (E2) | **0.9416** |
| E1 + K=100 candidates instead of 50 (E3; candidate recall 0.940 → 0.952) | **0.9415** |
| **E4 = final config**: translit + K=100 + 40k training entities + fast features (no TF-IDF) | **0.9421** |

Rejected (measured, hurt or no gain): legal-word canonicalization/removal, dot-joining, relative/expected-F0.5 decision rules.

## Known pitfalls (learned the hard way)
- pandas 3 string methods `.str.replace/.contains/.match` use RE2: `\w`/`\b` are **ASCII-only**. Use `[^\p{L}\p{N}\p{M}\s]`.
  `.str.findall` falls back to Python `re` (no `\p{}` there).
- Case-insensitive `[À-ɏ]` also matches every `s` (ſ folds to s).
- Write TSVs with `lineterminator="\n"` (Windows default is `\r\n`). Subprocess output on Windows needs `PYTHONIOENCODING=utf-8`.
- Never merge big tables on text columns: use integer codes / 64-bit hashes, chunk the pool, batch the queries.
- One experiment per fresh Python process (memory fragmentation). On the 8 GB laptop only ~1 GB was free → heavy swapping.
  On a bigger machine raise `chunk_rows` in `src/pipeline.run_blocking_and_features` (250k → 1M) and `chunk_entities` (10k → 50k),
  and experiments can run in parallel.

## Status at 2026-09-26 (continued on the same laptop)
- **Final model = E4**, saved in `cache/models/E4.joblib` (model, threshold 0.725, feature list, normalization steps, K=100, max_df=1000).
  TF-IDF features dropped: −0.001 F0.5 but 3.3× faster features (`experiments/phase12_ablation.tsv`); `src.features.FAST_FEATURE_COLUMNS`.
- **`src/predict.py`** = scalable prediction (one country at a time, hashed keys, streamed candidates, candidate file written
  incrementally — ~2 GB on test at K=100). Verified: `python -m src.predict --model cache/models/E4.joblib --split small_validation`
  reproduces E4 exactly (0.9421, candidate recall 0.9522).
- Training code moved to **`src/train.py`** (`python -m src.train E4`); `experiments/phase12_run.py` is a thin wrapper.
- `src/blocking.iter_candidates` uses a sorted-offset (CSR) lookup instead of a pandas merge per batch — 22× faster on the
  test set (~52 s per 10k entities on the laptop, ~2.5 h for the whole test set); verified identical to the merge version.
- `cache/models/E4.joblib` is NOT in git (cache/ is ignored): copy it to a new machine or retrain with `python -m src.train E4`
  (needs the same scikit-learn version, 1.9.0).
- Submission packaging: `submission_README.md` (becomes code/business_entity_resolution/README.md), filled `Documentation_template.md`
  (team name/members and the test candidate count `[TEST_CANDIDATES]` still to fill), `python -m src.package_submission --team NAME`.
- **Test run**: `python -m src.predict --model cache/models/E4.joblib --split test` → `output/matching_results.tsv` +
  `output/candidate_pairs.tsv` (≈ 3.5–4.5 h on the laptop). Then run the official validator.
- Optional: `--split validation` (full 441k validation entities) to measure the one-owner rule at realistic density.

## Next steps (Phase 12 → 15)
1. E2 (more training data) and E3 (K=100) each gave a significant +0.003 over E1 (paired bootstrap CI above 0).
   Next experiment: **E4 = both together** (translit + 60k+ training entities + K=100); expected ≈ 0.944–0.945.
   K=100 doubles the pairs to score on the test set (~173M), so the feature speed-up (step 2) matters even more.
2. **Speed up features**: E2 needed ~13 min per 4M pairs → the test set (~87M pairs at K=50) would take hours.
   Profile `src/features.add_features`; vectorize the Python loops (jaccard, legal forms, numbers) and/or parallelize chunks;
   optional cheap pre-filter stage (then `candidate_pairs.tsv` = the filtered set the final model scores).
3. **Scalable test run** (Phase 13): `src/blocking.build_index_hashed` + `iter_candidates` (already written, NOT yet tested) —
   process one country at a time, hashed keys, stream candidates in batches of ~10k entities, compute features, predict,
   keep only probabilities; `n_pool` = size of the whole pool so idf matches training.
4. **Final model**: train on more training entities (≥ 60k, more if memory allows) with the final config; save model + threshold.
5. **Full-density check**: run the whole pipeline on the full validation split (441k entities) to measure the one-owner rule
   and confirm the score at scale.
6. **Test prediction** → write both TSVs → run the official validator until PASS (Phase 14).
7. **Documentation** (Phase 15): README (reproduction steps), fill `Documentation_template.md` with measured results only,
   build the submission zip structure.
