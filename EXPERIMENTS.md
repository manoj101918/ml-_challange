# Experiment log

Every experiment is scored with `src/evaluation.py` (official macro F0.5) on the **small validation set**
(20,000 S1 entities, split from `src/splits.py`, seed 42). Occasionally confirmed on the full validation set (441,364 entities).
Change one thing at a time.

| # | Date | Blocking | Candidate recall | Features | Model | Threshold | Small-val F0.5 | Notes |
|---|---|---|---|---|---|---|---|---|
| 15 | 2026-09-26 | E4 on the **test set** | — | — | E4 | 0.725 | **LB 0.935** | 172,769,341 candidate pairs, 5,380,195 predicted pairs; validator PASS; distribution of matches per entity ≈ validation (experiments/phase14_checks.txt) |
| 14 | 2026-09-26 | #3 with K = 100 | 0.952 | #10 **without the 2 TF-IDF features** (3.3x faster) | HGB, 40k training entities (4.0M pairs) | 0.725 | **0.9421** | phase12_run.py E4 = candidate final config, model saved to cache/models/E4.joblib; +0.0039 vs E1 (CI [0.0028, 0.0050]); vs E2/E3 +0.0005 (CI incl. 0) |
| 13 | 2026-09-26 | #3 | 0.940 | #6 without name_tfidf / addr_tfidf | HGB, 20k | 0.70 | 0.9330 | phase12_ablation.py: −0.0011 vs #6 (0.9341) — TF-IDF is 64% of feature time, so dropped for speed |
| 12 | 2026-09-26 | #3 with **K = 100** | **0.952** | #10 | HGB, 20k training entities (4.0M pairs) | 0.725 | **0.9415** | phase12_run.py E3; +0.0033 vs E1, 95% CI [+0.0023, +0.0044]; twice the pairs to score; not yet combined with #11 |
| 11 | 2026-09-26 | #3 | 0.940 | #10 | HGB, **60k training entities** (3.0M pairs) | 0.70 | **0.9416** | phase12_run.py E2; +0.0034 vs E1, 95% CI [+0.0024, +0.0044]; singletons 0.934; P 0.983 / R 0.881 |
| 10 | 2026-09-26 | #3 | 0.940 | #6 + **transliteration** of Indian scripts (anyascii) in names and addresses | HGB, 20k training entities | 0.725 | **0.9382** | phase12_run.py E1; +0.0041 vs E0, paired-bootstrap 95% CI [+0.0030, +0.0054]; blocking recall unchanged, gain from features; singletons 0.929 |
| 9 | 2026-09-26 | #3 | 0.940 | #6 | HGB via new src/pipeline.py | 0.70 | 0.9341 | phase12_run.py E0: reproduces #7 exactly |
| 8 | 2026-09-26 | #3 | 0.940 | #6 | #6 | expected-F0.5 top-k per entity | 0.9331 | notebook 10; −0.001 vs #6 (CI incl. 0); worse on singletons → rejected |
| 7 | 2026-09-26 | #3 | 0.940 | #6 | #6 | global 0.70 + one-owner rule (resolve_conflicts) | 0.9341 | notebook 10; only 18 conflicting records at this sample density; kept |
| 6 | 2026-09-25 | #3 | 0.940 | 34 features (Phase 8) | **HistGradientBoosting** (400 iters, 63 leaves), trained on 20k tuning entities | 0.70 (3-fold OOF on tuning) | **0.9340** | notebook 09; P 0.979 / R 0.871; singletons 0.913; OOF tuning 0.932 |
| 5 | 2026-09-25 | #3 | 0.940 | 34 features (Phase 8) | random forest (100 trees) | 0.55 | 0.9168 | notebook 09 |
| 4 | 2026-09-25 | #3 | 0.940 | 34 features (Phase 8) | logistic regression | 0.65 | 0.8886 | notebook 09 |
| 3 | 2026-09-25 | 5 key types (name/address words, name/address word pairs, joined name), df ≤ 1000, top 50 by idf sum | **0.940** | #2 | rule | 0.60 | **0.7058** | notebook 07; blocking comparison in experiments/phase7_blocking_results.tsv; 0 entities without candidates |
| 2 | 2026-09-25 | same as #1 | 0.590 | #1 + normalization: names trade_name+invisible+accents+web; addresses invisible+accents+address_abbrev+null+zeros | rule | 0.55 | **0.5378** | notebook 06; +0.010 vs #1, paired-bootstrap 95% CI [+0.0065, +0.0134]; single-step table in experiments/phase6_results.tsv |
| 1 | 2026-09-25 | share a rare word (df ≤ 1000, same country), top 50 by idf sum | 0.588 | name Jaccard + address Jaccard (mean), basic cleaning | rule: score ≥ t | 0.45 (tuned on 20k train entities) | **0.5278** | notebook 05; P 0.738 / R 0.506; 4,922/40k entities got no candidates |
| 0 | 2026-09-25 | none | 0 | — | predict nothing | — | 0.0575 | the floor; = share of singletons |
