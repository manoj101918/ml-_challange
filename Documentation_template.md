# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TechCrafters  
**Team Members:** Manoj Kumar Pabbineedi, Pilla Prudhvi Lakshman, Marpu Giri Prasad, Tandasa Mourya  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
A classic, fully offline entity-resolution pipeline: **text normalization** (including transliteration of Indian scripts to Latin
letters) → **multi-key blocking** on rare word, word-pair and joined-name keys → **32 hand-designed similarity features** →
a **gradient-boosted tree classifier** (scikit-learn `HistGradientBoostingClassifier`) → a **precision-oriented decision rule**
(one global probability threshold chosen on out-of-fold training predictions, plus a "one owner per record" constraint).
On our held-out validation entities it reaches **macro F0.5 = 0.942** (pair precision 0.983, pair recall 0.881).

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from our exploratory analysis of the training data (all numbers measured on the training files):
- **Matches are many-to-one clusters.** 5.6% of Source 1 entities are singletons; most have 2–5 matches (max 11).
  No Source 2/3 record belongs to more than one Source 1 entity; 26% of S2/S3 records match no S1 entity at all (distractors).
- **Every true pair has the same country label** (7.6M of 7.6M), so all comparisons are restricted to the same country —
  implemented generically (the country string is part of every blocking key), so the unseen test country **France** needs no special code.
- **Names are unreliable on their own.** Only 10.8% of true pairs have the same lowercased name; conversely ~93% of S2/S3 records
  with exactly the same lowercased name as an S1 entity are *different* businesses (generic names such as "Family Center").
  Addresses are the strongest evidence: 95.4% of true pairs share an address word, 79.9% share a house/plot number.
- **Noise patterns in S2/S3:** random upper/lower case; fake accents on random letters (`Límited`); junk prefixes (`***`, `>>`, `#`);
  bracketed words; `null` / `<NULL>` inside addresses; abbreviations (`St`/`Street`, `Rd`, `R.`/`Rue`); state written as code vs full name;
  reordered address components; website-style names (`name.com`); trade-name patterns in Source 3 (`X dba Y`, `X née Y`,
  `X formerly known as Y`); landmark-based Indian addresses; **names in nine Indian scripts** (20–28% of Indian S2/S3 names),
  which are transliterations of the English name. No postal codes are present.

### 2.2 Solution Strategy
**Approach Type:** Blocking + pairwise classifier + decision rule (hybrid of rules and ML).  
**Core Innovation:** (1) combining five complementary blocking-key types with idf-based ranking, which raised candidate recall
from 0.59 (single words) to 0.95 at 100 candidates per entity; (2) transliterating Indian-script text before blocking and
feature computation; (3) a memory-efficient implementation (integer / 64-bit hashed keys, per-country processing, streaming)
that runs the full test set on an 8 GB laptop.

Validation protocol: Source 1 training entities were split 80/20 (seed 42); Source 2/3 were never split, so every entity searches
the full same-country pool with realistic distractors. All design choices (normalization steps, blocking parameters, model,
threshold) were selected on training entities (out-of-fold where needed) and measured on a fixed 20,000-entity validation sample
using a local re-implementation of the official macro F0.5. Improvements were accepted only if a paired bootstrap 95% confidence
interval of the per-entity score difference lay above zero.

---

## 3. Candidate Generation (Blocking)
- **Normalization first** (each step kept only if it improved validation F0.5): lowercase; Indian-script → Latin transliteration
  (`anyascii`, ISC licence, offline character table); keep the real name after trade-name markers (`dba`, `née`, `formerly known as`, …);
  remove zero-width characters and Latin accents; strip website decorations (`www.`, `.com`); symbols → spaces;
  addresses: expand street abbreviations, drop `null`, strip leading zeros from numbers.
  Canonicalizing or removing legal-form words (`pvt`→`private`, deleting `llc`/`ltd`) was tested and **rejected** (it increased false merges).
- **Blocking keys** (all prefixed with the country): name words, address words, name word pairs, address word pairs
  (e.g. `85_wayne`, rare even when both words are common), and the joined name without legal words (matches website names).
- **Rare keys only:** a key must appear in ≤ 1,000 pool records. Each candidate is scored by the sum of idf = log(N/df) over the keys it
  shares with the S1 entity, and the **top 100** per entity are kept. No entity is left without candidates.
- **Candidate pairs generated:** 100 per S1 entity (≈ 99.7 per entity on validation); on the test set
  **172,769,341 candidate pairs** for 1,732,544 S1 entities (99.7 per entity; only 4 entities without candidates),
  out of 6.72 × 10¹² possible same-country pairs → **reduction ratio 99.9974%**.
- **How we ensured true matches were not lost:** a diagnosis of every true pair on training entities showed that 98.4% share at least one
  key with df ≤ 1,000 and that the old single-word blocking lost most matches in the *ranking* step (look-alikes crowding them out);
  adding word-pair and joined-name keys and ranking by idf sum fixed this. Candidate recall on validation: **0.952** (K = 100) vs 0.940 (K = 50)
  vs 0.59 for the first word-based baseline.

---

## 4. Matching Model

**Features used (32 per pair):**
- Name features: exact match, word Jaccard, rapidfuzz `ratio` / `token_sort_ratio` / `token_set_ratio`, the same on the name without legal
  words (`ratio`, `token_set`, `partial_ratio`), joined-name ratio (spaces removed, for website names), length difference, and a flag for
  candidate names written in an Indian script.
- Legal-form features: same legal form / conflicting legal forms (e.g. `llc` vs `ltd`) / legal form missing on one side.
- Address features: candidate address empty, word Jaccard, rapidfuzz `ratio` / `token_set_ratio` / `token_sort_ratio` (missing when the
  candidate has no address), house/plot numbers shared, number Jaccard, number conflict, all candidate numbers contained in the S1 address.
- Other: blocking idf score (total / name keys / address keys), rank of the candidate among the entity's candidates, score relative to the
  entity's best candidate, name/address similarity relative to the entity's best candidate, number of candidates, candidate from Source 3.
- Character 3-gram TF-IDF cosine features were evaluated and dropped: they cost 64% of feature time for only −0.001 F0.5.

**Model type:** scikit-learn `HistGradientBoostingClassifier` (400 iterations, learning rate 0.1, 63 leaves, min 50 samples per leaf,
L2 = 1.0), trained on the ~4.0M candidate pairs of 40,000 training entities (positives = true matches; negatives = all other candidates,
i.e. hard negatives that already passed blocking). Logistic regression (0.889) and random forest (0.917) were weaker on the same features.  
**Threshold selection method:** macro-F0.5 maximisation over thresholds 0.50–0.90 on 3-fold out-of-fold predictions of the training entities
(folds grouped by S1 entity) → **0.725**. Predicted pairs are then filtered so that each S2/S3 record is assigned to at most one S1 entity
(the one with the highest probability), mirroring a property of the training ground truth. Per-entity "expected-F0.5" and relative
thresholds were tested and did not beat the global threshold. The model's probabilities are well calibrated (checked by probability buckets).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9421** on the 20,000-entity validation sample (pair precision 0.9825, pair recall 0.8814,
  singletons 0.930, entities with matches 0.943). Out-of-fold score on training entities: 0.9416.

  | step | validation macro F0.5 |
  |---|---|
  | predict nothing | 0.058 |
  | rule baseline (rare-word blocking + Jaccard) | 0.528 |
  | + normalization | 0.538 |
  | + multi-key blocking (candidate recall 0.59 → 0.94) | 0.706 |
  | + gradient boosting on similarity features | 0.934 |
  | + transliteration of Indian scripts | 0.938 |
  | + 100 candidates, more training data, faster features (final) | **0.942** |

- **Test-set predictions (no labels, so no score locally):** 5,380,195 predicted pairs; 6.5% of test entities predicted
  "no match"; 3.11 matches per entity on average — almost identical to the same model on validation (6.6%, 3.09),
  and **France, unseen in training, behaves like the other countries** (6.1% no match, 3.14 matches per entity vs 6.1% / 3.18 for US).
  Official validator: PASS (including `--check-ids` for the matching file).
- **Common false positives (wrong merges):** a different business at the *same address* (often with an Indian-script or invented brand
  name); records with the same generic name and no address; near-duplicate names and addresses that the ground truth treats as distinct
  businesses; pairs whose house numbers or legal forms conflict but whose other evidence is strong.
- **Common false negatives (missed matches):** true matches never reached by blocking (≈ 5% of true pairs; mostly Indian-script names with
  partial addresses, very generic names without address); candidates without an address where only a generic name is available;
  invented trade names at the same address (indistinguishable from a different business at that address); borderline scores between 0.3 and 0.7.

---

## 6. Conclusion
Careful error analysis mattered more than model complexity: most gains came from measuring *where* matches were lost (blocking ranking,
Indian-script names) and fixing those causes, each verified with a paired significance test on held-out entities. The final pipeline
uses only the provided data, small open-source libraries (pandas, NumPy, scikit-learn, rapidfuzz, anyascii) and a tree model of a few MB,
and runs end to end on an 8 GB laptop.

---

## Appendix

### A. Code Artefacts
- `src/preprocessing.py` — normalization steps (`NAME_STEPS`, `ADDRESS_STEPS`, transliteration)
- `src/blocking_keys.py`, `src/blocking.py` — blocking keys, idf ranking, scalable hashed/streaming candidate generation
- `src/features.py` — pair features; `src/decision.py` — threshold and one-owner rule; `src/evaluation.py` — local macro F0.5
- `src/pipeline.py`, `src/train.py` (`python -m src.train E4`) — trains the final model (`cache/models/E4.joblib`)
- `src/predict.py` — entry point that writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
  `python -m src.make_cache` → `python -m src.train E4` → `python -m src.predict --model cache/models/E4.joblib --split test`
- `notebooks/01–11` document every phase with outputs; `EXPERIMENTS.md` logs every experiment.

### B. Additional Results
Full experiment log with dates, settings and confidence intervals: `EXPERIMENTS.md`; blocking design comparison:
`experiments/phase7_blocking_results.tsv`; normalization ablation: `experiments/phase6_results.tsv`; feature ablation:
`experiments/phase12_ablation.tsv`; Phase 12 experiments: `experiments/phase12_results.tsv`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
