# VersionShift v2.2.1 — Final study design (frozen for Stage B; audit corrections applied)

**Title:** *From Exposure to Impact: Measuring and Explaining Conclusion Changes Across Machine-Learning Dependency Upgrades*
**Target venue:** Information and Software Technology (alternative: Journal of Systems and Software)
**Status, 2026-11-06:** design frozen for Stage B. The confirmatory Stage C has not started, and the protocol below must be registered before it does. Implementation readiness is established (34/34 tests); scientific validity across real historical binaries is not yet established.

## 1. Research questions

- **RQ1 Impact.** How often, and by how much, do scikit-learn upgrades change predictions, metrics and pairwise algorithm comparisons?
- **RQ2 Attribution.** Under which compatibility conditions can changes be attributed to default configurations, implementation behaviour or numeric dependencies?
- **RQ3 Prediction.** How reliably do changelog entries and release metadata predict the observed changes?
- **RQ4 Mitigation and accumulation.** How well do configuration pinning and environment locking prevent drift, across single upgrades and cumulatively?

## 2. Estimand and decision rule (decision_v22.py)

The unit is dataset k × ordered pair p=(A,B) × edge e, with orientation fixed by the frozen algorithm order LR, SVC, kNN, RF, GB, HGB:

Δ_kpe = (s_A,new − s_B,new) − (s_A,old − s_B,old), in pp, on identical frozen splits and seeds.

Two separate questions are answered:

| Question | Quantity | Inference |
|---|---|---|
| Q-MEAN direction | mean signed shift per pair, μ_p | 90% dataset-cluster t CI per pair (TOST) |
| Q-MEAN magnitude | M = mean over datasets of M_k = mean_p \|Δ_kp\| | one-sided 95% t bounds; Holm across the primary edge family |
| Q-PREV | π = share of datasets with M_k > m | exact Clopper–Pearson one-sided 95% bounds |

Labels per edge (m = 0.5 pp primary; 0.25 and 1.0 pp sensitivity):

| Label | Condition |
|---|---|
| CHANGED_MEAN | lower bound of M > m |
| WIDESPREAD | lower bound of π > 10% |
| EQUIVALENT_BALANCED | every pair's 90% CI for μ_p inside ±m (intersection–union test) **and** upper bound of π < 10% |
| UNIFORMLY_SMALL | EQUIVALENT_BALANCED **and** upper bound of M < m |
| INCONCLUSIVE | otherwise |

The 0.5-pp threshold is a practical-effect threshold, not the definition of stability.

**Correction to the round-2 review.** A mean signed effect pooled over pairs has no meaning, because A−B versus B−A orientation is arbitrary. Direction is therefore tested per pair, and equivalence requires all pairs to pass. As an intersection–union test this needs no multiplicity penalty.

**Consequences already visible before data.** With zero affected datasets, the prevalence bound drops below 10% only at K ≥ 29: the bound is 10.1% at K=28 and 9.8% at K=29. Stage B (12–20 datasets) can therefore never yield an equivalence label.

Rule check on the v2.1 known-truth generator (300 sims/cell, `results/RULE_CHECK_V22.json`):

- Pure null: UNIFORMLY_SMALL 100% at K ≥ 29; INCONCLUSIVE at K=16.
- Sparse +4 pp effect on 10% of datasets: equivalence 1–3%, previously 66–100% under v2.1's mean-only rule.
- Dense effect just below the margin (M=0.47 pp, π=0.40): WIDESPREAD 86–100%. This is the mean-versus-fraction distinction the reviewer asked for.
- Effect at the margin boundary: CHANGED_MEAN 5–7%.

## 3. Factors

- **Releases:** 14, from 0.21.3 to 1.9.1.
- **Primary edge family:** 0.21.3→0.22.2.post1 (positive mechanism edge) and 1.8.0→1.9.1 (negative-control edge), plus the coded-change edges added before Stage C.
- **Policies:**
  - D: shipped defaults.
  - E: archived partial explicit map.
  - P: strict full pin, fail-closed.
  - PT: documented translations only, reported separately.
- **Splits:** frozen centrally; native splitters only as a secondary arm.
- **Datasets:** OpenML-CC18 frame with an exclusion ledger. Categoricals are one-hot encoded centrally; NaNs get train-fold median imputation in numpy; n is subsampled to 5,000.

## 4. Attribution (attribution.py; analysis_v22.py attribution)

The four cells are:

- A = old defaults on the old runtime
- B = old defaults on the new runtime (P pinned from old)
- C = new defaults on the old runtime (P pinned from new, run in the old env)
- D = new defaults on the new runtime

The analysis reports the two ordered paths, the symmetric Shapley shares and the interaction D−B−C+A. Each unit gets an identifiability class (`full`, `runtime_first`, `defaults_first`, `total_only`); units without all needed cells get no numerical share. **Emulation fidelity** is measured by the A→B leg (runtime only), and by agreement between the 1.9.1 emulation (`emulate_plan.py`) and the observed edge on the same datasets.

## 5. Protocol amendments adopted in round 2

- **A-SM (selective missingness).** Strict P runs only for portable configurations. The analysis reports:
  - failure to transfer by estimator, as an outcome in its own right;
  - compatibility coverage (share of pair units with both algorithms portable);
  - a selection check comparing the total effect on all units with the total effect on compatible units;
  - mitigation estimated only on compatible units and labelled as such.
- **A-BL (blinding gate).** Every analysis that reveals cross-release outcomes requires a frozen, complete 78-case oracle frame whose hash is recorded. Otherwise it runs only with `--unblinded-exploratory`, which is written into the report header.
- **A-EO (earliest release).** Attempt 0.21.3 on CPython 3.7: the cp37 manylinux1 wheel exists, and uv supports 3.7 at Tier 2; `VS_PY37` can point at a system Python 3.7. If it cannot run reproducibly, the 0.21→0.22 transition is analysed **only** as emulation, labelled `emulation_only`, and the real-binary series starts at 0.22.2.post1. That transition is never described as observed cross-version evidence.
- **A-CO (changelog coding).** All 78 estimator × edge opportunities must be coded before any Stage B outcome is seen. Each records:
  - predicted impact category
  - trigger condition
  - applicable data types
  - estimated compatibility
  - release-note reference
  - coding uncertainty
  - coder and date

  `freeze` refuses incomplete or draft-coded frames; the 30 pre-filled rows are drafts that the author must confirm. A seeded 25% sample is then recoded with labels concealed, no earlier than 14 days after freezing (`recode-sample`, `agreement`). Raw agreement and Cohen's κ are reported with adjudicated disagreements, and described as **intra-rater** reliability, not independent inter-rater validation.

## 6. Stages

- **A — Mechanism validation:** done.
- **B — Historical binary pilot:** next; see `STAGE_B_PROTOCOL.md`.
- **C — Confirmatory run:** after registration.
- **D — External validation:** 5–8 pre-specified real notebooks or comparison scripts, with execution failures reported separately from conclusion changes.

## 7. Known limitations (declared)

- Single author: intra-rater coding reliability only.
- One ecosystem (scikit-learn) and one task type (tabular classification).
- CC18 may yield fewer than 48 eligible tasks without re-using Stage B tasks. `stage_b.py size` then applies the pre-stated rule.
- Era-matched stacks confound scikit-learn with numpy/scipy. The fixed-stack control edge (1.4.2→1.5.2) addresses this only partly.

## 8. v2.2.1 audit corrections (2026-11-06)

| Audit item | Correction | Code |
|---|---|---|
| P0 Holm not implemented | One-sided p-values per edge (t test vs margin; exact binomial vs 10%); Holm within each question family across primary edges, BH across secondary edges; labels assigned only after adjustment | `decision_v221.label_edges` |
| P0 Partial folds accepted | `validate()` requires exactly the frozen R × F folds per job, unique identities and valid scores; `pivot` refuses duplicates | `analysis_v221.validate` |
| P1 Oracle key set | `freeze` requires exactly the 78 expected keys once each, plus a build-provenance file matching releases, estimators and the changelog hash | `oracle_frame.freeze` |
| P1 Prevalence population | `superpopulation` (CP, assumes exchangeable tasks), `frame_census` (finite-frame proportion, no sampling CI) and `stratified` (Stage B weights, Kish effective n) | `decision_v221.prevalence` |
| P1 nanmean pair weights | Primary = complete-case datasets; coverage and per-pair availability reported; available-case and missingness bounds as sensitivity | `decision_v221.decide` |
| P1 Sizing from negative edge only | K = max(negative precision, positive power 90%, positive prevalence, floors 48 / 29) | `analysis_v221 size` |
| Section 2 one strong pair | π_max added as a pre-specified sensitivity flag at **1.0 pp**. The pre-data check showed 0.5 pp fires on rerun noise (12.9% of datasets under realistic noise, 35% under variance-only change); at 1.0 pp the null rates are 1.8% and 6.0%, and one +4 pp pair is detected in 100% of datasets (average-pair measure: 1.3%) | `rule_check_v221.py` |
| Section 3 runtime label | Component renamed `runtime_stack` (scikit-learn + numpy/scipy/BLAS/OpenMP + interpreter) | `attribution.py` |
| Section 3 populations | Strictly portable, translation-required and non-transferable pair units are classified by the more severe algorithm and never pooled; mitigation estimated separately for P and PT | `analysis_v221 mitigation` |
| Section 4 four envs first | `STAGE=B bash build_envs.sh` | `build_envs.sh` |
