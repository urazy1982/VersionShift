# VersionShift v2.1 — point-by-point response to the independent review of v2.0

**Date:** 2026-11-06 · **Verdict sought:** the review's "Conditional GO for v2.1". The full 14-release run is still NOT launched. The next step is Stage B (real-dataset cross-release pilot).

All changes below are implemented and tested. Tests: 23/23 pass (9 carried from v2.0, 14 new). Numbers come from files in `results/`.

## P0-1 — "Full pinning is not actually full pinning" · ACCEPTED, FIXED

`crossversion/worker.py` v2.1 no longer drops parameters. Policy **P** fails closed with a recorded portability status:

| Status | Trigger |
|---|---|
| `removed` | reference parameter does not exist in the target release |
| `unpinned_new` | target release has a parameter the reference lacks, so the pin would not be full |
| `ineffective` | placeholder value (`'warn'`, `'deprecated'`, `'auto_deprecated'`) transferred to a *different* release |
| `unsupported` | constructor or parameter validation rejects the value (now checked eagerly; at fit time on releases before 1.2) |
| `semantic_translation` | only in the separate **PT** arm, which applies documented translations from a JSON file; never pooled with P |

The job status `not_portable` is distinct from `fit_failed`. Each job also records constructor parameters and fitted effective attributes: `_gamma`, `n_iter_`, `n_estimators_`, `do_early_stopping_`, `_fit_method`, `effective_metric_`, coefficient shape and number of fitted estimators.

The fix exposed a real case on its first test. Inside 1.9.1, LR's own default `penalty='deprecated'` was flagged. Placeholders are now accepted only when the pin source and target are the same release. Six tests cover removed, new-parameter, placeholder, unsupported-value and translation cases.

## P0-2 — "The attribution decomposition is not yet causally justified" · ACCEPTED, REBUILT

`code/attribution.py` implements the reviewer's 2×2 exactly:

- A = old defaults on the old runtime; B = old defaults on the new runtime; C = new defaults on the old runtime; D = new defaults on the new runtime.
- Both ordered paths are kept.
- The symmetric Shapley shares are φ_runtime = ½[(B−A)+(D−C)] and φ_defaults = ½[(C−A)+(D−B)]. They sum to D−A exactly.
- The interaction I = D−B−C+A is always reported.
- Decomposition is on the signed outcome. Squared shifts are never split, because they are not additive.

Each unit gets an identifiability class: `full`, `runtime_first` (C not portable), `defaults_first` (B not portable) or `total_only`. Units in `total_only` get **no numerical share**. B and C are produced by strict P pins, so cells that fail closed become non-identifiable rather than approximated. Three tests check exact additivity, recovery of known effects and refusal to assign shares when cells are missing.

The v2.0 emulation is now described correctly. It runs both default sets in one runtime, so it identifies only the **B→D leg** (defaults, new runtime). Cells A and C need old binaries.

## P0-3 — "The statistical null is too narrowly calibrated" · ACCEPTED, NEW CALIBRATION

`code/calibrate_v21.py` generates fold-level scores from an explicit hierarchical model with a known truth. Sign flipping is no longer used. The model has paired splits, an algorithm × split interaction correlated across algorithms, seed noise only for RF/GB/HGB, heteroskedastic datasets, right-skewed noise, and RNG streams that are either shared or redrawn across versions.

There are 8 scenarios × K ∈ {12, 24, 48, 72} × 500 simulations, with truth taken from 40,000 simulated datasets (`results/CALIBRATION_V21.json`). Main findings, which were fixed before any real cross-release data:

| Finding | Evidence | Decision |
|---|---|---|
| Primary-estimand coverage | t interval 0.91–0.97 everywhere except sparse effects at small K (0.76 at K=12, 0.91 at K=24); percentile/BCa slightly worse (0.89 at K=12 under skew, 0.75 under sparse effects) | Use the dataset-cluster **t interval**; BCa as sensitivity |
| Size of the margin test (H0: E[M] ≤ 0.5 pp) | 0.4–2.0% just below the margin (truth 0.473); 6–11% just above it (truth 0.505) | Valid; mildly conservative below the margin |
| **Dilution of the primary estimand** | Sparse large effect (10% of datasets, +4 pp): truth M = 0.235 pp, so the mean-based test declares "equivalent" in 66–99.8% of runs | **Equivalence requires both** mean M below the margin **and** the Wilson upper bound of the prevalence P(M_k > margin) below 10%. Prevalence is promoted to a key secondary |
| Co-primary (decision-change excess) | Bootstrap test size 0–5.4% under N2/N3; Wilcoxon very conservative (0%, low power) | Bootstrap test + t interval; Wilcoxon is sensitivity only |
| Variance-only version change (N4) | Co-primary rejects 16–79% and S_k 11–61%, while M stays below the margin | Report as an instability effect, not a mean shift; S_k is **not** a mean-shift test and is demoted to diagnostic |
| Small K | Under sparse effects coverage is 0.71–0.76 at K=12 and 0.84–0.91 at K=24 | Stage B (12–20 datasets) is for heterogeneity estimation only; confirmatory K ≥ 48 |

## P1-4 — "Cross-version execution is not yet end-to-end complete" · ACCEPTED, FIXED

`controller.py` v2.1 adds the following:

- **Exclusion ledger.** Every task of the frame gets a status and reasons, and its hash is stored in the plan.
- **Wider eligibility.** Missing and categorical tasks are no longer excluded. Categoricals are one-hot encoded centrally and label-free. Missing values get train-fold median imputation in numpy inside the worker, so imputation cannot drift with the scikit-learn version. Large n is subsampled with stratification and a seed instead of excluded.
- **Resumable, idempotent runs.** `--resume` refuses if the RUN_LOCK differs, skips valid records, and quarantines partial or invalid files. Statuses are appended to `JOB_STATUS.jsonl`.
- **Completeness checks.** `collect` verifies every planned job and fold, and checks each prediction length against the frozen test size.
- **Replay.** `replay` independently recomputes every score from prediction bytes.
- **Introspection.** `introspect` dumps each release's resolved defaults and their adjacent-edge diffs. This gives an objective default-change detector for the oracle.

`code/analysis_v21.py` is the missing adapter: the same estimators serve both emulation regimes and real releases. An **end-to-end test** runs freeze → run → collect → replay → final inferential table. A/A gives exactly 0 pp and is declared equivalent. An injected +3 pp RF effect is recovered exactly (2.000 pp over 3 pairs).

The eligible dataset count is frozen by `freeze` before any power planning, as requested. It can only be produced on the Ubuntu machine with OpenML access.

## P1-5 — "Rerun noise is conflated with data-split variation" · ACCEPTED, MEASURED

New Experiment A2 (`code/run_noise_decomposition.py`, 44 datasets × 4 model seeds on fixed splits). Median rerun standard deviation of the 5-fold mean, in pp:

| Estimator | Fixed-split seed SD | Split-to-split SD | Seed share of variance |
|---|---|---|---|
| LR, SVC, kNN | 0 | 0.50–0.81 | 0 (deterministic) |
| HGB (n < 10k, no early stopping) | 0 | 0.69 | 0 |
| GB | 0.11 | 0.76 | 0.02 |
| RF, 100 trees | 0.37 | 0.18 | 0.51 |
| RF, 10 trees (0.21 default) | 0.90 | 0.33 | 0.77 |

The reviewer was right: for five of six estimators the v2.0 "rerun noise" was almost entirely split variation. The primary estimand is now the paired version-minus-version contrast on identical splits and seeds, in pp, with no noise denominator. VNR is kept only as a diagnostic, with its denominator labelled "split+seed".

## Section 4 — research questions · ACCEPTED

The study now has four RQs as proposed: impact, attribution under compatibility conditions, changelog prediction, and mitigation + accumulation. The old RQ5 is folded into RQ4.

## Section 5 — statistical hierarchy · ACCEPTED, with one amendment from the calibration

| Priority | Outcome | Analysis |
|---|---|---|
| Primary | M = mean over pairs of |Δδ| (pp), exactly paired | Dataset-cluster t interval; H0: E[M] ≤ margin; Holm across the pre-declared primary edge family |
| Co-primary | Decision-change excess over rerun baseline | Dataset-cluster bootstrap test + t interval; BH across secondary edges |
| Key secondary (new) | Prevalence P(M_k > margin) | Wilson interval; required for any equivalence claim |
| Secondary | Signed shift, VNR, S_k, prediction disagreement, Kendall τ | Descriptive or diagnostic |

The corrected repeated-CV t-test defines only the per-dataset decision. It is never used for population inference, as the reviewer noted.

## Section 6 — staged design and oracle · ACCEPTED

- **Stage A (mechanism validation).** Implemented here: emulation, negative controls, strict-pin tests, and A/A plus injected-effect end-to-end checks.
- **Stage B (real-dataset pilot).** 12–20 eligible CC18 tasks on 0.21.3→0.22.2 (positive edge) and 1.8.0→1.9.1 (negative edge), with D, P and PT. It estimates heterogeneity and portability rates, then freezes K.
- **Stage C (confirmatory run).** Takes place after OSF registration of the plan hash, the oracle-frame hash, the margin, the edge family and the analysis code hash.
- **Stage D (external validation).** A pre-specified set of 5–8 real published notebooks or comparison scripts. Execution failures are reported separately from conclusion changes.

The oracle is now a complete frame (`code/oracle_frame.py`): **78 estimator × edge opportunities** (13 edges × 6 estimators), including explicit no-change opportunities. Each has a coding status. Evaluation **refuses** to run unless every opportunity is `coded_complete`. The draft frame has 12 complete, 18 partial and 48 not yet coded. Default changes are cross-checked objectively by `introspect`.

The project is single-author, so a second human coder is not available. Instead: blinded delayed recoding of a 25% sample at least 14 days later (intra-rater κ), plus the automatic default-diff check. This is reported as a limitation.

## Section 7 — reproducibility · ACCEPTED

- `build_envs.sh` v2.1 compiles a lock with wheel hashes, installs with `--require-hashes`, and fails closed (renames the env `*.FAILED`) on any version mismatch, including the runtime scikit-learn version.
- Each env records OS, CPU (`lscpu`), uv version, interpreter compiler and threadpool/BLAS information.
- The worker records platform, threadpool/BLAS libraries and a numpy build-config hash per job.

## Section 2 / novelty wording · ACCEPTED

Verified: the Future Work paragraph of Montandon et al. states the intention to measure the magnitude of impact of behaviour-based DABCs. No follow-up was found in a search on 2026-11-06. The claim is now scoped: VersionShift **extends** that line from static exposure to controlled, counterfactual measurement of conclusion change. It does not claim to be first. This also creates a competitive-timing risk, which strengthens the case for running Stage B soon.

**Correction to my v2.0 document:** I cited [EMSE 2024, s10664-024-10563-4] as behavioural breaking-change work. It is "An extended study of syntactic breaking changes in the wild". It is now cited as such.

## Title · ACCEPTED

*From Exposure to Impact: Measuring and Explaining Conclusion Changes Across Machine-Learning Dependency Upgrades.*

## v2.1 pilot re-analysis (emulation; same 44 datasets; pilot only)

| Contrast | K | Primary M, pp (t 95% CI) | Holm p vs 0.5 pp | Co-primary E (t 95% CI) | Prevalence M_k > 0.5 (Wilson) |
|---|---|---|---|---|---|
| 0.21 → 1.9.1 defaults, raw features | 44 | 4.42 (3.13–5.71) | 0.0004 | 0.119 (0.078–0.160) | 0.82 (0.68–0.91) |
| same, standardised features | 44 | 1.47 (1.05–1.89) | 0.0004 | 0.078 (0.052–0.104) | 0.73 (0.58–0.84) |
| Negative control (kNN, GB, HGB pairs) | 44 | 0.00 (0–0) | — (equivalent) | 0.000 | 0.00 (0–0.08) |

The Holm p of 0.0004 is the bootstrap floor (B = 4,999 resamples).

## Open item for the author

Is the 0.5-pp margin right? The calibration shows its consequences, but the threshold is a substantive choice. It should be justified before registration, for example from the smallest difference a typical comparison paper reports as an improvement.

---

# Round 2 (2026-11-06) — decisions adopted in v2.2

| Reviewer decision | Implementation |
|---|---|
| Keep 0.5 pp as a practical-effect threshold; 0.25 / 1.0 pp sensitivity | `decision_v22.MARGINS`; every edge is reported at all three margins |
| Report signed shift, absolute shift and prevalence; equivalence needs a signed CI within ±0.5 and prevalence UB < 10%; "uniformly small" also needs the M upper bound < margin | `decide()` labels EQUIVALENT_BALANCED and UNIFORMLY_SMALL exactly so |
| Separate "mean comparison changed" from "substantial fraction changed" | Q-MEAN vs Q-PREV, with separate labels CHANGED_MEAN and WIDESPREAD |
| 29 datasets needed for zero-event UB < 10% | Verified: 10.1% at K=28, 9.8% at K=29 (`min_k_zero_events()`, unit-tested) |
| Attempt 0.21.3, fallback 0.22.2, never silently replaced | Amendment A-EO; `VS_PY37`; the fallback is written to BUILD_FAILURES.txt; `--emulation-only` labels the edge. Verified: PyPI has the cp37 manylinux1 x86-64 wheel (uploaded 2019-07-30); uv supports CPython 3.7 at Tier 2 |
| Code all 78 before Stage B outcomes; blinded 25% recode at 14 days or later; report raw agreement + κ; call it intra-rater | Amendment A-CO; `freeze` / `recode-sample` / `agreement`; analysis blinding gate (A-BL) |
| Selective missingness from fail-closed P | Amendment A-SM; `analysis_v22.py mitigation` reports coverage, a selection check, conditional mitigation and failure-to-transfer |
| Prioritise the real 0.21.3→0.22.2 edge | `STAGE_B_PROTOCOL.md` with go/no-go gates G1–G6 |

**One correction to the round-2 wording.** A "mean signed effect" pooled across algorithm pairs has no meaning, because pair orientation is arbitrary. Direction is therefore tested per pair (TOST), and equivalence requires all pairs (intersection–union test, so no multiplicity penalty). A unit test shows the labels do not change when pair orientations are flipped.

**One sizing clarification.** A ±0.25-pp precision target on a positive edge would need more than 1,000 datasets when per-dataset effects vary by about 4 pp, and it is not needed there. Stage C's K is therefore sized on the negative-control edge, where equivalence is claimed, with floors of 48 (coverage) and 29 (prevalence).

**On "cannot certify the 23 tests".** The v2.2 package contains all code, results JSON and the 34-test suite (`python3 -I -m unittest discover -s tests`), so you can re-run them yourself.

---

# Round 3 (2026-11-06) — v2.2 audit, corrections in v2.2.1

All six implementation findings were confirmed in the v2.2 code before fixing. In particular, `pivot_table(aggfunc='first')` silently dropped duplicate fold rows. Each finding now has a regression test (`tests/test_v221.py`, 20 tests). v2.2 files are kept unchanged as the record. The v2.2.1 analysis is `decision_v221.py` and `analysis_v221.py`, frozen by hash in `ANALYSIS_FREEZE_V221.json`.

| Finding | Status | Regression test |
|---|---|---|
| P0 Holm missing | Fixed: Holm per question family across primary edges; BH for secondary edges; labels only after adjustment | `HolmTests` (known p-values; raw p = 0.04 is not CHANGED_MEAN once Holm gives 0.06; CLI end to end) |
| P0 partial folds | Fixed: exact frozen fold set, unique identities, valid scores | `StrictFoldTests` (duplicate, missing, unexpected and reduced plans all rejected) |
| P1 oracle key set | Fixed: exact 78 keys + provenance | `OracleCompletenessTests` |
| P1 Clopper–Pearson population | Fixed: three explicit population modes | `PopulationTests` |
| P1 nanmean weights | Fixed: complete-case primary + coverage + sensitivity | `MissingnessTests` |
| P1 sizing | Fixed: K now also covers positive-edge power and prevalence | `SizingTests` |
| π_max | Adopted, **at 1.0 pp rather than 0.5 pp** (pre-data noise check, `results/RULE_CHECK_V221_PIMAX.json`) | `OneStrongPairTests` |
| runtime-stack label; unpooled populations | Done. Writing that test exposed a real bug in my first implementation: pair units must take the more severe class of their two algorithms | `MitigationPopulationTests` |

**On π_max at 1.0 pp.** As proposed, the maximum over 15 pairs at 0.5 pp would declare ANY_PAIR_WIDESPREAD with no version effect at all. Under the realistic-noise null, 12.9% of datasets exceed 0.5 pp on some pair, already above the 10% limit. The 1.0-pp threshold keeps the intended sensitivity (a single +4 pp pair is detected in 100% of datasets) at null rates of 1.8–6.0%. π_max at 0.5 pp is still reported, descriptively.

**Test runtime.** The full suite (54 tests) takes about 2.5 minutes on 2 CPUs. Most of that is three end-to-end tests that spawn subprocess runs. `VS_FAST_TESTS=1` skips those three and runs the other 51 in about 14 seconds.

---

# Round 4 (2026-11-06) — v2.2.2 audit received and verified; v2.2.3 amendment

- **Accepted as the reference.** v2.2.2 verified on receipt: manifest 62/62 matched; exactly the four declared files changed; all three gaps reproduced on v2.2.1 and closed in v2.2.2.
- **Tests now verified end to end.** The full suite passed 60/60 in 148 s, which the auditor could not run.
- **Decisions adopted.** The scientific decisions are adopted unchanged, including the caution that the Kish effective-n bounds are approximations.
- **Oracle chronology.** Adopted. A hash alone cannot prove when the coding happened, so the oracle freeze must also be timestamped externally (an OSF registration or a dated public commit). Any unblinded exploratory inspection must be reported.
- **v2.2.3 fixes three remaining gaps of the same kind** (see `docs/AMENDMENT_V223.md`):
  - absent P/PT jobs silently dropped by `mitigation`;
  - no plan-coverage check in `attribution`;
  - v2.2.2's all-or-nothing D check contradicting the 5% tolerance in protocol G2. Resolved with a ledgered dataset-level exclusion and a hard 5% limit.
- **Totals.** 66 tests (6 new). No design change.


## Round 5 (v2.2.5, 2026-11-06): CPU-handoff review, literature refresh, hybrid validation

- **v2.2.4 verified.**
  - All 69/69 manifest hashes match.
  - Only `analysis_v221.py`, `STAGE_B_PROTOCOL.md` and `test_v223.py` changed, and `test_v224.py` was added.
  - The fast and full suites pass.
  - The job-level G2 matches the protocol text.
  - Remaining caveat: G2 can pass while excluding datasets. A Stage C retention floor (≥ 90%) is recommended but left to the author (`docs/AMENDMENT_V225.md` §7).
- **Literature.** The gap still stands. sklearn-migrator (JOSS 2026) addresses portability of fitted models, not retraining or conclusions. seqbench (2026) already provides anytime-valid algorithm comparison, so lens D is not claimed as statistical novelty. SonarSource S6973 makes a testable mitigation arm possible. See `docs/LITERATURE_UPDATE_V225.md`.
- **Hybrid validation (secondary, pre-registered, calibrated):**
  - lens B: REML, HKSJ and prediction intervals;
  - lens C: Cauchy combination with BH, which detects the single-algorithm 1.2 pp scenario the primary misses, with 96% power; its FDR is approximate, about 0.11–0.12;
  - lens D: an anytime-valid e-process across Stage B → C; valid but conservative.
  - A Cauchy masking bug was found by calibration and fixed.
- **New arm.** Policy S (S6973 as written) was added to Stage B.
- **Emulation result (E-MIT).** D 4.42 pp → S6973 (after rewrite) 1.55 pp → E 0.07 pp → P 0 pp. Standardised features: S6973 = D (1.47 pp). As written, S6973 code fails on 1.9.1 (`max_features='auto'`).
- **Unchanged.** The primary design, gates and frozen analysis files.
- **Totals.** 13 new tests.
