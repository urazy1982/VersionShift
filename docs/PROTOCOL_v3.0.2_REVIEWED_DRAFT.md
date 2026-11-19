# When Library Defaults Change Scientific Conclusions
## VersionShift v3.0.2 — reviewed prospective extension protocol (DRAFT)

**Date:** 2026-11-16. **Implementation correction v3.0.3:** Original-edge fixed-stack intervention NOT supplied by v2.6 phase3; needs a separate run. **Status:** PROPOSED / NOT FROZEN / NO NEW v3 COMPARATIVE RESULTS.  This document supersedes conflicting clauses of `source_docs/PROTOCOL_v3.0.1_DRAFT.md`. The authoritative v2.6 code, historical results, original task frame and post-review E2 map remain read-only.

### Scientific question and scope
Can documented scikit-learn default/implementation changes alter balanced-accuracy comparisons and benchmark winners, and when do common effective configurations or fold-local preprocessing prevent those effects? **Scope is scikit-learn classification, not software ecosystems generally.** The two new edges were selected from version documentation; they **cannot** estimate prevalence across all upgrades. Longitudinal E5 is descriptive history of one library.

### Prior evidence must not be reclassified
The v2.6 paper reports 25/63 winner changes on its early edge, an M estimate of 5.0067 pp, and post-review/outcome-informed E2 M of 0.4921 pp. All are **prior observational/descriptive evidence**, not newly preregistered v3 findings. G0 must recover and verify them from the actual v2.6 archive and paper. Differences among v2.2 pilot, v2.6 full-frame and v3 results must always specify dataset frame, fold choices, preprocessing, paired estimators and policies.

### Research questions
- **RQ1, newly selected upgrades:** On 1.4.2→1.5.2 and 1.5.2→1.6.1, do shipped defaults change eight-estimator pairwise comparisons and winners under train-fold-only P1 preprocessing? Check practical materiality, ties, and independent external-frame replication.
- **RQ2, mechanisms/identification:** For LinearSVC and AdaBoost, compare observable effective policies on both runtimes under a common numerical stack; present genuinely missing AdaBoost SAMME.R on 1.6 as `REMOVED_BEHAVIOR`. For original 0.21→0.22, use a truly common numerical stack to isolate defaults from the wider runtime stack only where matched cells exist.
- **RQ3, mitigation:** Test the unchanged v2.6 six-estimator post-review E2 map **out of sample with respect to new upgrade edges**. Add separately frozen LinearSVC and AdaBoost entries. Report compatibility coverage and paired D0–D1 improvement; do not claim E2 is a prospectively designed intervention in the original v2.6 work.
- **RQ4, preprocessing:** P1 (fold-local, unscaled) vs P2 (fold-local, numeric standardized) on both new edges; also rerun original 0.21→0.22 on P1/P2, compared with archival transductive P0, retaining original folds and row keys.
- **RQ5, secondary E5:** All buildable consecutive historical scikit-learn edges, shipped defaults, six original estimators, same 48 allocated v3 evaluation tasks, P1. Report every edge and failure code, descriptive only.

### Exact new version edges, effective behavior
1. **E_LINEAR_DUAL:** scikit-learn 1.4.2→1.5.2. LinearSVC effective default dual=True (literal constructor placeholder `'warn'` in 1.4.2) → `dual='auto'`; the auto choice is determined by training-sample count, post-encoding feature count, penalty/loss/multi_class support. Do not assert universal `dual=False`: compute and record the resolved value **per fold**; label the source-supported optimization mechanism. Convergence and solver/numeric drift must be separated. For 1.4, compare explicit dual=True to dual='auto' in both runtimes where supported, and distinguish default-policy change from implementation change.
2. **E_ADA_SAMME:** scikit-learn 1.5.2→1.6.1. Old AdaBoost shipped behavior uses SAMME.R; new release implements only SAMME, with signature default `algorithm='deprecated'`. Old SAMME.R vs old SAMME, old SAMME vs new SAMME are the three identifiable cells. **Never create a new-release SAMME.R cell** or impute a zero shift. D1 in 1.6 omits `algorithm`, and this is behavior harmonization, not restoration.
3. **E_ORIGINAL:** scikit-learn 0.21.3→0.22.2.post1. Historical v2.6 analysis is immutable. RQ2 fixed-stack tests and RQ4b P1/P2 reruns are newly designated interventions and must carry new provenance.

Official source documentation: https://scikit-learn.org/1.4/modules/generated/sklearn.svm.LinearSVC.html ; https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html ; https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.AdaBoostClassifier.html ; https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.AdaBoostClassifier.html .

### Experimental frames and assignment
- **Core population:** exactly the 63 original eligible CC18 **task IDs** recovered from v2.6's frozen manifest, never guessed from a filename. Each must map to its OpenML dataset ID, revision, raw-row order and class labels. Source data are not inferred from already one-hot-encoded P0 matrices.
- **Pilot and evaluation:** allocate 15 tasks to v3 pilot and 48 to v3 evaluation using a deterministic metadata-only stratified rule, **before any new v3 outcomes**. These tasks were previously studied in v2.6; the 48 are *prospectively allocated relative to new v3 comparisons*, not never-before-observed datasets. Report the 63-task finite-frame census as context; do not call the 48 an externally independent sample.
- **External:** build a dated, frozen, unfiltered non-CC18 candidate catalog; select 20 eligible **task IDs**, excluding exact and alias dataset overlaps with the 63. Metadata-only decisions, exclusions, and revisions must be traceable. At least 10 rows/class, n capped at 5,000 by a frozen stratified sub-sampling rule, and ≤500 post-encoding features. If fewer than 20 eligible, report shortage; never replace based on outcome.
- **Repeated folds:** 6 repetitions × 5 stratified folds per task. Each fold has identical raw-row keys, y, train/test indices, and seeds across versions and policies. Units of uncertainty are tasks. All train/test encodings must be compatible with fixed folds. Abort on leakage/duplicate fold keys or source checksum differences.

### Preprocessing
- **P0:** original centrally encoded unscaled data **as archived**; potentially transductive feature handling. Never relabel it fold-local.
- **P1, primary:** per fold fit categorical vocabulary/encoder on train X only, numeric median from train X, categorical missing handling from train X, identical unseen-category policy on test; no numeric scaling. No labels enter transformations.
- **P2, sensitivity:** identical P1 + standardize *numeric* features using train-only mean and variance; one-hot columns retain 0/1. Document dtype, missingness, constants, sparse/dense strategy. Materialize once outside release-specific envs; hash each train/test matrix, test labels, row keys and fold indices. Validating the feature cap cannot inspect test targets or new outcomes.

### Configuration arms and identity
- **D0:** documented shipped defaults with evaluation-level seed/thread pinning only; verify actual effective parameters and warnings.
- **D1:** `config/E2_legacy_v26.json` is the **unchanged exact v2.6 six-estimator E2 file**, SHA-256 registered; `config/D1_new_estimators_v302.json` adds *only* LinearSVC and AdaBoost, selected from documentation. The resulting eight-estimator resolved configuration is a new derived artifact, not a byte-identical version of the old E2 file. An unsupported legacy E2 setting must remain `UNSUPPORTED`, with a separate denominator; never rewrite it silently.
- **D2:** explicit old effective defaults if supported; SAMME.R in 1.6.1 is always `REMOVED_BEHAVIOR`.
- **D3:** old binary+runtime A/A reference, excluded from mitigation comparisons, because its zero effect is constructed.
- **Portability:** publish `SUPPORTED`, `UNSUPPORTED`, `REMOVED_BEHAVIOR`, `INEFFECTIVE`, `API_REMOVED`, `FIT_FAILED`, `TIMED_OUT`, `MISSING_JOB` as disjoint typed outcomes; no attribution share where a required cell is missing.

### Numerical stacks and mechanism controls
Modern releases 1.4.2, 1.5.2, 1.6.1: independently installed scikit-learn packages but **identical** Python (candidate 3.11), NumPy (1.26.4), SciPy (1.13.1), BLAS/OpenMP and thread parameters. Treat versions as feasibility candidates until verified in Ubuntu-Semantic. Record full wheel hashes/pip freeze, version-specific signatures, fit/predict smoke output, warnings, and `threadpoolctl` backends. Do not run outcome comparisons until G1 and freeze pass.

Original edge: v2.6 phase3 actually compares 1.4.2 with 1.5.2 (NumPy 1.26.4, SciPy 1.12.0), and **does not** provide the proposed original-edge 0.21.3 vs 0.22.2.post1 common-stack cell. Reuse its execution/verification pattern but create a separately versioned new 0.22.2.post1 environment on the original Py3.7.12/NumPy1.17.2/SciPy1.3.1 stack; do not reuse the numeric results as if this cell had run. Verify BLAS runtime as well, or qualify the contrast. If a common-stack cell fails, report it; no replacement version. Use existing v2.6 phase2 historical environment builder for E5 rather than inventing a new independent implementation.

### Estimands, uncertainty, missingness
For task t, estimator e, version r, let s(e,t,r) be mean balanced accuracy in **percentage points** over 30 matched folds. For unordered pair (i,j): δ(t,i,j)=[s(i,new)−s(j,new)]−[s(i,old)−s(j,old)]. For eight estimators, M_t=(1/28) Σ_{i<j}|δ|, and M=(1/48) Σ_{t∈v3-evaluation} M_t. Report six-estimator continuity M over 15 pairs separately. For D0–D1, report paired per-task difference M_t(D0)−M_t(D1) and relative attenuation only where both full pairs are measurable, with failure fractions beside estimates. D1 missing cells are not imputed.

Predeclared primary family = two new edges, D0/P1/8 estimators; practical margin 0.5 pp. One-sided test H0:E[M_t]≤0.5 vs H1:E[M_t]>0.5, adjust Holm across both edges; report exact task counts, t-based one-sided bounds and task-bootstrap (20,000, seeded) sensitivity intervals. Do not use test p-values to make causal claims or claim sampling of all library upgrades. Overall equivalence requires all signed pairs' 90% task-level CIs within ±0.5 pp **and** a justified affected-task share upper bound below 10%; otherwise label inconclusive even if M small. Use finite-frame descriptors separate from superpopulation CI assumptions.

Secondary: task affected share; strict winner changes preserving ties; margin-qualified winners at 0, 0.25, 0.5, 1.0 pp with eligible denominators at both endpoints; material pair reversals; per-estimator score changes, balanced-accuracy prediction mismatches; convergence counts; algorithm-level leverage. Do not pool 28-pair and 15-pair panels. Causality: only the matched common-stack/intervention contrasts are interpreted as conditional mechanisms; no unconditional global percentage of drift assigned to defaults.

### Order and freeze gates
- **G0 archive:** require actual original v2.6 code archive, data/task/fold manifests, E2 file and phase2/phase3 code. Verify hashes and 5.0067 pp / 25-of-63 winners / 0.4921 pp anchors before copying. Stop if originals missing.
- **G1 feasibility:** all 3 modern isolated environments verify common stack, eight smoke constructors and effective LinearSVC/AdaBoost mechanisms. Original edge fixed-stack feasibility is recorded; failures lead to explicit `UNIDENTIFIABLE` cells, not substitution.
- **G2 prospective freeze:** metadata-driven 15/48 selection, external 20 plus complete eligible catalog, raw source data revisions, 6×5 folds, P1/P2 materialization and hashes, unmodified E2, external changelog code table for 2×8 estimator edges, full statistical scripts/hypotheses, and pipeline checks recorded. **No new cross-version outcomes may be inspected before freeze.** Local SHA lock, external immutable dated commit or OSF record and author approval all distinguished.
- **G3:** same-version A/A bitwise prediction identities and independent replay of stored labels and scores.
- **G4:** all 15 pilot tasks, report all objective failures without changing frozen heldout IDs/primary definitions.
- **G5:** all 48 core v3 evaluation tasks and specified arms; complete 30-fold case or typed failure.
- **G6:** 20 external tasks or documented, rule-based shortfall.
- **G7:** independently recompute fold balanced accuracy from raw y/pred and row index; reproduce summary tables, uncertainty and hashes; issue standalone reproducible release, transparency log of agent-authored files, failure map and manuscript scientific notes.

Workstream E5 is descriptive on old+new minor releases, original six algorithms, P1, v3 48 evaluation tasks. Do not alter the v2.6 historical results. Tuned-model sensitivity is explicitly optional and exploratory, frozen grid before execution, never added to primary after seeing findings.

### Approximate resource forecast
The v3.0.1 estimate of ~22,000 jobs × 30 folds is a **planning approximation**, not a locked count. Generate exact planned-job inventory from frozen task × release × estimator × arm × preprocessing × fold combinations, accounting for structurally absent AdaBoost cells. Limit one BLAS/OpenMP thread per worker, start with 8 workers on the 20-core / 30 GiB WSL profile, adjust only for documented OOM/timeouts. CPU-only; no GPU is scientifically required.