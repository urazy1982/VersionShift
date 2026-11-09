# VersionShift — Amendment v2.2.6 (pre-run, additive)

**Written:** 2026-11-07. **Status:** to be frozen by `addons/vs_freeze_amendment.py` **before** any Stage C, longitudinal or fixed-stack job starts. The freeze records this file's SHA-256 and the run directories that existed at that moment.

This amendment changes none of the frozen inputs: `controller.py`, `worker.py`, `analysis_v221.py`, PLAN (`523c42a3…`), the Stage B selection (`55554df0…`) and the 78-case oracle (`33f04fb6…`) stay as they are. It also changes nothing in the CPU completion kit: no folds, seeds, datasets, policies or gates. It only adds checks and descriptive outcomes, and records one correction.

## A1. Correction: mitigation by pinning (policy P) on 0.21.3 → 0.22.2.post1

In Stage B, P fails to transfer to 0.22.2.post1 for LR (`multi_class`, `solver`), SVC (`gamma`), RF (`n_estimators`), GB (`ccp_alpha`) and HGB (`warm_start`). Only kNN transfers. These failures follow from the release itself, so Stage C will reproduce them. With a single portable estimator there is no algorithm pair, so a pair-gap mitigation estimate for P on this edge **cannot be estimated by construction**. This will be reported as a finding (failure to transfer), not as missing data.

The P and S runs are still executed exactly as the kit specifies, for three reasons:
- The portability outcomes (ok, ineffective, unsupported, removed, unpinned_new) are reported per estimator.
- On 1.8.0 → 1.9.1 the strictly portable P population is estimable (5 of 6 estimators; LR is excluded because of `penalty`).
- Changing the frozen design would break comparability with Stage B.

The "available-pairs" P analysis suggested in the 2026-11-07 review is **withdrawn**. It gives no estimate on the early edge and simply duplicates the kit's strictly portable population on the late edge.

## A2. Descriptive outcomes at the level of conclusions (Stage C 47 and the 63-dataset frame)

These quantities were first computed on the 16 Stage B datasets on 2026-11-07, after the Stage B outcomes were known. On Stage B they are therefore **post hoc**. They are registered here before any Stage C outcome exists, so that the 47 pilot-excluded datasets serve as their out-of-sample replication. They are descriptive and add no hypothesis test or gate.

They are computed per edge under the default policy D. Let BA be the mean balanced accuracy over the 30 frozen folds, in percentage points.

- **O1 — change of best algorithm:** the number and share of datasets where the arg-max-BA estimator differs between the two releases.
- **O2 — flips of pairwise order:** the number of (dataset, pair) units, out of 15 pairs per dataset, where sign(BA_a − BA_b) differs between releases. Also reported: the number of datasets with at least one flip.
- **O3 — material flips:** O2 restricted to units where |BA_a − BA_b| ≥ 1.0 pp in both releases.
- **O4 — study-level ordering:** the mean rank of each estimator across datasets (rank 1 = best) in each release, and whether the order of mean ranks changes.

Ties in BA count as no sign (sign = 0), and a change into or out of a tie counts as a flip under O2 but never under O3.

Directional expectations, stated before seeing any data:
- On 0.21.3 → 0.22.2.post1: O1 > 0 and O2 > 0 on the 47 datasets.
- On 1.8.0 → 1.9.1: O1 = O2 = 0.

Either outcome will be reported. They are reported separately for the 47 pilot-excluded datasets and for the full 63-dataset frame (which includes the 16 pilot datasets).

## A3. Phase 2 changelog-oracle outcomes based on scores (14-release scan)

The kit's oracle outcome **H** (hash-based) is kept unchanged. H is positive when any frozen fold prediction hash changes on any of the 47 datasets. Because H fires on floating-point-level differences, two outcomes based on scores are added. Both use policy D, are computed on the 47 pilot-excluded datasets, and are evaluated per (adjacent edge, estimator) opportunity, of which there are 78:

- **S1:** at least one evaluable dataset has |ΔBA| ≥ 0.5 pp (mean over 30 folds, old vs new release).
- **S2:** at least 10% of evaluable datasets have |ΔBA| ≥ 0.5 pp.

An opportunity is evaluable on a dataset only if both releases have 30 'ok' folds with identical fold keys. Opportunities with no evaluable dataset are reported as NOT_ESTIMABLE and left out of precision and recall.

For each outcome in {H, S1, S2} and each oracle rule, TP/FP/FN/TN, precision and recall are reported:
- **strict rule:** `change_expected` or `change_possible`
- **lenient rule:** additionally `fp_level_change_possible`

The prevalence of each outcome is reported next to precision and recall, so that a trivially high recall under H can be seen for what it is. This is exploratory RQ3/RQ4 evidence. The 25% delayed blinded intra-rater recoding (not before 2026-11-20) is unaffected.

## A4. Fixed-stack verification (Phase 3)

Before `analyze-fixed` runs, `addons/vs_fixedstack_verify.py` must pass. It checks that:
- the run lock covers releases {1.4.2, 1.5.2}, policies {D, E} and the 47 holdout datasets;
- COLLECT_SUMMARY shows 1,128 planned jobs and none missing;
- every fold row records numpy 1.26.4;
- both environments' FREEZE.txt and PROVENANCE.json record numpy 1.26.4 and scipy 1.12.0.

A failure stops Phase 3. Nothing is substituted.

## A5. A/A controls

The existing Stage B control `AA_0213` (96 out of 96 job pairs bitwise identical, checked on 2026-11-07) is kept. The kit runs its four A/A controls under new names (`AA_0_21_3`, `AA_0_22_2_post1`, `AA_1_8_0`, `AA_1_9_1`). Both 0.21.3 controls will be reported. Their agreement is an additional determinism check, not a replacement for either.

## A6. Independent verification

After each phase, `addons/vs_verify_independent.py` recomputes the following from the raw `CROSS_RELEASE_FOLDS.csv.gz` files, with code independent of `analysis_v221.py`:
- M per edge;
- the counts of datasets above the 0.25, 0.5 and 1.0 pp margins;
- O1–O4;
- the A/A bitwise identity.

Where the kit's report exists, M is cross-checked against it (tolerance 1e-9 pp). A mismatch is reported and never silently resolved.

## A7. Execution rules (unchanged and restated)

- Thread variables are set to 1.
- At most 10 workers, and a single scheduler per output directory.
- Every failure, non-portable configuration and missing job is recorded.
- No folds are reduced, no datasets substituted, and no completed job is re-run under different seeds.
- If G2 or G7 fails, the analysis stops.

## External timestamp

After freezing, record the SHA-256 printed by the freeze script somewhere outside this machine before Phase 1 starts: for example, a dated e-mail to yourself and a colleague, a commit pushed to a private GitHub repository, or an OSF registration. The local freeze file alone does not prove when the amendment was written.
