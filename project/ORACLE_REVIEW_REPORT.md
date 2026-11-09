# VersionShift v2.2.5 — Independent review of the 78-opportunity scikit-learn changelog oracle

**Reviewer:** Claude (Anthropic AI model `claude-fable-5-1`), acting as an independent, blinded evidence coder commissioned by
Uraz Yavanoğlu (study author). **Coding date:** 2026-11-06. **Status of this document:** proposed coding for author
confirmation. Nothing here is author-confirmed, frozen or externally timestamped.

## 1. Scope, blinding and integrity statement

- Universe: the 78 opportunities (13 adjacent release edges × 6 estimators) defined by `code/oracle_frame.py` in
  `VersionShift_v2_2_5.zip`; the universe, schema, allowed categories, research questions, estimands, thresholds and G1–G7 gates were
  **not** modified.
- Inputs read: `code/oracle_frame.py`, `code/changelog_oracle.csv`, `crossversion/worker.py`, `crossversion/controller.py`
  (introspect/encoding only), `docs/STAGE_B_PROTOCOL.md`, `docs/FINAL_DESIGN_V22.md` (§A-CO/A-BL), `RESPONSE_TO_REVIEW.md`,
  the supplied `ORACLE_FRAME_WORKING.csv` + provenance, and the review queue (treated as unverified leads only).
- **Not opened:** `results/expA*`, `results/v21_analysis`, `evidence/AA_CONTROL/*`, any fold/score/prediction file, and the
  44-dataset emulation outputs. No classification was informed by measured cross-version outcomes. The
  `measured_effect` column of the evidence file is deliberately `NOT EXAMINED` for all 78 rows.
- Originals preserved: `ORACLE_FRAME_WORKING.csv` (sha256 `aab85be544c8d40c8cee4a62300ec559b5fa03141a878a9da02ba49c20bc478f`) and its provenance file were not modified.
  The working frame's hash equals `built_frame_sha256` in the provenance file; the builder and changelog hashes in the provenance
  match the zip.
- Primary sources: official scikit-learn release notes for every destination release (0.22.x, 0.23.x, 0.24.x, 1.0.x, 1.1.x, 1.2.x, 1.3.x,
  1.4.x, 1.5.x, 1.6.x, 1.7.x, 1.8.0, 1.9.0/1.9.1), read from the project's `doc/whats_new/v*.rst` sources and rendered pages;
  versioned API documentation (0.21, 0.22, 0.23, 0.24, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8) and tagged source code
  (`0.23.2`/`0.24.2` `neighbors/_base.py`, `1.8.0`/`1.9.1` `binning.py`, `1.9.1` `_forest.py`) where notes were ambiguous; GitHub PRs
  #13511, #19336, #22868, #26721, #28048, #28295, #29458, #29641, #31529, #32259, #32708. The locally installed scikit-learn 1.9.1
  was used to confirm 1.9.1 defaults. No reference, date or approval was fabricated; every URL in the frame was actually consulted.

## 2. Pipeline conditions used to evaluate triggers (from `worker.py` / `controller.py`)

Policy D = shipped defaults, plus `random_state=seed` for RF/GB/HGB and `n_jobs=1` for RF/kNN; `X` is dense float64 after central
one-hot encoding (missing category → own column); NaN is median-imputed on the training fold **before** `fit`, so no estimator ever
sees NaN; no `sample_weight`; `n ≤ 5000`; splits are frozen centrally (native splitters never run); outcome = hard labels from
`predict`; OpenMP/BLAS threads pinned to 1. Consequences used throughout: every "NaN present", "sample_weight given",
"native splits", "probability=True", "early stopping", "float32 input" trigger is *not met*; one-hot encoding makes
`n_features > 15` common (kNN `auto` heuristic) and makes exact distance/criterion ties common.

## 3. Coding conventions

- `predicted_impact_category` is the **changelog-based** prediction of whether policy-D predictions change on the edge, with triggers
  evaluated under §2. Mapping used consistently: documented default/semantic change on the exercised path → `change_expected`;
  official "Changed models"/"Fix changes results" entry whose trigger is plausibly met but dataset-dependent → `change_possible`;
  documented change to the exercised numeric path that is equivalent in exact arithmetic (fp round-off or tie-order only) →
  `fp_level_change_possible`; no entry touching the exercised path, or entries whose trigger is demonstrably unmet → `no_change_expected`.
- `coding_uncertainty`: `low` = mechanism and activation both documented; `medium` = activation inferred from documented
  parameters/code paths or absence of entries; `high` = mechanism inferred (tie-order from an efficiency note; retrospectively
  documented regression; unknown trigger prevalence). **Absence of entries was never treated as proof of no change** — such rows are
  `no_change_expected` with `medium` uncertainty and an explicit note.
- `n_entries`/`entry_categories` count every relevant destination-release entry, including those recorded as *not activated*.
- `estimated_compatibility` follows the builder's convention (`strict_pin_may_fail: <param>: <old> -> <new>` whenever any constructor
  default differs, `strict_pin_portable_expected` otherwise) but is **document-derived**: the reviewer could not run
  `controller.py introspect` on 13 historical environments. `auto_default_diff` is therefore filled with the documented diffs and
  the marker `introspect_not_run_by_reviewer`, never with fabricated introspect output.
- `coder` records the actual reviewer identity and that author confirmation is pending; `coded_on` is the actual coding date.

## 4. Result overview (`*` = row was blank in the working frame; `↑/↓` = category raised/lowered vs. an existing draft; letter = uncertainty)

| Edge | LR | SVC | kNN | RF | GB | HGB |
|---|---|---|---|---|---|---|
| 0.21.3->0.22.2.post1 | EXPECTED (l) | EXPECTED (l) | none (m) | EXPECTED (l) | fp-possible ↑ (m) | EXPECTED ↑ (m) |
| 0.22.2.post1->0.23.2 | none (m) | none (m) | none (m) | none (m) | none (m) | none (m) |
| 0.23.2->0.24.2 | none* (m) | fp-possible* (m) | fp-possible* (m) | none* (m) | none* (m) | none* (m) |
| 0.24.2->1.0.2 | none* (m) | none* (m) | fp-possible* (h) | possible* (m) | possible* (m) | none* (m) |
| 1.0.2->1.1.3 | fp-possible ↑ (m) | none (m) | fp-possible ↑ (m) | possible ↑ (l) | possible ↑ (l) | fp-possible ↑ (m) |
| 1.1.3->1.2.2 | none* (m) | none* (m) | none* (m) | none* (m) | none* (m) | none* (m) |
| 1.2.2->1.3.2 | none* (m) | none* (m) | none* (m) | none* (m) | none* (m) | none* (m) |
| 1.3.2->1.4.2 | EXPECTED* (l) | none* (m) | none* (m) | none* (m) | fp-possible* (m) | fp-possible* (m) |
| 1.4.2->1.5.2 | none* (m) | none* (m) | none* (m) | none* (m) | none* (m) | fp-possible* (m) |
| 1.5.2->1.6.1 | none* (m) | none* (m) | none* (m) | possible* (h) | possible* (h) | none* (m) |
| 1.6.1->1.7.2 | none* (m) | none* (m) | none* (m) | none* (m) | none* (m) | none* (m) |
| 1.7.2->1.8.0 | none (l) | none (m) | none (m) | possible ↑ (m) | possible ↑ (m) | none (m) |
| 1.8.0->1.9.1 | none (m) | none (l) | none (m) | none (l) | fp-possible (l) | EXPECTED ↑ (m) |

Category counts: {"no_change_expected": 53, "fp_level_change_possible": 11, "change_possible": 8, "change_expected": 6}; uncertainty: {"low": 10, "medium": 65, "high": 3}.

## 5. Principal scientific findings

1. **0.21.3→0.22.2.post1 (E+) is a four-estimator positive edge, not three.** LR (liblinear→lbfgs, ovr→auto), SVC (gamma
   auto→scale) and RF (10→100 trees) are confirmed from the 0.21 and 0.22 API docs. In addition the HGB default `max_bins`
   changed **256→255** (one bin reserved for missing values), so bin edges change for any feature with >255 distinct values; the draft
   had `change_possible` from the "Changed models" listing only — raised to `change_expected`. GB is not unchanged either: the 0.21
   default `presort='auto'` presorted dense data and 0.22 made `presort` a no-op, so the exercised split-search path changed
   (exact-arithmetic-equivalent → `fp_level_change_possible`).
2. **1.3.2→1.4.2 LR is an official "Changed model"** (#26721: lbfgs/newton-cg objective rescaled to the mean loss; "coefficients change
   for the same data") → `change_expected`. This edge was blank in the working frame.
3. **1.8.0→1.9.1 (E0, the planned negative-control edge) is not negative for HGB.** PR #29641 changed the *unweighted* quantile
   branch of `_find_binning_thresholds` from `np.percentile(method="midpoint")` (verified in the 1.8.0 source) to
   `method="averaged_inverted_cdf"` with duplicate-threshold removal (verified in the 1.9.1 source). The draft trigger
   ("sample_weight given or n_samples > subsample") and the release-note wording ("unweighted edges unchanged") hold only when a
   feature has ≤ `max_bins` distinct values. For high-cardinality numeric features the edges, and hence the model, change →
   `change_expected`. **This materially affects the Stage B design assumption that E0 is a negative control**; the author should
   decide whether E0 remains a negative control for the five other estimators only, or whether another edge is chosen.
4. **A regression never documented in its own release:** the 1.6.0 tree partitioner refactor (#29458, intended as a pure code move)
   left `FEATURE_THRESHOLD` initialised to 0.0 instead of 1e-7; it was fixed in 1.8.0 (#32259, "regression on almost-constant
   features"). Features varying by ≤1e-7 within a node were treated as constant before 1.6 and after 1.8 but splittable in
   1.6–1.7. RF and GB are coded `change_possible` on **both** 1.5.2→1.6.1 and 1.7.2→1.8.0, with high/medium uncertainty. The 1.6 coding
   depends on admitting a later release note as evidence (author decision §7).
5. **1.0.2→1.1.3 RF/GB:** the new best-split sort (#22868) is an official "Changed models" entry ("splits with tied criterion values may now
   differ"); the draft coded only the equivalent `max_features`/`loss` renames (`no_change_expected`) → raised to `change_possible`.
   The same release moved LR, HGB and the dense-float64 kNN back-end onto new numeric code (`fp_level_change_possible`).
6. **0.24.2→1.0.2 RF/GB:** the tree criterion rounding fix (#19336) is an official "Changed models" entry → `change_possible`.
7. **kNN** has no semantic default change on any edge, but three documented code-path changes affect which algorithm or
   back-end computes the (mathematically identical) neighbor sets: 0.24 `auto` heuristic (brute when `n_features>15`, verified in
   0.23.2/0.24.2 source), 1.0 KD/Ball-tree construction (#19473), 1.1 pairwise reduction back-end (#21987). All coded
   `fp_level_change_possible` (tie/round-off class).
8. **Compatibility estimates in the working frame are not evidence-based for 12 of 13 edges.** `estimated_compatibility` was
   `strict_pin_portable_expected` wherever the (not shipped) introspect JSON had no diff for the edge key, including
   0.21.3→0.22.2.post1 where `gamma`, `solver`, `multi_class`, `n_estimators`, `max_bins`, `presort` all differ. The builder emits
   "portable_expected" for edges the introspect never covered; this is a build-logic issue the author should fix
   (`build()` line 60: an edge absent from the introspect should yield `unknown_run_introspect`). The reviewed frame documents 15
   edge×estimator cells where a strict pin from the old release is expected to be non-portable (placeholders → `ineffective`,
   removed parameters → `removed`, new parameters → `unpinned_new`, or a value the new release rejects, e.g. `max_bins=256` in 0.22).

| Edge | Estimator | Draft category | Reviewed category |
|---|---|---|---|
| 0.21.3->0.22.2.post1 | GB | `no_change_expected` | `fp_level_change_possible` |
| 0.21.3->0.22.2.post1 | HGB | `change_possible` | `change_expected` |
| 1.0.2->1.1.3 | LR | `no_change_expected` | `fp_level_change_possible` |
| 1.0.2->1.1.3 | kNN | `no_change_expected` | `fp_level_change_possible` |
| 1.0.2->1.1.3 | RF | `no_change_expected` | `change_possible` |
| 1.0.2->1.1.3 | GB | `no_change_expected` | `change_possible` |
| 1.0.2->1.1.3 | HGB | `no_change_expected` | `fp_level_change_possible` |
| 1.7.2->1.8.0 | RF | `no_change_expected` | `change_possible` |
| 1.7.2->1.8.0 | GB | `no_change_expected` | `change_possible` |
| 1.8.0->1.9.1 | HGB | `no_change_expected` | `change_expected` |

Compatibility estimates changed vs. draft (15 cells):

| Edge | Estimator | Draft | Reviewed (abridged) |
|---|---|---|---|
| 0.21.3->0.22.2.post1 | LR | strict_pin_portable_expected | strict_pin_may_fail: solver: warn -> lbfgs; multi_class: warn -> auto |
| 0.21.3->0.22.2.post1 | SVC | strict_pin_portable_expected | strict_pin_may_fail: gamma: auto_deprecated -> scale |
| 0.21.3->0.22.2.post1 | RF | strict_pin_portable_expected | strict_pin_may_fail: n_estimators: warn -> 100 |
| 0.21.3->0.22.2.post1 | GB | strict_pin_portable_expected | strict_pin_may_fail: presort: auto -> deprecated |
| 0.21.3->0.22.2.post1 | HGB | strict_pin_portable_expected | strict_pin_may_fail: max_bins: 256 -> 255 |
| 0.22.2.post1->0.23.2 | HGB | strict_pin_portable_expected | strict_pin_may_fail: early_stopping, monotonic_cst: <absent> |
| 0.23.2->0.24.2 | GB | strict_pin_portable_expected | strict_pin_may_fail: presort: deprecated -> <absent> |
| 0.23.2->0.24.2 | HGB | strict_pin_portable_expected | strict_pin_may_fail: categorical_features: <absent> -> None |
| 0.24.2->1.0.2 | RF | strict_pin_portable_expected | strict_pin_may_fail: min_impurity_split: None -> <absent> |
| 0.24.2->1.0.2 | GB | strict_pin_portable_expected | strict_pin_may_fail: min_impurity_split: None -> <absent> |
| 1.1.3->1.2.2 | HGB | strict_pin_portable_expected | strict_pin_may_fail: interaction_cst, class_weight: <absent> |
| 1.3.2->1.4.2 | RF | strict_pin_portable_expected | strict_pin_may_fail: monotonic_cst: <absent> |
| 1.3.2->1.4.2 | HGB | strict_pin_portable_expected | strict_pin_may_fail: max_features: <absent> -> 1.0 |
| 1.5.2->1.6.1 | HGB | strict_pin_portable_expected | strict_pin_may_fail: categorical_features: warn -> from_dtype |
| 1.7.2->1.8.0 | LR | strict_pin_portable_expected | strict_pin_may_fail: multi_class: deprecated -> <absent> |

## 6. Disputed classifications and open interpretation questions

- **0.22.2.post1->0.23.2 / SVC** — DISPUTED: the official Changed-models list names SVC unconditionally; the trigger analysis relies on the random_state docstring. Author may prefer change_possible to honour the official listing literally.
- **1.5.2->1.6.1 / RF** — draft: not coded. Reviewer: change_possible from a LATER release note (1.8.0) describing a 1.6.0 regression. Author decision: whether the oracle may use notes published after the destination release.
- **1.8.0->1.9.1 / kNN** — DISPUTED (minor): author may prefer fp_level_change_possible to be conservative about tie handling.

Additional judgement calls the author may wish to revisit (all documented in `ORACLE_EVIDENCE_78.csv`):
- 0.21.3→0.22.2.post1 GB (`fp_level_change_possible` from the `presort` no-op): the mathematics is unchanged; only the sample
  accumulation order differs.
- 1.4.2→1.5.2 HGB (`fp_level_change_possible`: `predict` no longer routes through `predict_proba`): differs only at exact ties.
- 0.24.2→1.0.2 kNN (`fp_level_change_possible`, high uncertainty): tie-resolution change inferred from a construction-efficiency note.
- 1.2.2→1.3.2 RF/GB and 1.3.2→1.4.2 RF: the tree splitter was refactored for missing-value support; coded `no_change_expected`
  because the no-NaN branch is documented as unchanged and PR #28295 was verified to require a preceding missing-value split.

## 7. Limitations

- **Observed vs. predicted vs. measured.** Columns `observed_api_default_changes` (documented default diffs), `reviewed_predicted_impact_category`
  (prediction-level expectation) and `measured_effect` (not examined) are kept separate in the evidence file.
- **Dependency stack.** Each hash-locked environment ships a different NumPy/SciPy/BLAS build. LR's lbfgs is `scipy.optimize`
  L-BFGS-B (rewritten in C in SciPy 1.15), kNN brute uses BLAS GEMM, SVC's kernel path uses SciPy BLAS from 0.24. Round-off drift
  from these dependencies can flip hard labels near decision boundaries on *any* edge, independent of scikit-learn entries.
  This was **not** coded into the categories (RQ3 measures the predictiveness of the scikit-learn changelog; unpredicted changes are
  exactly what RQ3 should be able to detect) but is recorded per row in `dependency_stack_caveat`. The author may prefer a blanket
  `fp_level_change_possible` for LR/kNN/SVC; this would change the strict-rule positives not at all and the lenient-rule positives
  substantially.
- **Trigger prevalence.** Several triggers are dataset properties (features with >255 distinct values; near-constant features;
  `n_features>15`; criterion ties). The reviewer did not inspect the frozen dataset pool (to stay blind to Stage B artefacts), so
  `change_expected` rows with dataset-dependent triggers carry `medium` uncertainty.
- **Introspection.** `auto_default_diff`/`estimated_compatibility` are document-derived; `controller.py introspect` across all 14
  releases remains to be run and reconciled. The provenance file records an `introspect_sha256` but the JSON is not in the zip.
- **Release-note completeness.** Release notes are not exhaustive (scikit-learn states the Changed-models list "cannot be assured
  complete"); 1.7.x has no tree/neighbors sections at all. `no_change_expected` therefore means *no documented mechanism*, not a
  guarantee.
- **Single reviewer, single pass.** This is an independent coding, not inter-rater validation; the protocol's 14-day blinded
  intra-rater recode (§A-CO) still applies after the author's confirmation and freeze.
- The 0.22 "Changed models" entry for `StratifiedKFold`/`cv=int` is retained in the 0.22 rows for traceability but is never
  activated (frozen central splits).

## 8. Remaining author decisions and actions

1. Confirm or overrule each reviewed classification (78 rows); the coder field records the reviewer, not the author.
2. Decide whether later-release notes may document earlier regressions (1.5.2->1.6.1 RF/GB coded from the 1.8.0 fix note #32259).
3. Decide SVC 0.22.2.post1->0.23.2: official 'Changed models' listing vs documented probability=True-only RNG mechanism.
4. Decide kNN 1.8.0->1.9.1: tie-order-only change coded no_change_expected; conservative alternative is fp_level_change_possible.
5. Decide whether dependency-stack (SciPy/NumPy/BLAS) drift should be coded into the oracle or kept as a separate construct (reviewer kept it separate).
6. Run controller.py introspect on all 14 releases and reconcile auto_default_diff / estimated_compatibility (reviewer values are doc-derived).
7. Freeze (oracle_frame.py freeze) and externally timestamp AFTER confirmation; this review did not freeze.

## 9. Structural validation and freeze-readiness

Independent checks (`ORACLE_VALIDATION.json`): 78 rows, exactly the 78 expected keys once each, schema and row order
identical to the working frame, `pipeline_conditions` unchanged, 0 missing required fields, 0 invalid categories, 0 `draft` coders,
ISO dates on all rows, 105 distinct https references, evidence file aligned 1:1. The project's
`oracle_frame.py freeze` was executed **unmodified** as a dry run in a scratch directory (original provenance file copied beside the
reviewed frame): it passed; the resulting JSON was discarded and is not shipped. The same command on the supplied working frame
fails ("78 of 78 opportunities incomplete, draft-coded or invalid"), confirming the baseline status.

## 10. Final decision

**READY_FOR_AUTHOR_CONFIRMATION** — all 78 classifications carry independently justified official-source evidence and pass the structural
checks. The oracle is **not** author-confirmed, **not** frozen and **not** externally timestamped; the author must confirm or
overrule each row (in particular §6 and finding 3), run introspection, then freeze and timestamp.

## 11. Files

- `ORACLE_FRAME_REVIEWED.csv` — proposed coding, original 15-column schema, 78 rows (sha256 `33f04fb6c6da89654f3524cfaef290644b80e37afcd5148dcffa740e9bcfe2f6`).
- `ORACLE_FRAME_REVIEWED.csv.PROVENANCE.json` — byte-identical copy of the original build provenance, placed beside the reviewed frame so
  that `freeze` can locate it (its `built_frame_sha256` still refers to the working frame; see `ORACLE_REVIEW_PROVENANCE.json`).
- `ORACLE_REVIEW_PROVENANCE.json` — lineage: hashes of inputs, outputs and the unmodified project script used for the dry run.
- `ORACLE_EVIDENCE_78.csv` — one evidence record per opportunity.
- `ORACLE_VALIDATION.json`, `ORACLE_SHA256.txt`, this report, `ORACLE_CODING_SOURCE.json` (machine-readable coding),
  `tools/build_oracle_review.py`, `tools/validate_oracle_review.py`, `tools/write_report.py` (reproduce the deliverables).

## Appendix — per-opportunity detail

### 0.21.3->0.22.2.post1

- **LR — `change_expected`** (uncertainty low). Trigger: always (solver default liblinear -> lbfgs); additionally multinomial instead of OvR when n_classes>2 (multi_class ovr -> auto). Evidence: [DABC_semantic] solver default 'warn'(=liblinear) -> 'lbfgs' (announced 0.20, effective 0.22) ; [DABC_semantic] multi_class default 'warn'(=ovr) -> 'auto' (multinomial for lbfgs when n_classes>2) ; [Efficiency_not_exercised] liblinear solver faster/less memory (#14108, #14170, #14296); not used under D in 0.22 ; [splitter_changed_models] model_selection.StratifiedKFold / cv=int listed in 0.22 Changed models (#14704); NOT activated: splits are frozen centrally by controller.py. Compatibility: strict_pin_may_fail: solver: warn -> lbfgs; multi_class: warn -> auto (placeholders -> worker status ineffective).
- **SVC — `change_expected`** (uncertainty low). Trigger: always for unscaled input: gamma 1/n_features -> 1/(n_features*X.var()); identical only if X.var()==1. Evidence: [DABC_semantic] gamma default 'auto_deprecated'(=1/n_features) -> 'scale'(=1/(n_features*X.var())) ; [API_new_param] break_ties added (default False; no effect) (#12557) ; [Fix_not_exercised] zero/negative sample_weight now raises (#14286); sample_weight not used ; [splitter_changed_models] model_selection.StratifiedKFold / cv=int listed in 0.22 Changed models (#14704); NOT activated: splits are frozen centrally by controller.py. Compatibility: strict_pin_may_fail: gamma: auto_deprecated -> scale (placeholder -> ineffective); break_ties: <absent> -> False (unpinned_new).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [Feature_not_exercised] KNeighborsClassifier accepts precomputed sparse neighbors graph (#10482) ; [Fix_not_exercised] radius_neighbors list-of-arrays conversion (0.22.2.post1, #16076); radius API not used ; [splitter_changed_models] model_selection.StratifiedKFold / cv=int listed in 0.22 Changed models (#14704); NOT activated: splits are frozen centrally by controller.py. Compatibility: strict_pin_portable_expected (constructor parameters unchanged 0.21.3 -> 0.22.2.post1).
- **RF — `change_expected`** (uncertainty low). Trigger: always (n_estimators default 10 -> 100). Evidence: [DABC_semantic] n_estimators default 'warn'(=10) -> 100 (announced 0.20, effective 0.22) ; [API_new_param] ccp_alpha added (default 0.0, no pruning) (#12887); max_samples added (default None) (#14682) ; [splitter_changed_models] model_selection.StratifiedKFold / cv=int listed in 0.22 Changed models (#14704); NOT activated: splits are frozen centrally by controller.py. Compatibility: strict_pin_may_fail: n_estimators: warn -> 100 (placeholder -> ineffective); ccp_alpha, max_samples: <absent> (unpinned_new).
- **GB — `fp_level_change_possible`** (uncertainty medium). Trigger: always for dense input: 0.21 default presort='auto' used the presorted split-search path; 0.22 presort has no effect (per-node sorting) -> identical split thresholds in exact arithmetic, but a different accumulation order of per-sample sums. Evidence: [API_deprecation_changes_code_path] presort deprecated and 'has no effect' (#14907); 0.21 default 'auto' presorted dense data ; [API_new_param] ccp_alpha added (default 0.0) (#12887) ; [splitter_changed_models] model_selection.StratifiedKFold / cv=int listed in 0.22 Changed models (#14704); NOT activated: splits are frozen centrally by controller.py. Compatibility: strict_pin_may_fail: presort: auto -> deprecated (pinned 'auto' accepted with warning in 0.22; reverse pin 'deprecated' is a placeholder); ccp_alpha: <absent> (unpinned_new). **Draft → review:** draft: no_change_expected (0 entries). Reviewer: fp_level_change_possible.
- **HGB — `change_expected`** (uncertainty medium). Trigger: any feature with >255 distinct non-missing values in a training fold (max_bins 256 -> 255 changes the bin edges); features with <=255 distinct values keep identical midpoint edges; other listed fixes need early stopping or NaN. Evidence: [DABC_semantic] max_bins default 256 -> 255 (one bin reserved for missing values) (#13911) ; [changed_models_list] HistGradientBoostingClassifier listed in 0.22 'Changed models' (Fix, Feature, Enhancement) ; [Feature_not_exercised] native missing/infinite value support (#13911, #14406); no NaN after median imputation ; [Fix_not_exercised] train/validation binned separately (#13933); stratified early-stopping subsample (#14194); early stopping off by default (n_iter_no_change=None) ; [API_new_param] warm_start added (#14012) ; [splitter_changed_models] model_selection.StratifiedKFold / cv=int listed in 0.22 Changed models (#14704); NOT activated: splits are frozen centrally by controller.py. Compatibility: strict_pin_may_fail: max_bins: 256 -> 255 (pinned 256 rejected in 0.22: must be <=255 -> unsupported); warm_start: <absent> (unpinned_new). **Draft → review:** draft: change_possible (changed_models_list only). Reviewer: change_expected because the max_bins default change is a documented semantic default change.

### 0.22.2.post1->0.23.2

- **LR — `no_change_expected`** (uncertainty medium). Trigger: solver='liblinear' (libsvm/liblinear RNG change #13511) or solver='newton-cg' (convergence check #16266); D uses lbfgs. Evidence: [changed_models_list_conditional] LogisticRegression listed in 0.23 Changed models via liblinear RNG change (#13511); solver-specific ; [Fix_not_exercised] newton-cg convergence check '<' -> '<=' (#16266). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: probability=True (libsvm RNG is used only for the Platt-scaling CV shuffle; SVC.random_state doc: 'Ignored when probability is False'). Evidence: [changed_models_list_conditional] SVC listed in 0.23 'Changed models' (any model using libsvm): platform-dependent rand() replaced by mt19937 (#13511) ; [API_not_exercised] custom kernels must validate input (#11296). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** DISPUTED: the official Changed-models list names SVC unconditionally; the trigger analysis relies on the random_state docstring. Author may prefer change_possible to honour the official listing literally.
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no sklearn.neighbors section in 0.23.x; no pairwise euclidean change. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `no_change_expected`** (uncertainty medium). Trigger: read-only float32 input at predict (#16331); not the case (writable float64). Evidence: [Fix_not_exercised] read-only float32 input supported in tree predict/predict_proba (#16331); DecisionTreeClassifier listed in Changed models for this. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **GB — `no_change_expected`** (uncertainty medium). Trigger: n_iter_no_change set with n_classes>2 (MultinomialDeviance loss value average vs sum, 0.23.2 #17694, changes early-stopping decisions); read-only float32 predict (#16331). Under D n_iter_no_change=None and the loss value never feeds the fit. Evidence: [Fix_conditional] MultinomialDeviance average log-loss computed as sum (0.23.2, #17694): affects train_score_/early stopping only ; [Fix_not_exercised] read-only float32 predict (#16331); GradientBoostingClassifier listed in Changed models for this. Compatibility: strict_pin_portable_expected (constructor parameters unchanged; presort still present as deprecated).
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: n_samples>10000 (early_stopping='auto' then turns early stopping ON, previously off); max_depth set (#16182/#16183); sample_weight given (#14696). Evidence: [DABC_conditional] early_stopping='auto' introduced, n_iter_no_change default None -> 10; auto enables early stopping only when n_samples>10000 (#14516) ; [Feature_code_path] sample_weight support added to gradients/hessians (#14696); None under D ; [Fix_not_exercised] max_depth convention and max_leaf_nodes interplay (#16182, #16183); max_depth=None ; [API_new_param] monotonic_cst added (#15582) ; [changed_models_list] HGB listed in 0.23 Changed models with tag 'API'. Compatibility: strict_pin_may_fail: early_stopping, monotonic_cst: <absent> (unpinned_new); n_iter_no_change: None -> 10 (pinned None may be rejected).

### 0.23.2->0.24.2

- **LR — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [Fix_not_exercised] LogisticRegression no longer modifies sample_weight in place (0.24.2, #19182). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `fp_level_change_possible`** (uncertainty medium). Trigger: always (rbf kernel computations in fit/predict now go through the SciPy BLAS API). Evidence: [Efficiency_fp_level] SVC/SVR/NuSVC/NuSVR/OneClassSVM kernel computations use SciPy BLAS API (#16530). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: not coded. Reviewer: fp_level_change_possible.
- **kNN — `fp_level_change_possible`** (uncertainty medium). Trigger: n_features>15 after one-hot encoding (algorithm='auto' now resolves to brute instead of kd_tree; brute uses the GEMM expansion ||x||^2+||y||^2-2x.y whereas kd_tree computes exact squared differences) -> tie/near-tie reordering of neighbors. Evidence: [Enhancement_changes_code_path] algorithm='auto' selects 'brute' when n_features>15 (#17148); verified against 0.23.2 and 0.24.2 _base.py. Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** review-queue lead confirmed.
- **RF — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [API_not_exercised] X_idx_sorted deprecated/no effect in tree fit (#17614); forests never passed it ; [Feature_not_exercised] DecisionTreeRegressor 'poisson' criterion (#17386). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **GB — `no_change_expected`** (uncertainty medium). Trigger: criterion='mae' (deprecated #18326); default friedman_mse unaffected. Evidence: [API_deprecation_not_exercised] criterion='mae' deprecated (#18326) ; [API_removal] presort parameter removed in 0.24 (deprecated 0.22). Compatibility: strict_pin_may_fail: presort: deprecated -> <absent> (pinned parameter removed).
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: categorical_features set (native categorical path, #18394); not set under D (one-hot float64). Evidence: [API_new_param] categorical_features added, default None (#18394) ; [Efficiency_init_only] parallel histogram initialisation (#18341); memory cycle fix (#18334); uint8 predict (#18410); staged_predict (#16985). Compatibility: strict_pin_may_fail: categorical_features: <absent> -> None (unpinned_new).

### 0.24.2->1.0.2

- **LR — `no_change_expected`** (uncertainty medium). Trigger: solver='newton-cg' dense optimisation (#19571); D uses lbfgs. Evidence: [Efficiency_not_exercised] newton-cg optimised for dense X (#19571) ; [API_cross_cutting] keyword-only arguments enforced (#20002); worker passes kwargs. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [Fix_not_exercised] SVC/SVR raise instead of segfault on internal inconsistency (1.0.1, #21336). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `fp_level_change_possible`** (uncertainty high). Trigger: n_features<=15 (kd_tree path) with exact distance ties at the k-th neighbor: KDTree/BallTree construction algorithm changed (#19473) so partitioning/visit order and therefore tie resolution may differ; neighbor sets identical otherwise. Evidence: [Efficiency_changes_code_path] KDTree/BallTree construction worst case O(n^2) -> O(n) (#19473) ; [API_not_exercised] MinkowskiDistance weight parameter (1.0.2, #21741); p=2 resolves to 'euclidean'. Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: not coded. Reviewer: fp_level_change_possible (tie-order mechanism).
- **RF — `change_possible`** (uncertainty medium). Trigger: nodes where floating-point rounding in the impurity criterion previously produced a spurious non-improving split (#19336); RandomForestClassifier uses the same tree criterion/splitter code as DecisionTreeClassifier. Evidence: [Fix_changes_results] rounding-error fix in the splitting criterion; DecisionTreeClassifier listed in 1.0 'Changed models' (#19336) ; [API_removal] min_impurity_split removed in 1.0 (deprecated since 0.19/0.23) ; [Fix_not_exercised] bootstrap=False with max_samples raises (1.0.2, #21295). Compatibility: strict_pin_may_fail: min_impurity_split: None -> <absent> (pinned parameter removed). **Draft → review:** draft: not coded. Reviewer: change_possible.
- **GB — `change_possible`** (uncertainty medium). Trigger: same tree criterion rounding fix (#19336) in the regression trees (friedman_mse) built by GradientBoostingClassifier; loss='exponential' gradient sign fix (1.0.2, #22050) is not exercised (default deviance). Evidence: [Fix_changes_results] tree criterion rounding fix (#19336) ; [Fix_not_exercised] GradientBoostingClassifier exponential-loss gradient sign fixed (1.0.2, #22050); loss='deviance' under D ; [API_removal] min_impurity_split removed in 1.0. Compatibility: strict_pin_may_fail: min_impurity_split: None -> <absent> (pinned parameter removed). **Draft → review:** draft: not coded. Reviewer: change_possible.
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: tol=None previously (removed; default 1e-7 unchanged); multi-core cgroup quotas (OMP=1 in worker). Evidence: [API_import_path] HGB no longer experimental (#19799); worker tolerates both import styles ; [API_removal_not_exercised] tol=None removed (#19296) ; [Fix_not_exercised] rare segfault fixed (1.0.1, #21130). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).

### 1.0.2->1.1.3

- **LR — `fp_level_change_possible`** (uncertainty medium). Trigger: always under D (lbfgs): loss/gradient computation rewritten on the new _loss module and multiclass label-encoded (#21808, #20567, #21814); same objective and optimiser, fp-level coefficient drift; amplified when lbfgs stops at max_iter=100 without convergence. Evidence: [Efficiency_fp_level] LogisticRegression faster with lbfgs/newton-cg; multiclass label-encoded instead of one-hot (#21808, #20567, #21814) ; [API_attribute] n_iter_ now reports every OvR sub-problem (#21998) ; [Fix_not_exercised] newton-cg single-feature fix (1.1.2, #23608). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: no_change_expected (0 entries). Reviewer: fp_level_change_possible.
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [API_attribute] n_iter_ exposed (#21408) ; [Fix_not_exercised] non-finite dual-gap weights raise (#22149); parameter validation moved to fit (#21436). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `fp_level_change_possible`** (uncertainty medium). Trigger: brute path (n_features>15 or n_neighbors>=n_train//2): new low-level pairwise-distance reduction back-end for dense float64 (#21987, #22064, #22065, #22288, #22320) -> chunked GEMM distances, fp/tie reordering; kd_tree path unaffected. Evidence: [Efficiency_fp_level] pairwise-distance reductions refactored for dense float64; KNeighborsClassifier listed (#21987 et al.) ; [Fix] F-contiguous input support for the new back-end (1.1.2, #23990). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: no_change_expected (0 entries). Reviewer: fp_level_change_possible.
- **RF — `change_possible`** (uncertainty low). Trigger: ties in the split criterion value (new sort algorithm in best-split search, #22868: "splits with tied criterion values may now differ"); common with one-hot/discrete features. Evidence: [changed_models_list] RandomForestClassifier listed in 1.1 'Changed models': ~15% faster fitting, tied splits may differ (#22868) ; [DABC_syntactic_equivalent] max_features default 'auto' -> 'sqrt' (same meaning for classifiers) (#20803) ; [Fix_not_exercised] low-cardinality performance regression (1.1.1, #23410); invalid memory access (1.1.2, #23273). Compatibility: strict_pin_portable_expected (pinned max_features='auto' accepted with FutureWarning in 1.1; removed in 1.3). **Draft → review:** draft: no_change_expected (max_features equivalence only). Reviewer: change_possible because of the official 'Changed models' listing (#22868).
- **GB — `change_possible`** (uncertainty low). Trigger: ties in the split criterion value (#22868); loss rename deviance -> log_loss is equivalent. Evidence: [changed_models_list] GradientBoostingClassifier listed in 1.1 'Changed models' (#22868) ; [API_rename_equivalent] loss default 'deviance' -> 'log_loss' (equivalent) (#23036). Compatibility: strict_pin_portable_expected (pinned loss='deviance' accepted with FutureWarning in 1.1; removed in 1.3). **Draft → review:** draft: no_change_expected (rename only). Reviewer: change_possible (#22868).
- **HGB — `fp_level_change_possible`** (uncertainty medium). Trigger: always: binary/multiclass loss, gradient and hessian computation moved to the new _loss module (#20811, #20567, #21814); loss rename 'auto' -> 'log_loss' is equivalent. Evidence: [Efficiency_fp_level] HistGradientBoostingClassifier faster for binary and multiclass (#20811, #20567, #21814) ; [API_rename_equivalent] loss 'auto'/'binary_crossentropy'/'categorical_crossentropy' deprecated for 'log_loss' (#23040). Compatibility: strict_pin_portable_expected (pinned loss='auto' accepted with FutureWarning in 1.1). **Draft → review:** draft: no_change_expected (rename only). Reviewer: fp_level_change_possible.

### 1.1.3->1.2.2

- **LR — `no_change_expected`** (uncertainty medium). Trigger: solver='newton-cholesky' (new, not default); penalty='none' string deprecated. Evidence: [Feature_not_exercised] newton-cholesky solver added (#24637, #24767) ; [API_deprecation_not_exercised] penalty='none' deprecated for None (#23877). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] only class_weight_ deprecation for regressors (#22898). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: float32 input (new float32 reduction back-end #23865) or dense-sparse pairs; pipeline is float64 dense. Evidence: [Efficiency_not_exercised] pairwise-distance reductions for float32 (#23865) and dense-sparse pairs (#23604, #23585, #24556). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [API_rename] base_estimator_ -> estimator_ attribute (#23819) ; [Fix_not_exercised] min_samples_split=1 regression fixed (1.2.2, #25744); default 2. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **GB — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no GradientBoosting entries in 1.2.x. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: interaction_cst / class_weight set, or categorical_features with negative categories (#24283); none under D. Evidence: [API_new_param] interaction_cst (#21020) and class_weight (#22014) added, defaults None ; [Fix_not_exercised] negative categorical values treated as missing (#24283). Compatibility: strict_pin_may_fail: interaction_cst, class_weight: <absent> (unpinned_new).

### 1.2.2->1.3.2

- **LR — `no_change_expected`** (uncertainty medium). Trigger: solver='liblinear' with penalty='l1' (#25214) or 'newton-cholesky' with max_iter=0 (#26653). Evidence: [Fix_not_exercised] liblinear l1 convergence criterion (#25214); newton-cholesky max_iter=0 crash (#26653). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] only LinearSVC/LinearSVR dual='auto' (#26093). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: sparse-dense pairs (#25044) or brute with non-Euclidean metric (#24076); neither under D. Evidence: [Efficiency_not_exercised] CSR x dense squared Euclidean routine (#25044); brute non-Euclidean predict (#24076) ; [API_deprecation_not_exercised] non-Euclidean/non-Manhattan metrics deprecated for NearestNeighbors (#24083). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `no_change_expected`** (uncertainty medium). Trigger: NaN present in X (tree splitter gained missing-value support for splitter='best', #23595/#26376; the refactored splitter is a no-op without NaN). Evidence: [Feature_code_path] DecisionTree missing-value support with splitter='best' (#23595, #26376); forest support only in 1.4 ; [API_removal] max_features='auto' removed (deprecated 1.1); pinned value from 1.2.2 is 'sqrt' ; [Fix_not_exercised] deterministic tree pickles (1.3.2, #27580). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **GB — `no_change_expected`** (uncertainty medium). Trigger: NaN present in X (same splitter refactor); oob_score (#24882) not used. Evidence: [Feature_code_path] tree splitter missing-value refactor (#23595, #26376) ; [Feature_not_exercised] oob_scores_ exposed (#24882) ; [API_removal] loss='deviance' removed (deprecated 1.1); pinned value from 1.2.2 is 'log_loss'. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: categorical_features set with negative category codes (#25629); not under D. Evidence: [Fix_not_exercised] negative categorical values treated as missing (#25629) ; [Feature_not_exercised] loss='gamma' for the regressor (#22409). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).

### 1.3.2->1.4.2

- **LR — `change_expected`** (uncertainty low). Trigger: always under D (lbfgs): objective rescaled from summed to mean per-sample loss (#26721) -> different convergence path and coefficients for identical data/tol; binomial gradient numerically stabilised for extreme raw predictions (#28048). Evidence: [changed_models_list] LogisticRegression listed in 1.4 'Changed models': lbfgs/newton-cg objective uses average loss, coefficients change (#26721) ; [Fix_fp_level] binomial log-loss gradient stable for very large raw predictions; LogisticRegression listed (#28048). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: not coded. Reviewer: change_expected.
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no sklearn.svm section in 1.4.x. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: sparse input (1.4.1 partial revert #28235 of #28191 concerns Euclidean pairwise with sparse matrices); dense under D. Evidence: [Fix_not_exercised] partial revert for sparse Euclidean pairwise computation; KNeighborsClassifier listed (1.4.1, #28235) ; [Efficiency_not_exercised] dense-sparse predict_proba pairs (#27018). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `no_change_expected`** (uncertainty medium). Trigger: NaN present (forest missing-value support #26391; criterion re-init bug #28295/#28327 requires a prior split with missing values); monotonic_cst set (#13649). Evidence: [Feature_not_exercised] RandomForest missing-value support (#26391) ; [API_new_param] monotonic_cst added (#13649) ; [Fix_not_exercised] criterion not re-initialised when n_missing=0 after a split with missing values (1.4.1, #28295, #28327; closes #28254) ; [API_attribute] tree_.value stores weighted fractions instead of counts (1.4.1, #27639); predict argmax unaffected. Compatibility: strict_pin_may_fail: monotonic_cst: <absent> (unpinned_new).
- **GB — `fp_level_change_possible`** (uncertainty medium). Trigger: always: GradientBoostingClassifier loss/gradient path moved to the common _loss module (#26278, #28095) and binomial gradient stabilised (#28048); same mathematics, fp-level drift; material only for extreme raw predictions. Evidence: [Efficiency_fp_level] GradientBoostingClassifier faster for binary and multiclass (#26278, #28095) ; [Fix_fp_level] binomial log-loss gradient numerically stable; GradientBoostingClassifier listed (#28048). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: not coded. Reviewer: fp_level_change_possible.
- **HGB — `fp_level_change_possible`** (uncertainty medium). Trigger: binary classification (binomial gradient stabilisation #28048; multiclass loss untouched); categorical_features default None -> "warn" and new max_features=1.0 are behaviour-preserving for float64 numpy input. Evidence: [Fix_fp_level] binomial log-loss gradient stable for large raw predictions; HistGradientBoostingClassifier listed (#28048) ; [API_new_param] max_features added, default 1.0 (#27139) ; [DABC_placeholder] categorical_features default None -> 'warn' (from_dtype option, #26411); numpy input unaffected ; [Efficiency_memory] parent histogram reused for children (#27865). Compatibility: strict_pin_may_fail: max_features: <absent> -> 1.0 (unpinned_new); categorical_features: None -> warn. **Draft → review:** draft: not coded. Reviewer: fp_level_change_possible.

### 1.4.2->1.5.2

- **LR — `no_change_expected`** (uncertainty medium). Trigger: multi_class='ovr' explicitly requested (now deprecated); under D 'auto' already resolved to multinomial for lbfgs, and 1.5 'always multinomial' is equivalent. Evidence: [API_deprecation_equivalent] multi_class deprecated ('auto' -> 'deprecated' placeholder); multinomial always used for >=3 classes (#28703) ; [Enhancement_not_exercised] newton-cg verbose output (#27526). Compatibility: strict_pin_portable_expected (pinned multi_class='auto' accepted with FutureWarning in 1.5); reverse pin 'deprecated' is a placeholder (ineffective).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: C=inf (1.5.2, #29780); not under D. Evidence: [Fix_not_exercised] SVC/SVR accept C=inf again (1.5.2, #29780). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: Windows + OpenBLAS 0.3.26 (deadlock workaround #28692); Linux workers unaffected. Evidence: [Fix_platform] OpenBLAS 0.3.26 Windows deadlock workaround in pairwise distances (#28692). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no RandomForest/tree fit entries in 1.5.x. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **GB — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no GradientBoosting entries in 1.5.x; 1.5.2 non-OpenMP build performance fix (#29694) is speed-only. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **HGB — `fp_level_change_possible`** (uncertainty medium). Trigger: raw decision exactly at the class threshold (predict no longer routes through predict_proba, #27844: sign/argmax of raw scores vs argmax of probabilities, identical except at fp ties); binning thresholds computed from pre-sorted data (#28102; same percentile method). Evidence: [Efficiency_tie_level] HistGradientBoostingClassifier.predict no longer calls predict_proba (#27844) ; [Efficiency_code_path] data pre-sorted before computing binning thresholds (#28102) ; [Fix_not_exercised] monotonic_cst handling for non-categorical features (#28925). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: not coded. Reviewer: fp_level_change_possible.

### 1.5.2->1.6.1

- **LR — `no_change_expected`** (uncertainty medium). Trigger: solver='newton-cholesky' (multinomial support #28840, lbfgs fallback fixes #29818/#30100); D uses lbfgs. Evidence: [Enhancement_not_exercised] newton-cholesky full multinomial (#28840); fallback and n_iter_ fixes (#29818, #30100). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no sklearn.svm section in 1.6.x. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: metric='nan_euclidean' or X=None scoring (#25330, #30047); not under D. Evidence: [Enhancement_not_exercised] nan_euclidean metric and NaN inputs (#25330); X=None in predict/score (#30047). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `change_possible`** (uncertainty high). Trigger: any node in which some feature has a non-zero value range <=1e-7 (near-constant floating features; exactly-constant one-hot columns are unaffected). Evidence: [Fix_changes_results_retrospective] tree partitioner refactor #29458 (1.6.0, intended as a pure move of code) left FEATURE_THRESHOLD initialised to 0.0 instead of 1e-7, so features that vary by <=1e-7 within a node are no longer treated as constant; documented only by the 1.8.0 fix note "regression on almost-constant features" (#32259) ; [Fix_not_exercised] entropy/log_loss criterion used ln instead of log2 in 1.6.0, restored in 1.6.1 (#30557); criterion='gini' under D ; [Feature_not_exercised] ExtraTrees missing-value support (#28268). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: not coded. Reviewer: change_possible from a LATER release note (1.8.0) describing a 1.6.0 regression. Author decision: whether the oracle may use notes published after the destination release.
- **GB — `change_possible`** (uncertainty high). Trigger: same near-constant-feature partitioner regression (#29458/#32259) in the regression trees built by GB. Evidence: [Fix_changes_results_retrospective] tree partitioner refactor #29458 (1.6.0, intended as a pure move of code) left FEATURE_THRESHOLD initialised to 0.0 instead of 1e-7, so features that vary by <=1e-7 within a node are no longer treated as constant; documented only by the 1.8.0 fix note "regression on almost-constant features" (#32259). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: not coded. Reviewer: change_possible (retrospective evidence; see RF).
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: pandas/polars categorical input (categorical_features default "warn" -> "from_dtype"); numpy float64 input has no dtype categories; threshold search parallelised per feature (#28064) with OMP threads=1. Evidence: [DABC_placeholder_equivalent] categorical_features default 'warn' -> 'from_dtype' (announced 1.4, effective 1.6) ; [Efficiency_code_path] initial bin-threshold search parallelised across features (#28064); per-feature values unchanged. Compatibility: strict_pin_may_fail: categorical_features: warn -> from_dtype (pinned 'warn' is a placeholder -> ineffective).

### 1.6.1->1.7.2

- **LR — `no_change_expected`** (uncertainty medium). Trigger: solver='liblinear' with n_classes>2 (OvR scheme deprecated, raises in 1.8, #31241); newton-cholesky warm_start (1.7.2, #31866); class_weight='balanced' with sample_weight (#30057). Evidence: [Changed_models_message_only] lbfgs ConvergenceWarning message made more informative (#31316); text only ; [API_deprecation_not_exercised] liblinear one-vs-rest multiclass deprecated (#31241) ; [Fix_not_exercised] newton-cholesky multiclass warm_start (1.7.2, #31866). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] only LinearSVC class_weight/sample_weight interaction (#30057). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no sklearn.neighbors section in 1.7.x. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no sklearn.tree section and no forest entries in 1.7.x. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **GB — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no GradientBoosting entries in 1.7.x. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: X_val/y_val passed to fit (#27124); not under D. Evidence: [Feature_not_exercised] validation data can be passed to fit for early stopping (#27124). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).

### 1.7.2->1.8.0

- **LR — `no_change_expected`** (uncertainty low). Trigger: solver='liblinear' with n_classes>2 (now raises; structural failure for old configs); penalty -> l1_ratio migration is L2-equivalent (penalty 'l2' ~ l1_ratio 0.0). Evidence: [API_deprecation_equivalent] penalty deprecated (removal 1.10); l1_ratio default None -> 0.0 (pure L2, unchanged) (#32659) ; [API_removal_note] multi_class parameter removed in 1.8 (deprecated 1.5); liblinear multiclass raises (per 1.8 API doc) ; [API_not_exercised] liblinear raises when ;X;>1e30 (#31888); n_jobs deprecated (#32742). Compatibility: strict_pin_may_fail: multi_class: deprecated -> <absent> (removed); penalty: l2 -> deprecated; l1_ratio: None -> 0.0 (pinned None deprecated). **Draft → review:** draft: estimated_compatibility 'strict_pin_portable_expected' is contradicted by the 1.8 API doc (multi_class removed).
- **SVC — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no sklearn.svm bullets in 1.8.0. Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no sklearn.neighbors bullets in 1.8.0; pairwise changes are Array-API support only (#29822, #32597). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **RF — `change_possible`** (uncertainty medium). Trigger: any node in which some feature has a non-zero value range <=1e-7 (FEATURE_THRESHOLD restored to 1e-7, reversing the 1.6-1.7 behaviour, #32259); NaN present (#32274, #32351, #32280) is not met. Evidence: [Fix_changes_results] regression on almost-constant features fixed: FEATURE_THRESHOLD initialisation in _partitioner.pxd (#32259) ; [Fix_not_exercised] near-constant features with missing values no longer stop splitting early (#32274); last non-missing sample partition (#32351); decision_path NaN (#32280) ; [Fix_not_exercised] Bagging/IsolationForest sample_weight draws (#31414, #32825); not RandomForest in 1.8. Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: no_change_expected with trigger 'NaN present' (#32274 only). Reviewer: change_possible because #32259 applies without missing values.
- **GB — `change_possible`** (uncertainty medium). Trigger: same near-constant-feature fix (#32259) in GB regression trees; absolute_error fix (#32100) not exercised (friedman_mse). Evidence: [Fix_changes_results] FEATURE_THRESHOLD fix (#32259) ; [Fix_not_exercised] absolute_error criterion O(n log n) and sub-optimal splits (#32100) ; [Efficiency_not_exercised] weighted percentile rewrite (#31775); not used by GB classification defaults. Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: no_change_expected (0 entries). Reviewer: change_possible.
- **HGB — `no_change_expected`** (uncertainty medium). Trigger: none under pipeline conditions (no destination-release entry touches the exercised fit/predict path). Evidence: [no_relevant_entries] no HistGradientBoosting bullets in 1.8.0; weighted-percentile rewrite (#31775) not used by 1.8 binning (np.percentile, method=midpoint; verified in 1.8.0 source). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).

### 1.8.0->1.9.1

- **LR — `no_change_expected`** (uncertainty medium). Trigger: float32 input (lbfgs gradients now computed in float32, #32644); pipeline is float64. Evidence: [Efficiency_dtype_conditional] lbfgs gradients in float32 for float32 input; Array API support for lbfgs (#32644) ; [Feature_not_exercised] callbacks for lbfgs (#33322). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **SVC — `no_change_expected`** (uncertainty low). Trigger: probability=True (parameter deprecated, #32050); default False -> 'deprecated' behaves as False. Evidence: [API_deprecation] probability deprecated (default False -> 'deprecated'); probA_/probB_ deprecated (#32050, #33388). Compatibility: strict_pin_may_fail: probability: False -> deprecated (pinned False accepted with warning; reverse pin is a placeholder).
- **kNN — `no_change_expected`** (uncertainty medium). Trigger: kd_tree/ball_tree with many tied distances (simultaneous_sort quadratic-path fix #33252 changes only the ordering of returned ties, not the k-set); string labels with brute (#33048). Evidence: [Fix_tie_order] quadratic-time path in simultaneous_sort used by BallTree/KDTree with many ties (#33252) ; [Fix_not_exercised] string labels with brute (#33048); 1.9.1 radius_neighbors array-like radius (#34341). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** DISPUTED (minor): author may prefer fp_level_change_possible to be conservative about tie handling.
- **RF — `no_change_expected`** (uncertainty low). Trigger: sample_weight given (bootstrap draws now weighted, #31529; verified in 1.9.1 source that sample_weight=None keeps randint draws). Evidence: [Fix_conditional] bootstrap indices drawn with sample_weight (#31529); None under D -> unchanged randint path ; [Fix_not_exercised] missing-value fixes (#32119, #27630, #32193); friedman_mse deprecation for regressors (#32708). Compatibility: strict_pin_portable_expected (constructor parameters unchanged).
- **GB — `fp_level_change_possible`** (uncertainty low). Trigger: always (criterion 'friedman_mse' impurity scaling fixed by switching to 'squared_error', equivalent up to floating point, #32708). Evidence: [Fix_changes_results] friedman_mse impurity scaling incorrect; now squared_error, 'equivalent up to floating-point error' (#32708) ; [API_deprecation] criterion deprecated (default 'friedman_mse' -> 'deprecated'). Compatibility: strict_pin_may_fail: criterion: friedman_mse -> deprecated.
- **HGB — `change_expected`** (uncertainty medium). Trigger: any feature with >255 distinct values in a training fold: unweighted bin edges switch from np.percentile(method="midpoint") (1.8.0 source) to method="averaged_inverted_cdf" plus duplicate-threshold removal (1.9.1 source); features with <=255 distinct values keep midpoint edges. Evidence: [Fix_changes_results] weight-aware bin edges: midpoints when distinct values < max_bins, otherwise (weighted) averaged inverted CDF quantiles (#29641). Compatibility: strict_pin_portable_expected (constructor parameters unchanged). **Draft → review:** draft: no_change_expected with trigger 'sample_weight given or n_samples > subsample'. Reviewer: change_expected - the unweighted quantile method itself changed (source-verified on both sides).
