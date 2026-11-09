import csv, json, hashlib, shutil, sys, zipfile
from pathlib import Path

OUT = Path(sys.argv[1]); PROJ = Path(sys.argv[2]); WORK = Path(sys.argv[3])
rows = list(csv.DictReader(open(OUT / 'ORACLE_FRAME_REVIEWED.csv', newline='')))
ev = list(csv.DictReader(open(OUT / 'ORACLE_EVIDENCE_78.csv', newline='')))
val = json.loads((OUT / 'ORACLE_VALIDATION.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

ABBR = {'no_change_expected': 'none', 'fp_level_change_possible': 'fp-possible', 'change_possible': 'possible', 'change_expected': 'EXPECTED'}
edges = list(dict.fromkeys(r['edge'] for r in rows))
ests = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
grid = {(r['edge'], r['estimator']): r for r in rows}
evg = {(r['edge'], r['estimator']): r for r in ev}

def table():
    out = ['| Edge | ' + ' | '.join(ests) + ' |', '|---|' + '---|' * len(ests)]
    for e in edges:
        cells = []
        for s in ests:
            r = grid[(e, s)]; d = evg[(e, s)]['draft_predicted_impact_category']
            mark = '' if d in ('(not coded)', r['predicted_impact_category']) else ' ↑' if ABBR[r['predicted_impact_category']] != 'none' else ' ↓'
            star = '*' if d == '(not coded)' else ''
            cells.append('%s%s%s (%s)' % (ABBR[r['predicted_impact_category']], mark, star, r['coding_uncertainty'][0]))
        out.append('| %s | %s |' % (e, ' | '.join(cells)))
    return '\n'.join(out)

def per_edge_detail():
    out = []
    for e in edges:
        out.append('\n### %s\n' % e)
        for s in ests:
            r = grid[(e, s)]; d = evg[(e, s)]
            out.append('- **%s — `%s`** (uncertainty %s). Trigger: %s. Evidence: %s. Compatibility: %s.%s' % (
                s, r['predicted_impact_category'], r['coding_uncertainty'], r['trigger_condition'],
                d['relevant_changes'].replace('|', ';'), r['estimated_compatibility'],
                (' **Draft → review:** %s' % d['disputed_or_changed_vs_draft']) if d['disputed_or_changed_vs_draft'] not in ('consistent with draft', 'new coding (draft blank)') else ''))
    return '\n'.join(out)

changed = val['draft_classifications_changed']
changed_md = '\n'.join('| %s | %s | `%s` | `%s` |' % (c['edge'], c['estimator'], c['draft'], c['reviewed']) for c in changed)
disp_md = '\n'.join('- **%s / %s** — %s' % (d['edge'], d['estimator'], d['note']) for d in val['disputed_classifications_for_author'])
compat_md = '\n'.join('| %s | %s | %s | %s |' % (c['edge'], c['estimator'], c['draft'], c['reviewed']) for c in val['compatibility_changes'])

report = f"""# VersionShift v2.2.5 — Independent review of the 78-opportunity scikit-learn changelog oracle

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
- Originals preserved: `ORACLE_FRAME_WORKING.csv` (sha256 `{val['source_working_frame_sha256']}`) and its provenance file were not modified.
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

{table()}

Category counts: {json.dumps(val['predicted_impact_counts'])}; uncertainty: {json.dumps(val['coding_uncertainty_counts'])}.

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
{changed_md}

Compatibility estimates changed vs. draft ({len(val['compatibility_changes'])} cells):

| Edge | Estimator | Draft | Reviewed (abridged) |
|---|---|---|---|
{compat_md}

## 6. Disputed classifications and open interpretation questions

{disp_md}

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

{chr(10).join('%d. %s' % (i + 1, a) for i, a in enumerate(val['author_decisions_required']))}

## 9. Structural validation and freeze-readiness

Independent checks (`ORACLE_VALIDATION.json`): {val['row_count']} rows, exactly the 78 expected keys once each, schema and row order
identical to the working frame, `pipeline_conditions` unchanged, 0 missing required fields, 0 invalid categories, 0 `draft` coders,
ISO dates on all rows, {val['distinct_reference_urls']} distinct https references, evidence file aligned 1:1. The project's
`oracle_frame.py freeze` was executed **unmodified** as a dry run in a scratch directory (original provenance file copied beside the
reviewed frame): it passed; the resulting JSON was discarded and is not shipped. The same command on the supplied working frame
fails ("78 of 78 opportunities incomplete, draft-coded or invalid"), confirming the baseline status.

## 10. Final decision

**{val['final_decision']}** — all 78 classifications carry independently justified official-source evidence and pass the structural
checks. The oracle is **not** author-confirmed, **not** frozen and **not** externally timestamped; the author must confirm or
overrule each row (in particular §6 and finding 3), run introspection, then freeze and timestamp.

## 11. Files

- `ORACLE_FRAME_REVIEWED.csv` — proposed coding, original 15-column schema, 78 rows (sha256 `{val['frame_sha256']}`).
- `ORACLE_FRAME_REVIEWED.csv.PROVENANCE.json` — byte-identical copy of the original build provenance, placed beside the reviewed frame so
  that `freeze` can locate it (its `built_frame_sha256` still refers to the working frame; see `ORACLE_REVIEW_PROVENANCE.json`).
- `ORACLE_REVIEW_PROVENANCE.json` — lineage: hashes of inputs, outputs and the unmodified project script used for the dry run.
- `ORACLE_EVIDENCE_78.csv` — one evidence record per opportunity.
- `ORACLE_VALIDATION.json`, `ORACLE_SHA256.txt`, this report, `ORACLE_CODING_SOURCE.json` (machine-readable coding),
  `tools/build_oracle_review.py`, `tools/validate_oracle_review.py`, `tools/write_report.py` (reproduce the deliverables).

## Appendix — per-opportunity detail
{per_edge_detail()}
"""
(OUT / 'ORACLE_REVIEW_REPORT.md').write_text(report)

# provenance for the review + copy of the original provenance beside the frame
orig_prov = Path('/mnt/user-data/uploads/d0281c58-ORACLE_FRAME_WORKING.csv.PROVENANCE.json')
shutil.copyfile(orig_prov, OUT / 'ORACLE_FRAME_REVIEWED.csv.PROVENANCE.json')
(OUT / 'ORACLE_CODING_SOURCE.json').write_bytes((OUT / '_coding_source.json').read_bytes()); (OUT / '_coding_source.json').unlink()
tools = OUT / 'tools'; tools.mkdir(exist_ok=True)
for s in ['build_oracle_review.py', 'validate_oracle_review.py', 'write_report.py']:
    shutil.copyfile(WORK / s, tools / s)
rev_prov = dict(
    review_of='VersionShift v2.2.5 78-opportunity changelog oracle', reviewer='Claude (Anthropic AI model claude-fable-5-1); author confirmation pending',
    coded_on='2026-11-06', status='PROPOSED_NOT_FROZEN_NOT_AUTHOR_CONFIRMED',
    inputs=dict(working_frame_sha256=sha('/mnt/user-data/uploads/ff804fd2-ORACLE_FRAME_WORKING.csv'), working_provenance_sha256=sha(orig_prov),
                zip_oracle_frame_py_sha256=sha(PROJ / 'code/oracle_frame.py'), zip_changelog_oracle_csv_sha256=sha(PROJ / 'code/changelog_oracle.csv'),
                zip_worker_py_sha256=sha(PROJ / 'crossversion/worker.py'), review_queue_sha256=sha('/mnt/user-data/uploads/65605920-VersionShift_Oracle_78_Review_Queue.csv')),
    outputs=dict(reviewed_frame_sha256=sha(OUT / 'ORACLE_FRAME_REVIEWED.csv'), evidence_sha256=sha(OUT / 'ORACLE_EVIDENCE_78.csv')),
    original_provenance_copied_beside_reviewed_frame=True,
    note='ORACLE_FRAME_REVIEWED.csv.PROVENANCE.json is a byte-identical copy of the original build provenance; its built_frame_sha256 refers to '
         'the working frame, not to the reviewed frame. Outcome files (results/expA*, evidence/AA_CONTROL, emulation outputs) were not opened.')
(OUT / 'ORACLE_REVIEW_PROVENANCE.json').write_text(json.dumps(rev_prov, indent=1))

# hashes + zip
files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'ORACLE_SHA256.txt' and p.suffix != '.zip')
lines = ['%s  %s' % (sha(p), p.relative_to(OUT)) for p in files]
(OUT / 'ORACLE_SHA256.txt').write_text('\n'.join(lines) + '\n')
zpath = OUT / 'VersionShift_Oracle_Reviewed.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files + [OUT / 'ORACLE_SHA256.txt']:
        z.write(p, p.relative_to(OUT))
print('\n'.join(lines)); print('zip', zpath, zpath.stat().st_size)
