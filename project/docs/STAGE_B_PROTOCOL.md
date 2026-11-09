# Stage B protocol — historical binary pilot (v2.2.5 amended; originally v2.2.3, written 2026-11-06, before any Stage B outcome)

**Purpose (exploratory, not confirmatory).** On real historical binaries, Stage B measures four things:

- whether the 0.21.3→0.22.2 transition reproduces the emulated default effects;
- how often strict P is not portable;
- whether all four counterfactual cells can be built;
- whether the same datasets show metric changes and pairwise-conclusion changes.

Stage B also estimates heterogeneity, which fixes K for Stage C.

## Fixed in advance

| Item | Setting |
|---|---|
| Edges | E+ = 0.21.3 → 0.22.2.post1 (positive mechanism edge); E0 = 1.8.0 → 1.9.1 (negative-control edge) |
| Datasets | 16 eligible CC18 tasks chosen by `stage_b.py select` (seed 20261106; 4 strata: binary/multiclass × n below/above median) |
| Folds | 6 reruns × 5 folds, frozen centrally by `controller.py freeze` |
| Algorithms | LR, SVC, kNN, RF, GB, HGB |
| Policies | D; E; P pinned from the old release (cell B); P pinned from the new release, run in the old env (cell C); PT only with a translations file written before the run; **S** (SonarSource python:S6973 as written, `--pin-from` the old release; v2.2.5) |
| Analyses | `analysis_v221.py edge --plan-dir PLAN_DIR --primary 0.21.3:0.22.2.post1,1.8.0:1.9.1 --population stratified --selection STAGE_B_SELECTION.json` (Holm within the primary family), `mitigation --plan-dir PLAN_DIR --selection STAGE_B_SELECTION.json` (P and PT populations reported separately; `--no-pt` only if no translations file was written before the run), `attribution --plan-dir PLAN_DIR --selection STAGE_B_SELECTION.json`; `emulate_plan.py` on the same 16 datasets |
| Secondary lenses (v2.2.5) | After the primary analyses, for each edge and policy: `code/lenses_crossrelease_v225.py --inp <collected> --edge OLD:NEW --policy D --plan-dir PLAN_DIR --selection STAGE_B_SELECTION.json --oracle-frozen <hash> --margin 0.5 --order-seed 20261106 --out <dir>`. Lenses B/C/D are secondary and never change a primary label. The lens D e-process starts with the Stage B datasets and continues into Stage C datasets in frozen order (`docs/AMENDMENT_V225.md` §1) |
| Frozen analysis code | Hashes in `AMENDMENT_MANIFEST_V223.json` (v2.2.3 supersedes the v2.2.1 freeze for the amended files); any change to these files after Stage B outcomes are seen must be versioned and disclosed |

## Order of operations

1. Run `STAGE=B bash crossversion/build_envs.sh` (only 0.21.3, 0.22.2.post1, 1.8.0, 1.9.1), with `VS_PY37` set if needed. Record `BUILD_FAILURES.txt`. If 0.21.3 fails, apply amendment A-EO: E+ becomes emulation-only.
2. Run `controller.py introspect` on the four releases. Then `oracle_frame.py build --introspect`, code all 78 cases, `oracle_frame.py freeze`, and record the frozen hash (OSF or a dated commit).
3. Run `controller.py freeze --source openml-cc18`. Keep the exclusion ledger. Run `stage_b.py select` and record both hashes.
4. A/A check in each of the four environments: run `controller.py run --aa` with the same interpreter twice; predictions must be bitwise identical.
5. Run each edge with `controller.py run --selection STAGE_B_SELECTION.json --policies D,E,P,S --pin-from <old release>`, plus a separate cell-C run with `--policies P --pin-from <new release>`. Policy S jobs do not count toward G2; their failures are reported under portability, then `collect` and `replay`. Verify that `COLLECT_SUMMARY.json` counts exactly the selected jobs.
6. Run the analyses, passing `--oracle-frozen` (never `--unblinded-exploratory`), plus `emulate_plan.py` and the E+ emulation-only edge for comparison.
6b. Run the secondary lenses (row above) on E+ and E0 for policies D and S. Record the lens D wealth at the end of Stage B.
7. Run `analysis_v221.py size --negative <E0 report> --positive <E+ report> --eligible N --stage-b-used 16` and record the Stage C K decision. K is the maximum of: negative-edge precision, positive-edge power (90% power to clear the margin), positive-edge prevalence, and the floors 48 and 29.
8. Recode a 25% sample of the oracle frame (`recode-sample`), at least 14 days after step 2, then run `agreement`.

## Go / no-go for Stage C

| Gate | Criterion | If it fails |
|---|---|---|
| G1 Execution | All needed envs hash-locked; A/A bitwise identical; `replay` 100% | Fix and repeat Stage B with new labelled seeds |
| G2 Completeness | ≥ 95% of planned D jobs succeed (the denominator is dataset × release × estimator); every dataset with a failed D job is excluded from its analysed comparison and recorded in `plan_coverage.failure_ledger`; report the retained-dataset percentage separately and classify every failure | The analysis refuses; fix and repeat Stage B |
| G3 Blinding | Oracle frozen before step 6 | Stage B becomes unblinded exploratory; refreeze the oracle before Stage C and say so |
| G4 Identifiability | Share of units in each attribution class reported | If C is non-identifiable for > 50% of units, RQ2 is reported as single-path attribution (pre-declared) |
| G5 Emulation fidelity | Report the A→B runtime leg and the correlation of per-pair shifts between emulation and E+ | Descriptive only; no threshold, to avoid tuning on the pilot |
| G6 Sample size | `analysis_v221.py size` on both edges | Applies its pre-stated reuse/exclusion rule |
| G7 Input integrity | `analysis_v221.validate` passes: every job has exactly the frozen 6 × 5 folds, no duplicate identities | Fix collection; never analyse partial folds |

Prevalence in Stage B is a design-weighted estimate for the eligible CC18 frame (`--population stratified`), not a superpopulation claim. Stage C prevalence over the whole eligible frame is reported as a finite-frame census (`--population frame_census`); a superpopulation reading is a separate, explicitly labelled sensitivity analysis.

Stage B results are reported as exploratory in the paper, whatever they show. Stage C analyses exclude Stage B tasks when the eligible frame allows K ≥ 48 without them.
