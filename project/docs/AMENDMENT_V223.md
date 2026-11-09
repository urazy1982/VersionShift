# VersionShift v2.2.3 — integrity amendment on top of v2.2.2 (2026-11-06)

**Before any cross-version outcome analysis.** The v2.2.2 package is accepted as the reference and preserved unchanged; its manifest (`AMENDMENT_MANIFEST_V222.json`) verified 62/62 on receipt. Only `code/analysis_v221.py` and `docs/STAGE_B_PROTOCOL.md` change here. One test file is added.

## Verification of v2.2.2 on receipt
- 62/62 manifest hashes match. Compared with v2.2.1, exactly four files changed (`analysis_v221.py`, `stage_b.py`, `controller.py`, `STAGE_B_PROTOCOL.md`), as the amendment states.
- All three v2.2.1 gaps reported by the auditor were reproduced on v2.2.1 and are closed in v2.2.2.
- Fast tier 57 passed + 3 skipped. **Full suite 60/60 passed (148 s, 2 CPUs, scikit-learn 1.9.1)**, including the three slow end-to-end tests the auditor could not run.
- The A/A evidence replays: 120 rows, 12 jobs, bitwise-identical predictions, M = 0.0 pp, correctly labelled unblinded-exploratory.

## Gaps found in v2.2.2 and fixed here
1. **`mitigation` silently dropped absent P/PT jobs.** A deleted P/PT record for one dataset × algorithm disappeared from the population counts: it was neither counted as non-transferable nor flagged. The command now takes `--plan-dir` / `--selection` and requires a status record for every planned D@old, D@new, P@new and PT@new job. `--no-pt` declares that no PT arm was run. `not_portable` remains an outcome; an absent record is an error.
2. **`attribution` had no plan-coverage check.** Every cell now requires complete job records for its release and policy (plus D in the same release).
3. **Conflict between the new D check and protocol G2.** v2.2.2 refused the whole edge on any non-ok D row, so one fit failure would block Stage B, although G2 tolerates up to 5%. Now a dataset with a failed D job is excluded from every pair of the analysed edge and recorded in `plan_coverage.failure_ledger`. The analysis refuses if more than 5% of planned datasets are excluded. This makes G2 executable instead of contradicted.

`validate_plan_coverage` keeps the v2.2.2 call form, so the auditor's tests run unchanged.

## Tests
`tests/test_v223.py` adds 6 tests: absent pinned job refused; not_portable counted; PT required unless declared; absent attribution cell refused; G2 exclusion within 5% ledgered and removed from K; above 5% refused. **Total 66 tests.**
