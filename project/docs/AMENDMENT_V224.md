# VersionShift v2.2.4 — G2 job-count alignment

**Status:** Narrow integrity correction after v2.2.3 and before Stage B outcomes.

## Problem
v2.2.3's G2 evaluated the percentage of datasets with at least one failed shipped-default (D) job. The study's original requirement was a minimum 95% success rate of the planned D jobs. With 16 datasets, a single failed SVC job excluded 1/16 datasets (6.25%) and caused a STOP even though only 1/192 D jobs (0.521%) failed for a two-release edge.

## Fix
In `code/analysis_v221.py`, `validate_plan_coverage()` now enforces **failed D jobs / planned D jobs <= 5%**. Its output includes planned/failed D jobs and the job success share. Every failed D job still excludes the entire affected dataset from the applicable contrast and appears in the failure ledger. The excluded-dataset share and number retained are reported separately. Entirely absent planned jobs are always rejected, and successful jobs still must contain complete unique folds.

## Demonstration
With 16 datasets × 2 releases × 6 algorithms = 192 planned D jobs, one failed SVC job is allowed by G2 (191/192 successes), but only 15 datasets contribute to the edge analysis. A missing RF job is still rejected. The boundary of 12 failures in 240 jobs (5%) is inclusive, while 13 in 240 is rejected.

## Inferential limitation
Passing G2 does **not** certify the affected dataset sample as representative or sufficiently powered. A low D-job failure percentage can still exclude a high share of datasets. Report retained/eligible dataset counts and perform missingness sensitivity analyses. Do not silently replace excluded datasets, do not turn Stage B into confirmatory inference, and do not authorize Stage C without G1–G7 review.

## Provenance
Original v2.2.3 files are preserved in the separately uploaded archive. The v2.2.4 manifest records each packaged file's SHA-256. The amended analysis hash supersedes earlier freeze hashes only for this explicitly documented, pre-outcome code change.
