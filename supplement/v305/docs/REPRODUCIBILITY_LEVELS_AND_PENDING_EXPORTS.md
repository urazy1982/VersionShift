# Reproducibility levels and known gaps

| Level | Can be done with this public companion? | Evidence needed |
|---|---|---|
| A. Frozen-protocol identity | YES, online (or offline with frozen file) | commit-pinned file + SHA-256 |
| B. Numerical result table integrity | YES | aggregate CSV, expected inventory and summary JSON |
| C. Historical P1-D0 task-level M/winner descriptive summaries | YES, validate/report | historical_63_task_level.csv; no raw rescore |
| D. Modern 48-task primary inferential reanalysis | **YES** (added 19 Nov 2026) | `results/secondary/modern_D0_task_level_conclusions.csv` + `tools/reproduce_primary_inference.py` reproduce M, one-sided p, Holm p, lower bound and the seeded bootstrap interval exactly |
| D2. Secondary descriptive tables (winners, margins, reversals, D1/D2, P2, external, E5, typed cells) | YES, validate/report | `results/secondary/*.csv` + `tools/verify_secondary.py`; signed-pair vectors in `ALL_REGISTERED_EXTRACTED_STATISTICS.json` |
| E. Independently recalculate predictions to scores | **NO** | private y_true/y_pred NPZs, task/fold labels, verified replay source and original matrix cache |
| F. End-to-end training re-execution | **NO** | OpenML-licensed input availability, 83 materialized task matrices, 14 historical runtime environments, all frozen source hashes, frozen splits/configs |

Important: the private `VersionShift_v305_SCIENTIFIC_FINAL_VALIDATED_PRIVATE.zip` is approximately 1.6 GB; preserve privately with SHA-256 `f0e37943808537f757edde212ad53dafa833fe1d77393873aea1b7d59e3d9953`. Neither the ZIP nor its constituent private records should be blindly uploaded to GitHub. Historical P0 is recoverable from its archival fold score tables, not from raw predictions in this archive.

**Status of the earlier planned enhancement (now done for level D):** use a temporary private worktree to extract anonymized per-task modern aggregates and a sealed analysis-only script. Recalculate both raw and Holm p-values independently, include machine-readable confidence-interval values, compare against v3.0.5 frozen results, and publish only reviewed aggregates. Never re-register or rewrite the protocol to make this enhancement. Keep an explicit provenance and disclosure log.
