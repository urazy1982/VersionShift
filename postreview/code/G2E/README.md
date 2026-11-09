# VersionShift G2-E: boosting pilot, 8 workers

## Scope and integrity

Run exactly two diagnostic interventions on the original 16-task frozen pilot frame.

- On scikit-learn 0.22.2.post1, GB `presort='auto'` versus archived new-default GB (checks candidate legacy sorting behavior; **not** guaranteed restoration).
- On scikit-learn 0.21.3, HGB `max_bins=255` versus archived old-default HGB with `max_bins=256` (within-runtime binning sensitivity).

The old 0.21 HGB and new 0.22 HGB implementations may differ beyond `max_bins`. The 0.22 runtime may not accept 256; **never label this intervention a full old-default restoration**.

Each intervention gets an isolated immutable output and 30 frozen folds, with eight parallel subprocess workers. Existing original inputs and G2 pilot/full-frame results are read-only. The code records unsupported/failed fits instead of silently replacing parameters. All worker scripts are copies, not edits to the original VersionShift replication code.

## Install

Unzip `VersionShift_G2E_8workers_kit.zip` into `C:\VersionShift`, so scripts are at `/mnt/c/VersionShift/VersionShift_G2E_8workers/`.

## Execute

```bash
cd /mnt/c/VersionShift
python3 VersionShift_G2E_8workers/g2e_runner.py \
  --plan-dir "$HOME/versionshift_data/STAGE_B_PLAN_CACHED" \
  --selection "$HOME/versionshift_data/STAGE_B_SELECTION.json" \
  --old-python "$HOME/versionshift_envs/sk-0.21.3/bin/python" \
  --new-python "$HOME/versionshift_envs/sk-0.22.2.post1/bin/python" \
  --workers 8 \
  --out /mnt/c/VersionShift/V2_2_G2E_PILOT \
  2>&1 | tee /mnt/c/VersionShift/G2E_execution.log
```

## Analyze (no model fitting)

```bash
python3 VersionShift_G2E_8workers/g2e_analyze.py \
  --zip VersionShift_replication.zip \
  --results V2_2_G2E_PILOT \
  --out V2_2_G2E_ANALYSIS
```

## Upload

Upload `V2_2_G2E_PILOT/G2E_SUMMARY.json` and `V2_2_G2E_ANALYSIS/G2E_ANALYSIS.json`.
If an intervention fails, upload the summary: failures and compatibility limits are informative and should not be hidden.

## Important interpretation

This is a diagnostic pilot, not a complete two-factor randomized decomposition. The GB arm tests possible recovery; HGB tests within-old-runtime sensitivity to `max_bins`. Even all identical hashes would only prove equivalence on this frozen sample and setting. An additional full-frame or factorial experiment is contingent on these pilot results.
