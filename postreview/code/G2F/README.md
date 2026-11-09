# VersionShift G2-F: Full-frame boosting diagnostic

Uses the **unchanged** G2-E worker source. Runs only 47 remaining OpenML-CC18 datasets across 2 intervention arms using 8 subprocess workers; 94 jobs, 2,820 folds.

## Commands (WSL)

```bash
cd /mnt/c/VersionShift
unzip -o VersionShift_G2F_8workers_kit.zip
python3 VersionShift_G2F_8workers/g2f_runner.py \
  --plan-dir "$HOME/versionshift_data/STAGE_B_PLAN_CACHED" \
  --selection "$HOME/versionshift_data/STAGE_B_SELECTION.json" \
  --old-python "$HOME/versionshift_envs/sk-0.21.3/bin/python" \
  --new-python "$HOME/versionshift_envs/sk-0.22.2.post1/bin/python" \
  --workers 8 \
  --out /mnt/c/VersionShift/V2_2_G2F_HOLDOUT47 \
  2>&1 | tee G2F_execution.log
```

Python virtual environment paths are deliberately **not** symlink-resolved.

When complete, run:

```bash
python3 VersionShift_G2F_8workers/g2f_verify_full63.py \
  --zip VersionShift_replication.zip \
  --pilot V2_2_G2E_PILOT \
  --holdout V2_2_G2F_HOLDOUT47 \
  --out V2_2_G2F_FULL63
```

If G2-E pilot was saved in an alternative folder, change `--pilot` accordingly.

### Important
- The source archive and pilot results are read-only, and the run uses a new directory.
- Each 30-fold model job is independent. Existing valid records are reused and checked.
- Do not change the worker or runner mid-run; run-lock checks prevent a changed source from silently mixing outputs.
- GB tests `presort='auto'` in **new** version 0.22.2.post1 versus its own defaults.
- HGB tests `max_bins=255` in **old** version 0.21.3 versus its own defaults.
- The two interventions have different scientific meanings; neither constitutes a complete two-factor, six-estimator cross-version attribution.

Return `V2_2_G2F_HOLDOUT47/G2F_SUMMARY.json` and both files under `V2_2_G2F_FULL63/`.
