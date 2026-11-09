# VersionShift G3-E: Corrected six-estimator mitigation

The G3-D preflight passed in both 0.21.3 and 0.22.2.post1. This kit runs **only the missing HGB corrected explicit-map (E2) cells**. Other five estimator policies E and original default policies D are loaded read-only from `VersionShift_replication.zip`. E2 differs from the original failed E only for HGB: `early_stopping=False` becomes `n_iter_no_change=None`; the existing max_iter=200, learning_rate=0.1, max_bins=255 are unchanged.

This is a post-review protocol amendment, not a retroactive change to the original experiment.

## Execute 16-task pilot with eight CPU workers

```bash
cd /mnt/c/VersionShift
unzip -o VersionShift_G3E_8workers_kit.zip
python3 VersionShift_G3E_8workers/g3e_runner.py \
 --plan-dir "$HOME/versionshift_data/STAGE_B_PLAN_CACHED" \
 --selection "$HOME/versionshift_data/STAGE_B_SELECTION.json" \
 --old-python "$HOME/versionshift_envs/sk-0.21.3/bin/python" \
 --new-python "$HOME/versionshift_envs/sk-0.22.2.post1/bin/python" \
 --workers 8 --scope pilot --out /mnt/c/VersionShift/V2_2_G3E_PILOT \
 2>&1 | tee G3E_pilot.log
```

The expected output is 32 successful jobs, 960 saved fold scores, and `G3E_SUMMARY.json`. Failed jobs produce a failure rather than silently dropped rows. The script is **resume-safe** if invoked again with the exact same args and source files.

## Expand to the remaining 47 tasks

After pilot evidence is accepted, run the same command with `--scope holdout --out /mnt/c/VersionShift/V2_2_G3E_HOLDOUT47` (47 datasets × two releases × 30 folds = **2,820** fold scores; 94 model jobs).

## Combine and analyze results

```bash
cd /mnt/c/VersionShift
python3 - <<'PY'
import csv
from pathlib import Path
files=[Path('V2_2_G3E_PILOT/G3E_HGB_FOLDS.csv'),Path('V2_2_G3E_HOLDOUT47/G3E_HGB_FOLDS.csv')]
out=Path('V2_2_G3E_HGB_FULL63.csv')
with out.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['release','dataset','algo','policy','rep','fold','ba','pred_sha'])
    for p in files:
        with p.open(newline='') as inp:
            rows=csv.DictReader(inp)
            w.writerows([[r[k] for k in ['release','dataset','algo','policy','rep','fold','ba','pred_sha']] for r in rows])
print('Wrote:',out)
PY
python3 VersionShift_G3E_8workers/g3e_analyze.py \
 --zip VersionShift_replication.zip \
 --hgb V2_2_G3E_HGB_FULL63.csv \
 --selection "$HOME/versionshift_data/STAGE_B_SELECTION.json" \
 --out V2_2_G3E_ANALYSIS
```

Return `G3E_SUMMARY.json` from pilot and holdout and `V2_2_G3E_ANALYSIS/G3E_six_estimator_results.json` with `G3E_summary.txt`.

## Safety and interpretation

- Do not modify the original ZIP, experimental code, or frozen plan.
- Output directories are isolated. This runs CPU-only estimators; GPU is not needed.
- Eight workers process independent estimator–dataset–release jobs. Model BLAS threads are limited to one.
- The original E policy's five-estimator result is preserved; E2 is separately labeled and is only available as a post-review corrected map.
- After analysis, independently verify HGB prediction hashes, fold scores, and its effective parameters before claiming a corrected mitigation benefit.
