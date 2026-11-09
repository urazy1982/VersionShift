# G2 restored-old-defaults pilot

Copy this folder to `/mnt/c/VersionShift/V2_2_G2_RUNNER` without modifying the original source. Requires frozen dataset `.npz` and fold `.npz` files next to the original `PLAN.json` (the replication ZIP contains only a subset of metadata).

Run:

```bash
cd /mnt/c/VersionShift
python3 V2_2_G2_RUNNER/g2_runner.py \
 --plan-dir "$HOME/versionshift_data/STAGE_B_PLAN_CACHED" \
 --selection "$HOME/versionshift_data/STAGE_B_SELECTION.json" \
 --out /mnt/c/VersionShift/V2_2_G2_PILOT \
 --python "$HOME/versionshift_envs/sk-0.22.2.post1/bin/python"
```

If selection is located under the plan directory, use `--selection "$HOME/versionshift_data/STAGE_B_PLAN_CACHED/STAGE_B_SELECTION.json"`. The runner checks that selection matches the plan SHA-256 and stops if any freeze hash is inconsistent. Fresh run output must be empty; on a completed job it verifies and skips rather than overwrites. Re-run the identical command to resume.

This first execution generates 64 model jobs × 30 folds = 1,920 fits, not 1,920 separate jobs. Algorithms: LR, SVC, RF, kNN.  A, B are original saved data, not rerun here.

It deliberately does not infer causal attribution or numerical agreement with the original policy D; follow-on cross-runtime analysis is required. It preserves all original worker fit/predict/imputation logic except a new explicit restoration policy `R`.
