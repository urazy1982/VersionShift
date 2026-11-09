# VersionShift G2-D 8-worker execution

This is a parallel version of the holdout47 restoration runner. It invokes the **unchanged** historical worker using at most eight independent subprocesses. Run-lock hashes, individual results, fold replay checks and the final output schema remain unchanged apart from recording the worker-count setting and new runner fingerprint. Execute with a **new** output directory if you previously ran the sequential G2-D runner. Never start two coordinators on the same output directory.

```bash
cd /mnt/c/VersionShift
python3 VersionShift_G2D_8workers/g2d_runner.py \
  --plan-dir "$HOME/versionshift_data/STAGE_B_PLAN_CACHED" \
  --selection "$HOME/versionshift_data/STAGE_B_SELECTION.json" \
  --out /mnt/c/VersionShift/V2_2_G2_HOLDOUT47_8W \
  --python "$HOME/versionshift_envs/sk-0.22.2.post1/bin/python" \
  --workers 8 2>&1 | tee V2_2_G2_HOLDOUT47_8W.log
```

After a successful run, verify:

```bash
python3 VersionShift_G2D_8workers/g2d_verify_full63.py \
  --zip VersionShift_replication.zip \
  --pilot V2_2_G2_PILOT/G2_FOLD_SCORES.csv \
  --holdout V2_2_G2_HOLDOUT47_8W/G2_FOLD_SCORES.csv \
  --out V2_2_G2_FULL63
```

Stop if output reports any failed model job. Do not modify the original frozen inputs. Keep BLAS and OpenMP thread pools at one thread per job, as the unchanged worker does.
