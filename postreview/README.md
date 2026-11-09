# Post-review evidence (G0–G3)

These analyses were run **after** the original study and its first review, on the same frozen 63-task plan, folds and seeds. They are post-review diagnostics and sensitivity analyses, reported separately in the paper (Sections 3.9–3.10, 4.2, 4.4 and 4.5; Tables 5, 6 and 8). They do not change the frozen protocol or its primary results in `../results/`. In particular, **E2 is not the frozen policy E**: E2 is a corrected explicit map designed after the baseline was seen.

Every kit takes the frozen original archive as read-only input and refuses any other: `VersionShift_replication.zip`, SHA-256 `5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d`. It is the original-study part of this repository (everything outside `postreview/`) as archived before the post-review work, and is attached to the repository's first release.

| Gate | Code | What it does | Fits models? |
|---|---|---|---|
| G0-A | `code/G0A` | Intake audit of the frozen archive: CRCs, hashes, inventory | no |
| G0-B, G0-C | `code/G0B`, `code/G0C` | Locate fold-level records; evidence audit | no |
| G0-D | `code/G0D` | Independent recomputation of the D-policy point estimates (M = 5.006712 / 0.051310 pp) | no |
| G2 | `code/G2_PILOT`, `code/G2D`, `code/G2_COMPARE` | Restore old effective defaults for LR, SVC, RF and kNN on 0.22.2.post1 (policy R): pilot 16, then hold-out 47 with 8 workers; compare cells A/B/C | yes |
| G2-E, G2-F, G2-G | `code/G2E`, `code/G2F`, `code/G2G` | Boosting interventions: GB `presort='auto'` on 0.22.2.post1, HGB `max_bins=255` on 0.21.3; pilot, remaining 47, synthesis | yes (E, F) |
| G3-A/B | `code/G3_AB` | Dataset-cluster bootstrap of M and winner-margin sensitivity (Table 5 of the paper) | no |
| G3-D | `code/G3D` | Pre-flight: E2 HGB parameters accepted by both historical releases | no |
| G3-E | `code/G3E` | Fit the missing HGB E2 cells (126 jobs, 3,780 folds) and combine with the archived five-estimator E cells | yes |
| G3-G | `code/G3G` | Read-only replay of the E2 artifacts (specs, parameters, predictions, scores); paired 20,000-replicate bootstrap of D vs E2 | no |

The worker that produced all E2 fits is `code/G3E/worker.py`, SHA-256 `ee39adb1d9cda6030e62245dc6e10576e9b38056de94403c2355e3086431d90c`, the value recorded by all 126 jobs and cited in Section 4.5 of the paper. Each kit's worker is a modified copy of `../project/crossversion/worker.py`; the original is not edited.

## Results shipped here

| File | Content |
|---|---|
| `results/mitigation_D_vs_E2.csv` | Six-estimator D vs E2 by frame (pilot 16, hold-out 47, full 63) |
| `results/G3G_statistical_certificate.json` | Paired bootstrap of D vs E2 (20,000 replicates; full-frame reduction 4.51 pp, 95% interval 3.16–6.07; relative 90.17%, 83.49–95.33%) |
| `results/G3E_job_completion_user_report.json` | Completion of the E2 HGB runs: 32 + 94 jobs, 960 + 2,820 folds, worker and plan hashes |
| `results/G3G_provenance_replay.json` | G3-G replay verdict from the study machine: PASS, 126 jobs, 3,780 folds, 63 datasets, no errors. It checks archived artifacts and recomputes scores from stored predictions; it does not refit models |
| `results/master_results.json` | All G0–G3 headline numbers in one file |

Every output these summaries come from is in `raw/` (below).

## Re-running

Read-only checks run in about a minute on any machine with Python ≥ 3.10, numpy and pandas:

```bash
# the kits check the archive hash, so use the frozen archive attached to the v1.0 release, not a fresh zip
sha256sum VersionShift_replication.zip   # 5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d
python3 postreview/code/G3_AB/g3_robustness.py --zip VersionShift_replication.zip \
        --out /tmp/G3_AB --bootstraps 10000 --seed 20261108
```

The fitting kits (G2, G2-E/F, G3-E) need the per-release environments and the rebuilt plan cache from steps 1–2 of the main README. They write to fresh output directories, refuse to overwrite a run lock, and record unsupported fits instead of substituting parameters. Each kit's `README` gives its exact command, for example G3-E:

```bash
python3 postreview/code/G3E/g3e_runner.py \
  --plan-dir ~/versionshift_data/STAGE_B_PLAN_CACHED --selection ~/versionshift_data/STAGE_B_SELECTION.json \
  --old-python ~/versionshift_envs/sk-0.21.3/bin/python --new-python ~/versionshift_envs/sk-0.22.2.post1/bin/python \
  --workers 8 --scope pilot --out V2_2_G3E_PILOT        # then --scope holdout
python3 postreview/code/G3G/g3g_verify_local.py --plan-dir ~/versionshift_data/STAGE_B_PLAN_CACHED \
  --pilot V2_2_G3E_PILOT --holdout V2_2_G3E_HOLDOUT47_R2 --out G3G_PROVENANCE_REPLAY.json
```

## Raw outputs (`raw/`)

All post-review output directories from the study machine, unchanged (1,568 files, checksums in `raw/SHA256SUMS.txt`):

| Directory | Content |
|---|---|
| `V2_2_G0B`, `V2_2_G0C`, `V2_2_G0D` | Archive inventory, evidence audit, independent recomputation |
| `V2_2_G2_PILOT`, `V2_2_G2_HOLDOUT47` | Restoration fits (policy R): run lock, job specs, per-job records and prediction arrays, `G2_FOLD_SCORES.csv` |
| `V2_2_G2_ABC`, `V2_2_G2_FULL63` | Restoration analyses: cells A/B/C on the pilot; full-frame verification (7,560/7,560 folds match, A→C shift 0) |
| `V2_2_G2E_PILOT`, `V2_2_G2F_HOLDOUT47` | Boosting intervention fits (GB `presort`, HGB `max_bins`) |
| `V2_2_G2E_ANALYSIS`, `V2_2_G2F_FULL63`, `V2_2_G2G_ANALYSIS` | Boosting analyses; contrasts behind Table 6 |
| `V2_2_G3_AB` | Bootstrap and winner-margin sensitivity (Table 5) |
| `V2_2_G3D` | E2 pre-flight in both historical releases |
| `V2_2_G3E_PILOT`, `V2_2_G3E_HOLDOUT47_R2`, `V2_2_G3E_HGB_FULL63.csv` | E2 HGB fits (32 + 94 jobs, 3,780 folds) and their combined fold file |
| `V2_2_G3E_ANALYSIS`, `V2_2_G3G` | Six-estimator D vs E2 (Table 8) and the replay verdict |
| `logs/` | Execution logs. `G3E_holdout.log` records a first hold-out attempt that the runner stopped on a run-lock mismatch; it was re-run in a fresh directory as `V2_2_G3E_HOLDOUT47_R2`, which is the run reported |

## Verify

```bash
python3 postreview/verify_postreview.py --zip VersionShift_replication.zip
# add --plan-dir ~/versionshift_data/STAGE_B_PLAN_CACHED to also replay all 126 E2 jobs from their stored predictions
```

This re-runs G0-D, G2 A/B/C, G2-D, G2-F, G2-G, G3-A/B and G3-E from the frozen archive and `raw/`, and requires every output to match the archived one. It also checks that the combined E2 fold file is exactly the two E2 runs, and that both runs were made with `code/G3E/worker.py`. It takes a few seconds.

Two notes on the code:
- `G2_COMPARE/G2_compare_ABC.py` parses `rep` and `fold` with `int(float(...))`, because the merged result file stores them as `0.0`. With that one-line fix it reproduces the archived G2 A/B/C output exactly.
- G0-D exits with code 3 because its built-in comparison expects 165 pair flips on E+, while it counts 164 strict reversals. The paper reports both (165 sign changes, including one tie transition), and its recomputed metrics match the archived output.
