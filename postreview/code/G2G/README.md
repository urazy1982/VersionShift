# VersionShift v2.2 — G2-G boosting synthesis

Read-only numerical analysis; does not train models or change frozen data.

```bash
cd /mnt/c/VersionShift
unzip -o VersionShift_G2G_kit.zip
python3 VersionShift_G2G/g2g_analyze.py \
  --zip VersionShift_replication.zip \
  --pilot V2_2_G2E_PILOT \
  --holdout V2_2_G2F_HOLDOUT47 \
  --out V2_2_G2G_ANALYSIS
```

If the G2-E pilot used another directory name, update only `--pilot`.

Outputs: `G2G_analysis.json`, `G2G_dataset_comparisons.csv`. The script verifies 63 datasets × 30 folds for each contrast, worker/plan agreement, duplicate/missing rows, score range, and a signed HGB contrast identity. It uses existing baseline default-policy folds in `VersionShift_replication.zip`.

Interpretation: GB contrasts old shipped default against new shipped default; HGB provides total drift, within-old-runtime `max_bins=256` to 255 drift, and the residual comparison old-runtime `255` against new shipped `255`. These are not a fully identified causal factorial design.
