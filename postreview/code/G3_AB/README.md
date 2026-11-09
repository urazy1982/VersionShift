# VersionShift v2.2 — G3-A/B analysis

Read-only reproducibility and robustness test using the original frozen POS/NEG default-policy fold records. This stage performs **no estimator fits**, edits none of the frozen evidence, and is independent of ongoing GPU/LoRA research.

## Run in WSL

```bash
cd /mnt/c/VersionShift
unzip -o VersionShift_G3_AB_kit.zip
python3 VersionShift_G3_AB/g3_robustness.py \
  --zip VersionShift_replication.zip \
  --out V2_2_G3_AB \
  --bootstraps 10000 --seed 20261108
```

The source ZIP must match frozen SHA-256 `5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d` or analysis fails closed. Requires only Python 3 standard library. The output directory must be new or empty. No separate install or scikit-learn environment required.

## Output

- `G3_AB_results.json`: reproducible point estimates, bootstrap intervals and winner-margin sensitivity.
- `G3_AB_dataset_metrics.csv`: dataset-level paired comparisons, winner gaps, signed/strict/material flips.
- `G3_AB_winner_thresholds.csv`: paper-ready winner-sensitivity counts.
- `G3_AB_summary.txt`: concise console report.

### Inferential limits

The 63 tasks are the finite eligible CC18 study frame. Bootstrap confidence intervals describe sensitivity under *hypothetical resampling of the observed tasks*, rather than a census interval or unconditional generalization beyond the frame. Do not replace the manuscript's original 90% t CIs with these exploratory intervals without qualifying the inferential target. We require both old- and new-release winner margins (top-1 vs top-2) to exceed a specified threshold. This is a conservative criterion; it is not the same as the pairwise material-flip definition.

The sign-change function uses a numerical tie tolerance of 1e-12 pp; if the G0-D 165-vs-164 distinction differs, inspect the exact-tie convention before changing manuscript Table 4.
