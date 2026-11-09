VersionShift v2.2 — G0-D independent point-estimate recomputation

From WSL:
  cd /mnt/c/VersionShift
  python3 VersionShift_v2_2_G0D_recompute.py \
    --zip /mnt/c/VersionShift/VersionShift_replication.zip \
    --out /mnt/c/VersionShift/V2_2_G0D

Outputs: G0D_summary.txt, G0D_metrics.json
Exit 0: all compared full-frame manuscript point estimates matched.
Exit 2: missing or invalid fold coverage, duplicate fold identifiers, or other error.
Exit 3: a manuscript point estimate differs.

This script uses Python standard library only. It recomputes from merged
fold-level rows; it does NOT use paper reports as sources. It does not rerun
estimators, certify the prediction hashes against original predictions,
recompute confidence intervals, or certify causal attribution.

Any mismatch must be investigated; never edit the original archive.
