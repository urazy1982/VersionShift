VersionShift v2.2 — G0-B scientific evidence inventory
=====================================================

This package follows the successful archive CRC/integrity check (G0-A).
Run it on the original, unchanged ZIP from WSL:

  cd /mnt/c/VersionShift
  python3 VersionShift_v2_2_G0B_inventory.py \
    --zip /mnt/c/VersionShift/VersionShift_replication.zip \
    --out /mnt/c/VersionShift/V2_2_G0B

The script checks the ZIP SHA-256 against the G0-A fingerprint, then profiles
file paths, candidate entry points and JSON/CSV/JSONL shapes without executing
any code or extracting ZIP files.

Outputs:
  V2_2_G0B/G0B_summary.txt
  V2_2_G0B/G0B_profiles.json
  V2_2_G0B/G0B_file_index.csv

G0B_summary.txt and G0B_profiles.json locate the raw fold results and experiment
code inside the archive; G0-C uses them for its read-only recomputation.

This stage is not scientific certification and does not run experiments.
