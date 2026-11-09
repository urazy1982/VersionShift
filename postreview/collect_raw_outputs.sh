#!/usr/bin/env bash
# Collect the raw post-review (G0-G3) output directories from the study machine into one archive,
# so they can be added to postreview/raw/ (or deposited on Zenodo if too large for GitHub).
#
#   bash postreview/collect_raw_outputs.sh [SOURCE_DIR] [OUT_ZIP]
#
# SOURCE_DIR defaults to /mnt/c/VersionShift (where the G0-G3 kits wrote their outputs).
# Read-only on SOURCE_DIR. Run lock files, job specifications, prediction arrays and logs are all kept.
set -euo pipefail
SRC="${1:-/mnt/c/VersionShift}"
OUT="${2:-$HOME/VersionShift_postreview_raw.zip}"
cd "$SRC"

shopt -s nullglob
items=(V2_2_G0* V2_2_G2* V2_2_G3* G3D_preflight*.json G3G_*.json *.log)
# kit copies are code, already in postreview/code/
keep=()
for i in "${items[@]}"; do
  case "$i" in V2_2_G2_RUNNER) continue ;; esac
  keep+=("$i")
done
[ ${#keep[@]} -gt 0 ] || { echo "No post-review outputs found in $SRC"; exit 1; }

echo "Found in $SRC:"
du -sh "${keep[@]}" | sed 's/^/  /'
for need in V2_2_G3E_PILOT V2_2_G3E_HOLDOUT47_R2; do
  [ -e "$need" ] || echo "  WARNING: $need not found (needed for the G3-G replay of the E2 HGB arm)"
done

rm -f "$OUT"
zip -qr -X "$OUT" "${keep[@]}" -x '*/__pycache__/*' '*.pyc'
( cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" ) | tee "$OUT.sha256"
echo "Archive: $OUT ($(du -h "$OUT" | cut -f1)). Upload it to the chat, or attach it to a GitHub release / Zenodo record."
