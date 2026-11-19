#!/usr/bin/env bash
# Historical E5 descriptive census; never assigns confirmatory inference to E5.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
[[ -f runs/plan/FROZEN_PROTOCOL.json && -f runs/plan/EXTERNAL_TIMESTAMP_RECORD.json ]] || exit 2
for i in $(seq -w 1 13); do
  E="E5_$i"; OUT="runs/results/${E}_evaluation_P1_D0"
  LOG="runs/audit/${E}_ENVIRONMENT_FAILURE.txt"
  set +e
  if [[ -d "$OUT" ]]; then extra=--resume;else extra='';fi
  python3 -u -m versionshift_v3.run --project "$PROJECT" --frame evaluation --edge "$E" \
      --preprocessing P1 --policies D0 --env-root "$HOME/versionshift_envs" \
      --workers "${VS_WORKERS:-8}" --out "$OUT" $extra
  result=$?
  set -e
  if [[ $result != 0 ]];then
      echo "$E FAILED_TO_RUN exitcode=$result (see terminal log and frozen failure ledger)" > "$LOG"
      echo "E5_NONBUILDABLE_OR_FAILED: $E";continue
  fi
  python3 -m versionshift_v3.audit --run "$OUT" --out "runs/audit/${E}_evaluation.json"
done
echo 'E5 DESCRIPTIVE SCAN FINISHED; inspect failures and do not silently omit them.'
