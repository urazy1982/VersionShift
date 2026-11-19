#!/usr/bin/env bash
# Run primary modern v3 CPU experiments (pilot, evaluation, external) plus original-edge RQ4b.
# Fail closed; resume only with unchanged frozen locks.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
WORKERS="${VS_WORKERS:-8}"
function execute() {
    local EDGE="$1" FRAME="$2" PRE="$3" ENVROOT="$4" POL="$5"
    local OUT="$PROJECT/runs/results/${EDGE}_${FRAME}_${PRE}_${POL//,/-}"
    local ARGS=()
    if [[ -e "$OUT" ]]; then ARGS+=(--resume); fi
    python3 -u -m versionshift_v3.run --project "$PROJECT" --frame "$FRAME" --edge "$EDGE" \
         --preprocessing "$PRE" --policies "$POL" --env-root "$ENVROOT" \
         --workers "$WORKERS" --out "$OUT" "${ARGS[@]}"
    python3 -m versionshift_v3.audit --run "$OUT" --out "$PROJECT/runs/audit/$(basename "$OUT").json"
}
[[ -f runs/plan/FROZEN_PROTOCOL.json && -f runs/plan/EXTERNAL_TIMESTAMP_RECORD.json ]] || \
    { echo 'STOP freeze and external timestamp required';exit 2; }
if [[ ! -f runs/audit/AA_SAME_VERSION.json ]]; then
  python3 scripts/aa_control.py --project "$PROJECT" --env-root "$HOME/versionshift_v3_envs" \
      --out "$PROJECT/runs/audit/AA_SAME_VERSION.json"
else
  grep -q 'AA_SAME_VERSION_PASS' runs/audit/AA_SAME_VERSION.json || { echo 'STOP A/A status invalid';exit 2; }
fi
for FRAME in pilot evaluation external; do
  for PRE in P1 P2; do
    execute E_LINEAR_DUAL "$FRAME" "$PRE" "$HOME/versionshift_v3_envs" D0,D1,D2
    execute E_ADA_SAMME "$FRAME" "$PRE" "$HOME/versionshift_v3_envs" D0,D1,D2
  done
done
# Direct original edge preprocessing RQ4b, prior-study outcomes must stay read-only.
for PRE in P1 P2; do
    for FRAME in pilot evaluation; do
        execute E_ORIGINAL "$FRAME" "$PRE" "$HOME/versionshift_envs" D0,D1
    done
done
python3 scripts/aggregate_primary.py \
    --linear runs/audit/E_LINEAR_DUAL_evaluation_P1_D0-D1-D2.json \
    --adaboost runs/audit/E_ADA_SAMME_evaluation_P1_D0-D1-D2.json \
    --out runs/audit/PRIMARY_HOLM_48.json
python3 -m versionshift_v3.report --audits 'runs/audit/*.json' --out runs/report/CPU_STUDY_RESULTS.md
echo 'MAIN_CPU_STUDY_COMPLETED. Run descriptive E5 and old-edge fixed-stack separately.'
