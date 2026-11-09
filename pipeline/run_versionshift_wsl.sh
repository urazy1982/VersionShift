#!/usr/bin/env bash
# VersionShift WSL package - one entry point for all CPU phases.
#
#   bash run_versionshift_wsl.sh check     # pre-flight only (changes nothing)
#   bash run_versionshift_wsl.sh phase1    # A/A x4 + Stage C 47 (D/E/P/S) + 63/47 analysis + verification + bundle
#   bash run_versionshift_wsl.sh phase2    # build 14 historical envs + longitudinal scan + oracle outcomes (optional)
#   bash run_versionshift_wsl.sh phase3    # fixed numerical-stack control (optional, after phase2 env build)
#   bash run_versionshift_wsl.sh verify    # re-run independent verification only
#   bash run_versionshift_wsl.sh bundle    # rebuild the upload ZIP only
#
# Every phase is resumable: re-run the same command after an interruption.
# Never run two copies of this script at the same time.
set -Eeuo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${VS_PROJECT:-/mnt/c/VersionShift_v2_2_5}"
DATA="${VS_DATA:-$HOME/versionshift_data}"
ENVS="${VS_ENV_ROOT:-$HOME/versionshift_envs}"
WORKERS="${VS_WORKERS:-10}"
VENV="${VS_CONTROLLER_VENV:-$HOME/versionshift_controller}"
CMD="${1:-check}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OPENML_SKIP_PARQUET=true PYTHONDONTWRITEBYTECODE=1 VS_ENV_ROOT="$ENVS"

[[ -f "$VENV/bin/activate" ]] || { echo "STOP: controller venv not found at $VENV (set VS_CONTROLLER_VENV)"; exit 2; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$(command -v python3)"
LOGDIR="$DATA/package_logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/${CMD}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== VersionShift WSL package: $CMD  ($(date -Is))"
echo "   project=$PROJECT data=$DATA envs=$ENVS workers=$WORKERS python=$PY"
echo "   log=$LOG"

# Lock: one scheduler at a time
exec 9>"$DATA/.versionshift_package.lock"
flock -n 9 || { echo "STOP: another run_versionshift_wsl.sh is already running"; exit 2; }

KIT="$PY $PROJECT/versionshift_cpu_finish.py --project $PROJECT --data $DATA"

install_kit() {
  # Copy the three kit scripts into the project root without overwriting; refuse if a different file exists.
  for f in versionshift_cpu_finish.py versionshift_parallel_10_aa.py versionshift_policy_descriptives.py; do
    src="$PKG/kit/$f"; dst="$PROJECT/$f"
    if [[ -e "$dst" ]]; then
      if ! cmp -s "$src" "$dst"; then
        echo "STOP: $dst exists and differs from the kit copy."
        echo "      sha256 project: $(sha256sum "$dst" | cut -d' ' -f1)"
        echo "      sha256 kit:     $(sha256sum "$src" | cut -d' ' -f1)"
        echo "      Resolve consciously (keep the one used by any existing run); nothing was overwritten."
        exit 3
      fi
    else
      cp -n "$src" "$dst"; echo "installed $f"
    fi
  done
}

preflight() { $PY -I "$PKG/addons/vs_preflight.py" --project "$PROJECT" --data "$DATA" --envs "$ENVS" --phase "$1" --workers "$WORKERS"; }
freeze()    { $PY -I "$PKG/addons/vs_freeze_amendment.py" --amendment "$PKG/AMENDMENT_V226_PRE_RUN.md" --data "$DATA"; }
verify()    { $PY -I "$PKG/addons/vs_verify_independent.py" --data "$DATA" || echo "!! independent verification reported problems - see $DATA/INDEPENDENT_VERIFICATION/VERIFY_SUMMARY.json"; }
bundle()    { $PY -I "$PKG/addons/vs_make_upload_bundle.py" --data "$DATA" --envs "$ENVS" --out-dir "$PROJECT" --label "$1"; }

case "$CMD" in
  check)
    preflight phase1 || true
    ;;
  phase1)
    preflight phase1
    install_kit
    freeze
    echo; echo ">>> Record the amendment SHA-256 above OFF this machine (e-mail / git push / OSF)."
    if [[ -z "${VS_TIMESTAMP_DONE:-}" ]]; then
      read -r -p ">>> Type YES once the SHA-256 is recorded externally (or set VS_TIMESTAMP_DONE=1): " ans
      [[ "$ans" == "YES" ]] || { echo "STOP: re-run after recording the SHA-256"; exit 4; }
    fi
    $KIT prepare
    $KIT status
    $PY -u "$PROJECT/versionshift_cpu_finish.py" --project "$PROJECT" --data "$DATA" run --workers "$WORKERS"
    $PY -u "$PROJECT/versionshift_cpu_finish.py" --project "$PROJECT" --data "$DATA" analyze
    verify
    bundle phase1
    echo "PHASE 1 COMPLETE. Upload the ZIP printed above."
    ;;
  phase2)
    [[ -f "$DATA/STAGE_C_CPU_COMPLETION/reports/CPU_COMPLETION_ASSESSMENT.json" ]] || { echo "STOP: finish phase1 first"; exit 2; }
    [[ -f "$DATA/AMENDMENT_V226_FREEZE.json" ]] || { echo "STOP: amendment not frozen"; exit 2; }
    preflight phase2 || echo "(missing historical envs are expected before the build step)"
    ( cd "$PROJECT" && STAGE=ALL bash crossversion/build_envs.sh )
    if [[ -s "$ENVS/BUILD_FAILURES.txt" ]]; then echo "-- BUILD_FAILURES.txt:"; cat "$ENVS/BUILD_FAILURES.txt"; fi
    preflight phase2
    $PY -u "$PROJECT/versionshift_cpu_finish.py" --project "$PROJECT" --data "$DATA" longitudinal --workers "$WORKERS"
    $PY -u "$PROJECT/versionshift_cpu_finish.py" --project "$PROJECT" --data "$DATA" analyze-long
    $PY -I "$PKG/addons/vs_oracle_score_outcome.py" --project "$PROJECT" --data "$DATA"
    verify
    bundle phase2
    echo "PHASE 2 COMPLETE. Upload the ZIP printed above."
    ;;
  phase3)
    [[ -f "$DATA/AMENDMENT_V226_FREEZE.json" ]] || { echo "STOP: amendment not frozen"; exit 2; }
    preflight phase3
    $PY -u "$PROJECT/versionshift_cpu_finish.py" --project "$PROJECT" --data "$DATA" fixedstack --workers "$WORKERS"
    $PY -I "$PKG/addons/vs_fixedstack_verify.py" --data "$DATA" --envs "$ENVS"
    $PY -u "$PROJECT/versionshift_cpu_finish.py" --project "$PROJECT" --data "$DATA" analyze-fixed
    verify
    bundle phase3
    echo "PHASE 3 COMPLETE. Upload the ZIP printed above."
    ;;
  verify) verify ;;
  bundle) bundle "${2:-manual}" ;;
  *) echo "usage: bash run_versionshift_wsl.sh {check|phase1|phase2|phase3|verify|bundle}"; exit 1 ;;
esac
