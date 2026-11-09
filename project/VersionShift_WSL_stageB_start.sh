#!/usr/bin/env bash
# VersionShift v2.2.5: fail-closed local Stage B preparation.
# Run from the extracted VersionShift_v2_2_5 project root.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null || true
if [[ ! -f SHA256_MANIFEST_V225.json ]]; then
  echo 'ERROR: Run this script from inside VersionShift_v2_2_5, with the script copied there.' >&2
  exit 2
fi
command -v python3 >/dev/null || { echo 'ERROR: python3 missing'; exit 2; }
command -v uv >/dev/null || { echo 'ERROR: uv missing. Install uv before running.'; exit 2; }
python3 - <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path('.'); m=json.loads((r/'SHA256_MANIFEST_V225.json').read_text())['files']; bad=[]
for name,expected in m.items():
    f=r/name
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected: bad.append(name)
print(f'Manifest: {len(m)-len(bad)}/{len(m)} valid')
if bad: print('Invalid files:',bad); sys.exit(3)
PY
mkdir -p STAGE_B_LOCAL_LOGS
printf 'Started UTC: '; date -u +%FT%TZ
printf 'Python: '; python3 --version
printf 'uv: '; uv --version
printf 'Running four historical environment builds.\n'
STAGE=B bash crossversion/build_envs.sh 2>&1 | tee STAGE_B_LOCAL_LOGS/build_envs.log
for version in 0.21.3 0.22.2.post1 1.8.0 1.9.1; do
  py="${VS_ENV_ROOT:-$HOME/versionshift_envs}/sk-$version/bin/python"
  if [[ -x "$py" ]]; then
    "$py" -c 'import sklearn,sys;print("OK",sys.executable,sklearn.__version__)'
  else
    echo "UNAVAILABLE: $version (see build log and BUILD_FAILURES.txt)"
  fi
done
if [[ ! -x "${VS_ENV_ROOT:-$HOME/versionshift_envs}/sk-0.22.2.post1/bin/python" || ! -x "${VS_ENV_ROOT:-$HOME/versionshift_envs}/sk-1.8.0/bin/python" || ! -x "${VS_ENV_ROOT:-$HOME/versionshift_envs}/sk-1.9.1/bin/python" ]]; then
  echo 'STOP: required historical environments unavailable. Do not run comparisons.'
  exit 4
fi
if ! python3 -c 'import numpy,pandas,sklearn,openml' 2>/dev/null; then
  echo 'STOP: controller Python lacks numpy, pandas, sklearn, or openml. Install controller dependencies in a separate modern venv.'
  exit 5
fi
args=()
for version in 0.21.3 0.22.2.post1 1.8.0 1.9.1; do
  py="${VS_ENV_ROOT:-$HOME/versionshift_envs}/sk-$version/bin/python"
  [[ -x "$py" ]] && args+=(--release "$version=$py")
done
python3 crossversion/controller.py introspect "${args[@]}" --out INTROSPECT_STAGE_B.json
python3 code/oracle_frame.py build --introspect INTROSPECT_STAGE_B.json --out ORACLE_FRAME.csv
printf '\nSTOP (intentional): verify and code all 78 oracle rows, externally timestamp, and freeze BEFORE viewing outcomes.\n'
printf 'After author verification: python3 code/oracle_frame.py freeze --frame ORACLE_FRAME.csv --out ORACLE_FROZEN.json\n'
printf 'Then follow docs/STAGE_B_PROTOCOL.md; do not use --unblinded-exploratory for confirmatory claims.\n'
