#!/usr/bin/env bash
# VersionShift v2.1 — one isolated, HASH-LOCKED environment per scikit-learn release (uv).
#  * era-matched resolution with --exclude-newer, compiled to a lock file with wheel hashes
#  * install with --require-hashes from the lock (no re-resolution)
#  * FAIL CLOSED: the environment is renamed *.FAILED if any installed version differs from the lock
#    or scikit-learn differs from the requested release; failures are logged, never substituted
#  * provenance: lock, pip freeze, uv version, OS, CPU, threadpool/BLAS info, compiler of the interpreter
set -euo pipefail
ROOT="${VS_ENV_ROOT:-$HOME/versionshift_envs}"
mkdir -p "$ROOT"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 2; }
uv --version > "$ROOT/UV_VERSION.txt"; uname -a > "$ROOT/UNAME.txt"; (lscpu || true) > "$ROOT/LSCPU.txt" 2>/dev/null

# release  python  exclude-newer   (VERIFY dates against PyPI upload times before freezing)
read -r -d '' RELEASES <<'EOF' || true
0.21.3        3.7   2019-10-01
0.22.2.post1  3.8   2020-05-15
0.23.2        3.8   2020-10-15
0.24.2        3.9   2021-06-15
1.0.2         3.9   2022-02-28
1.1.3         3.10  2022-12-31
1.2.2         3.10  2023-05-31
1.3.2         3.11  2023-12-31
1.4.2         3.11  2024-06-15
1.5.2         3.12  2024-11-15
1.6.1         3.12  2025-03-15
1.7.2         3.12  2025-11-15
1.8.0         3.13  2026-02-15
1.9.1         3.13  2026-11-01
EOF

build() {   # name python date requirement-lines...
  local NAME="$1" PY="$2" DATE="$3"; shift 3
  local DEST="$ROOT/sk-$NAME"
  if [[ -e "$DEST" || -e "$DEST.FAILED" ]]; then echo "exists, skipping: $DEST"; return; fi
  local WORK; WORK="$(mktemp -d)"; printf '%s\n' "$@" threadpoolctl > "$WORK/requirements.in"
  echo "== $NAME (python $PY, index as of $DATE)"
  if ! uv pip compile --python-version "$PY" --exclude-newer "${DATE}T00:00:00Z" --generate-hashes \
        "$WORK/requirements.in" -o "$WORK/lock.txt"; then
    echo "$NAME LOCK_FAILED" >> "$ROOT/BUILD_FAILURES.txt"; return; fi
  # 0.21.3 needs CPython 3.7 (cp37 manylinux1 wheel, PyPI upload 2019-07-30). uv supports 3.7 at Tier 2;
  # if no managed 3.7 is available, point VS_PY37 at a system/deadsnakes/conda python3.7.
  local PYREQ="$PY"; if [[ "$PY" == "3.7" && -n "${VS_PY37:-}" ]]; then PYREQ="$VS_PY37"; fi
  if ! uv venv --python "$PYREQ" "$DEST"; then
    echo "$NAME PYTHON_UNAVAILABLE" >> "$ROOT/BUILD_FAILURES.txt"
    [[ "$NAME" == "0.21.3" ]] && echo "0.21.3 FALLBACK: analyse 0.21->0.22 as EMULATION-ONLY (--emulation-only); real series starts at 0.22.2.post1" >> "$ROOT/BUILD_FAILURES.txt"
    return; fi
  if ! VIRTUAL_ENV="$DEST" uv pip install --require-hashes --no-deps -r "$WORK/lock.txt"; then
    mv "$DEST" "$DEST.FAILED"; echo "$NAME INSTALL_FAILED" >> "$ROOT/BUILD_FAILURES.txt"; return; fi
  cp "$WORK/requirements.in" "$WORK/lock.txt" "$DEST/"
  VIRTUAL_ENV="$DEST" uv pip freeze > "$DEST/FREEZE.txt"
  # fail closed: every locked name==version must be what is installed; sklearn must be the requested release
  if ! "$DEST/bin/python" - "$DEST/lock.txt" "$DEST/FREEZE.txt" "$1" <<'PY'
import re, sys
lock = dict(re.findall(r'^([A-Za-z0-9_.\-]+)==([^\s\\;]+)', open(sys.argv[1]).read(), re.M))
frz = dict(re.findall(r'^([A-Za-z0-9_.\-]+)==([^\s;]+)', open(sys.argv[2]).read(), re.M))
norm = lambda d: {k.lower().replace('_', '-'): v for k, v in d.items()}
lock, frz = norm(lock), norm(frz)
bad = {k: (v, frz.get(k)) for k, v in lock.items() if frz.get(k) != v}
import sklearn
want = sys.argv[3].split('==')[-1]
if sklearn.__version__ != want: bad['scikit-learn(runtime)'] = (want, sklearn.__version__)
if bad: print('VERSION MISMATCH', bad); sys.exit(1)
PY
  then mv "$DEST" "$DEST.FAILED"; echo "$NAME VERSION_MISMATCH" >> "$ROOT/BUILD_FAILURES.txt"; return; fi
  "$DEST/bin/python" -c "import sys,sysconfig,platform,json;from threadpoolctl import threadpool_info;import sklearn,numpy,scipy;print(json.dumps(dict(python=sys.version,compiler=platform.python_compiler(),cc=sysconfig.get_config_var('CC'),machine=platform.machine(),sklearn=sklearn.__version__,numpy=numpy.__version__,scipy=scipy.__version__,threadpools=threadpool_info()),default=str,indent=1))" > "$DEST/PROVENANCE.json"
  rm -rf "$WORK"; echo "OK $NAME"
}

# STAGE=B builds only the four Stage-B environments (audit recommendation); default builds all 14 + controls.
STAGE_B_SET=" 0.21.3 0.22.2.post1 1.8.0 1.9.1 "
while read -r REL PY DATE; do
  [[ -z "$REL" ]] && continue
  if [[ "${STAGE:-}" == "B" && "$STAGE_B_SET" != *" $REL "* ]]; then continue; fi
  build "$REL" "$PY" "$DATE" "scikit-learn==$REL"
done <<< "$RELEASES"

[[ "${STAGE:-}" == "B" ]] && { echo "STAGE=B: four Stage-B environments only"; exit 0; }
# Numeric-stack control: two scikit-learn releases with numpy/scipy held fixed.
build "1.4.2-fixedstack" 3.11 2024-11-15 "scikit-learn==1.4.2" "numpy==1.26.4" "scipy==1.12.0"
build "1.5.2-fixedstack" 3.11 2024-11-15 "scikit-learn==1.5.2" "numpy==1.26.4" "scipy==1.12.0"
echo "Done. Environments: $ROOT/sk-*; failures (never substituted): $ROOT/BUILD_FAILURES.txt"
