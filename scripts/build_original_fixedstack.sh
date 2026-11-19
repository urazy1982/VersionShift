#!/usr/bin/env bash
# Feasibility setup for new old-edge attribution cell. Never substitutes package versions.
set -euo pipefail
OLD="${VS_OLD_0213_PYTHON:-$HOME/versionshift_envs/sk-0.21.3/bin/python}"
ROOT="${VS_OLDSTACK_ROOT:-$HOME/versionshift_v3_oldedge_stack}"
mkdir -p "$ROOT"
[[ -x "$OLD" ]] || { echo "STOP 0.21.3 environment missing: $OLD"; exit 2; }
"$OLD" - <<'PY'
import sys,numpy,scipy,sklearn
assert sys.version_info[:3]==(3,7,12)
assert numpy.__version__=='1.17.2' and scipy.__version__=='1.3.1' and sklearn.__version__=='0.21.3'
print('ORIGINAL_0213_STACK_VERIFIED')
PY
[[ -e "$ROOT/sk-0.21.3" ]] || ln -s "$(dirname "$(dirname "$OLD")")" "$ROOT/sk-0.21.3"
NEW="$ROOT/sk-0.22.2.post1"
if [[ ! -e "$NEW" ]]; then
    uv venv --python "$OLD" "$NEW"
    uv pip install --python "$NEW/bin/python" --only-binary=:all: \
        'numpy==1.17.2' 'scipy==1.3.1' 'scikit-learn==0.22.2.post1' 'joblib==0.14.1' 'threadpoolctl==2.1.0'
fi
"$NEW/bin/python" - <<'PY'
import sys,numpy,scipy,sklearn
assert sys.version_info[:3]==(3,7,12)
assert numpy.__version__=='1.17.2' and scipy.__version__=='1.3.1' and sklearn.__version__=='0.22.2.post1'
print('NEW_0222_ON_ORIGINAL_STACK_VERIFIED')
PY
uv pip freeze --python "$NEW/bin/python" > "$ROOT/oldstack_0222_FREEZE.txt"
sha256sum "$ROOT/oldstack_0222_FREEZE.txt" "$NEW/bin/python" > "$ROOT/oldstack_0222_SHA256.txt"
echo "OLD_EDGE_FIXED_STACK_ENV_READY $ROOT"
echo 'NEXT verify BLAS vendor and OpenMP consistency and run matching new G0 controls. Do not claim full causal isolation until BLAS is matched.'
