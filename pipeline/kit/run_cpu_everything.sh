#!/usr/bin/env bash
# CPU-only phases A/A, Stage C 47+63, RQ3/RQ4 14-release, fixed-stack.
# Stops on any missing historical interpreter or failed frozen-integrity gate.
# Stage D still needs externally prespecified real artifacts; cannot invent them.
set -Eeuo pipefail
cd /mnt/c/VersionShift_v2_2_5
source "$HOME/versionshift_controller/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OPENML_SKIP_PARQUET=true

python3 versionshift_cpu_finish.py prepare
python3 -u versionshift_cpu_finish.py run --workers 10
python3 -u versionshift_cpu_finish.py analyze

# Builds only missing, genuinely distinct historical runtime stacks. Preserves existing locks.
export VS_PY37="$HOME/anaconda3/envs/versionshift_py37/bin/python"
STAGE=ALL bash crossversion/build_envs.sh
python3 -u versionshift_cpu_finish.py longitudinal --workers 10
python3 -u versionshift_cpu_finish.py analyze-long
python3 -u versionshift_cpu_finish.py fixedstack --workers 10
python3 -u versionshift_cpu_finish.py analyze-fixed

echo 'INTERNAL CPU STUDY COMPLETED. External timestamp/recode and Stage D real artifacts remain independent requirements.'
