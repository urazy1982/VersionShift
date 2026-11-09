#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/VersionShift_v2_2_5
source "$HOME/versionshift_controller/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
python3 versionshift_cpu_finish.py prepare
python3 -u versionshift_cpu_finish.py run --workers 10
python3 -u versionshift_cpu_finish.py analyze
