# VersionShift v2.2.2 Stage B Ubuntu handoff

This is a *new code amendment* relative to the v2.2.1 analysis freeze. The original `ANALYSIS_FREEZE_V221.json` remains an historical record and does not certify these edited files.

## Prerequisites
Ubuntu/WSL, `uv`, stable network for PyPI and OpenML-CC18, adequate CPU/RAM. Ensure current working directory is this package root. Real 0.21.3 requires CPython 3.7. Do not replace it silently.

## 1 — Build historical release environments

```bash
STAGE=B bash crossversion/build_envs.sh
cat "$HOME/versionshift_envs/BUILD_FAILURES.txt" 2>/dev/null || true
```

Build only those releases successfully resolved. If 0.21.3 is absent, label the positive historical edge *emulation-only* and do not claim binary evidence for 0.21→0.22.

## 2 — Freeze inputs, including oracle

```bash
python3 crossversion/controller.py introspect \
 --release 0.21.3="$HOME/versionshift_envs/sk-0.21.3/bin/python" \
 --release 0.22.2.post1="$HOME/versionshift_envs/sk-0.22.2.post1/bin/python" \
 --release 1.8.0="$HOME/versionshift_envs/sk-1.8.0/bin/python" \
 --release 1.9.1="$HOME/versionshift_envs/sk-1.9.1/bin/python" \
 --out INTROSPECT_STAGE_B.json
python3 code/oracle_frame.py build --introspect INTROSPECT_STAGE_B.json --out ORACLE_FRAME.csv
```

Manually substantiate all 78 rows with original release-note evidence; do not automatically mark the 30 draft rows confirmed. Then run:

```bash
python3 code/oracle_frame.py freeze --frame ORACLE_FRAME.csv --out ORACLE_FROZEN.json
python3 crossversion/controller.py freeze --source openml-cc18 --plan-dir PLAN_CC18
python3 code/stage_b.py select --plan-dir PLAN_CC18 --n 16 --seed 20261106 --out STAGE_B_SELECTION.json
```

If `select` fails because a stratum is undersized, report the exact eligible stratum counts and register a revised selection allocation **before** examining version outcomes. Do not silently borrow from other strata.

## 3 — Run Stage B on selected datasets only

First run A/A in each historical interpreter, then one edge at a time. Example for defaults D on the positive historical edge:

```bash
python3 crossversion/controller.py run --plan-dir PLAN_CC18 --out B_POS_D \
 --selection STAGE_B_SELECTION.json \
 --release 0.21.3="$HOME/versionshift_envs/sk-0.21.3/bin/python" \
 --release 0.22.2.post1="$HOME/versionshift_envs/sk-0.22.2.post1/bin/python" \
 --policies D
python3 crossversion/controller.py collect --plan-dir PLAN_CC18 --out B_POS_D
python3 crossversion/controller.py replay --plan-dir PLAN_CC18 --out B_POS_D
python3 code/analysis_v221.py edge --plan-dir PLAN_CC18 --selection STAGE_B_SELECTION.json \
 --inp B_POS_D/CROSS_RELEASE_FOLDS.csv.gz \
 --edges 0.21.3:0.22.2.post1 --primary 0.21.3:0.22.2.post1 \
 --population stratified --oracle-frozen ORACLE_FROZEN.json --out B_POS_D/EDGE_REPORT.json
```

Repeat for negative control (1.8.0→1.9.1). For pinning run P and PT separately with explicit `--pin-from` and reference-release inclusion; use distinct output directories. The 2×2 attribution requires reverse pinning, documented failures, and all four cells; never infer missing cells.

## 4 — Stop/go

Inspect G1–G7 from `docs/STAGE_B_PROTOCOL.md`. Stage B is exploratory. Stage C remains unauthorized until the actual binary outcomes and oracle fidelity are reviewed.
