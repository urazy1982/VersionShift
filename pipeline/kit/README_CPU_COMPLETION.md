# VersionShift v2.2.5 — CPU completion kit

**Scope:** finish all currently specified and executable *internal* CPU experiments without changing `controller.py`, `worker.py`, the 63-dataset frozen plan, Stage B selection, or the frozen 78-case oracle. This is an additive, resumable execution amendment, not a new confirmatory preregistration.

**No GPU is needed.** The project uses six CPU scikit-learn classifiers. The user's RTX 5090 can remain idle.

## Scientific boundary (read before executing)

- Original OpenML-CC18 plan contains **63 eligible datasets** (PLAN hash `523c42a3dabee44e66d43d538b228939b01fd7fe1b68285f542f0980906492f4`).
- Stage B selection was **16 datasets** (selection hash `55554df012a30d7000d3b5840babd23f8e10b9daca57b10e8cf6099fce92943c`). They have already been analysed.
- Previously unseen complement is **47 datasets**, **below** the prespecified `K >= 48` floor. The protocol's pilot-reuse fallback allows a **63-dataset finite-frame analysis** with pilot-reuse disclosure and a separate **47-dataset pilot-excluded sensitivity**. Do not call the 63 an untouched confirmatory sample.
- Frozen oracle frame SHA-256: `33f04fb6c6da89654f3524cfaef290644b80e37afcd5148dcffa740e9bcfe2f6`. Local `ORACLE_FROZEN.json` is insufficient by itself to prove an external timestamp or author confirmation. Those controls remain outstanding unless you have retained independent proof.
- The 25% delayed blinded intra-rater oracle recoding must be performed **no sooner than 20 November 2026**, 14 days after the 6 November freeze; that future check cannot be truthfully completed earlier.
- A 5–8-artifact Stage D external validation requires *prespecified real notebooks or comparison scripts*. None are included in this kit. Do not invent external artifacts or substitute synthetic notebooks.
- The 14-release longitudinal and fixed NumPy/SciPy-stack controls are separate CPU-only phases requiring genuine historic environments. These phases are non-GPU and may be time-intensive.

## Installation (WSL Ubuntu)

Extract ZIP into `C:\\VersionShift_v2_2_5`, or another temporary folder, then copy these three scripts into the root of the existing project **without modifying the frozen originals**:

```bash
cd /mnt/c/VersionShift_v2_2_5
cp -n VersionShift_CPU_Completion_Kit/versionshift_cpu_finish.py .
cp -n VersionShift_CPU_Completion_Kit/versionshift_parallel_10_aa.py .
cp -n VersionShift_CPU_Completion_Kit/versionshift_policy_descriptives.py .
source ~/versionshift_controller/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
```

If a script already exists, compare hashes and resolve consciously; do not overwrite it blindly. The kit's scripts are kept alongside the original controller and worker and record immutable run locks.

### Phase 1 — Complete all four A/A controls and the 47-dataset historical confirmatory-candidate collection

```bash
python3 versionshift_cpu_finish.py prepare
python3 versionshift_cpu_finish.py status
python3 -u versionshift_cpu_finish.py run --workers 10 \
  2>&1 | tee "$HOME/versionshift_cpu_completion.log"
python3 -u versionshift_cpu_finish.py analyze \
  2>&1 | tee "$HOME/versionshift_cpu_analysis.log"
```

`run` is resumable at the run-directory level; re-run the same command after interruption. **Never launch a second scheduler on the same output directory.** Run each phase after the preceding command exits successfully. `analyze` requires all four A/A controls to be completed and replayed.

Runs executed and collected:

- Four `AA_<release>` runs, each 16 datasets × two same-interpreter labels × six algorithms × D; bitwise label arrays verified.
- `STAGE_C47_POS` 47 datasets × (0.21.3,0.22.2.post1) × six algorithms × policies D/E/P/S; reverse P in `STAGE_C47_POS_REVERSE`.
- `STAGE_C47_NEG` 47 datasets × (1.8.0,1.9.1) × six algorithms × policies D/E/P/S; reverse P in `STAGE_C47_NEG_REVERSE`.
- Each run is checked with the original `controller.py collect` and `replay`.

Phase 1 schedules up to **6,408 job records** (768 A/A + 5,640 new historical-policy jobs), each running the frozen 6×5 folds (A/A 6×5, too). Nonportable P and S configurations count as recorded outcomes, not successful fits. The runner never silently omits them.

`analyze` produces `~/versionshift_data/STAGE_C_CPU_COMPLETION/` including:

- The merged full-frame 63-dataset descriptive analysis with both primary edges and original Holm logic.
- The 47-dataset never-seen subframe sensitivity, kept explicitly separate.
- D/E/P/S descriptive mitigation results, strict P compatibility and conditional mitigation, and four-cell identifiability decomposition.
- Secondary lenses B/C/D explicitly labelled exploratory; lens C's known FDR inflation is not hidden.
- 16-dataset default-regime emulator (on modern sklearn) and a **qualified** G5 fidelity diagnostic (emulated R19 is not exactly old 0.22.2 defaults).
- A scientific status/gate report and SHA-256 manifest of analytical outputs.

If G2/G7 fails, the original frozen analysis refuses; do not reduce folds or substitute datasets to force completion.

### Phase 2 — Longitudinal RQ3/RQ4 scan over all 14 releases

**Only after Phase 1:** build the remaining historical CPU environments and fixed numerical-stack controls:

```bash
cd /mnt/c/VersionShift_v2_2_5
export VS_PY37="$HOME/anaconda3/envs/versionshift_py37/bin/python"
STAGE=ALL bash crossversion/build_envs.sh
```

Examine `~/versionshift_envs/BUILD_FAILURES.txt`, individual `FREEZE.txt` and `PROVENANCE.json`. Do not substitute modern wheels for missing historic versions. When all 14 real historical interpreters exist:

```bash
python3 -u versionshift_cpu_finish.py longitudinal --workers 10 \
  2>&1 | tee "$HOME/versionshift_longitudinal.log"
python3 versionshift_cpu_finish.py analyze-long
```

This runs 47 previously unseen datasets × 14 genuine releases × six estimators × D/E = **7,896 job records**, and records all 13 adjacent edges. It compares the frozen 78-case changelog classifications with observed differences in **hard prediction hashes** and in mean algorithm scores. Strict/lenient positive definitions and coverage are reported. This is exploratory RQ3/RQ4 evidence; selected coded-change confirmatory edges must already have been registered, not selected post hoc.

### Phase 3 — Fixed numerical-stack control

```bash
python3 -u versionshift_cpu_finish.py fixedstack --workers 10 \
  2>&1 | tee "$HOME/versionshift_fixedstack.log"
python3 versionshift_cpu_finish.py analyze-fixed
```

Requires `sk-1.4.2-fixedstack` and `sk-1.5.2-fixedstack`, each with NumPy 1.26.4 and SciPy 1.12.0 recorded in the lock. This adds **1,128 CPU job records** on 47 datasets. The measured effect is a **fixed-numeric-stack control**, not automatic causal attribution to sklearn in all other experiments.

## One-command internal CPU execution

After copying the scripts to the project root (installation above), run the provided `run_cpu_everything.sh` from the project root:

```bash
cp -n VersionShift_CPU_Completion_Kit/run_cpu_everything.sh .
bash run_cpu_everything.sh 2>&1 | tee "$HOME/versionshift_cpu_all.log"
```

It first completes Phase 1 and only then attempts environment builds and Phases 2–3. If a missing historical Python interpreter blocks a phase, the command stops without deleting completed results; fix that environment and re-run the identical command to resume. The scientific status remains qualified as described at the top. **No GPU involved.**

## Live progress (another WSL terminal)

```bash
cd /mnt/c/VersionShift_v2_2_5
source ~/versionshift_controller/bin/activate
watch -n 10 'python3 versionshift_cpu_finish.py status; free -h | grep -E "Mem:|Swap:"'
```

Monitor swap and disk. The 10-worker concurrency has already been tested on the local 20-core machine, but memory requirements vary by dataset. If a worker fails, preserve its artifacts and logs; never delete the run-lock or repeat a completed job under different seeds.

## Postprocessing and handoff

```bash
cd ~/versionshift_data
zip -qr /mnt/c/VersionShift_v2_2_5/VersionShift_Completed_CPU_Reports.zip \
  STAGE_C_CPU_COMPLETION STAGE_C_HOLDOUT47_SELECTION.json \
  STAGE_C_COMPLETION_PROTOCOL_RECORD.json
```

Upload this reports-only ZIP (or direct JSON/CSV files) and accompanying logs for an independent scientific audit. Full raw predictions can be packaged separately and verified against individual job SHA-256 hashes.

## What this kit does NOT claim

No access from this chat to your private WSL CPU runtime; the scripts must be run there. Until the terminal reports and output files are returned, **Stage C, the 14-release scan, the fixed-stack control, the delayed oracle recode, and Stage D external validation are not executed or completed**. A valid frozen local oracle does not prove the required external public timestamp. No GPU will resolve those prerequisites.
