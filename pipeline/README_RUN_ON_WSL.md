# VersionShift — WSL run package (v2.2.6, 2026-11-07)

This package contains:
- the CPU completion kit, **unchanged** (`kit/`, hashes identical to the kit you uploaded);
- an additive amendment, frozen before any new job runs;
- five add-on scripts:
  - pre-flight checks;
  - an amendment freeze with a record of what existed at freeze time;
  - independent verification from the raw outputs;
  - the missing fixed-stack checks;
  - a results bundle to upload.

Nothing here modifies the frozen files: `controller.py`, `worker.py`, `analysis_v221.py`, the plan, the Stage B selection and the oracle. GPU is not used.

## What the phases produce

| Phase | Runs | Jobs | Rough wall time, 10 workers* | Needed for the paper? |
|---|---|---|---|---|
| **phase1** | 4 A/A controls; Stage C on the 47 new datasets × both upgrades × D/E/P/S (+ reverse P); 63-frame and 47-holdout analysis | 6,408 | ~6–10 h | **Yes** — replication section |
| phase2 | builds 14 historical environments; 14-release scan × D/E on 47 datasets; oracle H/S1/S2 outcomes | 7,896 | ~10–14 h, plus environment builds | Optional (RQ3/RQ4) |
| phase3 | fixed NumPy/SciPy stack, 1.4.2 vs 1.5.2 × D/E on 47 datasets | 1,128 | ~1.5–2 h | Optional (attribution) |

\*Extrapolated from your Stage B job timings: about 18 CPU-hours for 16 datasets. Actual time depends on dataset sizes.

## Expected layout (same as Stage B)

```
/mnt/c/VersionShift_v2_2_5/            project (v2.2.5; ORACLE_FROZEN.json expects this exact path)
~/versionshift_data/STAGE_B_PLAN_CACHED/   PLAN.json + all 63 *.npz and *_folds.npz
~/versionshift_data/STAGE_B_SELECTION.json
~/versionshift_data/STAGE_B_RUNS/          AA_0213, EDGE_0213_0222(_REVERSE), EDGE_180_190(_REVERSE)
~/versionshift_envs/sk-0.21.3, sk-0.22.2.post1, sk-1.8.0, sk-1.9.1
~/versionshift_controller/                 controller venv (numpy, pandas, scipy)
```

The kit hard-codes `~/versionshift_envs`, so keep the environments there. The other paths can be overridden with `VS_PROJECT`, `VS_DATA` and `VS_CONTROLLER_VENV`.

## Steps

**0. Unpack** this ZIP to `C:\VersionShift_WSL_Package`, then open a WSL terminal:

```bash
cd /mnt/c/VersionShift_WSL_Package
sha256sum -c SHA256SUMS.txt          # every line must say OK
```

**1. Pre-flight check.** This is read-only and changes nothing.

```bash
bash run_versionshift_wsl.sh check
```

Every line should be PASS. A WARN is acceptable. Fix every FAIL before going on. The most likely one is a missing plan `.npz`, because the uploaded snapshot's plan folder lacked 4 of the 63 datasets. Your `STAGE_B_PLAN_CACHED` on WSL should have all of them.

**2. Self-test on your real Stage B data.** This is also read-only.

```bash
bash run_versionshift_wsl.sh verify
```

The values must match these lines exactly (spacing may differ):

```
AA_0213          A/A PASS: 96 pairs, 0 differ
EDGE_0213_0222   0.21.3->0.22.2.post1: K=16 M=3.8116037 pp | >0.5pp 15 | best changes 6 | flips 47/240 (material 15)
EDGE_180_190     1.8.0->1.9.1: K=16 M=0.0383746 pp | >0.5pp 0 | best changes 0 | flips 0/240 (material 0)
```

These are the values I computed from the uploaded evidence. If yours differ, stop and send me the output.

**3. Phase 1.** Use `tmux` so that closing the terminal doesn't kill the run.

```bash
tmux new -s vs
cd /mnt/c/VersionShift_WSL_Package
bash run_versionshift_wsl.sh phase1
```

The script runs these steps in order:
- pre-flight;
- installs the three kit scripts into the project root (it refuses to overwrite a different existing file);
- freezes the amendment and prints its **SHA-256**.

At that point it **pauses**. Send the SHA-256 somewhere off this machine: an e-mail to yourself (and ideally a colleague), a commit pushed to a private GitHub repository, or OSF. Then type `YES`. The run then continues with the kit's `prepare`, `run` (4 A/A runs, then the Stage C runs, each collected and replayed), `analyze`, independent verification, and the bundle.

Detach from tmux with `Ctrl-b d` and reattach with `tmux attach -t vs`. To watch progress from another terminal:

```bash
source ~/versionshift_controller/bin/activate
watch -n 30 'python3 /mnt/c/VersionShift_v2_2_5/versionshift_cpu_finish.py status; free -h | grep -E "Mem|Swap"'
```

**If it is interrupted** (reboot, crash, Ctrl-C), run exactly `bash run_versionshift_wsl.sh phase1` again. It resumes and does not repeat completed jobs. The amendment step reports "ALREADY FROZEN"; type `YES` again, or set `VS_TIMESTAMP_DONE=1`.

**4. Upload** the ZIP printed at the end, `C:\VersionShift_v2_2_5\VersionShift_Results_phase1_<date>.zip`, in the chat. It contains:
- the reports and merged fold CSVs;
- every run's lock, summary and fold CSV;
- the A/A identity files, the amendment and its freeze record;
- the pre-flight and verification reports, logs, and environment provenance.

It does not contain the raw `.npz` prediction files. Those can be packaged with `python3 addons/vs_make_upload_bundle.py --with-raw`.

## Optional phases

```bash
export VS_PY37="$HOME/anaconda3/envs/versionshift_py37/bin/python"   # only if 0.21.3 must be rebuilt
bash run_versionshift_wsl.sh phase2     # environment build + 14-release scan + oracle H/S1/S2
bash run_versionshift_wsl.sh phase3     # fixed-stack control (needs envs from phase2's build)
```

Phase 2 stops if any of the 14 historical environments cannot be built. Check `~/versionshift_envs/BUILD_FAILURES.txt`, and never substitute modern wheels. Phase 3 stops if the fixed-stack check fails (wrong numpy/scipy, missing jobs, wrong datasets).

## Rules (from the protocol and the amendment)

- Run only one copy at a time. A lock file enforces this.
- Do not delete run directories or `RUN_LOCK.json`. Do not change folds, seeds or datasets. Do not re-run completed jobs under other seeds.
- If a job fails, keep its logs and results. Failures and non-portable configurations are recorded outcomes.
- If `analyze` stops on G2 or G7, do not force it. Send me the log.
- The delayed 25% blinded oracle recoding cannot be done before **2026-11-20**. Stage D needs 5–8 real, pre-selected notebooks or scripts. Neither is part of this package.

## Files

| File | Purpose |
|---|---|
| `run_versionshift_wsl.sh` | single entry point: `check`, `verify`, `phase1`, `phase2`, `phase3`, `bundle` |
| `AMENDMENT_V226_PRE_RUN.md` | additive pre-run amendment (A1 P-mitigation correction; A2 conclusion-level outcomes O1–O4; A3 oracle S1/S2; A4 fixed-stack checks; A5 A/A; A6 verification) |
| `addons/vs_preflight.py` | read-only checks of hashes, plan artifacts, Stage B evidence, interpreters, resources |
| `addons/vs_freeze_amendment.py` | write-once freeze with SHA-256 and the state of the run directories |
| `addons/vs_verify_independent.py` | recomputes M, margins, O1–O4 and A/A identity from raw folds without `analysis_v221.py`; cross-checks the kit |
| `addons/vs_oracle_score_outcome.py` | Phase 2 oracle outcomes H, S1, S2 with precision and recall |
| `addons/vs_fixedstack_verify.py` | Phase 3 lock, inventory and numpy/scipy checks |
| `addons/vs_make_upload_bundle.py` | results ZIP with a SHA-256 manifest |
| `kit/` | original CPU completion kit, unchanged |
