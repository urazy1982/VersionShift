# VersionShift — replication package

Code, frozen protocol and fold-level results for the paper

> Uraz Yavanoğlu. *When a Library Upgrade Changes the Verdict: Default-Configuration Drift and the Stability of Machine-Learning Comparisons.* Submitted to *Information and Software Technology*, 2026.

The study runs six scikit-learn classifiers (LR, SVC, kNN, RF, GB, HGB) on all 63 eligible OpenML-CC18 tasks. It uses frozen splits (6 × 5-fold CV) and two real upgrades of the library:

| Upgrade | Role | Mean shift in pairwise gaps *M* (90% CI) | Best classifier changed |
|---|---|---|---|
| 0.21.3 → 0.22.2.post1 (E+) | defaults changed | **5.01 pp** (3.72–6.30) | 25 of 63 tasks |
| 1.8.0 → 1.9.1 (E0) | negative control | 0.05 pp | 0 of 63 tasks |

The effect replicates on the 47 tasks that were not used in the 16-task pilot (5.41 pp). In post-review analyses, restoring the old effective defaults on the newer release reproduced all 7,560 archived prediction hashes for LR, SVC, RF and kNN, and the corrected explicit map E2 cut the six-estimator shift from 5.01 to 0.49 pp with no changed winners (`postreview/`).

## Contents

| Folder | What it is |
|---|---|
| `project/` | The study code, **v2.2.5**, with its frozen design. It contains `crossversion/` (the controller, the worker, and `build_envs.sh` for the per-release environments), `code/` (the analysis), `tests/`, `docs/` (the protocol and amendments), and the frozen changelog-coding frame (`ORACLE_*`). Every file listed in `SHA256_MANIFEST_V225.json` is byte-identical to the frozen version. |
| `pipeline/` | The **v2.2.6** run package that produced the Phase 1 results: the pre-registered amendment, the CPU completion kit, and the add-ons for pre-flight checks, independent verification and bundling. `finish/` holds the final analysis step and a checker for a rebuilt plan cache. |
| `results/` | The verified Phase 1 bundle. It has every fold-level result file with scores and prediction hashes (`STAGE_B_RUNS/*/CROSS_RELEASE_FOLDS.csv.gz`), the merged files and all analysis reports, the frozen plan and selections (`data/`), environment lock files and provenance (`envs/`), run logs, and the 16-task pilot reports (`STAGE_B_SCIENCE_V225/`). |
| `paper/` | `verify_results.py` checks the shipped results. `reproduce_paper.py` regenerates every number and figure in the manuscript from `results/`. |
| `postreview/` | Post-review diagnostics G0–G3 on the same frozen frame: independent recomputation, restoration of old effective defaults, boosting interventions, bootstrap and winner-margin sensitivity, and the corrected six-estimator explicit map E2 with its replay audit. All raw outputs are in `postreview/raw/`; `postreview/verify_postreview.py` re-runs every read-only analysis against them. See `postreview/README.md`. |

## 1. Check the results and regenerate the paper (minutes, no experiments)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 paper/verify_results.py     # hashes, A/A identity, independent recomputation -> ALL CHECKS PASSED
python3 paper/reproduce_paper.py    # writes paper/output/: PAPER_NUMBERS.json, CSVs, figures
```

`verify_results.py` runs these checks:
- It confirms the hashes of all files.
- It confirms that the two executions of each of the five A/A controls are bitwise identical.
- It recomputes the conclusion outcomes from the fold-level files with code independent of the analysis module, and compares them with the recorded summary. The outcomes are the best classifier, pair flips, material flips and mean-rank order, for the 63-task, 47-task hold-out and 16-task pilot frames on both upgrades.

## 2. Re-run the experiments

You need Linux (or WSL2 on Windows), [uv](https://docs.astral.sh/uv/), and a conda-forge Python 3.7 for scikit-learn 0.21.3, because uv cannot provide that interpreter. The pre-flight check asks for at least 5 GB of free disk. Phase 1 took roughly 6–10 hours with 10 single-threaded workers on an Intel Core Ultra 7 265K. No GPU is used.

The scripts expect this layout. Override it with `VS_PROJECT`, `VS_DATA` and `VS_CONTROLLER_VENV`:

```
/mnt/c/VersionShift_v2_2_5   -> project/          (ORACLE_FROZEN.json records this path)
~/versionshift_envs/         one environment per release (hard-coded by the kit)
~/versionshift_data/         plan cache, selections, runs
~/versionshift_controller/   controller venv: pip install -r requirements.txt
```

**Step 1. Build the four release environments.** Each build is hash-locked and fail-closed.

```bash
cd project
STAGE=B VS_PY37=/path/to/python3.7 bash crossversion/build_envs.sh
```

Compare the result with `results/envs/*/lock.txt` and `PROVENANCE.json`.

**Step 2. Rebuild the plan cache.** The datasets are not redistributed; download them from OpenML. Folds and seeds are derived deterministically from the dataset names.

```bash
python3 project/fetch_cc18_parallel_safe.py 8                      # warm the OpenML cache (8 connections)
python3 project/crossversion/controller.py freeze --source openml-cc18 --reps 6 \
        --plan-dir ~/versionshift_data/STAGE_B_PLAN_CACHED
python3 pipeline/finish/check_plan_cache.py --plan-dir ~/versionshift_data/STAGE_B_PLAN_CACHED --install
cp results/data/STAGE_B_SELECTION.json results/data/STAGE_C_HOLDOUT47_SELECTION.json ~/versionshift_data/
```

**Step 3. Run the Stage B pilot** (16 tasks, A/A control and both upgrades under policies D/E/P/S). Follow `project/docs/STAGE_B_PROTOCOL.md`, steps 4–6. The runs go to `~/versionshift_data/STAGE_B_RUNS/{AA_0213,EDGE_0213_0222,EDGE_180_190}` (plus their `_REVERSE` runs). `project/versionshift_run_stageb_science.sh` produces the pilot reports.

**Step 4. Run Phase 1.** This covers four A/A controls, Stage C on the 47 unseen tasks, and the 63- and 47-task analyses.

```bash
cd pipeline
sha256sum -c SHA256SUMS.txt
bash run_versionshift_wsl.sh check        # pre-flight, read-only
bash run_versionshift_wsl.sh verify       # must reproduce the pilot values in README_RUN_ON_WSL.md
bash run_versionshift_wsl.sh phase1       # use tmux; the run is resumable
bash finish/finish_phase1_analysis.sh     # lenses, emulation fidelity, assessment, verification, bundle
```

**Step 5. Regenerate the paper from your own bundle.**

```bash
python3 paper/reproduce_paper.py --results <unpacked bundle>
```

`phase2` and `phase3` in `run_versionshift_wsl.sh` (a 14-release scan and a fixed numerical-stack control) are optional extensions. The paper does not use them.

## Notes on provenance

- Frozen files are never edited. Changes after the freeze are additive and disclosed in `pipeline/AMENDMENT_V226_PRE_RUN.md` (frozen before any Phase 1 job; hash in `results/data/AMENDMENT_V226_FREEZE.json`). The pre-flight check refuses to run if a frozen file differs.
- The paper reports two defects found in the frozen kit, both worked around without changing it:
  - **`combine()` re-run.** The kit's `combine()` fails when `analyze` is re-run; it reads a gzip temp file as plain text. `pipeline/addons/vs_finish_analyze.py` replaces only that function, with explicit gzip handling.
  - **Lens S on E0.** The lens for policy S cannot be computed on E0, because the coverage check counts only default-policy jobs. The finishing step instead proves that policy S equals policy D on that upgrade, bitwise on all 22,680 folds, and records it in `results/STAGE_C_CPU_COMPLETION/reports/LENSES_NEG_S_EXPLORATORY.json`.
- Policy E (an explicit hyperparameter map) cannot be applied to HGB on E+. The comparison of policies D and E therefore uses the other five estimators; see `paper/reproduce_paper.py`, function `policy_e_vs_d`.
- Provenance records keep the absolute paths of the original machine (for example `/home/<user>/versionshift_envs`). They are left as recorded because they are covered by hashes.

## Frozen archive

The original-study part of this repository (everything except `postreview/`) was archived before the post-review work as `VersionShift_replication.zip`, SHA-256 `5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d`. That archive is attached to the first release, and the post-review kits accept only that exact file.

## Citation

See `CITATION.cff`. Please cite the paper once it is published.

## License

Code: MIT (see `LICENSE`). Results and documentation: CC BY 4.0. The OpenML-CC18 datasets are not included; they remain under their original OpenML terms.
