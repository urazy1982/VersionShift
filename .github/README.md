# VersionShift

**Measuring the impact of dependency upgrades on machine-learning experiments.**

VersionShift is a controlled study of what scikit-learn upgrades do to the results and conclusions of standard classifier comparisons. Data, folds, seeds, preprocessing and experiment code are held fixed; only the library release changes.

> Manuscript: U. Yavanoğlu, *VersionShift: Measuring the Impact of Dependency Upgrades on Machine-Learning Experiments* (under submission). Please cite this repository via `CITATION.cff`.

## Main results (v3.0.5)

| Upgrade | Changed default | Mean pairwise gap shift M (95% CI) | Holm p (H0: M ≤ 0.5 pp) | Winners changed (48 tasks) |
|---|---|---|---|---|
| 1.4.2 → 1.5.2 | `LinearSVC` `dual` | 0.995 pp (0.637–1.400) | 0.0082 | 3 (0 with >0.5-pp leads) |
| 1.5.2 → 1.6.1 | AdaBoost SAMME.R removed | 2.053 pp (1.337–2.872) | 0.0003 | 5 (1 with >0.5-pp leads) |

- Each upgrade's drift came entirely from the estimator whose default changed. Pinning its configuration removed the drift on the seven-estimator panel the explicit map could express.
- Re-running 0.21.3 → 0.22.2.post1 on 63 tasks reproduced the archived drift (5.03 pp, 25 changed winners). Across 13 consecutive releases this was the only large shift.
- Replacing the numerical stack (Python/NumPy/SciPy/BLAS) changed no prediction in 3,024 matched jobs.

Eight classifiers, 48 OpenML-CC18 evaluation tasks (plus 15 pilot and 20 external), 30 folds per task, 33 registered run groups, 29,472 jobs.

## Repository layout

| Location | Status |
|---|---|
| Repository root (`versionshift_v3/`, `config/`, `runs/plan/`, `environments/`, `scripts/`, `tests/`, `verify_release.py`, root `README.md`, …) | **Frozen before outcome runs.** Byte-identical to release [`v3.0.5`](https://github.com/urazy1982/VersionShift/releases/tag/v3.0.5), commit `2330e78`, protocol SHA-256 `7a62fb45…`. The root README describes the package as it was frozen, before any results existed. |
| [`supplement/v305/`](supplement/v305/) | **Post-results** aggregate results, analysis code, figures and verifiers. |
| `.github/README.md` (this page), `CITATION.cff`, `RELEASE_NOTES_v3.0.5-results.md` | Post-results documentation. |

## Verify in one command

```bash
cd supplement/v305
pip install -r analysis_code/ANALYSIS_REQUIREMENTS.txt   # numpy, scipy (only for the primary-inference step)
python3 verify_all.py
```

This checks the frozen release unchanged, every supplement file hash and the consistency of all aggregate tables, and recomputes the two primary tests (M, p, Holm p, bootstrap CI) from the 48 published task-level values. Nothing is trained.

Full re-execution needs the OpenML datasets, 14 historical environments and the frozen matrices; see `RUN_INSTRUCTIONS.md` at the repository root.

## Earlier versions

v2.2.5 (16-task pilot) and v2.6 (63-task historical study; manuscript withdrawn before publication, superseded by this study) are earlier stages of the same programme. The v2.6 archive is at https://doi.org/10.5281/zenodo.24080651.

## Licence and funding

Code: MIT (see `LICENSE`). Datasets are not redistributed and keep their OpenML licences. The work was funded by UAYTECH Corporation, which is owned by the author.
