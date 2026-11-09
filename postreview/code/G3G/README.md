# VersionShift G3-G — server audit status

## Completed

A fresh 20,000-replicate, paired dataset-bootstrap sensitivity calculation was run on the uploaded per-dataset D/E2 summary. Dataset identities were aligned before resampling. The primary point estimates reproduce the provided six-estimator report.

| Frame | D shift (pp) | E2 shift (pp) | Paired reduction (pp) | Relative reduction | 95% bootstrap CI: paired reduction (pp) | 95% bootstrap CI: relative reduction |
|---|---:|---:|---:|---:|---:|---:|
| Pilot 16 | 3.8116 | 0.3818 | 3.4298 | 89.98% | [2.1511, 4.8890] | [76.33%, 98.90%] |
| Holdout 47 | 5.4136 | 0.5297 | 4.8839 | 90.22% | [3.1499, 6.8887] | [82.47%, 95.63%] |
| Full 63 | 5.0067 | 0.4921 | 4.5146 | 90.17% | [3.1567, 6.0716] | [83.49%, 95.33%] |

The full-frame E2 mean has 95% paired-dataset-bootstrap percentile interval [0.2315, 0.8078] pp. The intervals describe sensitivity to resampling tasks; they do not establish a population guarantee beyond the 63 eligible tasks. They are different from the frozen manuscript's original parametric intervals.

## Still unverified

The server did **not** receive the raw G3-E HGB job files (`jobs/*.json`, `results/*.json`, `results/*.npz`) or the original frozen dataset/fold NPZ files from the local working environment. Hence it is **not scientifically valid** to declare independent fitted-parameter provenance and fold replay complete. The previously reported 126 successful HGB jobs indicate local verification by the G3-E runner, which checks parameter values, job/worker hashes, 30-fold coverage, and recomputed scores; the current server audit has not independently repeated those checks.

The included `g3g_verify_local.py` performs the missing independent read-only audit in the Windows/WSL environment. It can also audit copies of the same folders if later made accessible here.
