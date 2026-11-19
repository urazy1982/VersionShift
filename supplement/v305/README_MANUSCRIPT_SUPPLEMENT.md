# VersionShift v3.0.5 — post-results public supplement

Supplement to the manuscript **"VersionShift: Measuring the Impact of Dependency Upgrades on Machine-Learning Experiments"** (U. Yavanoğlu). It is added after the outcome runs and does not modify or replace the prospectively frozen protocol:

- Frozen protocol (commit-pinned): https://github.com/urazy1982/VersionShift/blob/2330e78171fad7bda3d8bc69a3698204ad35a545/runs/plan/FROZEN_PROTOCOL.json
- SHA-256 `7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55`; GitHub release `v3.0.5` published 2026-11-17 16:40:51 UTC (a public dated commitment, not a notarisation).

## Contents

| Path | What it is |
|---|---|
| `results/*.csv`, `results/validated_summary.json` | Primary family, historical edge (63 tasks, P0/P1/P2 × D0/D1, era and fixed stack), run inventory (33 groups, 29,472 jobs), earlier v2.2.5 pilot values |
| `results/secondary/` | Modern task-level M_t and winners (pilot, evaluation, external; P1/P2), margin-qualified winners, per-estimator shifts, matched D0/D1 contrasts, D2 mechanism contrasts, P2 sensitivity, external frame, all 13 E5 edges, all 33 D0 summaries, typed unsupported/removed reasons, full extracted statistics (JSON) |
| `analysis_code/` | The statistics module used for the registered analysis and the extraction script that produced `results/secondary/` from the private archive (it needs that archive to run) |
| `tools/` | Verifiers (see below) |
| `figures/` | Summary figures; `figures/manuscript/` holds the manuscript figures |
| `docs/` | Claim ladder, reproducibility levels, literature sources, integration notes |

## Verify

```bash
cd supplement/v305
python3 verify_all.py
```

`verify_all.py` runs the frozen v3.0.5 verifier unchanged (in a temporary copy of the frozen file set), the companion SHA-256 manifest, aggregate-summary checks, secondary-table consistency checks and, if `numpy` and `scipy` are installed (`pip install -r analysis_code/ANALYSIS_REQUIREMENTS.txt`), `tools/reproduce_primary_inference.py`, which recomputes the two primary effects from the 48 published task-level values: M, one-sided t-test p, Holm-adjusted p, the one-sided lower bound and the seeded 20,000-resample bootstrap interval. No model is fitted and no GPU is needed.

What cannot be done from public files: recompute fold scores from predictions, or re-run training (see `docs/REPRODUCIBILITY_LEVELS_AND_PENDING_EXPORTS.md`).

## Interpretive restrictions

- Tasks form a finite allocated frame; the two modern edges were selected because they contain documented changes. No prevalence claim about upgrades in general.
- Secondary results are descriptive; the only inferential family is the two-edge primary test.
- Historical D1 (E2) results are outcome-informed; modern D1 is a seven-estimator comparison (HGB's E2 setting is rejected by modern releases).
- P1/P2 and D0/D1 differences are conditional contrasts, not causal shares.
- The zero stack contrast concerns hard-label balanced accuracy only.
- UNSUPPORTED and REMOVED_BEHAVIOR records are structural outcomes, never zeros.
- Earlier versions (v2.2.5 pilot; v2.6, an unpublished and withdrawn manuscript archived at https://doi.org/10.5281/zenodo.24080651) are stages of the same programme and are not pooled with v3.0.5.

## Not included

Raw OpenML data, feature matrices, labels, per-fold predictions, private author records and environment binaries.
