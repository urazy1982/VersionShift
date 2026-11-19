# VersionShift v3.0.5-results (post-results supplement)

This release adds the public, aggregate results of the VersionShift v3.0.5 study. It does not change any file of the frozen preregistration release **v3.0.5** (commit `2330e78171fad7bda3d8bc69a3698204ad35a545`, published 2026-11-17 16:40:51 UTC, protocol SHA-256 `7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55`).

Added:
- `supplement/v305/`: aggregate results (primary family, historical edge, run inventory, modern task-level conclusions, margin-qualified winners, per-estimator shifts, D0/D1/D2 contrasts, P2 sensitivity, external frame, all 13 E5 edges, typed unsupported/removed records), analysis code, figures and verifiers.
- `supplement/v305/tools/reproduce_primary_inference.py`: recomputes the registered primary tests from the 48 published task-level values.
- `.github/README.md` (landing page), `CITATION.cff`.

Verification: `cd supplement/v305 && python3 verify_all.py` → `ALL_PUBLIC_CHECKS_PASS`.

Not included: raw data, matrices, labels, predictions, environment binaries, private author records.

Manuscript: "VersionShift: Measuring the Impact of Dependency Upgrades on Machine-Learning Experiments" (under submission). The earlier v2.6 archive (withdrawn manuscript) remains at https://doi.org/10.5281/zenodo.24080651.
