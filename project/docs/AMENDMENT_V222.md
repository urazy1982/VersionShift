# VersionShift v2.2.2 — limited integrity amendment (2026-11-06)

Preserves the original v2.2.1 ZIP unchanged. These changes are new and were introduced before actual cross-version outcome analysis.

1. `controller.py run --selection STAGE_B_SELECTION.json` executes only the frozen Stage B task subset, locks the selection hash, and makes collect/replay use that subset.
2. `analysis_v221.py edge --plan-dir PLAN_DIR` refuses fully absent planned datasets and algorithm jobs, not merely duplicate/partial folds. Existing complete-fold checks still apply.
3. `stage_b.py select` fails closed if an equal-stratum sample cannot be allocated, instead of using an uncontrolled remainder fill with invalid probability weights. The Stage B protocol prespecifies 16 tasks divisible by four strata.
4. No changelog predictions were guessed or confirmed; the 78-row oracle remains unfrozen. `--unblinded-exploratory` is ONLY for A/A diagnostic reporting.
5. Stage B weighted Kish+Beta intervals are approximate and should not be interpreted as exact design-based coverage. Stage B is a pilot; do not assert confirmatory equivalence or population prevalence from these intervals.
6. Current environment has Python 3.13/sklearn 1.8.0, with no installed Python 3.7 and no OpenML package; historical binary evidence remains pending an Ubuntu/WSL run.

The analysis-code freeze for v2.2.1 must NOT be represented as applying to this amended code. New SHA-256 manifest is included.
