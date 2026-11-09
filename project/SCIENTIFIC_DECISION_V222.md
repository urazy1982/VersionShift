# Independent technical audit and scientific decision — VersionShift v2.2.2
Date: 2026-11-06

## Evidence verified
- Original v2.2.1 package: 11 frozen analysis-file SHA-256 hashes matched the supplied freeze (before amendment).
- Original fast tier: 54 discovered; 51 completed/passed and three skipped. The author's asserted full 54/54 suite was not independently reproduced.
- Amended fast tier: 60 discovered; **57 passed, three skipped**, 24.856 seconds, CPython 3.13, scikit-learn 1.8.0. The later amended run also passed after compatibility adjustments.
- Full slow suite: attempted but exceeded local execution time; no success claim.
- Real-data A/A: three bundled-dataset smoke preparation; a prior 18-job/180-fold LR/SVC/kNN A/A passed replay (original v2.2.1 controller). Amended v2.2.2 selection-aware test: one bundled iris dataset, six classifiers, A and B mapped to identical sklearn 1.8.0 runtime; 12 successful jobs, 120/120 fold scores replayed; mean cross-release shift M=0.0 pp; no equivalence claim at K=1.

## Newly identified defects and corrections
1. **Absent whole jobs**: v2.2.1 required all folds for present successful jobs but did not require the intended universe of datasets, algorithms and releases. Added mandatory frozen-plan coverage validation for oracle-frozen `edge` analysis. In unblinded exploratory mode a missing plan is permitted but marked `NOT_CHECKED_EXPLORATORY`. Six new tests cover this and selection.
2. **Stage B subset ignored**: controller previously executed all eligible tasks even after `stage_b.py select`. Added `run --selection`, hash lock and subset-aware collect/replay.
3. **Stratum remainder fill**: previous selection filled excess slots from a pooled remainder while using nominal stratum weights. Now fails closed if intended equal allocation cannot be realized; this can require a preregistered design amendment if the CC18 eligibility frame is uneven.
4. **Stage B weighted uncertainty**: Kish effective-n beta intervals are approximate, not exact design-based confidence bounds. Treat Stage B prevalence as descriptive/design-weighted; reserve confirmatory inference for a validated full-frame design.
5. **Oracle chronology**: code checks frozen hash but a hash alone cannot authenticate that coding predates all analyst exposure to outcomes. Record an external timestamp/commit and report any unblinded exploratory inspection.

## Not done, no fabricated evidence
- 78 changelog opportunities are not author-confirmed and therefore not frozen.
- Python 3.7 and OpenML package are unavailable in this evaluation container. No 0.21.3→0.22.2.post1 or 1.8.0→1.9.1 cross-version binary pilot was executed.
- Neither the 16-dataset Stage B pilot nor Stage C were executed here.
- Do not treat the zero A/A control as scientific evidence of historical equivalence.

## Decision
GO: Stage B prerequisite setup (oracle verification, environment builds, frozen dataset frame, A/A controls).
CONDITIONAL GO: run historical binary pilot once the frozen selection and oracle pass the gates.
NO-GO: inferential claims and Stage C until actual historical binary evidence and G1–G7 checks exist.

## Statistical cautions
- `pi_max` threshold 1.0 pp is a justified exploratory sensitivity choice under the reported simulation, not universal control of family-wise false positives.
- The finite CC18 frame may be strongly heterogeneous; Stage B 16-task weighted intervals do not establish precise population stability.
- A 2×2 Shapley decomposition is valid as an algebraic contrast when all four cells exist, but its runtime-stack component is not uniquely a scikit-learn implementation effect.
