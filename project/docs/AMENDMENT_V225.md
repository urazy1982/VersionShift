# Amendment v2.2.5: hybrid validation lenses, policy S, Experiment E-MIT

Written 2026-11-06. It is **pre-outcome**: no Stage B historical-binary result exists yet. Every Stage B inference is still pending on Ubuntu.

## 0. What does not change

- **Primary analysis:** `decision_v221.py`, `analysis_v221.py` (the v2.2.4 job-level G2), `attribution.py`, `oracle_frame.py` and `stage_b.py` are byte-identical to v2.2.4.
- **Settings and gates:** margins, Holm/BH families, populations and gates G1–G7 are unchanged.
- **Labels:** the primary labels in the paper come only from `decision_v221`.
- **New material:** everything below is **secondary** and pre-registered here before any Stage B outcome.

## 1. Secondary lenses (`code/lenses_v225.py`)

Each lens answers a question the primary analysis does not address.

| Lens | Question | Method | Fixed settings |
|---|---|---|---|
| B: heterogeneity | How large a shift should the *next* dataset expect? | REML random-effects per pair over datasets (within-dataset variance from reruns), HKSJ CI, 95% prediction interval, I² | Per pair; reported, not tested |
| C: discovery | *Which* datasets changed, without a threshold? | Rerun-calibrated t per dataset × pair; Cauchy combination over pairs (exact-zero pairs excluded, p capped at 0.99); BH at q = 0.10; Storey π̂1 | q = 0.10, λ = 0.5 |
| D: continuation | Can Stage B evidence carry into Stage C without inflating error? | Betting e-process for H0: E[M_k] ≤ margin (M_k ≥ 0), aGRAPA λ ≤ 0.5/m, Ville threshold 1/α | margin 0.5 pp, α = 0.05, order seed 20261106; Stage B datasets enter first, then Stage C datasets in frozen random order |

The cross-release CLI `code/lenses_crossrelease_v225.py` runs behind the same gates as the primary analysis: `AN.gate` (oracle frozen), `AN.validate` (6 × 5 folds, scores in [0, 1]) and `AN.coverage_or_exploratory`.

## 2. Calibration (`code/calibrate_lenses_v225.py`)

Results are in `results/CALIBRATION_LENSES_V225.json`: 300 simulations per scenario, K = 16 and K = 48. The generator has a known truth; lens C truth is "dataset has any true shift".

| Scenario | C FDR (16/48) | C power | B PI coverage | D reject (16/48) |
|---|---|---|---|---|
| N1 pure null | 0 / 0 | — | 1.00 | 0 / 0 |
| N2 RNG stream changed | 0.113 / 0.107 | — | 0.97 / 0.98 | 0 / 0 |
| N3 heterogeneous, skewed | 0.110 / 0.107 | — | 0.97 / 1.00 | 0 / 0 |
| N4 variance-only change | 0.117 / 0.120 | — | 0.98 / 1.00 | 0 / 0 |
| A1 sparse large | 0.096 / 0.086 | 1.00 | 0.91 / 0.90 | 0 / 0 |
| B0 at margin boundary | 0 / 0 | 1.00 | 0.95 | 0 / 0 |
| A3 dense meaningful | 0 / 0 | 1.00 | 0.95 | **0.97 / 1.00** |
| **C2 single algorithm, 1.2 pp** | 0 / 0 | **0.96** | 0.95 / 0.99 | 0 / 0 |

Caveats, to be stated in the paper:
- **Lens C FDR runs at about 0.11–0.12 against a nominal 0.10** under RNG-stream and variance-only nulls. Rerun-t with 6 reruns is anti-conservative when variance changes across versions. Lens C is labelled "approximate FDR" and is never used for a primary claim.
- **Lens D is valid but conservative.** It does not reject at the margin boundary (B0) and needs dense meaningful shifts. Its role is to license continuation, not to discover.
- **Lens B CI coverage** falls to 0.81 with sparse large effects at K = 16, a known small-K HKSJ problem.

**Why C2 matters.** C2 is the main argument for the hybrid design. When one algorithm (RF) moves 1.2 pp everywhere, the mean |Δ| over the 15 pairs is diluted below the 0.5 pp margin, so the primary analysis and lens D stay silent. Lens C finds the affected datasets with 96% power.

## 3. Structural fact (reported, not assumed)

A pair shift is the difference of two algorithm shifts: Δ_AB = δ_A − δ_B. A change in one algorithm moves exactly 5 of the 15 pairs, so a shift confined to a single pair is impossible. This sets the resolution of every pair-level claim. Attribution is to algorithms, not to pairs.

## 4. Policy S: SonarSource python:S6973 as written

`crossversion/worker.py` adds policy `S`. It pins only the parameters the rule names, using values read from the pin-source release:

| Estimator | Parameters pinned |
|---|---|
| SVC | C, kernel, gamma |
| kNN | n_neighbors |
| RF | min_samples_leaf, max_features |
| GB, HGB | learning_rate |
| LR | none (not covered by the rule) |

- **Failures:** removed or ineffective values raise `NotPortable`, and values the new release rejects are classed `unsupported` (fail-closed, as for P).
- **Controller:** `controller.py` requires `--pin-from` for S.
- **Stage B:** policy S is added to the policy list (protocol updated).
- **Verification:** within one release, S, P and D are bitwise identical, checked end to end on 108 jobs.

## 5. Experiment E-MIT (emulation, scikit-learn 1.9.1, 44 datasets)

`code/policy_mitigation_v225.py` → `results/EXP_EMIT_V225.json`. No new fits: each policy maps to an already-fitted configuration, verified by `effective_signature`.

| Policy | Raw M (pp) | Raw label | Raw prev. | Std M (pp) | Std label | Lens C disc. (raw) |
|---|---|---|---|---|---|---|
| D defaults | 4.42 | CHANGED_MEAN + WIDESPREAD | 0.82 | 1.47 | CHANGED_MEAN + WIDESPREAD | 43 |
| S6973 as written | — | **hard failure:** RF `max_features='auto'` rejected by 1.9.1 | — | — | — | — |
| S6973 after rewrite to 'sqrt' | 1.55 | CHANGED_MEAN + WIDESPREAD | 0.73 | **1.47 (= D)** | CHANGED_MEAN + WIDESPREAD | 41 |
| E archived explicit map | 0.07 | INCONCLUSIVE | 0.02 | 0.07 | INCONCLUSIVE | 5 |
| P full pin | 0 | UNIFORMLY_SMALL | 0 | 0 | UNIFORMLY_SMALL | 0 |

**What this shows.**
- **On raw features, S6973 removes the SVC gamma shift.** It leaves the RF n_estimators shift (10 → 100), which the rule does not cover.
- **On standardised features**, where the gamma change has no effect, S6973 gives no protection at all.
- **As written, the rule fails outright** on the next major release.
- **Interpretation.** A deployed lint rule does not protect conclusions. Protection comes from pinning *the parameters whose defaults changed*, and that needs the changelog oracle (RQ3) or full pinning.
- **Scope.** This is an emulation result. Stage B tests it on real binaries.

## 6. Bug found and fixed during calibration

In the first lens C implementation, exact-zero pairs (shift identically 0, so p = 1) entered the Cauchy statistic as tan(−π/2) ≈ −3·10¹⁴. That masked every real effect: power was 0 even for a 4 pp shift. The fix excludes exact-zero pairs and caps p at 0.99. A regression test is in `tests/test_v225.py`.

## 7. Recommendation, not implemented

G2 at the job level can pass while excluding several datasets. Stage B already reports retention separately (15/16 in the CPU continuation). For Stage C, a retention floor of ≥ 90% of datasets is recommended. Adding it is a design decision for the author, so it is not added here.

## 8. Files

- **New:** `code/lenses_v225.py`, `code/calibrate_lenses_v225.py`, `code/policy_mitigation_v225.py`, `code/lenses_crossrelease_v225.py`, `tests/test_v225.py`, `results/CALIBRATION_LENSES_V225.json`, `results/EXP_EMIT_V225.json`, `docs/LITERATURE_UPDATE_V225.md`, this file.
- **Changed:** `crossversion/worker.py`, `crossversion/controller.py` (policy S), `docs/STAGE_B_PROTOCOL.md`, `RESPONSE_TO_REVIEW.md`, `README.md`.
- **Hashes:** `SHA256_MANIFEST_V225.json`.
