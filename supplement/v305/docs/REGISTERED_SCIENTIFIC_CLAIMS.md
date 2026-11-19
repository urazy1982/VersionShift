# Claim ladder: permitted and prohibited

## Defensible main claims

- In the frozen 48-task D0/P1 evaluation frame, changing scikit-learn from 1.4.2 to 1.5.2 was accompanied by mean pairwise comparative shift M=0.995320 pp, with conditional Holm-adjusted p=0.008169 against a 0.5-pp margin.
- In the same frame, 1.5.2→1.6.1 had M=2.052814 pp, Holm p=0.000293.
- The old 0.21.3→0.22.2.post1 historical edge has much greater P1-D0 drift (M=5.028388 pp on the 63-task six-estimator frame), with 25/63 reported winner changes.
- Historical fold-local standardized preprocessing (P2) has an observed conditional association with lower M (P2−P1=-3.564088 pp; task-bootstrap CI [-5.164011,-2.204983]). This is descriptive, not a universal causal effect.
- In the tested historical six-estimator hard-label outputs, no changes were observed between the era and common-stack environments across 3,024 matched supported predictions. Different intended environments were verified, but full per-process cryptographic binary telemetry is unavailable.
- Unsupported and removed estimator behaviors constitute distinct portability outcomes. They must remain in the registered inventory.

## Defensible secondary claims (descriptive; added with the post-results supplement)

- On the 48-task evaluation frame (D0/P1), the best of eight classifiers changed on 3/48 (E_LINEAR_DUAL) and 5/48 (E_ADA_SAMME) tasks; with leading margins strictly above 0.5 pp in both releases, 0/25 and 1/24 eligible tasks changed winner.
- Drift was confined to the mechanism estimator: LinearSVC +3.541 pp and AdaBoost +3.993 pp mean score change; six-estimator continuity M was 0.000 and 0.003 pp.
- On the matched seven-estimator panel (HGB excluded because the E2 setting is rejected by modern parameter validation), D1 reduced M from 1.138 to 0.000 pp and from 2.346 to 0.005 pp. This is not an equivalence claim and not an HGB mitigation result.
- Fold-local standardisation (P2) reduced E_LINEAR_DUAL drift to 0.011 pp but left E_ADA_SAMME drift unchanged (2.053 pp).
- Across 13 consecutive releases (six legacy estimators, era stacks) only 0.21.3→0.22.2.post1 was large (M=5.136 pp); all other edges had M≤0.233 pp.

## Claims that CANNOT be made from this release

- 'Scikit-learn is generally unsafe to update' or 'all library upgrades change winners'.
- 'The p-values demonstrate population prevalence across libraries, tasks and dates'.
- 'D1 causally removes 90% of upgrade effects' (historical outcome-informed setting).
- 'Preprocessing guarantees stability' (single historical six-estimator case frame).
- 'Python/BLAS/NumPy have no effect' (only observed hard labels and balanced accuracy tested).
- 'All 29,472 jobs are independent experiments' (these are job record counts).
- 'The modern edge values include unsupported/removed cells imputed as zeros'.
- 'The public package alone reproduces the 719,820 fold scores from raw labels'. (The 48-task primary inference *is* reproducible from the published task-level aggregates; fold scores are not.)
- 'Most upgrades change the best classifier' (most modern winner changes were near-ties; E5 shows most consecutive edges are stable).
- 'A release tag or GitHub repository can never be modified'.
