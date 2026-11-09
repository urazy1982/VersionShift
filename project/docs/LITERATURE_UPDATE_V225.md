# VersionShift v2.2.5 — literature refresh and novelty positioning (2026-11-06)

**Conclusion:** the core gap still stands. As of 2026-11-06, no study measures whether scikit-learn upgrades change *ML conclusions* beyond rerun noise, attributes those changes counterfactually, and tests practitioner safeguards. Three new or newly found items narrow how the contribution is worded, and one strengthens it.

## New items since the v2.0 search

| Work | What it does | Consequence for VersionShift |
|---|---|---|
| [sklearn-migrator](https://science.ecosyste.ms/projects/808550) (Valdes Gonzalez, JOSS 11(121), May 2026) | Migrates already-trained models across 32 versions (0.21.3–1.7.2, 1,024 pairs, 21 estimators); checks migrated-model prediction agreement (max abs diff < 1e-2); one configuration per model; no retraining | Addresses *artefact portability*, not *re-execution*. Cite as complementary: VersionShift retrains under each release and measures conclusion change |
| [seqbench](https://cran.r-universe.dev/seqbench/doc/manual.html) (Leite, Ospina, Ferraz; R, manual 2026-10-08) | Anytime-valid sequential comparison of two algorithms on paired losses, with a practical-equivalence margin; betting confidence sequences (Waudby-Smith & Ramdas) | Anytime-valid benchmarking **exists**, so lens D is not claimed as a statistical novelty. The new element is the target: upgrade-induced conclusion change, with Stage B → C continuation |
| [SonarSource rule python:S6973](https://rules.sonarsource.com/python/rspec-6973/) | Industrial lint rule: set scikit-learn hyperparameters explicitly, because defaults "can lead to non-reproducible results across different versions". For the six study estimators it covers SVC (C, kernel, gamma), kNN (n_neighbors), RF (min_samples_leaf, max_features) and GB/HGB (learning_rate), but **not** RF n_estimators and not LogisticRegression | **Strengthens RQ4.** We can test a deployed industrial safeguard. Pilot: it leaves most of the 0.22 drift in place (E-MIT below) |
| [Dilhara et al., TOSEM 30(4) 2021](https://dl.acm.org/doi/10.1145/3453478) | ML library usage and evolution in 3,340 projects; ML libraries are updated more often, and 22% of updates are downgrades | Motivation: upgrades are frequent, and so are downgrades (a reverse edge is a natural extension) |
| [Byam](https://link.springer.com/article/10.1007/s10664-026-10835-1) (EMSE 2026) | LLM repair of *compilation-breaking* dependency updates; notes that semantic changes can remain after compilation succeeds | Same gap from the other side: silent semantic changes are unaddressed |
| scikit-learn 0.22 notes ([GitHub](https://github.com/scikit-learn/scikit-learn/blob/24a50e59d2faa53ea68c179aa855e7c33fa06d15/doc/whats_new/v0.22.rst)) | "Changed models" list includes HistGradientBoosting and **StratifiedKFold** | Confirms the splitter-change arm |

Earlier anchors still stand: Montandon et al. (JSS 2025; static DABC exposure; measuring impact magnitude named as their future work), Jayasuriya et al. (FSE 2024; behavioural breaking changes), Bouthillier et al. (MLSys 2021; benchmark variance), Pham et al. (ASE 2020; implementation-level variance in deep learning).

## Methods literature for the secondary lenses

- **Random-effects meta-analysis with prediction intervals** (Higgins–Thompson–Spiegelhalter; REML; Hartung–Knapp–Sidik–Jonkman). Known small-K coverage problems ([Matrai et al. 2024](https://arxiv.org/abs/2408.08080v1); [Nagashima et al.](https://arxiv.org/pdf/2110.07856); [Kronthaler & Held 2025](https://arxiv.org/pdf/2510.13216)). Benchmark tables are treated as meta-analytic data. Prediction intervals in ML benchmarking are uncommon but not new, so they are not claimed as novelty.
- **Cauchy combination test** (Liu & Xie, JASA 2020). Valid under arbitrary dependence in the tails; it needs truncation near p = 1, which the v2.2.5 calibration confirmed (a bug found and fixed).
- **BH-FDR and Storey π0**: standard.
- **E-values and anytime validity**: Grünwald et al.; Ramdas et al.; [Pérez-Ortiz et al., "Sequentializing a test", 2025](https://ar5iv.labs.arxiv.org/html/2501.03982); [anytime-valid LLM evaluation, 2026](https://www.alphaxiv.org/overview/2607.17409).

From the v2.0 reference list, Waudby-Smith & Ramdas, Liu & Xie, Storey and the HKSJ papers are cited from memory and should be fetched before submission.

## Novelty statement (for the manuscript)

The study's combination is new:
1. **Measured impact.** Counterfactual measurement of how library evolution changes ML *conclusions*, in noise-calibrated units, on real historical binaries.
2. **Hybrid validation.** A frozen design-based primary analysis, plus three pre-registered secondary lenses:
   - heterogeneity and prediction;
   - threshold-free discovery;
   - anytime-valid continuation.

   Each lens was calibrated on a known-truth generator, and each catches a failure mode the primary misses.
3. **Mitigation tested.** An empirical test of an industrial safeguard (S6973) and of explicit and full pinning.
4. **Changelog oracle.** Precision and recall of release notes as predictors of conclusion change.

None of the statistical methods is new in isolation, and the paper should say so. The novelty is the empirical question and the validation design.
