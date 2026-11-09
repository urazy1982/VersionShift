"""Build ORACLE_FRAME_REVIEWED.csv and ORACLE_EVIDENCE_78.csv for VersionShift v2.2.5.

Independent, blinded changelog coding (no Stage B / A-A outcome files were read).
Reviewer: Claude (Anthropic AI model claude-fable-5-1), commissioned by U. Yavanoglu, 2026-11-06.
"""
import csv, json, sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/claude/outputs/oracle_review')
OUT.mkdir(parents=True, exist_ok=True)
WORKING = Path('/mnt/user-data/uploads/ff804fd2-ORACLE_FRAME_WORKING.csv')

RELEASES = ['0.21.3', '0.22.2.post1', '0.23.2', '0.24.2', '1.0.2', '1.1.3', '1.2.2', '1.3.2', '1.4.2', '1.5.2',
            '1.6.1', '1.7.2', '1.8.0', '1.9.1']
ESTIMATORS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
EDGES = ['%s->%s' % (o, n) for o, n in zip(RELEASES, RELEASES[1:])]

CODER = ('Claude (Anthropic AI model claude-fable-5-1), independent blinded reviewer commissioned by '
         'U. Yavanoglu; author confirmation pending')
CODED_ON = '2026-11-06'
PIPE = 'frozen central splits; float64 dense, no sample_weight, n<=5000, median-imputed NaN'
DTYPES = 'dense float64; categorical one-hot; NaN median-imputed'
INTROSPECT_NOTE = 'introspect_not_run_by_reviewer'

def WN(v):   # rendered release notes (versioned) + the GitHub source actually read
    return 'https://scikit-learn.org/%s/whats_new/v%s.html' % (v, v)
def RST(v):
    return 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/whats_new/v%s.rst' % v
def PR(n):
    return 'https://github.com/scikit-learn/scikit-learn/pull/%d' % n
def API(v, cls):
    return 'https://scikit-learn.org/%s/modules/generated/sklearn.%s.html' % (v, cls)
def SRC(tag, path):
    return 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/%s/%s' % (tag, path)

DEP_CAVEAT = {
    'LR': ('lbfgs is scipy.optimize.minimize(L-BFGS-B); the hash-locked environments necessarily ship different '
           'SciPy/NumPy/BLAS builds per release, so fp-level coefficient drift from the dependency stack is possible '
           'on every edge even with identical scikit-learn code (amplified when max_iter=100 is hit without convergence). '
           'Not coded into the category (RQ3 measures changelog predictiveness); see report.'),
    'SVC': 'libsvm is vendored and compiled per wheel; compiler/BLAS (0.24+) differences may give fp-level kernel/decision drift.',
    'kNN': 'brute path uses BLAS GEMM for the Euclidean expansion; different NumPy/OpenBLAS builds may reorder tied/near-tied neighbors.',
    'RF': 'numpy legacy RandomState stream is stable across NumPy versions; tree code is vendored Cython (compiler-dependent fp only).',
    'GB': 'numpy legacy RandomState stream is stable across NumPy versions; tree code is vendored Cython (compiler-dependent fp only).',
    'HGB': 'OpenMP threads pinned to 1 by the worker; binning uses np.percentile whose numerics are stable across NumPy versions for a fixed method.',
}

# ---------------------------------------------------------------------------------------------------------------
# Row coding.  Each entry: category, trigger, uncertainty, entries [(entry_category, text)], refs [urls], compat,
# autodiff (documented default diffs the project's introspect would report), activation note, evidence types,
# disputed-vs-draft note, uncertainty reason.
# ---------------------------------------------------------------------------------------------------------------
R = {}

def row(edge, est, cat, trig, unc, entries, refs, compat, autodiff, activation, evidence_type, disputed, unc_reason):
    R[(edge, est)] = dict(cat=cat, trig=trig, unc=unc, entries=entries, refs=refs, compat=compat, autodiff=autodiff,
                          activation=activation, evidence_type=evidence_type, disputed=disputed, unc_reason=unc_reason)

NONE_TRIG = 'none under pipeline conditions (no destination-release entry touches the exercised fit/predict path)'
SPLIT = ('splitter_changed_models', 'model_selection.StratifiedKFold / cv=int listed in 0.22 Changed models (#14704); '
         'NOT activated: splits are frozen centrally by controller.py')

# ===== Edge 1: 0.21.3 -> 0.22.2.post1 ==========================================================================
E = EDGES[0]
row(E, 'LR', 'change_expected',
    'always (solver default liblinear -> lbfgs); additionally multinomial instead of OvR when n_classes>2 (multi_class ovr -> auto)',
    'low',
    [('DABC_semantic', "solver default 'warn'(=liblinear) -> 'lbfgs' (announced 0.20, effective 0.22)"),
     ('DABC_semantic', "multi_class default 'warn'(=ovr) -> 'auto' (multinomial for lbfgs when n_classes>2)"),
     ('Efficiency_not_exercised', 'liblinear solver faster/less memory (#14108, #14170, #14296); not used under D in 0.22'),
     SPLIT],
    [API('0.21', 'linear_model.LogisticRegression'), API('0.22', 'linear_model.LogisticRegression'), WN('0.22'), RST('0.22')],
    'strict_pin_may_fail: solver: warn -> lbfgs; multi_class: warn -> auto (placeholders -> worker status ineffective)',
    'solver: warn -> lbfgs; multi_class: warn -> auto (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED: policy D resolves solver=liblinear in 0.21.3 and solver=lbfgs in 0.22.2.post1; different optimiser, different coefficients.',
    'versioned_api_doc;official_release_note', '',
    'mechanism certain; magnitude dataset-dependent')

row(E, 'SVC', 'change_expected',
    "always for unscaled input: gamma 1/n_features -> 1/(n_features*X.var()); identical only if X.var()==1",
    'low',
    [('DABC_semantic', "gamma default 'auto_deprecated'(=1/n_features) -> 'scale'(=1/(n_features*X.var()))"),
     ('API_new_param', 'break_ties added (default False; no effect) (#12557)'),
     ('Fix_not_exercised', 'zero/negative sample_weight now raises (#14286); sample_weight not used'),
     SPLIT],
    [API('0.21', 'svm.SVC'), API('0.22', 'svm.SVC'), WN('0.22'), RST('0.22')],
    'strict_pin_may_fail: gamma: auto_deprecated -> scale (placeholder -> ineffective); break_ties: <absent> -> False (unpinned_new)',
    'gamma: auto_deprecated -> scale; break_ties: <absent> -> False (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED: no feature scaling in the pipeline, so X.var() != 1 for essentially every dataset.',
    'versioned_api_doc;official_release_note', '', 'mechanism certain')

row(E, 'kNN', 'no_change_expected', NONE_TRIG, 'medium',
    [('Feature_not_exercised', 'KNeighborsClassifier accepts precomputed sparse neighbors graph (#10482)'),
     ('Fix_not_exercised', 'radius_neighbors list-of-arrays conversion (0.22.2.post1, #16076); radius API not used'),
     SPLIT],
    [WN('0.22'), RST('0.22'), API('0.22', 'neighbors.KNeighborsClassifier')],
    'strict_pin_portable_expected (constructor parameters unchanged 0.21.3 -> 0.22.2.post1)',
    'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED: no 0.22.x entry touches dense-euclidean kneighbors or the kd_tree/brute fit path.',
    'official_release_note;versioned_api_doc', '',
    'absence of entries is not proof of absence of change; 0.22 also made broad validation changes (check_array, FutureWarning)')

row(E, 'RF', 'change_expected', 'always (n_estimators default 10 -> 100)', 'low',
    [('DABC_semantic', "n_estimators default 'warn'(=10) -> 100 (announced 0.20, effective 0.22)"),
     ('API_new_param', 'ccp_alpha added (default 0.0, no pruning) (#12887); max_samples added (default None) (#14682)'),
     SPLIT],
    [API('0.21', 'ensemble.RandomForestClassifier'), API('0.22', 'ensemble.RandomForestClassifier'), WN('0.22'), RST('0.22')],
    'strict_pin_may_fail: n_estimators: warn -> 100 (placeholder -> ineffective); ccp_alpha, max_samples: <absent> (unpinned_new)',
    'n_estimators: warn -> 100; ccp_alpha: <absent> -> 0.0; max_samples: <absent> -> None (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED: 10 vs 100 trees with the same random_state yields different ensembles.',
    'versioned_api_doc;official_release_note', '', 'mechanism certain')

row(E, 'GB', 'fp_level_change_possible',
    "always for dense input: 0.21 default presort='auto' used the presorted split-search path; 0.22 presort has no effect "
    '(per-node sorting) -> identical split thresholds in exact arithmetic, but a different accumulation order of per-sample sums',
    'medium',
    [('API_deprecation_changes_code_path', "presort deprecated and 'has no effect' (#14907); 0.21 default 'auto' presorted dense data"),
     ('API_new_param', 'ccp_alpha added (default 0.0) (#12887)'),
     SPLIT],
    [API('0.21', 'ensemble.GradientBoostingClassifier'), API('0.22', 'ensemble.GradientBoostingClassifier'), WN('0.22'), RST('0.22'), PR(14907)],
    "strict_pin_may_fail: presort: auto -> deprecated (pinned 'auto' accepted with warning in 0.22; reverse pin 'deprecated' is a placeholder); ccp_alpha: <absent> (unpinned_new)",
    'presort: auto -> deprecated; ccp_alpha: <absent> -> 0.0 (doc-derived; %s)' % INTROSPECT_NOTE,
    'PARTIALLY ACTIVATED: the exercised best-split code path changed (presorted -> per-node introsort); exact-arithmetic result identical.',
    'versioned_api_doc;official_release_note', 'draft: no_change_expected (0 entries). Reviewer: fp_level_change_possible.',
    'whether fp accumulation-order differences flip any tie between candidate splits is dataset-dependent')

row(E, 'HGB', 'change_expected',
    'any feature with >255 distinct non-missing values in a training fold (max_bins 256 -> 255 changes the bin edges); '
    'features with <=255 distinct values keep identical midpoint edges; other listed fixes need early stopping or NaN',
    'medium',
    [('DABC_semantic', 'max_bins default 256 -> 255 (one bin reserved for missing values) (#13911)'),
     ('changed_models_list', "HistGradientBoostingClassifier listed in 0.22 'Changed models' (Fix, Feature, Enhancement)"),
     ('Feature_not_exercised', 'native missing/infinite value support (#13911, #14406); no NaN after median imputation'),
     ('Fix_not_exercised', 'train/validation binned separately (#13933); stratified early-stopping subsample (#14194); early stopping off by default (n_iter_no_change=None)'),
     ('API_new_param', 'warm_start added (#14012)'),
     SPLIT],
    [API('0.21', 'ensemble.HistGradientBoostingClassifier'), API('0.22', 'ensemble.HistGradientBoostingClassifier'), WN('0.22'), RST('0.22')],
    'strict_pin_may_fail: max_bins: 256 -> 255 (pinned 256 rejected in 0.22: must be <=255 -> unsupported); warm_start: <absent> (unpinned_new)',
    'max_bins: 256 -> 255; warm_start: <absent> -> False (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED on datasets with high-cardinality numeric features (common at n_train in the thousands); not activated on all-categorical one-hot data.',
    'versioned_api_doc;official_release_note',
    'draft: change_possible (changed_models_list only). Reviewer: change_expected because the max_bins default change is a documented semantic default change.',
    'trigger prevalence in the frozen dataset pool not inspected by the reviewer (blinding)')

# ===== Edge 2: 0.22.2.post1 -> 0.23.2 ==========================================================================
E = EDGES[1]
row(E, 'LR', 'no_change_expected',
    "solver='liblinear' (libsvm/liblinear RNG change #13511) or solver='newton-cg' (convergence check #16266); D uses lbfgs",
    'medium',
    [('changed_models_list_conditional', 'LogisticRegression listed in 0.23 Changed models via liblinear RNG change (#13511); solver-specific'),
     ('Fix_not_exercised', "newton-cg convergence check '<' -> '<=' (#16266)")],
    [WN('0.23'), RST('0.23'), PR(13511), PR(16266)],
    'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED: lbfgs path untouched in 0.23.x.', 'official_release_note', '',
    'listed as a changed model, but only through solvers not used under D')

row(E, 'SVC', 'no_change_expected',
    "probability=True (libsvm RNG is used only for the Platt-scaling CV shuffle; SVC.random_state doc: 'Ignored when probability is False')",
    'medium',
    [('changed_models_list_conditional', "SVC listed in 0.23 'Changed models' (any model using libsvm): platform-dependent rand() replaced by mt19937 (#13511)"),
     ('API_not_exercised', 'custom kernels must validate input (#11296)')],
    [WN('0.23'), RST('0.23'), PR(13511), API('0.23', 'svm.SVC')],
    'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED under D (probability=False): the SMO solver itself is deterministic; RNG only feeds probability CV.',
    'official_release_note;versioned_api_doc',
    'DISPUTED: the official Changed-models list names SVC unconditionally; the trigger analysis relies on the random_state docstring. '
    'Author may prefer change_possible to honour the official listing literally.',
    'official listing vs documented mechanism conflict')

row(E, 'kNN', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no sklearn.neighbors section in 0.23.x; no pairwise euclidean change')],
    [WN('0.23'), RST('0.23')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')

row(E, 'RF', 'no_change_expected', 'read-only float32 input at predict (#16331); not the case (writable float64)', 'medium',
    [('Fix_not_exercised', 'read-only float32 input supported in tree predict/predict_proba (#16331); DecisionTreeClassifier listed in Changed models for this')],
    [WN('0.23'), RST('0.23'), PR(16331)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no fit-path change documented')

row(E, 'GB', 'no_change_expected',
    'n_iter_no_change set with n_classes>2 (MultinomialDeviance loss value average vs sum, 0.23.2 #17694, changes early-stopping decisions); '
    'read-only float32 predict (#16331). Under D n_iter_no_change=None and the loss value never feeds the fit',
    'medium',
    [('Fix_conditional', 'MultinomialDeviance average log-loss computed as sum (0.23.2, #17694): affects train_score_/early stopping only'),
     ('Fix_not_exercised', 'read-only float32 predict (#16331); GradientBoostingClassifier listed in Changed models for this')],
    [WN('0.23'), RST('0.23'), PR(17694), PR(16331)], 'strict_pin_portable_expected (constructor parameters unchanged; presort still present as deprecated)',
    'none documented (%s)' % INTROSPECT_NOTE, 'NOT ACTIVATED: gradients/leaf updates untouched.', 'official_release_note', '',
    'train_score_ values change for multiclass but predictions should not')

row(E, 'HGB', 'no_change_expected',
    "n_samples>10000 (early_stopping='auto' then turns early stopping ON, previously off); max_depth set (#16182/#16183); sample_weight given (#14696)",
    'medium',
    [('DABC_conditional', "early_stopping='auto' introduced, n_iter_no_change default None -> 10; auto enables early stopping only when n_samples>10000 (#14516)"),
     ('Feature_code_path', 'sample_weight support added to gradients/hessians (#14696); None under D'),
     ('Fix_not_exercised', 'max_depth convention and max_leaf_nodes interplay (#16182, #16183); max_depth=None'),
     ('API_new_param', 'monotonic_cst added (#15582)'),
     ('changed_models_list', "HGB listed in 0.23 Changed models with tag 'API'")],
    [WN('0.23'), RST('0.23'), API('0.23', 'ensemble.HistGradientBoostingClassifier'), API('1.2', 'ensemble.HistGradientBoostingClassifier')],
    'strict_pin_may_fail: early_stopping, monotonic_cst: <absent> (unpinned_new); n_iter_no_change: None -> 10 (pinned None may be rejected)',
    'early_stopping: <absent> -> auto; monotonic_cst: <absent> -> None; n_iter_no_change: None -> 10 (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED at n<=5000: early stopping stays off on both sides; sample_weight=None path should be arithmetically identical.',
    'official_release_note;versioned_api_doc', '',
    'the sample_weight refactor touched the exercised gradient code; bitwise identity not guaranteed by the notes')

# ===== Edge 3: 0.23.2 -> 0.24.2 ================================================================================
E = EDGES[2]
row(E, 'LR', 'no_change_expected', NONE_TRIG, 'medium',
    [('Fix_not_exercised', 'LogisticRegression no longer modifies sample_weight in place (0.24.2, #19182)')],
    [WN('0.24'), RST('0.24')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no solver change documented')

row(E, 'SVC', 'fp_level_change_possible', 'always (rbf kernel computations in fit/predict now go through the SciPy BLAS API)', 'medium',
    [('Efficiency_fp_level', 'SVC/SVR/NuSVC/NuSVR/OneClassSVM kernel computations use SciPy BLAS API (#16530)')],
    [WN('0.24'), RST('0.24'), PR(16530)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'ACTIVATED at fp level: kernel values may differ in the last bits; support set/decision values can drift; hard labels rarely.',
    'official_release_note', 'draft: not coded. Reviewer: fp_level_change_possible.', 'fp-only mechanism; flips only near the margin')

row(E, 'kNN', 'fp_level_change_possible',
    "n_features>15 after one-hot encoding (algorithm='auto' now resolves to brute instead of kd_tree; brute uses the GEMM expansion "
    '||x||^2+||y||^2-2x.y whereas kd_tree computes exact squared differences) -> tie/near-tie reordering of neighbors',
    'medium',
    [('Enhancement_changes_code_path', "algorithm='auto' selects 'brute' when n_features>15 (#17148); verified against 0.23.2 and 0.24.2 _base.py")],
    [WN('0.24'), RST('0.24'), PR(17148), SRC('0.23.2', 'sklearn/neighbors/_base.py'), SRC('0.24.2', 'sklearn/neighbors/_base.py')],
    'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    "ACTIVATED on wide datasets: fitted _fit_method changes kd_tree -> brute (observable in fitted_summary); neighbor sets identical in exact arithmetic.",
    'official_release_note;source_code_tag', 'review-queue lead confirmed.', 'exact/near ties are frequent in one-hot heavy data')

row(E, 'RF', 'no_change_expected', NONE_TRIG, 'medium',
    [('API_not_exercised', 'X_idx_sorted deprecated/no effect in tree fit (#17614); forests never passed it'),
     ('Feature_not_exercised', "DecisionTreeRegressor 'poisson' criterion (#17386)")],
    [WN('0.24'), RST('0.24')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no forest/tree fit-path change documented')

row(E, 'GB', 'no_change_expected', "criterion='mae' (deprecated #18326); default friedman_mse unaffected", 'medium',
    [('API_deprecation_not_exercised', "criterion='mae' deprecated (#18326)"),
     ('API_removal', 'presort parameter removed in 0.24 (deprecated 0.22)')],
    [WN('0.24'), RST('0.24'), API('0.24', 'ensemble.GradientBoostingClassifier'), API('0.22', 'ensemble.GradientBoostingClassifier')],
    'strict_pin_may_fail: presort: deprecated -> <absent> (pinned parameter removed)', 'presort: deprecated -> <absent> (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED for predictions; portability of a strict 0.23.2 pin fails (presort removed).', 'official_release_note;versioned_api_doc', '',
    'no fit-path change documented')

row(E, 'HGB', 'no_change_expected', 'categorical_features set (native categorical path, #18394); not set under D (one-hot float64)', 'medium',
    [('API_new_param', 'categorical_features added, default None (#18394)'),
     ('Efficiency_init_only', 'parallel histogram initialisation (#18341); memory cycle fix (#18334); uint8 predict (#18410); staged_predict (#16985)')],
    [WN('0.24'), RST('0.24'), PR(18394), PR(18341)], 'strict_pin_may_fail: categorical_features: <absent> -> None (unpinned_new)',
    'categorical_features: <absent> -> None (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED: central one-hot float64 does not enter the native categorical path (review-queue lead confirmed).',
    'official_release_note', '', 'histogram-initialisation parallelism does not change accumulated values')

# ===== Edge 4: 0.24.2 -> 1.0.2 =================================================================================
E = EDGES[3]
row(E, 'LR', 'no_change_expected', "solver='newton-cg' dense optimisation (#19571); D uses lbfgs", 'medium',
    [('Efficiency_not_exercised', 'newton-cg optimised for dense X (#19571)'),
     ('API_cross_cutting', 'keyword-only arguments enforced (#20002); worker passes kwargs')],
    [WN('1.0'), RST('1.0')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no lbfgs change documented')

row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('Fix_not_exercised', 'SVC/SVR raise instead of segfault on internal inconsistency (1.0.1, #21336)')],
    [WN('1.0'), RST('1.0')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no numeric change documented')

row(E, 'kNN', 'fp_level_change_possible',
    'n_features<=15 (kd_tree path) with exact distance ties at the k-th neighbor: KDTree/BallTree construction algorithm changed (#19473) '
    'so partitioning/visit order and therefore tie resolution may differ; neighbor sets identical otherwise',
    'high',
    [('Efficiency_changes_code_path', 'KDTree/BallTree construction worst case O(n^2) -> O(n) (#19473)'),
     ('API_not_exercised', "MinkowskiDistance weight parameter (1.0.2, #21741); p=2 resolves to 'euclidean'")],
    [WN('1.0'), RST('1.0'), PR(19473)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'CONDITIONALLY ACTIVATED (tie-order class, mapped to fp_level_change_possible): only kd_tree datasets with duplicated/tied points.',
    'official_release_note', 'draft: not coded. Reviewer: fp_level_change_possible (tie-order mechanism).',
    'the note is an efficiency note; whether tie resolution changed is inferred from the construction change, not stated')

row(E, 'RF', 'change_possible',
    'nodes where floating-point rounding in the impurity criterion previously produced a spurious non-improving split (#19336); '
    'RandomForestClassifier uses the same tree criterion/splitter code as DecisionTreeClassifier',
    'medium',
    [('Fix_changes_results', "rounding-error fix in the splitting criterion; DecisionTreeClassifier listed in 1.0 'Changed models' (#19336)"),
     ('API_removal', 'min_impurity_split removed in 1.0 (deprecated since 0.19/0.23)'),
     ('Fix_not_exercised', 'bootstrap=False with max_samples raises (1.0.2, #21295)')],
    [WN('1.0'), RST('1.0'), PR(19336), API('1.0', 'ensemble.RandomForestClassifier')],
    'strict_pin_may_fail: min_impurity_split: None -> <absent> (pinned parameter removed)',
    'min_impurity_split: None -> <absent> (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED where the rounding pathology occurs (near-pure nodes); official listing names only DecisionTree*, forests inherit the code.',
    'official_release_note;versioned_api_doc', 'draft: not coded. Reviewer: change_possible.', 'prevalence of the rounding pathology unknown')

row(E, 'GB', 'change_possible',
    'same tree criterion rounding fix (#19336) in the regression trees (friedman_mse) built by GradientBoostingClassifier; '
    "loss='exponential' gradient sign fix (1.0.2, #22050) is not exercised (default deviance)",
    'medium',
    [('Fix_changes_results', 'tree criterion rounding fix (#19336)'),
     ('Fix_not_exercised', "GradientBoostingClassifier exponential-loss gradient sign fixed (1.0.2, #22050); loss='deviance' under D"),
     ('API_removal', 'min_impurity_split removed in 1.0')],
    [WN('1.0'), RST('1.0'), PR(19336), PR(22050), API('1.0', 'ensemble.GradientBoostingClassifier')],
    'strict_pin_may_fail: min_impurity_split: None -> <absent> (pinned parameter removed)',
    'min_impurity_split: None -> <absent> (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED where the rounding pathology occurs.', 'official_release_note;versioned_api_doc',
    'draft: not coded. Reviewer: change_possible.', 'prevalence unknown')

row(E, 'HGB', 'no_change_expected', 'tol=None previously (removed; default 1e-7 unchanged); multi-core cgroup quotas (OMP=1 in worker)', 'medium',
    [('API_import_path', 'HGB no longer experimental (#19799); worker tolerates both import styles'),
     ('API_removal_not_exercised', 'tol=None removed (#19296)'),
     ('Fix_not_exercised', 'rare segfault fixed (1.0.1, #21130)')],
    [WN('1.0'), RST('1.0'), PR(19799)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED (review-queue lead confirmed: API/import only).', 'official_release_note', '', 'no numeric change documented')

# ===== Edge 5: 1.0.2 -> 1.1.3 ==================================================================================
E = EDGES[4]
row(E, 'LR', 'fp_level_change_possible',
    'always under D (lbfgs): loss/gradient computation rewritten on the new _loss module and multiclass label-encoded (#21808, #20567, #21814); '
    'same objective and optimiser, fp-level coefficient drift; amplified when lbfgs stops at max_iter=100 without convergence',
    'medium',
    [('Efficiency_fp_level', 'LogisticRegression faster with lbfgs/newton-cg; multiclass label-encoded instead of one-hot (#21808, #20567, #21814)'),
     ('API_attribute', 'n_iter_ now reports every OvR sub-problem (#21998)'),
     ('Fix_not_exercised', 'newton-cg single-feature fix (1.1.2, #23608)')],
    [WN('1.1'), RST('1.1'), PR(21808), PR(20567)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'ACTIVATED at fp level on every dataset.', 'official_release_note',
    'draft: no_change_expected (0 entries). Reviewer: fp_level_change_possible.',
    'unconverged lbfgs runs (common on unscaled data at max_iter=100) can amplify fp drift into label changes')

row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('API_attribute', 'n_iter_ exposed (#21408)'), ('Fix_not_exercised', 'non-finite dual-gap weights raise (#22149); parameter validation moved to fit (#21436)')],
    [WN('1.1'), RST('1.1')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'n_iter_ bookkeeping added to libsvm; no numeric change documented')

row(E, 'kNN', 'fp_level_change_possible',
    'brute path (n_features>15 or n_neighbors>=n_train//2): new low-level pairwise-distance reduction back-end for dense float64 (#21987, #22064, #22065, #22288, #22320)'
    ' -> chunked GEMM distances, fp/tie reordering; kd_tree path unaffected',
    'medium',
    [('Efficiency_fp_level', 'pairwise-distance reductions refactored for dense float64; KNeighborsClassifier listed (#21987 et al.)'),
     ('Fix', 'F-contiguous input support for the new back-end (1.1.2, #23990)')],
    [WN('1.1'), RST('1.1'), PR(21987), PR(22320)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'ACTIVATED at fp/tie level on wide datasets.', 'official_release_note',
    'draft: no_change_expected (0 entries). Reviewer: fp_level_change_possible.', 'fp-only mechanism')

row(E, 'RF', 'change_possible',
    'ties in the split criterion value (new sort algorithm in best-split search, #22868: "splits with tied criterion values may now differ"); '
    'common with one-hot/discrete features',
    'low',
    [('changed_models_list', "RandomForestClassifier listed in 1.1 'Changed models': ~15% faster fitting, tied splits may differ (#22868)"),
     ('DABC_syntactic_equivalent', "max_features default 'auto' -> 'sqrt' (same meaning for classifiers) (#20803)"),
     ('Fix_not_exercised', 'low-cardinality performance regression (1.1.1, #23410); invalid memory access (1.1.2, #23273)')],
    [WN('1.1'), RST('1.1'), PR(22868), PR(20803)],
    "strict_pin_portable_expected (pinned max_features='auto' accepted with FutureWarning in 1.1; removed in 1.3)",
    'max_features: auto -> sqrt (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED wherever criterion ties occur.', 'official_release_note',
    "draft: no_change_expected (max_features equivalence only). Reviewer: change_possible because of the official 'Changed models' listing (#22868).",
    'official wording is "may"; the project has frozen random_state so tie order is the only free variable')

row(E, 'GB', 'change_possible', 'ties in the split criterion value (#22868); loss rename deviance -> log_loss is equivalent', 'low',
    [('changed_models_list', "GradientBoostingClassifier listed in 1.1 'Changed models' (#22868)"),
     ('API_rename_equivalent', "loss default 'deviance' -> 'log_loss' (equivalent) (#23036)")],
    [WN('1.1'), RST('1.1'), PR(22868), PR(23036)],
    "strict_pin_portable_expected (pinned loss='deviance' accepted with FutureWarning in 1.1; removed in 1.3)",
    'loss: deviance -> log_loss (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED wherever criterion ties occur.', 'official_release_note',
    'draft: no_change_expected (rename only). Reviewer: change_possible (#22868).', 'official wording is "may"')

row(E, 'HGB', 'fp_level_change_possible',
    'always: binary/multiclass loss, gradient and hessian computation moved to the new _loss module (#20811, #20567, #21814); '
    "loss rename 'auto' -> 'log_loss' is equivalent",
    'medium',
    [('Efficiency_fp_level', 'HistGradientBoostingClassifier faster for binary and multiclass (#20811, #20567, #21814)'),
     ('API_rename_equivalent', "loss 'auto'/'binary_crossentropy'/'categorical_crossentropy' deprecated for 'log_loss' (#23040)")],
    [WN('1.1'), RST('1.1'), PR(20811), PR(23040), API('1.1', 'ensemble.HistGradientBoostingClassifier')],
    "strict_pin_portable_expected (pinned loss='auto' accepted with FutureWarning in 1.1)",
    'loss: auto -> log_loss (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED at fp level (gradient/hessian arithmetic path).', 'official_release_note;versioned_api_doc',
    'draft: no_change_expected (rename only). Reviewer: fp_level_change_possible.', 'fp-only mechanism')

# ===== Edge 6: 1.1.3 -> 1.2.2 ==================================================================================
E = EDGES[5]
row(E, 'LR', 'no_change_expected', "solver='newton-cholesky' (new, not default); penalty='none' string deprecated", 'medium',
    [('Feature_not_exercised', 'newton-cholesky solver added (#24637, #24767)'), ('API_deprecation_not_exercised', "penalty='none' deprecated for None (#23877)")],
    [WN('1.2'), RST('1.2')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no lbfgs change documented')
row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'only class_weight_ deprecation for regressors (#22898)')],
    [WN('1.2'), RST('1.2')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'kNN', 'no_change_expected', 'float32 input (new float32 reduction back-end #23865) or dense-sparse pairs; pipeline is float64 dense', 'medium',
    [('Efficiency_not_exercised', 'pairwise-distance reductions for float32 (#23865) and dense-sparse pairs (#23604, #23585, #24556)')],
    [WN('1.2'), RST('1.2'), PR(23865)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED (float64 dense path unchanged).', 'official_release_note', '', 'back-end dispatch code changed; float64 kernels not listed as changed')
row(E, 'RF', 'no_change_expected', NONE_TRIG, 'medium',
    [('API_rename', 'base_estimator_ -> estimator_ attribute (#23819)'), ('Fix_not_exercised', 'min_samples_split=1 regression fixed (1.2.2, #25744); default 2')],
    [WN('1.2'), RST('1.2')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no tree fit-path change documented')
row(E, 'GB', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no GradientBoosting entries in 1.2.x')],
    [WN('1.2'), RST('1.2')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'HGB', 'no_change_expected', 'interaction_cst / class_weight set, or categorical_features with negative categories (#24283); none under D', 'medium',
    [('API_new_param', 'interaction_cst (#21020) and class_weight (#22014) added, defaults None'),
     ('Fix_not_exercised', 'negative categorical values treated as missing (#24283)')],
    [WN('1.2'), RST('1.2'), API('1.2', 'ensemble.HistGradientBoostingClassifier')],
    'strict_pin_may_fail: interaction_cst, class_weight: <absent> (unpinned_new)',
    'interaction_cst: <absent> -> None; class_weight: <absent> -> None (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note;versioned_api_doc', '', 'no numeric change documented for the default path')

# ===== Edge 7: 1.2.2 -> 1.3.2 ==================================================================================
E = EDGES[6]
row(E, 'LR', 'no_change_expected', "solver='liblinear' with penalty='l1' (#25214) or 'newton-cholesky' with max_iter=0 (#26653)", 'medium',
    [('Fix_not_exercised', 'liblinear l1 convergence criterion (#25214); newton-cholesky max_iter=0 crash (#26653)')],
    [WN('1.3'), RST('1.3')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no lbfgs change documented')
row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', "only LinearSVC/LinearSVR dual='auto' (#26093)")],
    [WN('1.3'), RST('1.3')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'kNN', 'no_change_expected', 'sparse-dense pairs (#25044) or brute with non-Euclidean metric (#24076); neither under D', 'medium',
    [('Efficiency_not_exercised', 'CSR x dense squared Euclidean routine (#25044); brute non-Euclidean predict (#24076)'),
     ('API_deprecation_not_exercised', 'non-Euclidean/non-Manhattan metrics deprecated for NearestNeighbors (#24083)')],
    [WN('1.3'), RST('1.3')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no dense-Euclidean change documented')
row(E, 'RF', 'no_change_expected',
    "NaN present in X (tree splitter gained missing-value support for splitter='best', #23595/#26376; the refactored splitter is a no-op without NaN)",
    'medium',
    [('Feature_code_path', "DecisionTree missing-value support with splitter='best' (#23595, #26376); forest support only in 1.4"),
     ('API_removal', "max_features='auto' removed (deprecated 1.1); pinned value from 1.2.2 is 'sqrt'"),
     ('Fix_not_exercised', 'deterministic tree pickles (1.3.2, #27580)')],
    [WN('1.3'), RST('1.3'), PR(23595), PR(26376)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED: median imputation removes all NaN before fit.', 'official_release_note', '',
    'the exercised splitter code was refactored; identity relies on the no-NaN branch being arithmetically unchanged')
row(E, 'GB', 'no_change_expected', 'NaN present in X (same splitter refactor); oob_score (#24882) not used', 'medium',
    [('Feature_code_path', 'tree splitter missing-value refactor (#23595, #26376)'), ('Feature_not_exercised', 'oob_scores_ exposed (#24882)'),
     ('API_removal', "loss='deviance' removed (deprecated 1.1); pinned value from 1.2.2 is 'log_loss'")],
    [WN('1.3'), RST('1.3')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'as for RF')
row(E, 'HGB', 'no_change_expected', 'categorical_features set with negative category codes (#25629); not under D', 'medium',
    [('Fix_not_exercised', 'negative categorical values treated as missing (#25629)'), ('Feature_not_exercised', "loss='gamma' for the regressor (#22409)")],
    [WN('1.3'), RST('1.3')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no numeric change documented for the default path')

# ===== Edge 8: 1.3.2 -> 1.4.2 ==================================================================================
E = EDGES[7]
row(E, 'LR', 'change_expected',
    'always under D (lbfgs): objective rescaled from summed to mean per-sample loss (#26721) -> different convergence path and '
    'coefficients for identical data/tol; binomial gradient numerically stabilised for extreme raw predictions (#28048)',
    'low',
    [('changed_models_list', "LogisticRegression listed in 1.4 'Changed models': lbfgs/newton-cg objective uses average loss, coefficients change (#26721)"),
     ('Fix_fp_level', 'binomial log-loss gradient stable for very large raw predictions; LogisticRegression listed (#28048)')],
    [WN('1.4'), RST('1.4'), PR(26721), PR(28048)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'ACTIVATED on every dataset.', 'official_release_note', 'draft: not coded. Reviewer: change_expected.',
    'mechanism official and unconditional; label-level magnitude dataset-dependent')
row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no sklearn.svm section in 1.4.x')],
    [WN('1.4'), RST('1.4')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'kNN', 'no_change_expected', 'sparse input (1.4.1 partial revert #28235 of #28191 concerns Euclidean pairwise with sparse matrices); dense under D', 'medium',
    [('Fix_not_exercised', 'partial revert for sparse Euclidean pairwise computation; KNeighborsClassifier listed (1.4.1, #28235)'),
     ('Efficiency_not_exercised', 'dense-sparse predict_proba pairs (#27018)')],
    [WN('1.4'), RST('1.4'), PR(28235)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED (dense path).', 'official_release_note', '', 'the revert touched shared pairwise code; dense float64 kernels not listed as changed')
row(E, 'RF', 'no_change_expected',
    'NaN present (forest missing-value support #26391; criterion re-init bug #28295/#28327 requires a prior split with missing values); monotonic_cst set (#13649)',
    'medium',
    [('Feature_not_exercised', 'RandomForest missing-value support (#26391)'), ('API_new_param', 'monotonic_cst added (#13649)'),
     ('Fix_not_exercised', 'criterion not re-initialised when n_missing=0 after a split with missing values (1.4.1, #28295, #28327; closes #28254)'),
     ('API_attribute', 'tree_.value stores weighted fractions instead of counts (1.4.1, #27639); predict argmax unaffected')],
    [WN('1.4'), RST('1.4'), PR(26391), PR(28295), PR(27639)], 'strict_pin_may_fail: monotonic_cst: <absent> (unpinned_new)',
    'monotonic_cst: <absent> -> None (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED: no NaN reaches the trees; #28295 verified to require a preceding missing-value split.', 'official_release_note;pull_request', '',
    'tree_.value semantic change alters an attribute, not predictions')
row(E, 'GB', 'fp_level_change_possible',
    'always: GradientBoostingClassifier loss/gradient path moved to the common _loss module (#26278, #28095) and binomial gradient '
    'stabilised (#28048); same mathematics, fp-level drift; material only for extreme raw predictions',
    'medium',
    [('Efficiency_fp_level', 'GradientBoostingClassifier faster for binary and multiclass (#26278, #28095)'),
     ('Fix_fp_level', 'binomial log-loss gradient numerically stable; GradientBoostingClassifier listed (#28048)')],
    [WN('1.4'), RST('1.4'), PR(26278), PR(28095), PR(28048)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'ACTIVATED at fp level.', 'official_release_note', 'draft: not coded. Reviewer: fp_level_change_possible.', 'fp-only mechanism')
row(E, 'HGB', 'fp_level_change_possible',
    'binary classification (binomial gradient stabilisation #28048; multiclass loss untouched); categorical_features default None -> "warn" '
    'and new max_features=1.0 are behaviour-preserving for float64 numpy input',
    'medium',
    [('Fix_fp_level', 'binomial log-loss gradient stable for large raw predictions; HistGradientBoostingClassifier listed (#28048)'),
     ('API_new_param', 'max_features added, default 1.0 (#27139)'),
     ('DABC_placeholder', "categorical_features default None -> 'warn' (from_dtype option, #26411); numpy input unaffected"),
     ('Efficiency_memory', 'parent histogram reused for children (#27865)')],
    [WN('1.4'), RST('1.4'), PR(28048), PR(27139), API('1.4', 'ensemble.HistGradientBoostingClassifier')],
    'strict_pin_may_fail: max_features: <absent> -> 1.0 (unpinned_new); categorical_features: None -> warn',
    'max_features: <absent> -> 1.0; categorical_features: None -> warn (doc-derived; %s)' % INTROSPECT_NOTE,
    'ACTIVATED at fp level for binary tasks only (review-queue lead confirmed: max_features is a portability, not a model change).',
    'official_release_note;versioned_api_doc', 'draft: not coded. Reviewer: fp_level_change_possible.', 'fp-only mechanism; multiclass datasets unaffected')

# ===== Edge 9: 1.4.2 -> 1.5.2 ==================================================================================
E = EDGES[8]
row(E, 'LR', 'no_change_expected',
    "multi_class='ovr' explicitly requested (now deprecated); under D 'auto' already resolved to multinomial for lbfgs, and 1.5 'always multinomial' is equivalent",
    'medium',
    [('API_deprecation_equivalent', "multi_class deprecated ('auto' -> 'deprecated' placeholder); multinomial always used for >=3 classes (#28703)"),
     ('Enhancement_not_exercised', 'newton-cg verbose output (#27526)')],
    [WN('1.5'), RST('1.5'), PR(28703)],
    "strict_pin_portable_expected (pinned multi_class='auto' accepted with FutureWarning in 1.5); reverse pin 'deprecated' is a placeholder (ineffective)",
    'multi_class: auto -> deprecated (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED (binary: single binomial problem both sides; multiclass: multinomial both sides).', 'official_release_note', '',
    'equivalence asserted from the documented resolution rule, not from a bitwise statement')
row(E, 'SVC', 'no_change_expected', 'C=inf (1.5.2, #29780); not under D', 'medium',
    [('Fix_not_exercised', 'SVC/SVR accept C=inf again (1.5.2, #29780)')],
    [WN('1.5'), RST('1.5')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'kNN', 'no_change_expected', 'Windows + OpenBLAS 0.3.26 (deadlock workaround #28692); Linux workers unaffected', 'medium',
    [('Fix_platform', 'OpenBLAS 0.3.26 Windows deadlock workaround in pairwise distances (#28692)')],
    [WN('1.5'), RST('1.5')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no numeric change documented')
row(E, 'RF', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no RandomForest/tree fit entries in 1.5.x')],
    [WN('1.5'), RST('1.5')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'GB', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no GradientBoosting entries in 1.5.x; 1.5.2 non-OpenMP build performance fix (#29694) is speed-only')],
    [WN('1.5'), RST('1.5')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'HGB', 'fp_level_change_possible',
    'raw decision exactly at the class threshold (predict no longer routes through predict_proba, #27844: sign/argmax of raw scores vs argmax of probabilities, '
    'identical except at fp ties); binning thresholds computed from pre-sorted data (#28102; same percentile method)',
    'medium',
    [('Efficiency_tie_level', 'HistGradientBoostingClassifier.predict no longer calls predict_proba (#27844)'),
     ('Efficiency_code_path', 'data pre-sorted before computing binning thresholds (#28102)'),
     ('Fix_not_exercised', 'monotonic_cst handling for non-categorical features (#28925)')],
    [WN('1.5'), RST('1.5'), PR(27844), PR(28102)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'ACTIVATED at tie/fp level only.', 'official_release_note', 'draft: not coded. Reviewer: fp_level_change_possible.', 'tie-level mechanism; very low prevalence expected')

# ===== Edge 10: 1.5.2 -> 1.6.1 =================================================================================
E = EDGES[9]
row(E, 'LR', 'no_change_expected', "solver='newton-cholesky' (multinomial support #28840, lbfgs fallback fixes #29818/#30100); D uses lbfgs", 'medium',
    [('Enhancement_not_exercised', 'newton-cholesky full multinomial (#28840); fallback and n_iter_ fixes (#29818, #30100)')],
    [WN('1.6'), RST('1.6')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED (review-queue lead confirmed: newton-cholesky only).', 'official_release_note', '', 'no lbfgs change documented')
row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no sklearn.svm section in 1.6.x')],
    [WN('1.6'), RST('1.6')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'kNN', 'no_change_expected', "metric='nan_euclidean' or X=None scoring (#25330, #30047); not under D", 'medium',
    [('Enhancement_not_exercised', 'nan_euclidean metric and NaN inputs (#25330); X=None in predict/score (#30047)')],
    [WN('1.6'), RST('1.6')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no dense-Euclidean change documented')
RETRO = ('Fix_changes_results_retrospective',
         'tree partitioner refactor #29458 (1.6.0, intended as a pure move of code) left FEATURE_THRESHOLD initialised to 0.0 instead of 1e-7, so features '
         'that vary by <=1e-7 within a node are no longer treated as constant; documented only by the 1.8.0 fix note "regression on almost-constant features" (#32259)')
row(E, 'RF', 'change_possible',
    'any node in which some feature has a non-zero value range <=1e-7 (near-constant floating features; exactly-constant one-hot columns are unaffected)',
    'high',
    [RETRO, ('Fix_not_exercised', "entropy/log_loss criterion used ln instead of log2 in 1.6.0, restored in 1.6.1 (#30557); criterion='gini' under D"),
     ('Feature_not_exercised', 'ExtraTrees missing-value support (#28268)')],
    [WN('1.8'), RST('1.8'), PR(32259), PR(29458), WN('1.6'), RST('1.6')], 'strict_pin_portable_expected (constructor parameters unchanged)',
    'none documented (%s)' % INTROSPECT_NOTE,
    'CONDITIONALLY ACTIVATED; prevalence of near-constant (but not constant) floating features in the pool unknown.',
    'later_release_note;pull_request',
    'draft: not coded. Reviewer: change_possible from a LATER release note (1.8.0) describing a 1.6.0 regression. '
    'Author decision: whether the oracle may use notes published after the destination release.',
    'the change is undocumented in 1.6.x notes; trigger prevalence unknown')
row(E, 'GB', 'change_possible', 'same near-constant-feature partitioner regression (#29458/#32259) in the regression trees built by GB', 'high',
    [RETRO], [WN('1.8'), RST('1.8'), PR(32259), PR(29458), WN('1.6'), RST('1.6')], 'strict_pin_portable_expected (constructor parameters unchanged)',
    'none documented (%s)' % INTROSPECT_NOTE, 'CONDITIONALLY ACTIVATED.', 'later_release_note;pull_request',
    'draft: not coded. Reviewer: change_possible (retrospective evidence; see RF).', 'as for RF')
row(E, 'HGB', 'no_change_expected',
    'pandas/polars categorical input (categorical_features default "warn" -> "from_dtype"); numpy float64 input has no dtype categories; '
    'threshold search parallelised per feature (#28064) with OMP threads=1',
    'medium',
    [('DABC_placeholder_equivalent', "categorical_features default 'warn' -> 'from_dtype' (announced 1.4, effective 1.6)"),
     ('Efficiency_code_path', 'initial bin-threshold search parallelised across features (#28064); per-feature values unchanged')],
    [WN('1.6'), RST('1.6'), PR(28064), API('1.5', 'ensemble.HistGradientBoostingClassifier'), API('1.6', 'ensemble.HistGradientBoostingClassifier')],
    "strict_pin_may_fail: categorical_features: warn -> from_dtype (pinned 'warn' is a placeholder -> ineffective)",
    'categorical_features: warn -> from_dtype (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED (review-queue lead: performance-only change should not be assumed bitwise identical without validation - noted as uncertainty).',
    'official_release_note;versioned_api_doc', '', 'parallel-by-feature does not reorder within-feature arithmetic')

# ===== Edge 11: 1.6.1 -> 1.7.2 =================================================================================
E = EDGES[10]
row(E, 'LR', 'no_change_expected',
    "solver='liblinear' with n_classes>2 (OvR scheme deprecated, raises in 1.8, #31241); newton-cholesky warm_start (1.7.2, #31866); class_weight='balanced' with sample_weight (#30057)",
    'medium',
    [('Changed_models_message_only', 'lbfgs ConvergenceWarning message made more informative (#31316); text only'),
     ('API_deprecation_not_exercised', 'liblinear one-vs-rest multiclass deprecated (#31241)'),
     ('Fix_not_exercised', 'newton-cholesky multiclass warm_start (1.7.2, #31866)')],
    [WN('1.7'), RST('1.7'), PR(31316), PR(31241)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    "NOT ACTIVATED. Note: the worker counts warnings whose message contains 'onverg'; verify the new message still matches.", 'official_release_note', '',
    'no numeric lbfgs change documented')
row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'only LinearSVC class_weight/sample_weight interaction (#30057)')],
    [WN('1.7'), RST('1.7')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'kNN', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no sklearn.neighbors section in 1.7.x')],
    [WN('1.7'), RST('1.7')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'RF', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no sklearn.tree section and no forest entries in 1.7.x')],
    [WN('1.7'), RST('1.7')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'GB', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no GradientBoosting entries in 1.7.x')],
    [WN('1.7'), RST('1.7')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'HGB', 'no_change_expected', 'X_val/y_val passed to fit (#27124); not under D', 'medium',
    [('Feature_not_exercised', 'validation data can be passed to fit for early stopping (#27124)')],
    [WN('1.7'), RST('1.7'), PR(27124)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'no numeric change documented')

# ===== Edge 12: 1.7.2 -> 1.8.0 =================================================================================
E = EDGES[11]
row(E, 'LR', 'no_change_expected',
    "solver='liblinear' with n_classes>2 (now raises; structural failure for old configs); penalty -> l1_ratio migration is L2-equivalent (penalty 'l2' ~ l1_ratio 0.0)",
    'low',
    [('API_deprecation_equivalent', 'penalty deprecated (removal 1.10); l1_ratio default None -> 0.0 (pure L2, unchanged) (#32659)'),
     ('API_removal_note', 'multi_class parameter removed in 1.8 (deprecated 1.5); liblinear multiclass raises (per 1.8 API doc)'),
     ('API_not_exercised', 'liblinear raises when |X|>1e30 (#31888); n_jobs deprecated (#32742)')],
    [WN('1.8'), RST('1.8'), PR(32659), API('1.8', 'linear_model.LogisticRegression')],
    "strict_pin_may_fail: multi_class: deprecated -> <absent> (removed); penalty: l2 -> deprecated; l1_ratio: None -> 0.0 (pinned None deprecated)",
    'multi_class: deprecated -> <absent>; penalty: l2 -> deprecated; l1_ratio: None -> 0.0 (doc-derived; %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED for predictions (L2 objective identical); strict pin from 1.7.2 is NOT portable (multi_class removed).',
    'official_release_note;versioned_api_doc',
    "draft: estimated_compatibility 'strict_pin_portable_expected' is contradicted by the 1.8 API doc (multi_class removed).",
    'equivalence of penalty=l2 and l1_ratio=0.0 is documented')
row(E, 'SVC', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no sklearn.svm bullets in 1.8.0')],
    [WN('1.8'), RST('1.8')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'kNN', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no sklearn.neighbors bullets in 1.8.0; pairwise changes are Array-API support only (#29822, #32597)')],
    [WN('1.8'), RST('1.8')], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note', '', 'absence of entries is not proof of absence of change')
row(E, 'RF', 'change_possible',
    'any node in which some feature has a non-zero value range <=1e-7 (FEATURE_THRESHOLD restored to 1e-7, reversing the 1.6-1.7 behaviour, #32259); '
    'NaN present (#32274, #32351, #32280) is not met',
    'medium',
    [('Fix_changes_results', 'regression on almost-constant features fixed: FEATURE_THRESHOLD initialisation in _partitioner.pxd (#32259)'),
     ('Fix_not_exercised', 'near-constant features with missing values no longer stop splitting early (#32274); last non-missing sample partition (#32351); decision_path NaN (#32280)'),
     ('Fix_not_exercised', 'Bagging/IsolationForest sample_weight draws (#31414, #32825); not RandomForest in 1.8')],
    [WN('1.8'), RST('1.8'), PR(32259), PR(32274)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'CONDITIONALLY ACTIVATED (same trigger as 1.5.2->1.6.1, reversed).', 'official_release_note;pull_request',
    "draft: no_change_expected with trigger 'NaN present' (#32274 only). Reviewer: change_possible because #32259 applies without missing values.",
    'trigger prevalence unknown')
row(E, 'GB', 'change_possible', 'same near-constant-feature fix (#32259) in GB regression trees; absolute_error fix (#32100) not exercised (friedman_mse)', 'medium',
    [('Fix_changes_results', 'FEATURE_THRESHOLD fix (#32259)'), ('Fix_not_exercised', 'absolute_error criterion O(n log n) and sub-optimal splits (#32100)'),
     ('Efficiency_not_exercised', 'weighted percentile rewrite (#31775); not used by GB classification defaults')],
    [WN('1.8'), RST('1.8'), PR(32259), PR(32100)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'CONDITIONALLY ACTIVATED.', 'official_release_note;pull_request', 'draft: no_change_expected (0 entries). Reviewer: change_possible.', 'trigger prevalence unknown')
row(E, 'HGB', 'no_change_expected', NONE_TRIG, 'medium',
    [('no_relevant_entries', 'no HistGradientBoosting bullets in 1.8.0; weighted-percentile rewrite (#31775) not used by 1.8 binning (np.percentile, method=midpoint; verified in 1.8.0 source)')],
    [WN('1.8'), RST('1.8'), SRC('1.8.0', 'sklearn/ensemble/_hist_gradient_boosting/binning.py')], 'strict_pin_portable_expected (constructor parameters unchanged)',
    'none documented (%s)' % INTROSPECT_NOTE, 'NOT ACTIVATED.', 'official_release_note;source_code_tag', '', 'absence of entries is not proof of absence of change')

# ===== Edge 13: 1.8.0 -> 1.9.1 =================================================================================
E = EDGES[12]
row(E, 'LR', 'no_change_expected', 'float32 input (lbfgs gradients now computed in float32, #32644); pipeline is float64', 'medium',
    [('Efficiency_dtype_conditional', 'lbfgs gradients in float32 for float32 input; Array API support for lbfgs (#32644)'),
     ('Feature_not_exercised', 'callbacks for lbfgs (#33322)')],
    [WN('1.9'), RST('1.9'), PR(32644)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED (float64).', 'official_release_note', '', 'the Array-API refactor touched the exercised float64 path; identity not stated')
row(E, 'SVC', 'no_change_expected', "probability=True (parameter deprecated, #32050); default False -> 'deprecated' behaves as False", 'low',
    [('API_deprecation', "probability deprecated (default False -> 'deprecated'); probA_/probB_ deprecated (#32050, #33388)")],
    [WN('1.9'), RST('1.9'), PR(32050)], 'strict_pin_may_fail: probability: False -> deprecated (pinned False accepted with warning; reverse pin is a placeholder)',
    'probability: False -> deprecated (confirmed in installed 1.9.1: SVC().get_params(); %s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED.', 'official_release_note;local_introspection_1.9.1', '', 'no numeric change documented')
row(E, 'kNN', 'no_change_expected',
    'kd_tree/ball_tree with many tied distances (simultaneous_sort quadratic-path fix #33252 changes only the ordering of returned ties, not the k-set); string labels with brute (#33048)',
    'medium',
    [('Fix_tie_order', 'quadratic-time path in simultaneous_sort used by BallTree/KDTree with many ties (#33252)'),
     ('Fix_not_exercised', 'string labels with brute (#33048); 1.9.1 radius_neighbors array-like radius (#34341)')],
    [WN('1.9'), RST('1.9'), PR(33252)], 'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'NOT ACTIVATED for the majority vote (order of ties within the k-set is irrelevant to the mode); would matter for weights="distance" ties.',
    'official_release_note', 'DISPUTED (minor): author may prefer fp_level_change_possible to be conservative about tie handling.', 'tie-order mechanism only')
row(E, 'RF', 'no_change_expected', 'sample_weight given (bootstrap draws now weighted, #31529; verified in 1.9.1 source that sample_weight=None keeps randint draws)', 'low',
    [('Fix_conditional', 'bootstrap indices drawn with sample_weight (#31529); None under D -> unchanged randint path'),
     ('Fix_not_exercised', 'missing-value fixes (#32119, #27630, #32193); friedman_mse deprecation for regressors (#32708)')],
    [WN('1.9'), RST('1.9'), PR(31529), SRC('1.9.1', 'sklearn/ensemble/_forest.py')], 'strict_pin_portable_expected (constructor parameters unchanged)',
    'none documented (%s)' % INTROSPECT_NOTE, 'NOT ACTIVATED (source-verified).', 'official_release_note;source_code_tag', '', 'source verified for the None branch')
row(E, 'GB', 'fp_level_change_possible', "always (criterion 'friedman_mse' impurity scaling fixed by switching to 'squared_error', equivalent up to floating point, #32708)", 'low',
    [('Fix_changes_results', "friedman_mse impurity scaling incorrect; now squared_error, 'equivalent up to floating-point error' (#32708)"),
     ('API_deprecation', "criterion deprecated (default 'friedman_mse' -> 'deprecated')")],
    [WN('1.9'), RST('1.9'), PR(32708)], 'strict_pin_may_fail: criterion: friedman_mse -> deprecated',
    'criterion: friedman_mse -> deprecated (confirmed in installed 1.9.1; %s)' % INTROSPECT_NOTE,
    'ACTIVATED at fp level.', 'official_release_note;local_introspection_1.9.1', '', 'official wording: equivalent up to fp')
row(E, 'HGB', 'change_expected',
    'any feature with >255 distinct values in a training fold: unweighted bin edges switch from np.percentile(method="midpoint") (1.8.0 source) to '
    'method="averaged_inverted_cdf" plus duplicate-threshold removal (1.9.1 source); features with <=255 distinct values keep midpoint edges',
    'medium',
    [('Fix_changes_results', 'weight-aware bin edges: midpoints when distinct values < max_bins, otherwise (weighted) averaged inverted CDF quantiles (#29641)')],
    [WN('1.9'), RST('1.9'), PR(29641), SRC('1.8.0', 'sklearn/ensemble/_hist_gradient_boosting/binning.py'), SRC('1.9.1', 'sklearn/ensemble/_hist_gradient_boosting/binning.py')],
    'strict_pin_portable_expected (constructor parameters unchanged)', 'none documented (%s)' % INTROSPECT_NOTE,
    'ACTIVATED on datasets with high-cardinality numeric features; the release note wording ("unweighted edges unchanged") holds only for the <=max_bins branch.',
    'official_release_note;source_code_tag',
    "draft: no_change_expected with trigger 'sample_weight given or n_samples > subsample'. Reviewer: change_expected - the unweighted quantile method itself changed (source-verified on both sides).",
    'trigger prevalence in the pool not inspected (blinding)')

assert len(R) == 78, len(R)

# ---------------------------------------------------------------------------------------------------------------
# Emit the reviewed frame (original schema) and the evidence file
# ---------------------------------------------------------------------------------------------------------------
working = list(csv.DictReader(open(WORKING, newline='')))
cols = list(working[0].keys())
wmap = {(r['edge'], r['estimator']): r for r in working}
assert set(wmap) == set(R)

frame_rows, ev_rows = [], []
for edge in EDGES:
    for est in ESTIMATORS:
        d = R[(edge, est)]; w = wmap[(edge, est)]
        refs = '; '.join(dict.fromkeys(d['refs']))
        frame_rows.append(dict(
            edge=edge, estimator=est, coding_status='coded_complete', n_entries=len(d['entries']),
            entry_categories=';'.join(sorted({c for c, _ in d['entries']})),
            predicted_impact_category=d['cat'], trigger_condition=d['trig'], applicable_data_types=DTYPES,
            estimated_compatibility=d['compat'], release_note_reference=refs, coding_uncertainty=d['unc'],
            coder=CODER, coded_on=CODED_ON, auto_default_diff=d['autodiff'], pipeline_conditions=PIPE))
        ev_rows.append(dict(
            edge=edge, estimator=est, destination_release_notes_read=edge.split('->')[1],
            draft_predicted_impact_category=w['predicted_impact_category'] or '(not coded)',
            draft_coding_status=w['coding_status'],
            reviewed_predicted_impact_category=d['cat'],
            relevant_changes=' | '.join('[%s] %s' % (c, t) for c, t in d['entries']),
            activation_under_pipeline_conditions=d['activation'],
            trigger_condition=d['trig'],
            observed_api_default_changes=d['autodiff'],
            estimated_strict_pin_compatibility=d['compat'],
            evidence_types=d['evidence_type'],
            source_urls=refs,
            dependency_stack_caveat=DEP_CAVEAT[est],
            coding_uncertainty=d['unc'], uncertainty_reason=d['unc_reason'],
            disputed_or_changed_vs_draft=d['disputed'] or 'consistent with draft' if w['predicted_impact_category'] else (d['disputed'] or 'new coding (draft blank)'),
            measured_effect='NOT EXAMINED (blinded; Stage B / A-A outcome files were not opened)',
            reviewer=CODER, coded_on=CODED_ON))

with open(OUT / 'ORACLE_FRAME_REVIEWED.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=cols, lineterminator='\n'); wr.writeheader(); wr.writerows(frame_rows)
with open(OUT / 'ORACLE_EVIDENCE_78.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()), lineterminator='\n'); wr.writeheader(); wr.writerows(ev_rows)
json.dump({'%s|%s' % k: v for k, v in R.items()}, open(OUT / '_coding_source.json', 'w'), indent=1)
print('rows', len(frame_rows), 'categories', {c: sum(r['predicted_impact_category'] == c for r in frame_rows)
                                             for c in ['no_change_expected', 'fp_level_change_possible', 'change_possible', 'change_expected']})
