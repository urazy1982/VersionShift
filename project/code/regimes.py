"""VersionShift v2.0 — configuration regimes for within-runtime default emulation.

A *regime* is the constructor configuration a practitioner would obtain from a
given scikit-learn era when relying on shipped defaults (D) or on the archived
partial explicit map (E). Emulating an old regime inside the newest runtime
isolates the *default-change* (DABC) component of an upgrade from the
*implementation* component, which needs the real old binaries (crossversion/).

Regime sources (official changelogs / API pages):
  * 0.22: SVC gamma 'auto' -> 'scale'; RandomForest n_estimators 10 -> 100;
          LogisticRegression solver 'liblinear' -> 'lbfgs', multi_class 'warn'(ovr) -> 'auto'.
  * 1.1:  RandomForestClassifier max_features 'auto' -> 'sqrt' (identical meaning for classifiers).
  * 1.8:  LogisticRegression penalty deprecated, l1_ratio None -> 0.0 (identical L2 meaning).
  * kNN, GradientBoosting, HistGradientBoosting: no default change between 0.21 and 1.9
    affecting n_samples < 10,000 (HGB early_stopping='auto' only activates above 10k).

Emulation fidelity notes:
  * 1.9 removed multiclass support for solver='liblinear' and the multi_class
    parameter; 0.21 one-vs-rest is emulated with OneVsRestClassifier, which fits
    the same binary problems and normalizes probabilities the same way.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.multiclass import OneVsRestClassifier

ALGORITHMS = ('LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB')
SCALE_SENSITIVE = ('LR', 'SVC', 'kNN')
REGIMES = ('R21', 'R19')          # 0.21-era defaults vs 1.9.1 shipped defaults
POLICIES = ('D', 'E')             # shipped defaults vs archived partial explicit map

# Archived EXPLICIT map from VersionShift v1.x original_models.py (unchanged).
EXPLICIT = {
    'LR': dict(C=1.0, solver='lbfgs', max_iter=1000, tol=1e-4),
    'SVC': dict(C=1.0, kernel='rbf', gamma='scale'),
    'kNN': dict(n_neighbors=5, weights='uniform', algorithm='brute', metric='euclidean'),
    'RF': dict(n_estimators=300, max_features='sqrt', min_samples_leaf=1, bootstrap=True),
    'HGB': dict(max_iter=200, learning_rate=0.1, early_stopping=False, max_bins=255),
    'GB': dict(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0),
}

# Parameters whose shipped default differs between the 0.21 era and 1.9.1 (semantic change only).
R21_DEFAULT_OVERRIDES = {
    'LR': dict(solver='liblinear'),          # + one-vs-rest for multiclass
    'SVC': dict(gamma='auto'),
    'RF': dict(n_estimators=10),
    'kNN': {}, 'GB': {}, 'HGB': {},
}


def _base(algorithm, kwargs, seed):
    kw = dict(kwargs)
    if algorithm in ('RF', 'GB', 'HGB'):
        kw['random_state'] = int(seed)
    if algorithm in ('RF', 'kNN'):
        kw['n_jobs'] = 1
    cls = dict(LR=LogisticRegression, SVC=SVC, kNN=KNeighborsClassifier, RF=RandomForestClassifier,
               GB=GradientBoostingClassifier, HGB=HistGradientBoostingClassifier)[algorithm]
    return cls(**kw)


def config(algorithm, regime, policy):
    """Constructor kwargs plus whether 0.21 one-vs-rest semantics apply."""
    if regime not in REGIMES or policy not in POLICIES or algorithm not in ALGORITHMS:
        raise ValueError('unknown regime/policy/algorithm')
    kw = dict(EXPLICIT[algorithm]) if policy == 'E' else {}
    ovr = False
    if regime == 'R21':
        for k, v in R21_DEFAULT_OVERRIDES[algorithm].items():
            kw.setdefault(k, v)          # explicit pins win over era defaults
        if algorithm == 'LR':
            ovr = True                   # multi_class default was 'ovr' in 0.21, unpinned by E
    return kw, ovr


def build(algorithm, regime, policy, seed, n_classes):
    kw, ovr = config(algorithm, regime, policy)
    est = _base(algorithm, kw, seed)
    if ovr and n_classes > 2:
        est = OneVsRestClassifier(est)
    return est


def effective_signature(algorithm, regime, policy, n_classes):
    """Hashable description of what is actually fitted; equal signatures => identical fits."""
    kw, ovr = config(algorithm, regime, policy)
    est = _base(algorithm, kw, 0)
    params = tuple(sorted((k, repr(v)) for k, v in est.get_params(deep=False).items()))
    return (algorithm, params, bool(ovr and n_classes > 2))
