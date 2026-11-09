#!/usr/bin/env python3
"""VersionShift G3-D historical-compatible explicit-map API preflight.
Run separately with scikit-learn 0.21.3 and 0.22.2.post1.
Diagnostic only: no original result directories are modified.
"""
import argparse
import inspect
import json
import sys
import warnings
import hashlib
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

CONFIG = {
    'LR': (LogisticRegression, {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 0.0001}),
    'SVC': (SVC, {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}),
    'kNN': (KNeighborsClassifier, {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'brute', 'metric': 'euclidean'}),
    'RF': (RandomForestClassifier, {'n_estimators': 300, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'bootstrap': True}),
    'GB': (GradientBoostingClassifier, {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0}),
    'HGB': (HistGradientBoostingClassifier, {'max_iter': 200, 'learning_rate': 0.1, 'n_iter_no_change': None, 'max_bins': 255}),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    X, y = load_iris(return_X_y=True)
    # Small deterministic smoke test only, not an empirical study result.
    Xtr, Xte = X[::2], X[1::2]
    ytr = y[::2]
    report = {'sklearn': sklearn.__version__, 'python': sys.version.split()[0], 'role': 'API-and-fit smoke test; not a mitigation claim', 'tests': {}}
    for label, (cls, base) in CONFIG.items():
        kw = dict(base)
        if 'random_state' in inspect.signature(cls).parameters and label in ('RF','GB','HGB'):
            kw['random_state'] = 42
        if label in ('RF','kNN'):
            kw['n_jobs'] = 1
        item = {'requested': kw}
        try:
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter('always')
                est = cls(**kw)
                actual = est.get_params(deep=False)
                if any(actual.get(k) != v for k,v in kw.items()):
                    raise ValueError('Constructor does not retain requested parameters')
                est.fit(Xtr, ytr)
                pred = np.asarray(est.predict(Xte), dtype=np.int64)
            item.update(status='PASS', predictions=int(pred.size), prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(), warnings=[str(w.message) for w in warn])
        except Exception as exc:
            item.update(status='FAIL', error=repr(exc))
        report['tests'][label] = item
    report['status'] = 'PASS' if all(v['status']=='PASS' for v in report['tests'].values()) else 'FAIL'
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({'version':report['sklearn'],'status':report['status'],'per_estimator':{k:v['status'] for k,v in report['tests'].items()}},indent=2))
    if report['status']!='PASS': sys.exit(1)
if __name__ == '__main__': main()
