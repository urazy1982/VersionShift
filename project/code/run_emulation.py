"""Experiment A — within-runtime default-regime emulation (scikit-learn 1.9.1, CPU, offline).

For every dataset, R replicates x 5 stratified folds. Replicate r has its own split
seed and model seeds; regimes and policies are evaluated on *identical* splits and
seeds (paired). Fits with identical effective configuration are computed once and
shared, which makes the regime contrast exactly zero by construction where no
default changed (built-in negative controls: kNN, GB, HGB).
"""
import argparse, hashlib, json, os, sys, time, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regimes import ALGORITHMS, SCALE_SENSITIVE, REGIMES, build, effective_signature
from suite import load_all, h

TREE = ('RF', 'GB', 'HGB')


def balanced_accuracy(y, p, k):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(k) if np.any(y == c)]))


def cells_for(k):
    cells = []
    for a in ALGORITHMS:
        preps = ('raw', 'std') if a in SCALE_SENSITIVE else ('raw',)
        policies = ('D', 'E') if a not in TREE else ('D',)
        for prep in preps:
            for pol in policies:
                for g in REGIMES:
                    cells.append((a, g, pol, prep))
    return cells


def run_dataset(args):
    ds, reps = args
    X, y = ds['X'], ds['y']; k = int(y.max() + 1)
    rows, dis = [], []
    cells = cells_for(k)
    t0 = time.time()
    for r in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=h('split', ds['name'], r))
        for f, (tr, te) in enumerate(skf.split(X, y)):
            Xs = {'raw': (X[tr], X[te])}
            sc = StandardScaler().fit(X[tr]); Xs['std'] = (sc.transform(X[tr]), sc.transform(X[te]))
            cache = {}
            preds = {}
            for (a, g, pol, prep) in cells:
                sig = (effective_signature(a, g, pol, k), prep)
                if sig not in cache:
                    seed = h('model', ds['name'], r, f, a)          # same seed across regimes/policies
                    est = build(a, g, pol, seed, k)
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter('always', ConvergenceWarning)
                        t = time.time(); est.fit(Xs[prep][0], y[tr]); fit_s = time.time() - t
                    p = np.asarray(est.predict(Xs[prep][1]), dtype=np.int64)
                    nconv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
                    cache[sig] = (p, balanced_accuracy(y[te], p, k), nconv, fit_s, len(cache))
                p, ba, nconv, fit_s, fit_id = cache[sig]
                preds[(a, g, pol, prep)] = p
                rows.append(dict(dataset=ds['name'], origin=ds['origin'], hetero=ds['hetero'], n=len(y),
                                 p=X.shape[1], k=k, rep=r, fold=f, algo=a, regime=g, policy=pol, prep=prep,
                                 ba=ba, conv_warn=nconv, fit_s=fit_s, fit_id=fit_id,
                                 pred_sha=hashlib.sha256(p.tobytes()).hexdigest()[:16]))
            for (a, g, pol, prep) in cells:
                if g == 'R21':
                    p0, p1 = preds[(a, 'R21', pol, prep)], preds[(a, 'R19', pol, prep)]
                    dis.append(dict(dataset=ds['name'], rep=r, fold=f, algo=a, policy=pol, prep=prep,
                                    n_test=len(te), n_disagree=int((p0 != p1).sum())))
    return rows, dis, ds['name'], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = load_all()
    if a.limit: data = data[:a.limit]
    import sklearn, scipy
    meta = dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__,
                python=sys.version.split()[0], reps=a.reps, folds=5, datasets=len(data),
                started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    rows, dis = [], []
    # largest datasets first for load balance
    order = sorted(data, key=lambda d: -(len(d['y']) * d['X'].shape[1] * (d['y'].max() + 1)))
    with Pool(a.workers) as pool:
        for r, d, name, secs in pool.imap_unordered(run_dataset, [(ds, a.reps) for ds in order]):
            rows += r; dis += d
            print(f'{name}: {secs:.1f}s', flush=True)
    pd.DataFrame(rows).to_csv(out / 'fold_scores.csv.gz', index=False)
    pd.DataFrame(dis).to_csv(out / 'prediction_disagreement.csv.gz', index=False)
    meta['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    meta['datasets_detail'] = [dict(name=d['name'], origin=d['origin'], n=len(d['y']), p=d['X'].shape[1],
                                    k=int(d['y'].max() + 1), hetero=d['hetero'],
                                    spec={k: v for k, v in d.get('spec', {}).items()}) for d in data]
    (out / 'RUN_META.json').write_text(json.dumps(meta, indent=2, default=str))


if __name__ == '__main__':
    main()
