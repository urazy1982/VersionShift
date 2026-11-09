"""Experiment A2 (v2.1) — separate the rerun noise floor into its parts.

For every dataset and the replicate-0 split set (identical to Experiment A), each
stochastic estimator is refit with S model seeds on the SAME folds:
  * fixed-split seed noise   : variance across model seeds, splits held fixed
  * split-to-split noise     : from Experiment A (fresh split per replicate), minus seed noise
  * paired cross-regime shift: same split and same seed, only the configuration changes
Deterministic estimators (LR, SVC, kNN; GB without subsampling is checked) have zero
seed noise by construction; this is measured, not assumed.
"""
import argparse, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regimes import build
from suite import load_all, h
from run_emulation import balanced_accuracy

CELLS = [('RF', 'R21'), ('RF', 'R19'), ('GB', 'R19'), ('HGB', 'R19'), ('LR', 'R19'), ('SVC', 'R19'), ('kNN', 'R19')]


def seed_for(name, f, a, s):
    # s == 0 reproduces Experiment A's replicate-0 seed exactly
    return h('model', name, 0, f, a) if s == 0 else h('model', name, 0, f, a, 'seedrep', s)


def run(args):
    ds, S = args
    X, y = ds['X'], ds['y']; k = int(y.max() + 1)
    skf = StratifiedKFold(5, shuffle=True, random_state=h('split', ds['name'], 0))
    rows = []
    for f, (tr, te) in enumerate(skf.split(X, y)):
        for a, g in CELLS:
            for s in range(S):
                est = build(a, g, 'D', seed_for(ds['name'], f, a, s), k)
                est.fit(X[tr], y[tr]); p = np.asarray(est.predict(X[te]), np.int64)
                rows.append(dict(dataset=ds['name'], origin=ds['origin'], fold=f, algo=a, regime=g, seed_rep=s,
                                 ba=balanced_accuracy(y[te], p, k)))
    return rows, ds['name']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = sorted(load_all(), key=lambda d: -(len(d['y']) * d['X'].shape[1] * (d['y'].max() + 1)))
    rows = []; t0 = time.time()
    with Pool(a.workers) as pool:
        for r, name in pool.imap_unordered(run, [(d, a.seeds) for d in data]):
            rows += r; print(name, f'{time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out / 'seed_reruns.csv.gz', index=False)
    (out / 'META.json').write_text(json.dumps(dict(seeds=a.seeds, datasets=len(data), seconds=time.time() - t0)))


if __name__ == '__main__':
    main()
