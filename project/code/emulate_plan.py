"""Run the 0.21-era vs current-default emulation on a FROZEN Stage-B plan (same datasets, folds, seeds),
emitting rows in the CROSS_RELEASE_FOLDS format so analysis_v22 can compare emulated and real effects.
Releases are labelled emu_R21 / emu_R19 and must be analysed with --emulation-only."""
import argparse, json, sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regimes import build

ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--datasets', default=None, help='comma list (default: all in plan)')
    a = ap.parse_args(); plan_dir = Path(a.plan_dir); plan = json.loads((plan_dir / 'PLAN.json').read_text())
    keep = set(a.datasets.split(',')) if a.datasets else None
    rows = []
    for d in plan['datasets']:
        if keep and d['name'] not in keep: continue
        z = np.load(plan_dir / d['dataset_npz'], allow_pickle=False); X = np.asarray(z['X'], float); y = z['y']
        folds = np.load(plan_dir / d['folds_npz'], allow_pickle=False); k = int(y.max()) + 1
        for key in d['fold_keys']:
            tr, te = folds[key + '_train'], folds[key + '_test']
            Xtr, Xte = X[tr], X[te]
            if np.isnan(Xtr).any() or np.isnan(Xte).any():
                med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
                Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
            r, f = key[1:].split('_f')
            for g in ('R21', 'R19'):
                for alg in ALGOS:
                    est = build(alg, g, 'D', d['seeds'][key], k); est.fit(Xtr, y[tr]); p = est.predict(Xte)
                    ba = float(np.mean([np.mean(p[y[te] == c] == c) for c in range(k) if np.any(y[te] == c)]))
                    rows.append(dict(condition='emu_' + g, release='emu_' + g, dataset=d['name'], algo=alg, policy='D',
                                     status='ok', portability='ok', rep=int(r), fold=int(f), ba=ba, k=k, n=d['n']))
    pd.DataFrame(rows).to_csv(a.out, index=False); print('rows', len(rows))


if __name__ == '__main__':
    main()
