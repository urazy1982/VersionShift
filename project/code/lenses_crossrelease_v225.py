"""Apply the v2.2.5 secondary lenses (B, C, D) to real cross-release runs, behind the same gates as the primary:
blinding gate, strict fold validation and frozen-plan coverage (analysis_v221). Secondary analyses never change
the primary labels.

  python3 code/lenses_crossrelease_v225.py --inp RUN/CROSS_RELEASE_FOLDS.csv.gz --edge 0.21.3:0.22.2.post1 \
      --policy D --plan-dir PLAN --selection STAGE_B_SELECTION.json --oracle-frozen ORACLE_FROZEN.json --out L.json
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_v221 as AN
import lenses_v225 as L

ALGOS = AN.ALGOS
PAIRS = list(itertools.combinations(ALGOS, 2))


def rerun_pair_shift(ok, old, new, policy_old, policy_new):
    """X[k,p,r] (pp) on complete-case datasets (every algorithm present in both cells)."""
    def cell(rel, pol):
        s = ok[(ok.release == rel) & (ok.policy == pol)]
        return s.pivot(index=['dataset', 'rep', 'fold'], columns='algo', values='ba')
    o, n = cell(old, policy_old), cell(new, policy_new)
    ds = sorted(set(o.index.get_level_values('dataset')) & set(n.index.get_level_values('dataset')))
    keep = [d for d in ds if o.loc[d].reindex(columns=ALGOS).notna().all().all() and n.loc[d].reindex(columns=ALGOS).notna().all().all()]
    X = np.stack([np.stack([((n.loc[d][a] - n.loc[d][b]) - (o.loc[d][a] - o.loc[d][b])).groupby(level='rep').mean().sort_index().to_numpy() * 100
                            for a, b in PAIRS]) for d in keep]) if keep else np.zeros((0, len(PAIRS), 0))
    return X, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', nargs='+', required=True); ap.add_argument('--edge', required=True)
    ap.add_argument('--policy', default='D'); ap.add_argument('--policy-new', default=None)
    ap.add_argument('--plan-dir'); ap.add_argument('--selection'); ap.add_argument('--oracle-frozen')
    ap.add_argument('--unblinded-exploratory', action='store_true'); ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--folds', type=int, default=5); ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--order-seed', type=int, default=20261106); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    head = AN.gate(a); df = pd.concat([pd.read_csv(p) for p in a.inp], ignore_index=True)
    ok = AN.validate(df, a.reps, a.folds); old, new = a.edge.split(':'); pn = a.policy_new or a.policy
    cov = AN.coverage_or_exploratory(a, df, {(old, a.policy), (new, pn)})
    ok = ok[~ok.dataset.astype(str).isin(cov['excluded_datasets_failed_default_jobs'])]
    X, ds = rerun_pair_shift(ok, old, new, a.policy, pn)
    names = ['%s-%s' % p for p in PAIRS]
    Mk = np.abs(X.mean(2)).mean(1) if len(ds) else np.array([])
    rep = dict(header=head, role='SECONDARY (pre-registered v2.2.5); never changes primary labels', edge=a.edge,
               policies=dict(old=a.policy, new=pn), plan_coverage=cov, datasets=ds,
               lensB_random_effects_by_pair=L.lens_b(X, names) if len(ds) > 2 else 'needs >= 3 datasets',
               lensC_discovery=L.lens_c(X) if len(ds) else None,
               lensD_anytime_valid=L.lens_d(Mk, a.margin, order_seed=a.order_seed) if len(ds) and np.any(Mk > 0) else
                                   dict(rejected=False, note='all M_k are zero or no datasets'),
               caveats=['lens C FDR approximately controlled: 10.7-12.0% observed at nominal 10% (CALIBRATION_LENSES_V225.json)',
                        'lens B prediction intervals assume roughly normal true effects; sparse mixtures under-cover at small K',
                        'lens D is conservative; its role is valid continuation across Stage B -> C'])
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ('edge', 'datasets', 'lensC_discovery', 'lensD_anytime_valid')}, default=float)[:800])


if __name__ == '__main__':
    main()
