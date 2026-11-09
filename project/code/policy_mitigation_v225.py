"""Experiment E-MIT (v2.2.5): do practitioner pinning policies protect conclusions across the 0.21 -> 1.9.1
default-change wave? Uses the existing Experiment A fits (44 datasets x 6 reruns x 5 folds, scikit-learn 1.9.1);
no new model fits are needed because every policy's configuration is either identical to an already-fitted
configuration (verified via regimes.effective_signature) or structurally unchanged across regimes.

Policies (code written in the 0.21 era, then run unchanged under 1.9.1 defaults):
  D      shipped defaults (no explicit hyperparameters)
  S6973  SonarSource rule python:S6973 compliance with the values in effect at writing time:
           SVC C, kernel, gamma pinned; kNN n_neighbors; RF min_samples_leaf, max_features; GB/HGB learning_rate;
           LogisticRegression is not covered by the rule.
           RF max_features='auto' (the 0.21 default) is REJECTED by 1.9.1 -> hard failure as written ('S6973');
           'S6973fix' = the same code after the documented equivalent rewrite max_features='sqrt'.
  E      the archived explicit map (pins SVC gamma, RF n_estimators, LR solver/max_iter; not LR multi_class)
  P      full pin of every constructor parameter (identical configurations -> zero within one runtime;
         in real binaries the P residual is the runtime-stack effect, measured in Stage B)
Outputs per policy and feature condition: decision_v221 labels and quantities + lenses B/C/D.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regimes as RG
import decision_v221 as DV
import lenses_v225 as L

ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']; TREES = {'RF', 'GB', 'HGB'}
PAIRS = list(itertools.combinations(range(6), 2)); PAIR_NAMES = ['%s-%s' % (ALGOS[a], ALGOS[b]) for a, b in PAIRS]


def shift_array(df, algo, policy, prep, datasets):
    pp = 'raw' if algo in TREES else prep
    sub = df[(df.algo == algo) & (df.policy == policy) & (df.prep == pp)]
    piv = sub.pivot(index=['dataset', 'rep', 'fold'], columns='regime', values='ba')
    d = (piv['R19'] - piv['R21']).unstack('fold').reindex(pd.MultiIndex.from_product([datasets, range(6)]))
    return d.to_numpy().reshape(len(datasets), 6, 5) * 100          # pp


def verify_structural_zeros():
    """S6973 SVC: explicit 0.21 values == R21 shipped config, so its R19 run equals the fitted R21 D run."""
    s21 = dict(RG.effective_signature('SVC', 'R21', 'D', 2)[1])
    pinned = dict(C=repr(1.0), kernel=repr('rbf'), gamma=repr('auto'))
    assert all(s21[k] == v for k, v in pinned.items()), s21
    return 'SVC under S6973 pins C=1.0, kernel=rbf, gamma=auto = the 0.21 configuration (fitted as R21/D)'


def policy_shifts(df, prep, datasets):
    zero = np.zeros((len(datasets), 6, 5))
    D = {a: shift_array(df, a, 'D', prep, datasets) for a in ALGOS}
    E_lr = shift_array(df, 'LR', 'E', prep, datasets)
    pol = {
        'D': D,
        'S6973fix': dict(D, SVC=zero),                                       # n_estimators + LR drift remain
        'E': {a: (E_lr if a == 'LR' else zero) for a in ALGOS},              # only unpinned LR multi_class drifts
        'P': {a: zero for a in ALGOS},
    }
    return pol


def pair_rerun_shift(shifts):
    A = np.stack([shifts[a] for a in ALGOS], 1)                          # (K,6,R,F)
    return np.stack([(A[:, a] - A[:, b]).mean(2) for a, b in PAIRS], 1)  # (K,P,R)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inp', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = pd.read_csv(Path(a.inp) / 'fold_scores.csv.gz')
    datasets = sorted(df.dataset.unique()); k_of = df.drop_duplicates('dataset').set_index('dataset').k
    rep = dict(note=__doc__, structural_check=verify_structural_zeros(), K=len(datasets),
               S6973_as_written=dict(status='hard_failure', estimator='RF', datasets_affected=len(datasets),
                                     reason="max_features='auto' rejected by scikit-learn 1.9.1 (InvalidParameterError)"),
               conditions={})
    for prep in ('raw', 'std'):
        out = {}
        for name, sh in policy_shifts(df, prep, datasets).items():
            X = pair_rerun_shift(sh); D = X.mean(2)
            dec = DV.label_edges({'0.21->1.9.1': DV.decide(D, 0.5, pair_names=PAIR_NAMES)}, ['0.21->1.9.1'])['0.21->1.9.1']
            Mk = np.abs(D).mean(1)
            algo_shift = {al: float(np.abs(sh[al].mean((1, 2))).mean()) for al in ALGOS}
            c = L.lens_c(X); b = L.lens_b(X, PAIR_NAMES)
            out[name] = dict(labels=dec['labels'], sensitivity_flags=dec['sensitivity_flags'], M_pp=dec['q_mean']['M_pp'],
                             M_lower95=dec['q_mean']['M_one_sided_lower95'], M_upper95=dec['q_mean']['M_one_sided_upper95'],
                             prevalence_avg=dec['q_prev']['primary_avg_pair']['estimate'],
                             prevalence_any_pair_1pp=dec['q_prev']['sensitivity_any_pair_pi_max']['estimate'],
                             mean_abs_algorithm_shift_pp=algo_shift,
                             lensC=dict(discoveries=c['discoveries'], storey_pi1=c['storey_pi1']),
                             lensD=L.lens_d(Mk, 0.5) if np.any(Mk > 0) else dict(rejected=False, stopping_K=None, note='all M_k = 0'),
                             lensB_widest_PI={p: v['pi'] for p, v in sorted(b.items(), key=lambda kv: -(kv[1]['pi'][1] - kv[1]['pi'][0]))[:3]},
                             multiclass_only_note=('E residual comes only from LR multi_class on %d multiclass datasets'
                                                   % int((k_of > 2).sum())) if name == 'E' else None)
        rep['conditions'][prep] = out
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for prep, out in rep['conditions'].items():
        for name, r in out.items():
            print('%-4s %-9s M=%.3f [%.3f,%.3f] labels=%s flags=%s prevAvg=%.2f prevAny1pp=%.2f lensC_disc=%d pi1=%.2f lensD=%s' % (
                prep, name, r['M_pp'], r['M_lower95'], r['M_upper95'], r['labels'], r['sensitivity_flags'], r['prevalence_avg'],
                r['prevalence_any_pair_1pp'], r['lensC']['discoveries'], r['lensC']['storey_pi1'],
                (r['lensD'].get('stopping_K') if r['lensD'].get('rejected') else 'no reject')))


if __name__ == '__main__':
    main()
