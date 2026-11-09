"""Calibration of the v2.2.5 secondary lenses on the v2.1 known-truth generator (pre-data).

Adds scenario C2_single_algorithm_1p2pp: one algorithm (RF) shifts by +1.2 pp on every dataset, which moves
its 5 pairs and gives M = 0.40 pp < 0.5 pp margin -> the realistic dilution case. (A shift confined to ONE
pair is structurally impossible: every pair shift is a difference of two algorithm shifts.)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_v21 as CV
from estimands_v21 import pair_index, dataset_level
import decision_v221 as DV
import lenses_v225 as L

A, R, F = 6, 6, 5
IA, IB = pair_index(A)
PAIR = 9                                   # (kNN, RF): involves the shifted algorithm in A1 and C2
SCEN = dict(CV.SCENARIOS)
SCEN['C2_single_algorithm_1p2pp'] = dict(CV.SCENARIOS['A1_sparse_large_effect'], effect='single')


def true_shift(rng, K, sc):
    s = np.zeros((K, A))
    if sc['effect'] == 'sparse':
        s[rng.random(K) < 0.10, 3] = 4.0
    elif sc['effect'] in ('dense',):
        s = sc['dense_sd'] * 100 * rng.standard_normal((K, A))
    elif sc['effect'] == 'single':
        s[:, 3] = 1.2
    return s                                                       # pp, per algorithm


def generate(rng, K, sc):
    """Same model as calibrate_v21.generate, returning the true per-algorithm shifts (pp) as well."""
    sigma = 0.015 * np.exp(sc['hetero'] * rng.standard_normal(K) - 0.5 * sc['hetero'] ** 2)
    tau = 0.006 * np.exp(sc['hetero'] * rng.standard_normal(K) - 0.5 * sc['hetero'] ** 2)
    mu = 0.80 + 0.05 * rng.standard_normal((K, A)); u = 0.02 * rng.standard_normal((K, 1, 1, R, F))
    e = np.sqrt(sc['rho']) * CV.noise(rng, (K, 1, R, F), sc['skew']) + np.sqrt(1 - sc['rho']) * CV.noise(rng, (K, A, R, F), sc['skew'])
    s_old = rng.standard_normal((K, A, R, F)); s_new = s_old if sc['rng_shared'] else rng.standard_normal((K, A, R, F))
    sh = true_shift(rng, K, sc)
    base = mu[:, :, None, None] + sigma[:, None, None, None] * e
    st = (tau[:, None, None, None] * CV.STOCH[None, :, None, None])
    old = base + st * s_old
    new = base + sh[:, :, None, None] / 100 + st * s_new
    if sc['var_change'] > 0:
        new = new + sc['var_change'] * sigma[:, None, None, None] * CV.noise(rng, (K, A, R, F), sc['skew'])
    S = np.stack([old, new], 1) + u                                 # no clipping: keeps the truth exact
    return S, sh


def per_rerun_pair_shift(S):
    d = S[:, :, IA] - S[:, :, IB]                                   # (K,2,P,R,F)
    return (d[:, 1] - d[:, 0]).mean(3) * 100                        # (K,P,R) pp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sims', type=int, default=300); ap.add_argument('--Ks', default='16,48')
    ap.add_argument('--out', required=True); a = ap.parse_args()
    rng = np.random.default_rng(20261109); report = {}; t0 = time.time()
    for name, sc in SCEN.items():
        Sbig, shbig = generate(rng, 20000, sc)
        tp_big = shbig[:, IA] - shbig[:, IB]                        # true pair shifts (pp)
        truth = dict(M=float(dataset_level(Sbig)['M'].mean()), mu_pair=float(tp_big[:, PAIR].mean()),
                     pi1_any_change=float(np.mean(np.abs(tp_big).max(1) > 1e-9)))
        rows = {}
        for K in [int(x) for x in a.Ks.split(',')]:
            acc = dict(prim_reject=0, d_reject=0, d_stop=[], b_ci_cover=0, b_pi_cover=[], c_fdr=[], c_power=[], c_pi1=[], c_any_disc=0)
            for i in range(a.sims):
                S, sh = generate(rng, K, sc); X = per_rerun_pair_shift(S); Mk = np.abs(X.mean(2)).mean(1)
                tp = sh[:, IA] - sh[:, IB]
                acc['prim_reject'] += DV.p_mean_exceeds(Mk, 0.5) < 0.05
                d = L.lens_d(Mk, 0.5, order_seed=i); acc['d_reject'] += d['rejected']
                if d['rejected']: acc['d_stop'].append(d['stopping_K'])
                b = L.random_effects(X.mean(2)[:, PAIR], X.var(2, ddof=1)[:, PAIR] / R)
                acc['b_ci_cover'] += b['ci'][0] <= truth['mu_pair'] <= b['ci'][1]
                _, shn = generate(rng, 400, sc); newtrue = (shn[:, IA] - shn[:, IB])[:, PAIR]
                acc['b_pi_cover'].append(np.mean((newtrue >= b['pi'][0]) & (newtrue <= b['pi'][1])))
                c = L.lens_c(X); rej = np.array(c['changed']); changed = np.abs(tp).max(1) > 1e-9
                acc['c_any_disc'] += rej.any()
                acc['c_fdr'].append((rej & ~changed).sum() / max(rej.sum(), 1))
                if changed.any(): acc['c_power'].append((rej & changed).sum() / changed.sum())
                acc['c_pi1'].append(c['storey_pi1'])
            n = a.sims
            rows[K] = dict(primary_reject_M_gt_margin=acc['prim_reject'] / n, lensD_reject=acc['d_reject'] / n,
                           lensD_median_stopping_K=float(np.median(acc['d_stop'])) if acc['d_stop'] else None,
                           lensB_ci_coverage=acc['b_ci_cover'] / n, lensB_pi_coverage_mean=float(np.mean(acc['b_pi_cover'])),
                           lensC_fdr=float(np.mean(acc['c_fdr'])), lensC_any_discovery_rate=acc['c_any_disc'] / n,
                           lensC_power=float(np.mean(acc['c_power'])) if acc['c_power'] else None,
                           lensC_pi1_mean=float(np.mean(acc['c_pi1'])))
        report[name] = dict(truth=truth, rows=rows)
        print(name, {k: round(v, 3) for k, v in truth.items()}, {K: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for K, r in rows.items()}, '%.0fs' % (time.time() - t0), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
