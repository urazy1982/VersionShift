"""VersionShift v2.1 null calibration (review P0-3): independent null-generating experiments.

Fold-level scores are generated from a hierarchical model, not by sign-flipping observed
statistics. Each scenario fixes the truth; coverage and rejection rates are estimated
for the primary, co-primary and secondary estimands from estimands_v21.

score[k,v,a,r,f] = mu[k,a] + shift[k,a]*[v=new]
                 + u[k,r,f]                         split/fold effect common to all algorithms (paired across versions)
                 + sigma_k * e[k,a,r,f]             algorithm x split interaction, correlated across algorithms (rho), paired
                 + tau_k * s[k,v,a,r,f]             seed noise of stochastic algorithms (RF, GB, HGB): shared across
                                                    versions ('same RNG stream') or redrawn ('RNG stream changed')
                 + extra[k,a,r,f]*[v=new]           variance-only version change (scenario N4)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimands_v21 import dataset_level, infer

A, R, F = 6, 6, 5
STOCH = np.array([0, 0, 0, 1, 1, 1], float)          # LR SVC kNN deterministic; RF GB HGB stochastic

SCENARIOS = {
    'N1_pure_null':            dict(rng_shared=True,  hetero=0.0, skew=False, rho=0.3, var_change=0.0, effect=None),
    'N2_rng_stream_changed':   dict(rng_shared=False, hetero=0.0, skew=False, rho=0.3, var_change=0.0, effect=None),
    'N3_hetero_skew_corr':     dict(rng_shared=False, hetero=0.6, skew=True,  rho=0.7, var_change=0.0, effect=None),
    'N4_variance_only_change': dict(rng_shared=False, hetero=0.6, skew=True,  rho=0.7, var_change=0.5, effect=None),
    'A1_sparse_large_effect':  dict(rng_shared=False, hetero=0.6, skew=True,  rho=0.7, var_change=0.0, effect='sparse'),
    'A2_dense_small_effect':   dict(rng_shared=False, hetero=0.6, skew=True,  rho=0.7, var_change=0.0, effect='dense', dense_sd=0.004),
    'B0_at_margin_boundary':   dict(rng_shared=False, hetero=0.6, skew=True,  rho=0.7, var_change=0.0, effect='dense', dense_sd=0.0043),
    'A3_dense_meaningful':     dict(rng_shared=False, hetero=0.6, skew=True,  rho=0.7, var_change=0.0, effect='dense', dense_sd=0.008),
}


def noise(rng, shape, skew):
    if not skew:
        return rng.standard_normal(shape)
    g = rng.gamma(2.0, 1.0, shape)                   # right-skewed, standardised
    return (g - 2.0) / np.sqrt(2.0)


def generate(rng, K, sc, fold_sd=0.015, seed_sd=0.006):
    sigma = fold_sd * np.exp(sc['hetero'] * rng.standard_normal(K) - 0.5 * sc['hetero'] ** 2)
    tau = seed_sd * np.exp(sc['hetero'] * rng.standard_normal(K) - 0.5 * sc['hetero'] ** 2)
    mu = 0.80 + 0.05 * rng.standard_normal((K, A))
    u = 0.02 * rng.standard_normal((K, 1, 1, R, F))
    common = noise(rng, (K, 1, R, F), sc['skew']); own = noise(rng, (K, A, R, F), sc['skew'])
    e = np.sqrt(sc['rho']) * common + np.sqrt(1 - sc['rho']) * own
    s_old = rng.standard_normal((K, A, R, F))
    s_new = s_old if sc['rng_shared'] else rng.standard_normal((K, A, R, F))
    shift = np.zeros((K, A))
    if sc['effect'] == 'sparse':
        hit = rng.random(K) < 0.10
        shift[hit, 3] = 0.04                         # RF-like +4 pp on 10% of datasets
    elif sc['effect'] == 'dense':
        shift = sc['dense_sd'] * rng.standard_normal((K, A))  # every algorithm moves a little
    base = mu[:, :, None, None] + sigma[:, None, None, None] * e
    old = base + (tau[:, None, None, None] * STOCH[None, :, None, None]) * s_old
    new = base + shift[:, :, None, None] + (tau[:, None, None, None] * STOCH[None, :, None, None]) * s_new
    if sc['var_change'] > 0:
        new = new + sc['var_change'] * sigma[:, None, None, None] * noise(rng, (K, A, R, F), sc['skew'])
    S = np.stack([old, new], 1) + u
    return np.clip(S, 0, 1)


def truth(sc, rng, K_big=40000, chunk=4000):
    M, E = [], []
    for _ in range(K_big // chunk):
        dl = dataset_level(generate(rng, chunk, sc)); M.append(dl['M']); E.append(dl['E'])
    M = np.concatenate(M); E = np.concatenate(E)
    return float(M.mean()), float(E.mean()), float(M.std() / np.sqrt(len(M))), float(E.std() / np.sqrt(len(E)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--Ks', default='12,24,48,72'); ap.add_argument('--margin', type=float, default=0.5)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    Ks = [int(x) for x in a.Ks.split(',')]; rng = np.random.default_rng(20261106)
    report = dict(model=__doc__, margin_pp=a.margin, sims=a.sims, Ks=Ks, scenarios={})
    t0 = time.time()
    for name, sc in SCENARIOS.items():
        tM, tE, seM, seE = truth(sc, rng)
        rows = []
        for K in Ks:
            c_ = dict(covM_pct=0, covM_bca=0, covM_t=0, covE_pct=0, covE_bca=0, covE_t=0, rejM_pct=0, rejM_bca=0,
                      eqM=0, rejE_boot=0, rejE_w=0, rejS=0)
            tolM, tolE = 3 * seM, 3 * seE
            inside = lambda ci, t, tol: ci[0] - tol <= t <= ci[1] + tol
            for _ in range(a.sims):
                dl = dataset_level(generate(rng, K, sc)); inf = infer(dl, margin_pp=a.margin, B=499, rng=rng)
                p, c, s = inf['primary'], inf['co_primary'], inf['secondary']
                c_['covM_pct'] += inside(p['ci95'], tM, tolM); c_['covM_bca'] += inside(p['ci95_bca'], tM, tolM)
                c_['covM_t'] += inside(p['ci95_t'], tM, tolM)
                c_['covE_pct'] += inside(c['ci95'], tE, tolE); c_['covE_bca'] += inside(c['ci95_bca'], tE, tolE)
                c_['covE_t'] += inside(c['ci95_t'], tE, tolE)
                c_['rejM_pct'] += p['exceeds_margin']; c_['rejM_bca'] += p['one_sided_lower95_bca'] > a.margin
                c_['eqM'] += p['below_margin_equivalence']
                c_['rejE_boot'] += c['p_boot_two_sided'] < 0.05; c_['rejE_w'] += c['p_wilcoxon'] < 0.05
                c_['rejS'] += (s['S_p_wilcoxon'] is not None and s['S_p_wilcoxon'] < 0.05)
            rows.append(dict(K=K, **{k: v / a.sims for k, v in c_.items()}))
            print(name, K, rows[-1], f'{time.time() - t0:.0f}s', flush=True)
        report['scenarios'][name] = dict(settings=sc, truth_M_pp=tM, truth_M_mcse=seM, truth_E=tE, truth_E_mcse=seE,
                                         M_exceeds_margin=tM > a.margin, rows=rows)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
