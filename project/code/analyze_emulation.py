"""Experiment A analysis — the four-level impact ladder, plus Experiment B design calibration.

Ladder:  L1 predictions -> L2 per-algorithm metric -> L3 pairwise conclusion -> L4 benchmark ranking.
Every version effect is reported against the comparison's own *rerun noise floor*
(fresh split + model seeds, same release), which is the quantity a reader implicitly
accepts when a paper reports a single run.

Experiment B compares, on the same data budget, the v1.7 six-block decisive-reversal
excess against a continuous excess-squared-shift statistic, using algorithm pairs whose
configuration is identical across regimes as exact negative controls.
"""
import argparse, hashlib, importlib.util, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
FROZEN_SHA = 'dfce98097c54b9e3bf86ee9d2f02b77d100b2a3885df1589ba9efc7f7e7e01a5'
ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
NULL_ALGOS = {'kNN', 'GB', 'HGB'}         # no default change 0.21 -> 1.9.1 (n < 10k)
TREES = {'RF', 'GB', 'HGB'}


def frozen_metrics():
    p = HERE / 'frozen_metrics_v36.py'
    if hashlib.sha256(p.read_bytes()).hexdigest() != FROZEN_SHA:
        raise SystemExit('frozen posterior source changed')
    spec = importlib.util.spec_from_file_location('frozen_v36', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def boot_mean_ci(x, B=4000, seed=1):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) < 2: return [float('nan')] * 2
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def condition_frame(df, policy, prep):
    """Per-fold BA matrix for one pipeline condition; trees reuse raw (scale-invariant) fits."""
    parts = []
    for a in ALGOS:
        if policy == 'E' and a in TREES:
            continue
        pp = 'raw' if a in TREES else prep
        sub = df[(df.algo == a) & (df.policy == policy) & (df.prep == pp)]
        parts.append(sub)
    return pd.concat(parts)


# ---------------------------------------------------------------- L1, L2
def ladder_l1(dis):
    g = dis.groupby(['algo', 'policy', 'prep'])
    out = g.apply(lambda s: pd.Series(dict(
        prediction_disagreement_rate=s.n_disagree.sum() / s.n_test.sum(),
        folds_with_any_change=(s.n_disagree > 0).mean(),
        datasets_with_any_change=s.groupby('dataset').n_disagree.sum().gt(0).mean())), include_groups=False)
    return out.reset_index()


def ladder_l2(df):
    rows = []
    rep = df.groupby(['dataset', 'origin', 'hetero', 'algo', 'policy', 'prep', 'regime', 'rep']).ba.mean().unstack('regime')
    for key, s in rep.groupby(level=['algo', 'policy', 'prep']):
        per_ds = []
        for ds, t in s.groupby(level='dataset'):
            shift = (t.R19 - t.R21).to_numpy()
            noise = t.R19.std(ddof=1)
            per_ds.append(dict(dataset=ds, origin=t.index.get_level_values('origin')[0],
                               hetero=t.index.get_level_values('hetero')[0],
                               mean_shift_pp=100 * shift.mean(), noise_sd_pp=100 * noise,
                               vnr=abs(shift.mean()) / noise if noise > 0 else (0.0 if abs(shift.mean()) < 1e-15 else np.inf)))
        p = pd.DataFrame(per_ds)
        finite = p.vnr[np.isfinite(p.vnr)]
        rows.append(dict(algo=key[0], policy=key[1], prep=key[2], K=len(p),
                         mean_shift_pp=p.mean_shift_pp.mean(), ci95_pp=boot_mean_ci(p.mean_shift_pp),
                         mean_abs_shift_pp=p.mean_shift_pp.abs().mean(),
                         median_noise_sd_pp=p.noise_sd_pp.median(),
                         median_vnr=float(np.median(p.vnr)), share_vnr_gt1=float((p.vnr > 1).mean()),
                         share_vnr_gt2=float((p.vnr > 2).mean()), max_abs_shift_pp=p.mean_shift_pp.abs().max(),
                         hetero_mean_abs_shift_pp=p[p.hetero == True].mean_shift_pp.abs().mean(),
                         homo_mean_abs_shift_pp=p[p.hetero == False].mean_shift_pp.abs().mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- L3
def corrected_state(d, n_folds_per_rep=5, alpha=0.05):
    """Nadeau-Bengio / Bouckaert-Frank corrected repeated-CV t test on per-fold differences."""
    d = np.asarray(d, float); n = len(d); m = d.mean(); v = d.var(ddof=1)
    if v == 0:
        return 'A>B' if m > 0 else ('B>A' if m < 0 else '~')
    t = m / np.sqrt((1 / n + 1 / (n_folds_per_rep - 1)) * v)
    p = 2 * stats.t.sf(abs(t), n - 1)
    return ('A>B' if m > 0 else 'B>A') if p < alpha else '~'


def pair_table(cf, posterior):
    """Per dataset x pair: states under several budgets plus continuous shifts."""
    piv = cf.pivot_table(index=['dataset', 'regime', 'rep', 'fold'], columns='algo', values='ba')
    out = []
    algos = [a for a in ALGOS if a in piv.columns]
    for ds, t in piv.groupby(level='dataset'):
        for a, b in itertools.combinations(algos, 2):
            d = (t[a] - t[b])
            get = lambda g, reps: d.xs(g, level='regime')[d.xs(g, level='regime').index.get_level_values('rep').isin(reps)].to_numpy()
            full = {g: get(g, range(6)) for g in ('R21', 'R19')}
            h1, h2 = [0, 1, 2], [3, 4, 5]
            st = lambda x: corrected_state(x)
            rec = dict(dataset=ds, pair=f'{a}-{b}', null_pair=(a in NULL_ALGOS and b in NULL_ALGOS),
                       s21=st(full['R21']), s19=st(full['R19']),
                       w19=(st(get('R19', h1)), st(get('R19', h2))), w21=(st(get('R21', h1)), st(get('R21', h2))),
                       x=(st(get('R21', h1)), st(get('R19', h2))), x2=(st(get('R21', h2)), st(get('R19', h1))))
            per_rep = {g: np.array([get(g, [r]).mean() for r in range(6)]) for g in ('R21', 'R19')}
            noise = per_rep['R19'].std(ddof=1)
            rec['delta21_pp'] = 100 * full['R21'].mean(); rec['delta19_pp'] = 100 * full['R19'].mean()
            rec['shift_pp'] = rec['delta19_pp'] - rec['delta21_pp']
            rec['noise_sd_pp'] = 100 * noise
            rec['vnr'] = abs(rec['shift_pp']) / rec['noise_sd_pp'] if noise > 0 else (0.0 if rec['shift_pp'] == 0 else np.inf)
            # --- Experiment B: three blocks of two replicates each (10 folds per block)
            blk = {g: [get(g, [2 * i, 2 * i + 1]) for i in range(3)] for g in ('R21', 'R19')}
            ps = {g: [posterior(x)['state'] for x in blk[g]] for g in blk}
            rev = lambda s1, s2: int((s1, s2) in (('A>B', 'B>A'), ('B>A', 'A>B')))
            rec['six_R'] = rev(ps['R21'][0], ps['R19'][1])
            rec['six_Nold'] = rev(ps['R21'][1], ps['R21'][2])
            rec['six_Nnew'] = rev(ps['R19'][0], ps['R19'][2])
            m = {g: [x.mean() for x in blk[g]] for g in blk}
            # exchangeable construction: under no version effect, R19_b1 == R21_b1 in distribution
            # (seed-paired), so cross=(b1-b0)^2 and null=(b1-b2)^2 swap under relabelling b0<->b2.
            rec['cont_cross'] = (m['R19'][1] - m['R21'][0]) ** 2
            rec['cont_null'] = (m['R21'][1] - m['R21'][2]) ** 2
            out.append(rec)
    return pd.DataFrame(out)


def ladder_l3(pt):
    chg = lambda s: s[0] != s[1]
    revf = lambda s: s in (('A>B', 'B>A'), ('B>A', 'A>B'))
    res = []
    for label, sub in [('all_pairs', pt), ('changed_config_pairs', pt[~pt.null_pair]), ('null_control_pairs', pt[pt.null_pair])]:
        full = list(zip(sub.s21, sub.s19))
        within = [chg(x) for x in sub.w19] + [chg(x) for x in sub.w21]
        cross = [chg(x) for x in sub.x] + [chg(x) for x in sub.x2]
        res.append(dict(set=label, slots=len(sub), datasets=sub.dataset.nunique(),
                        full_budget_paired_change=np.mean([chg(x) for x in full]),
                        full_budget_paired_reversal=np.mean([revf(x) for x in full]),
                        half_budget_cross_change=np.mean(cross), half_budget_within_change=np.mean(within),
                        excess_change=np.mean(cross) - np.mean(within),
                        half_cross_reversal=np.mean([revf(x) for x in sub.x] + [revf(x) for x in sub.x2]),
                        half_within_reversal=np.mean([revf(x) for x in sub.w19] + [revf(x) for x in sub.w21]),
                        median_vnr=float(np.median(sub.vnr)), share_vnr_gt1=float((sub.vnr > 1).mean()),
                        share_vnr_gt2=float((sub.vnr > 2).mean()),
                        mean_abs_shift_pp=float(sub.shift_pp.abs().mean())))
    # excess change per dataset with cluster bootstrap
    per = pt[~pt.null_pair].groupby('dataset').apply(lambda s: np.mean(
        [a[0] != a[1] for a in list(s.x) + list(s.x2)]) - np.mean([a[0] != a[1] for a in list(s.w19) + list(s.w21)]),
        include_groups=False)
    return pd.DataFrame(res), dict(excess_change_changed_pairs_dataset_mean=float(per.mean()),
                                   ci95=boot_mean_ci(per.to_numpy()), K=int(len(per)))


# ---------------------------------------------------------------- L4
def avg_ranks(M):
    return pd.DataFrame(M).rank(axis=1, ascending=False).mean(axis=0)


def ladder_l4(cf, B=2000, seed=7):
    mean = cf.groupby(['dataset', 'regime', 'algo']).ba.mean().unstack('algo')
    half = cf.assign(h=np.where(cf.rep < 3, 'h1', 'h2')).groupby(['dataset', 'regime', 'h', 'algo']).ba.mean().unstack('algo')
    algos = [a for a in ALGOS if a in mean.columns]
    M21 = mean.xs('R21', level='regime')[algos]; M19 = mean.xs('R19', level='regime')[algos]
    ds = M19.index.to_numpy()
    r21, r19 = avg_ranks(M21), avg_ranks(M19)
    tau = stats.kendalltau(r21, r19).statistic
    hh = {(g, h): half.xs((g, h), level=['regime', 'h'])[algos].loc[ds] for g in ('R21', 'R19') for h in ('h1', 'h2')}
    rng = np.random.default_rng(seed)
    bt = dict(tau_cross=[], tau_within=[], win_cross=[], win_within=[], top2_cross=[])
    for _ in range(B):
        idx = rng.integers(0, len(ds), len(ds))
        a = avg_ranks(M21.to_numpy()[idx]); b = avg_ranks(M19.to_numpy()[idx])
        w1 = avg_ranks(hh[('R19', 'h1')].to_numpy()[idx]); w2 = avg_ranks(hh[('R19', 'h2')].to_numpy()[idx])
        c1 = avg_ranks(hh[('R21', 'h1')].to_numpy()[idx])
        bt['tau_cross'].append(stats.kendalltau(c1, w2).statistic)
        bt['tau_within'].append(stats.kendalltau(w1, w2).statistic)
        bt['win_cross'].append(int(np.argmin(c1.values) != np.argmin(w2.values)))
        bt['win_within'].append(int(np.argmin(w1.values) != np.argmin(w2.values)))
        bt['top2_cross'].append(int(set(np.argsort(a.values)[:2]) != set(np.argsort(b.values)[:2])))
    fr21 = stats.friedmanchisquare(*[M21[a] for a in algos]); fr19 = stats.friedmanchisquare(*[M19[a] for a in algos])
    k, N = len(algos), len(ds)
    cd = 2.850 * np.sqrt(k * (k + 1) / (6 * N))      # Nemenyi q_0.05 for k=6
    sig = lambda r: {(a, b) for a, b in itertools.combinations(algos, 2) if abs(r[a] - r[b]) > cd}
    s21, s19 = sig(r21), sig(r19)
    q = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
    return dict(K=int(N), algorithms=algos,
                avg_rank_R21={a: round(float(r21[a]), 3) for a in algos},
                avg_rank_R19={a: round(float(r19[a]), 3) for a in algos},
                winner_R21=str(r21.idxmin()), winner_R19=str(r19.idxmin()),
                kendall_tau_full=float(tau),
                friedman_p_R21=float(fr21.pvalue), friedman_p_R19=float(fr19.pvalue), nemenyi_cd=float(cd),
                nemenyi_significant_pairs_R21=sorted('-'.join(x) for x in s21),
                nemenyi_significant_pairs_R19=sorted('-'.join(x) for x in s19),
                nemenyi_pairs_changed=sorted('-'.join(x) for x in s21 ^ s19),
                bootstrap=dict(B=B, tau_cross_half_mean=float(np.mean(bt['tau_cross'])), tau_cross_ci=q(bt['tau_cross']),
                               tau_within_half_mean=float(np.mean(bt['tau_within'])), tau_within_ci=q(bt['tau_within']),
                               p_winner_differs_cross=float(np.mean(bt['win_cross'])),
                               p_winner_differs_within=float(np.mean(bt['win_within'])),
                               p_top2_set_differs_full=float(np.mean(bt['top2_cross']))))


# ---------------------------------------------------------------- Experiment B
def design_power(pt, Ks=(8, 16, 24, 32, 44), sims=4000, seed=11):
    """Same data budget, two dataset-level statistics:
    six-block   e_k = mean_pairs[R - (N_old + N_new)/2]  (v1.7 decisive-reversal excess, frozen posterior)
    continuous  S_k = log(sum_pairs cross^2) - log(sum_pairs null^2)  (symmetric about 0 under H0)
    Power: resample datasets of the changed-configuration pair set.
    Size: resample datasets of the exact-null pair set with random sign flips (H0 imposed;
    valid because S_k is exchangeable-symmetric under H0)."""
    def per_dataset(sub):
        g = sub.groupby('dataset')
        six = g.apply(lambda s: (s.six_R - 0.5 * (s.six_Nold + s.six_Nnew)).mean(), include_groups=False)
        cont = g.apply(lambda s: np.log(s.cont_cross.sum() + 1e-12) - np.log(s.cont_null.sum() + 1e-12), include_groups=False)
        return six.to_numpy(), cont.to_numpy()
    rng = np.random.default_rng(seed)
    def t_rej(x):
        if np.allclose(x, x[0]): return 0
        return int(stats.ttest_1samp(x, 0).pvalue < 0.05)
    def w_rej(x):
        if np.allclose(x, 0): return 0
        return int(stats.wilcoxon(x, zero_method='zsplit').pvalue < 0.05)
    res = {}
    has = lambda a: pt.pair.str.split('-').apply(lambda x: a in x)
    rf_only = has('RF') & ~has('LR') & ~has('SVC')
    sets = [('changed_config_pairs', pt[~pt.null_pair], False),
            ('rf_pairs_without_LR_SVC', pt[rf_only], False),
            ('null_control_pairs', pt[pt.null_pair], True)]
    for label, sub, flip in sets:
        six, cont = per_dataset(sub)
        rows = []
        for K in Ks:
            r6 = rc = rw = 0
            for _ in range(sims):
                idx = rng.integers(0, len(six), K)
                sg = rng.choice([-1.0, 1.0], K) if flip else 1.0
                r6 += t_rej(six[idx] * sg); rc += t_rej(cont[idx] * sg); rw += w_rej(cont[idx] * sg)
            rows.append(dict(K=K, sixblock_reversal_excess_t=r6 / sims, continuous_logratio_t=rc / sims,
                             continuous_logratio_wilcoxon=rw / sims))
        res[label] = dict(K_pool=int(len(six)), mean_sixblock=float(six.mean()), share_datasets_sixblock_nonzero=float((six != 0).mean()),
                          mean_logratio=float(cont.mean()), median_logratio=float(np.median(cont)),
                          logratio_ci95=boot_mean_ci(cont), H0_imposed_by_sign_flip=flip, sims=sims, rates=rows)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inp', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); inp, out = Path(a.inp), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp / 'fold_scores.csv.gz'); dis = pd.read_csv(inp / 'prediction_disagreement.csv.gz')
    post = frozen_metrics().posterior
    report = dict(meta=json.loads((inp / 'RUN_META.json').read_text()))
    report['meta'].pop('datasets_detail', None)
    l1 = ladder_l1(dis); l1.to_csv(out / 'L1_predictions.csv', index=False)
    l2 = ladder_l2(df); l2.to_csv(out / 'L2_algorithm_metric.csv', index=False)
    report['L1'] = l1.to_dict('records'); report['L2'] = l2.to_dict('records')
    report['L3'] = {}; report['L4'] = {}; report['ExpB'] = {}
    for prep in ('raw', 'std'):
        cf = condition_frame(df, 'D', prep)
        pt = pair_table(cf, post); pt.to_csv(out / f'pairs_D_{prep}.csv.gz', index=False)
        l3, l3ci = ladder_l3(pt)
        l3.to_csv(out / f'L3_pairs_D_{prep}.csv', index=False)
        report['L3'][f'D_{prep}'] = dict(table=l3.to_dict('records'), **l3ci)
        report['L4'][f'D_{prep}'] = ladder_l4(cf)
        if (cf.origin == 'synthetic').any():
            report['L4'][f'D_{prep}_synthetic_only'] = ladder_l4(cf[cf.origin == 'synthetic'])
        report['ExpB'][f'D_{prep}'] = design_power(pt)
    # E-policy (LR, SVC, kNN only): does the partial explicit map remove the regime shift?
    report['E_policy_note'] = ('E pins SVC gamma and RF n_estimators, so its R21/R19 contrast is identically zero '
                               'for those; LR multi_class is NOT pinned by E, so multiclass LR still shifts.')
    (out / 'EXP_A_B_REPORT.json').write_text(json.dumps(report, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    print(json.dumps({k: report[k] for k in ('L4',)}, indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
