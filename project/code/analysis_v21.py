"""VersionShift v2.1 analysis: one adapter for emulation regimes and for real releases.

Usage
  python3 analysis_v21.py emulation --inp results/expA --noise results/expA2 --out results/v21_analysis
  python3 analysis_v21.py crossrelease --inp RUN/CROSS_RELEASE_FOLDS.csv.gz --edges old:new[,old:new] --policy D --out ...

Primary family = the pre-declared contrasts given on the command line (Holm-adjusted
p-values vs the margin). Everything else is secondary (BH within the edge).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimands_v21 import dataset_level, infer, holm, bh, array_from_long, pair_index

ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
NULL_ALGOS = ['kNN', 'GB', 'HGB']
TREES = {'RF', 'GB', 'HGB'}


def emulation_long(df, prep, policy='D'):
    parts = []
    for a in ALGOS:
        pp = 'raw' if a in TREES else prep
        parts.append(df[(df.algo == a) & (df.policy == policy) & (df.prep == pp)])
    out = pd.concat(parts).rename(columns={'regime': 'condition'})
    return out


def contrast(long, old, new, algos, margin, rng, label):
    S, ds, dropped = array_from_long(long, old, new, algos)
    dl = dataset_level(S)
    res = infer(dl, margin_pp=margin, B=4999, rng=rng)
    res.update(label=label, old=old, new=new, algorithms=algos, K=len(ds), datasets_dropped_incomplete=dropped,
               prevalence_M_gt_margin=float(np.mean(dl['M'] > margin)),
               prevalence_ci95=wilson(int(np.sum(dl['M'] > margin)), len(ds)))
    ia, ib = pair_index(len(algos))
    pairs = ['%s-%s' % (algos[a], algos[b]) for a, b in zip(ia, ib)]
    res['per_pair_mean_abs_shift_pp'] = dict(zip(pairs, np.abs(dl['pair_shift']).mean(0).round(4).tolist()))
    return res, dl


def wilson(x, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - w)), float(min(1, c + w))]


def noise_components(seed_csv, fold_csv):
    s = pd.read_csv(seed_csv); a = pd.read_csv(fold_csv)
    seed = s.groupby(['dataset', 'algo', 'regime', 'seed_rep']).ba.mean().groupby(['dataset', 'algo', 'regime']).std(ddof=1) * 100
    A = a[(a.policy == 'D') & (a.prep == 'raw')]
    tot = A.groupby(['dataset', 'algo', 'regime', 'rep']).ba.mean().groupby(['dataset', 'algo', 'regime']).std(ddof=1) * 100
    j = pd.concat([seed.rename('seed_sd_pp'), tot.rename('total_rerun_sd_pp')], axis=1).dropna()
    j['split_sd_pp'] = np.sqrt(np.maximum(j.total_rerun_sd_pp ** 2 - j.seed_sd_pp ** 2, 0))
    j['seed_variance_share'] = np.where(j.total_rerun_sd_pp > 0, j.seed_sd_pp ** 2 / j.total_rerun_sd_pp ** 2, 0)
    return j.groupby(['algo', 'regime']).median().round(4).reset_index()


def finish(results, out):
    prim = [r['primary']['p_vs_margin'] for r in results]
    for r, p in zip(results, holm(prim)):
        r['primary']['p_vs_margin_holm'] = float(p)
    co = [r['co_primary']['p_wilcoxon'] for r in results]
    for r, p in zip(results, bh(co)):
        r['co_primary']['p_wilcoxon_bh'] = float(p)
    return results


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('emulation'); e.add_argument('--inp', required=True); e.add_argument('--noise'); e.add_argument('--out', required=True)
    e.add_argument('--margin', type=float, default=0.5)
    c = sub.add_parser('crossrelease'); c.add_argument('--inp', required=True); c.add_argument('--edges', required=True)
    c.add_argument('--policy', default='D'); c.add_argument('--out', required=True); c.add_argument('--margin', type=float, default=0.5)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    report = dict(margin_pp=a.margin, contrasts=[])
    if a.cmd == 'emulation':
        df = pd.read_csv(Path(a.inp) / 'fold_scores.csv.gz')
        for prep in ('raw', 'std'):
            long = emulation_long(df, prep)
            r, _ = contrast(long, 'R21', 'R19', ALGOS, a.margin, rng, 'primary: 0.21-era -> 1.9.1 defaults, %s features' % prep)
            report['contrasts'].append(r)
        long = emulation_long(df, 'raw')
        nc, _ = contrast(long, 'R21', 'R19', NULL_ALGOS, a.margin, rng, 'negative control: algorithms without default change')
        report['negative_control'] = nc
        report['leg_identified'] = ('Emulation runs old and new defaults in ONE runtime (1.9.1): it identifies only the B->D leg '
                                    '(defaults, new runtime) of the 2x2; cells A and C need old binaries (crossversion/).')
        if a.noise:
            report['noise_components'] = noise_components(Path(a.noise) / 'seed_reruns.csv.gz', Path(a.inp) / 'fold_scores.csv.gz').to_dict('records')
    else:
        df = pd.read_csv(a.inp); df = df[(df.status == 'ok') & (df.policy == a.policy)]
        for edge in a.edges.split(','):
            old, new = edge.split(':')
            algos = [x for x in ALGOS if x in set(df.algo)]
            r, _ = contrast(df, old, new, algos, a.margin, rng, 'edge %s -> %s (policy %s)' % (old, new, a.policy))
            report['contrasts'].append(r)
    report['contrasts'] = finish(report['contrasts'], out)
    (out / 'V21_REPORT.json').write_text(json.dumps(report, indent=1, default=float))
    for r in report['contrasts'] + ([report['negative_control']] if 'negative_control' in report else []):
        p, c = r['primary'], r['co_primary']
        print('%s | K=%d | M=%.3f pp CI %s BCa %s | exceeds margin: %s, equivalent: %s | E=%.4f CI %s p_w=%.2g | prevalence %.2f' % (
            r['label'], r['K'], p['estimate_pp'], np.round(p['ci95'], 3).tolist(), np.round(p['ci95_bca'], 3).tolist(), p['exceeds_margin'],
            p['below_margin_equivalence'], c['estimate'], np.round(c['ci95'], 4).tolist(), c['p_wilcoxon'], r['prevalence_M_gt_margin']))


if __name__ == '__main__':
    main()
