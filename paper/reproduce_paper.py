#!/usr/bin/env python3
"""Reproduce every number, table input and figure in the VersionShift manuscript.

Read-only with respect to results/: it reads the verified Phase 1 bundle and writes to --out.

    python3 paper/reproduce_paper.py                       # results/ -> paper/output/
    python3 paper/reproduce_paper.py --tex manuscript.tex  # also check the numbers appear in the manuscript

Outputs
    PAPER_NUMBERS.json              every quantity quoted in the text
    policy_E_vs_D_10pairs.json      policy E vs D on the five estimators where E runs (Section on protections)
    levels_{POS,NEG}.csv            mean balanced accuracy per release x dataset x estimator (policy D)
    algo_change_{POS,NEG}.csv       per-dataset change in mean balanced accuracy per estimator
    Mk_{POS,NEG}.csv                per-dataset M_k
    figures/fig_algo_change.pdf, fig_winners.pdf, fig_mk.pdf (+ .png)

POS = 0.21.3 -> 0.22.2.post1 (edge E+), NEG = 1.8.0 -> 1.9.1 (edge E0, negative control).
"""
import argparse, glob, itertools, json, os, re
from pathlib import Path

import numpy as np
import pandas as pd

A = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
ED = {'POS': ('0.21.3', '0.22.2.post1'), 'NEG': ('1.8.0', '1.9.1')}


def J(p):
    with open(p) as f:
        return json.load(f)


def mean_levels(merged_csv, policy='D'):
    df = pd.read_csv(merged_csv, dtype={'release': str})
    d = df[(df.policy == policy) & (df.status == 'ok')]
    return d.groupby(['release', 'dataset', 'algo']).ba.mean().unstack('algo') * 100


def mk(o, n, algos):
    return pd.Series({ds: np.mean([abs((n.loc[ds, a] - n.loc[ds, b]) - (o.loc[ds, a] - o.loc[ds, b]))
                                   for a, b in itertools.combinations(algos, 2)]) for ds in o.index})


def collect(F, out_dir):
    R = F / 'STAGE_C_CPU_COMPLETION' / 'reports'
    out = {}
    for frame, f in (('FULL63', 'FULL63_FRAME_CENSUS.json'), ('HOLDOUT47', 'HOLDOUT47_FINITE_SUBFRAME_SENSITIVITY.json')):
        d = J(R / f)
        for code, (o, n) in ED.items():
            e = f'{o}:{n}'; r = d['margins']['0.5'][e]; q = r['q_mean']
            x = dict(K=r['K_complete_case'], M=q['M_pp'], M_sd=q['M_k_sd'], lb=q['M_one_sided_lower95'],
                     ub=q['M_one_sided_upper95'], all_tost=q['pairs_all_tost_equivalent'], labels=r['labels'],
                     adjusted=r.get('adjusted'),
                     per_pair={p['pair']: dict(mean=p['mean_signed_pp'], ci90=p['ci90'], tost=p['tost_equivalent'])
                               for p in q['per_pair']})
            for m in ('0.25', '0.5', '1.0'):
                x[f'affected_{m}'] = d['margins'][m][e]['q_prev']['primary_avg_pair']['affected']
                x[f'labels_{m}'] = d['margins'][m][e]['labels']
            x['coverage'] = d.get('plan_coverage')
            out[f'{frame}_{code}'] = x
    v = J(F / 'INDEPENDENT_VERIFICATION' / 'VERIFY_SUMMARY.json')
    for k, fr in v['frames'].items():
        lab, e = k.split('_', 1); code = 'POS' if e.startswith('0.21') else 'NEG'
        out.setdefault(f'{lab}_{code}', {})['conclusions'] = {
            kk: fr[kk] for kk in fr if kk.startswith('O') or kk.startswith('mean_ba')
            or kk in ('M_pp', 'K_complete', 'datasets_M_gt_0_5', 'cross_check')}
    out['aa'] = {k: (v['runs'][k] if v['runs'][k].get('kind') == 'AA' else None) for k in v['runs'] if k.startswith('AA')}
    for k in sorted(glob.glob(str(F / 'STAGE_B_RUNS' / 'AA_*' / 'AA_IDENTITY.json'))):
        out['aa_identity_' + Path(k).parent.name] = J(k)

    lev = {}
    for code, (o, n) in ED.items():
        m = mean_levels(F / 'STAGE_C_CPU_COMPLETION' / 'merged' / f'{code}.csv.gz')[A]
        for rel in (o, n):
            mm = m.loc[rel]
            lev[rel] = dict(mean_ba={a: float(mm[a].mean()) for a in A},
                            wins=mm.idxmax(1).value_counts().reindex(A, fill_value=0).astype(int).to_dict(),
                            mean_rank=mm.rank(1, ascending=False).mean().round(3).to_dict())
        ch = m.loc[n] - m.loc[o]
        lev[f'{code}_algo_change'] = {a: dict(mean=float(ch[a].mean()), median=float(ch[a].median()),
                                              min=float(ch[a].min()), max=float(ch[a].max()),
                                              n_abs_gt_1=int((ch[a].abs() > 1).sum()), n_exact0=int((ch[a] == 0).sum()))
                                      for a in A}
        Mk = mk(m.loc[o], m.loc[n], A)
        lev[f'{code}_Mk'] = dict(median=float(Mk.median()), max=float(Mk.max()), max_ds=str(Mk.idxmax()),
                                 q75=float(Mk.quantile(.75)))
        Mk.to_csv(out_dir / f'Mk_{code}.csv'); ch.to_csv(out_dir / f'algo_change_{code}.csv'); m.to_csv(out_dir / f'levels_{code}.csv')
    out['levels'] = lev
    out['policies'] = J(R / 'POLICIES_FULL63_DESCRIPTIVE.json')
    for c in ('POS', 'NEG'):
        out['mitigation_' + c] = J(R / f'MITIGATION_{c}_FULL63.json'); out['mitigation_' + c].pop('header', None)
        out['attribution_' + c] = J(R / f'ATTRIBUTION_{c}_FULL63.json'); out['attribution_' + c].pop('header', None)
        l = J(R / f'LENSES_{c}_D_EXPLORATORY.json')
        out['lensC_' + c] = l['lensC_discovery']; out['lensD_' + c] = l['lensD_anytime_valid']
        out['lensB_' + c] = {p: dict(mu=x['mu'], pi=x['pi'], I2=x['I2']) for p, x in l['lensB_random_effects_by_pair'].items()}
        out['seqwealth_' + c] = {k: x for k, x in J(R / f'LENS_D_STAGEB_FIRST_{c}_EXPLORATORY.json').items() if k != 'trace'}
    out['emulation'] = J(R / 'EMULATION_FIDELITY_STAGE_B16.json')
    out['assessment'] = J(R / 'CPU_COMPLETION_ASSESSMENT.json')
    jobs = {}
    for f in sorted(glob.glob(str(F / 'STAGE_B_RUNS' / '*' / 'COLLECT_SUMMARY.json'))):
        s = J(f); jobs[Path(f).parent.name] = dict(planned=s['planned_jobs'], status=s['status_counts'])
    out['jobs'] = jobs; out['jobs_total'] = sum(j['planned'] for j in jobs.values())
    out['folds_total_rows'] = int(sum(len(pd.read_csv(f)) for f in glob.glob(str(F / 'STAGE_B_RUNS' / '*' / 'CROSS_RELEASE_FOLDS.csv.gz'))))
    holm = J(F / 'STAGE_B_SCIENCE_V225' / 'EDGES_HOLM.json')['margins']['0.5']
    out['stageB_pilot'] = {k: {kk: vv for kk, vv in holm[k]['q_mean'].items() if kk != 'per_pair'}
                           for k in ('0.21.3:0.22.2.post1', '1.8.0:1.9.1')}
    out['envs'] = {Path(f).parent.name: {k: J(f).get(k) for k in ('python', 'sklearn', 'numpy', 'scipy')}
                   for f in sorted(glob.glob(str(F / 'envs' / 'sk-*' / 'PROVENANCE.json')))}
    out['amendment_freeze'] = J(F / 'data' / 'AMENDMENT_V226_FREEZE.json')
    (out_dir / 'PAPER_NUMBERS.json').write_text(json.dumps(out, indent=1, default=str))
    return out


def policy_e_vs_d(F, out_dir):
    """Policy E cannot be applied to HGB on E+, so D and E are compared on the other five estimators (10 pairs)."""
    o, n = ED['POS']
    res = {}
    for name, pol, algos in (('D_15pairs', 'D', A), ('D_10pairs_noHGB', 'D', A[:5]), ('E_10pairs_noHGB', 'E', A[:5])):
        m = mean_levels(F / 'STAGE_C_CPU_COMPLETION' / 'merged' / 'POS.csv.gz', pol)
        mo, mn = m.loc[o][algos].dropna(), m.loc[n][algos].dropna()
        ds = mo.index.intersection(mn.index); mo, mn = mo.loc[ds], mn.loc[ds]
        Mk = mk(mo, mn, algos)
        res[name] = dict(M=round(float(Mk.mean()), 4), gt05=int((Mk > 0.5).sum()),
                         best_changes=int((mo.idxmax(1) != mn.idxmax(1)).sum()), K=int(len(ds)))
    (out_dir / 'policy_E_vs_D_10pairs.json').write_text(json.dumps(res, indent=1))
    return res


def figures(out_dir, d):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    FIG = out_dir / 'figures'; FIG.mkdir(exist_ok=True)
    INK, INK2, GRID, BLUE, ORANGE = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
    # Figure: per-dataset change in mean balanced accuracy by estimator, E+
    ch = pd.read_csv(out_dir / 'algo_change_POS.csv', index_col=0)[A]
    fig, ax = plt.subplots(figsize=(6.3, 2.9)); rng = np.random.default_rng(1)
    for i, a in enumerate(A):
        y = ch[a].values; x = i + rng.uniform(-0.22, 0.22, len(y))
        ax.scatter(x, y, s=14, color=BLUE, alpha=0.75, linewidths=0.4, edgecolors='#fcfcfb', zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.mean(y)] * 2, color=INK, lw=1.6, zorder=4)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1); ax.set_xticks(range(6)); ax.set_xticklabels(A)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_ylabel('Change in mean balanced accuracy (pp)'); ax.set_xlim(-0.6, 5.6)
    ax.text(5.55, ax.get_ylim()[1] * 0.97, 'dot = dataset (n = 63)\nbar = mean', ha='right', va='top', color=INK2, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'fig_algo_change.pdf'); fig.savefig(FIG / 'fig_algo_change.png', dpi=200); plt.close(fig)
    # Figure: datasets on which each estimator is best, by release
    L = d['levels']; w0 = [L['0.21.3']['wins'][a] for a in A]; w1 = [L['0.22.2.post1']['wins'][a] for a in A]
    fig, ax = plt.subplots(figsize=(6.3, 2.6)); x = np.arange(6); bw = 0.36
    b0 = ax.bar(x - bw / 2 - 0.01, w0, bw, color=BLUE, label='0.21.3', zorder=3)
    b1 = ax.bar(x + bw / 2 + 0.01, w1, bw, color=ORANGE, label='0.22.2.post1', zorder=3)
    for bars in (b0, b1):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, str(int(r.get_height())), ha='center', va='bottom', fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(A); ax.set_ylabel('Datasets on which estimator is best')
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(frameon=False, loc='upper left'); ax.set_ylim(0, 44)
    fig.tight_layout(); fig.savefig(FIG / 'fig_winners.pdf'); fig.savefig(FIG / 'fig_winners.png', dpi=200); plt.close(fig)
    # Figure: per-dataset M_k for both edges
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for ax, c, title in ((axs[0], 'POS', '0.21.3 → 0.22.2.post1'), (axs[1], 'NEG', '1.8.0 → 1.9.1')):
        s = pd.read_csv(out_dir / f'Mk_{c}.csv', index_col=0).iloc[:, 0].sort_values().values
        ax.scatter(s, np.arange(len(s)), s=10, color=BLUE, zorder=3, linewidths=0.4, edgecolors='#fcfcfb')
        ax.axvline(0.5, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2); ax.text(0.5, len(s) + 1, ' 0.5 pp margin', fontsize=7, color=INK, va='bottom')
        ax.set_title(title, fontsize=8, color=INK, loc='left'); ax.set_xlabel('$M_k$ (pp)'); ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_ylim(-1, len(s) + 5)
    axs[1].set_xlim(-0.02, 0.6); axs[0].set_ylabel('Datasets (sorted)')
    fig.tight_layout(); fig.savefig(FIG / 'fig_mk.pdf'); fig.savefig(FIG / 'fig_mk.png', dpi=200); plt.close(fig)


def check_tex(tex_path, d, e):
    tex = Path(tex_path).read_text()
    f2 = lambda x: f'{x:.2f}'
    F, H, N, HN, P = d['FULL63_POS'], d['HOLDOUT47_POS'], d['FULL63_NEG'], d['HOLDOUT47_NEG'], d['stageB_pilot']
    C = lambda k: d[k]['conclusions']; L = d['levels']; At = d['attribution_NEG']['summary']['full']['runtime_stack']
    checks = {
        'M E+': f2(F['M']), 'lb': f2(F['lb']), 'ub': f2(F['ub']), 'M unseen': f2(H['M']), 'lb u': f2(H['lb']), 'ub u': f2(H['ub']),
        'pilot M': f2(P['0.21.3:0.22.2.post1']['M_pp']), 'M E0': f2(N['M']), 'E0 ub': f2(N['ub']), 'E0 unseen ub': f2(HN['ub']),
        'affected': f"{F['affected_0.25']}\\,/\\,{F['affected_0.5']}\\,/\\,{F['affected_1.0']}",
        'O1': f" {C('FULL63_POS')['O1_best_algorithm_changes']} of the 63",
        'O2': f"{C('FULL63_POS')['O2_pair_flips']}\\,/\\,{C('FULL63_POS')['O2_pair_units']}",
        'rank old': f"GB ({L['0.21.3']['mean_rank']['GB']:.2f})", 'rank new': f"LR ({L['0.22.2.post1']['mean_rank']['LR']:.2f})",
        'runtime stack': f"{At['mean_abs']:.2f}", 'emulation': f"{d['emulation']['emulator_M_pp']:.2f}$ against {d['emulation']['real_M_pp']:.2f}",
        'D5': f2(e['D_10pairs_noHGB']['M']), 'E5': f2(e['E_10pairs_noHGB']['M']), 'jobs': f"{d['jobs_total']:,}",
    }
    missing = [(k, v) for k, v in checks.items() if v not in tex]
    print(f'manuscript check: {len(checks) - len(missing)}/{len(checks)} numbers found' + (f'; missing {missing}' if missing else ''))
    return not missing


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=root / 'results')
    ap.add_argument('--out', type=Path, default=root / 'paper' / 'output')
    ap.add_argument('--tex', help='optional manuscript .tex to check numbers against')
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = collect(a.results, a.out)
    e = policy_e_vs_d(a.results, a.out)
    if not a.no_figures:
        figures(a.out, d)
    print(f"E+ (63 tasks): M = {d['FULL63_POS']['M']:.2f} pp, 90% CI {d['FULL63_POS']['lb']:.2f}-{d['FULL63_POS']['ub']:.2f}; "
          f"best classifier changed on {d['FULL63_POS']['conclusions']['O1_best_algorithm_changes']} tasks")
    print(f"E+ hold-out (47): M = {d['HOLDOUT47_POS']['M']:.2f} pp | E0 (63): M = {d['FULL63_NEG']['M']:.2f} pp")
    print(f"policy D vs E (5 estimators): {e['D_10pairs_noHGB']['M']:.2f} vs {e['E_10pairs_noHGB']['M']:.2f} pp; "
          f"best changes {e['D_10pairs_noHGB']['best_changes']} vs {e['E_10pairs_noHGB']['best_changes']}")
    print('outputs in', a.out)
    if a.tex and not check_tex(a.tex, d, e):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
