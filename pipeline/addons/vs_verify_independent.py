#!/usr/bin/env python3
"""Independent recomputation from raw CROSS_RELEASE_FOLDS.csv.gz (Amendment v2.2.6, A2/A5/A6).

Does NOT import analysis_v221.py. For every collected run under <data>/STAGE_B_RUNS:
  * A/A runs: bitwise identity of saved fold predictions (AA1 vs AA2).
  * Cross-release runs with policy D: for each adjacent release pair in RUN_LOCK order,
    M (pp), margin counts, and conclusion-level outcomes O1-O4.
If the kit's merged Stage C frame exists, the same is computed for the 63-dataset frame,
the 47 pilot-excluded datasets and the 16 Stage B datasets, and M is cross-checked against
the kit's FULL63_FRAME_CENSUS.json / HOLDOUT47_FINITE_SUBFRAME_SENSITIVITY.json.
Exit code 0 = all checks consistent; 2 = an A/A difference or a cross-check mismatch.
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
PAIRS = list(itertools.combinations(ALGOS, 2))
TOL = 1e-9
problems = []


def aa_identity(root):
    lock = json.loads((root / 'RUN_LOCK.json').read_text())
    res = root / 'results'
    n = differ = missing = 0
    for ds in sorted(lock['selected_datasets']):
        for algo in ALGOS:
            l, r = res / f'AA1__{ds}__{algo}__D.json', res / f'AA2__{ds}__{algo}__D.json'
            if not (l.exists() and r.exists()):
                missing += 1
                continue
            a, b = json.loads(l.read_text()), json.loads(r.read_text())
            if a.get('status') != 'ok' or b.get('status') != 'ok' or a.get('fold_keys') != b.get('fold_keys'):
                differ += 1
                n += 1
                continue
            with np.load(l.with_suffix('.npz'), allow_pickle=False) as za, \
                    np.load(r.with_suffix('.npz'), allow_pickle=False) as zb:
                same = set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)
            differ += (not same)
            n += 1
    out = dict(kind='AA', job_pairs=n, pairs_with_difference=differ, missing_pairs=missing,
               status='PASS' if n and not differ and not missing else 'FAIL')
    if out['status'] != 'PASS':
        problems.append(f'A/A {root.name}: {out}')
    return out


def edge_outcomes(df, old, new, datasets=None):
    d = df[(df.policy == 'D') & (df.status == 'ok')]
    if datasets is not None:
        d = d[d.dataset.astype(str).isin(datasets)]
    d = d[d.release.astype(str).isin([old, new])]
    cnt = d.groupby(['dataset', 'release', 'algo']).size()
    m = d.groupby(['dataset', 'release', 'algo']).ba.mean() * 100
    complete, incomplete = [], []
    for ds in sorted(d.dataset.astype(str).unique()):
        ok = all(cnt.get((ds, r, a), 0) == 30 for r in (old, new) for a in ALGOS)
        (complete if ok else incomplete).append(ds)
    if not complete:
        return dict(old=old, new=new, K_complete=0, incomplete=incomplete, status='NOT_ESTIMABLE')
    M, best_change, flips, mflips, ds_flip, ds_mflip = [], [], 0, 0, 0, 0
    ranks_o, ranks_n, examples = [], [], []
    for ds in complete:
        o = pd.Series({a: m[(ds, old, a)] for a in ALGOS})
        n = pd.Series({a: m[(ds, new, a)] for a in ALGOS})
        M.append(np.mean([abs((n[a] - n[b]) - (o[a] - o[b])) for a, b in PAIRS]))
        if o.idxmax() != n.idxmax():
            best_change.append(ds)
            examples.append(dict(dataset=ds, best_old=o.idxmax(), best_new=n.idxmax()))
        f = mf = 0
        for a, b in PAIRS:
            go, gn = o[a] - o[b], n[a] - n[b]
            if np.sign(go) != np.sign(gn):
                f += 1
                if abs(go) >= 1.0 and abs(gn) >= 1.0:
                    mf += 1
        flips += f
        mflips += mf
        ds_flip += f > 0
        ds_mflip += mf > 0
        ranks_o.append(o.rank(ascending=False))
        ranks_n.append(n.rank(ascending=False))
    M = np.array(M)
    ro = pd.concat(ranks_o, axis=1).mean(axis=1)
    rn = pd.concat(ranks_n, axis=1).mean(axis=1)
    K = len(complete)
    return dict(old=old, new=new, K_complete=K, incomplete=incomplete,
                M_pp=float(M.mean()), M_sd=float(M.std(ddof=1)) if K > 1 else None,
                datasets_M_gt_0_25=int((M > 0.25).sum()), datasets_M_gt_0_5=int((M > 0.5).sum()),
                datasets_M_gt_1_0=int((M > 1.0).sum()),
                O1_best_algorithm_changes=len(best_change), O1_share=len(best_change) / K, O1_examples=examples,
                O2_pair_flips=int(flips), O2_pair_units=15 * K, O2_datasets_with_flip=int(ds_flip),
                O3_material_flips_ge_1pp=int(mflips), O3_datasets_with_material_flip=int(ds_mflip),
                O4_mean_rank_old=ro.round(4).to_dict(), O4_mean_rank_new=rn.round(4).to_dict(),
                O4_order_old=list(ro.sort_values(kind='stable').index),
                O4_order_new=list(rn.sort_values(kind='stable').index),
                O4_order_changed=list(ro.sort_values(kind='stable').index) != list(rn.sort_values(kind='stable').index),
                mean_ba_change_by_algo_pp={a: float(np.mean([m[(ds, new, a)] - m[(ds, old, a)] for ds in complete]))
                                           for a in ALGOS},
                status='MEASURED')


def run_report(root):
    lock = json.loads((root / 'RUN_LOCK.json').read_text())
    rels = list(lock['releases'])
    if lock.get('pin_from') is None and set(rels) == {'AA1', 'AA2'}:
        return aa_identity(root)
    if 'D' not in lock['policies']:
        return dict(kind='reverse_or_policy_only', note='no policy D; not an edge comparison', policies=lock['policies'])
    df = pd.read_csv(root / 'CROSS_RELEASE_FOLDS.csv.gz', dtype={'release': str, 'dataset': str})
    return dict(kind='cross_release', edges=[edge_outcomes(df, a, b) for a, b in zip(rels, rels[1:])])


def kit_M(report_path, edge):
    try:
        r = json.loads(Path(report_path).read_text())
        return r['margins']['0.5'][edge]['q_mean']['M_pp']
    except Exception:  # noqa
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    ap.add_argument('--out', default=None, help='default <data>/INDEPENDENT_VERIFICATION')
    a = ap.parse_args()
    dat = Path(a.data)
    out = Path(a.out) if a.out else dat / 'INDEPENDENT_VERIFICATION'
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), tool='vs_verify_independent.py', runs={}, frames={})
    for root in sorted((dat / 'STAGE_B_RUNS').glob('*')):
        if not ((root / 'RUN_LOCK.json').is_file() and (root / 'COLLECT_SUMMARY.json').is_file()):
            continue
        try:
            rep = run_report(root)
        except Exception as e:  # noqa
            rep = dict(kind='error', error=repr(e))
            problems.append(f'{root.name}: {e!r}')
        summary['runs'][root.name] = rep
        if rep.get('kind') == 'AA':
            print(f"{root.name:<32} A/A {rep['status']}: {rep['job_pairs']} pairs, {rep['pairs_with_difference']} differ")
        for e in rep.get('edges', []):
            if e['status'] == 'MEASURED':
                print(f"{root.name:<32} {e['old']}->{e['new']}: K={e['K_complete']} M={e['M_pp']:.7f} pp | "
                      f">0.5pp {e['datasets_M_gt_0_5']} | best changes {e['O1_best_algorithm_changes']} | "
                      f"flips {e['O2_pair_flips']}/{e['O2_pair_units']} (material {e['O3_material_flips_ge_1pp']})")
    comp = dat / 'STAGE_C_CPU_COMPLETION'
    merged = comp / 'merged'
    csel = dat / 'STAGE_C_HOLDOUT47_SELECTION.json'
    bsel = dat / 'STAGE_B_SELECTION.json'
    if (merged / 'POS.csv.gz').is_file() and (merged / 'NEG.csv.gz').is_file() and csel.is_file():
        hold = set(json.loads(csel.read_text())['stage_b_datasets'])
        pilot = set(json.loads(bsel.read_text())['stage_b_datasets'])
        for code, old, new in (('POS', '0.21.3', '0.22.2.post1'), ('NEG', '1.8.0', '1.9.1')):
            df = pd.read_csv(merged / f'{code}.csv.gz', dtype={'release': str, 'dataset': str})
            for label, ds in (('FULL63', None), ('HOLDOUT47', hold), ('PILOT16', pilot)):
                r = edge_outcomes(df, old, new, ds)
                key = f'{label}_{old}:{new}'
                kitfile = {'FULL63': comp / 'reports/FULL63_FRAME_CENSUS.json',
                           'HOLDOUT47': comp / 'reports/HOLDOUT47_FINITE_SUBFRAME_SENSITIVITY.json'}.get(label)
                if kitfile is not None and r['status'] == 'MEASURED':
                    km = kit_M(kitfile, f'{old}:{new}')
                    r['kit_M_pp'] = km
                    r['cross_check'] = ('NO_KIT_REPORT' if km is None else
                                        ('MATCH' if abs(km - r['M_pp']) <= TOL else 'MISMATCH'))
                    if r['cross_check'] == 'MISMATCH':
                        problems.append(f'{key}: independent M {r["M_pp"]} vs kit {km}')
                summary['frames'][key] = r
                if r['status'] == 'MEASURED':
                    print(f"{key:<32} K={r['K_complete']} M={r['M_pp']:.7f} pp | best changes "
                          f"{r['O1_best_algorithm_changes']}/{r['K_complete']} | flips {r['O2_pair_flips']}/"
                          f"{r['O2_pair_units']} | kit check {r.get('cross_check', '-')}")
    summary['problems'] = problems
    summary['status'] = 'CONSISTENT' if not problems else 'PROBLEMS_FOUND'
    (out / 'VERIFY_SUMMARY.json').write_text(json.dumps(summary, indent=2, default=str))
    print('\nINDEPENDENT VERIFICATION:', summary['status'], '->', out / 'VERIFY_SUMMARY.json')
    for p in problems:
        print('  PROBLEM:', p)
    sys.exit(2 if problems else 0)


if __name__ == '__main__':
    main()
