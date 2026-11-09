#!/usr/bin/env python3
"""Amendment v2.2.6 A3: hash-based (H) and score-based (S1, S2) outcomes for the 78
changelog-oracle opportunities, from the 14-release longitudinal run (policy D, 47 datasets).

Requires the frozen amendment record. Writes (write-once) to
<data>/STAGE_C_CPU_COMPLETION/longitudinal/ORACLE_SCORE_OUTCOMES_V226.{csv,json}.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
LONG = ['0.21.3', '0.22.2.post1', '0.23.2', '0.24.2', '1.0.2', '1.1.3', '1.2.2',
        '1.3.2', '1.4.2', '1.5.2', '1.6.1', '1.7.2', '1.8.0', '1.9.1']
EXPECTED_ORACLE = '33f04fb6c6da89654f3524cfaef290644b80e37afcd5148dcffa740e9bcfe2f6'
RULES = {'strict': {'change_expected', 'change_possible'},
         'lenient': {'change_expected', 'change_possible', 'fp_level_change_possible'}}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outcomes(rows, datasets, edges):
    d = rows[(rows.policy == 'D') & (rows.status == 'ok')].copy()
    d['fold_key'] = d.rep.astype(str) + '_' + d.fold.astype(str)
    g = {k: v.set_index('fold_key') for k, v in d.groupby(['dataset', 'release', 'algo'])}
    out = []
    for a, b in edges:
        for algo in ALGOS:
            n_eval = n_hash = n_score = 0
            for ds in sorted(datasets):
                o, n = g.get((ds, a, algo)), g.get((ds, b, algo))
                if o is None or n is None or len(o) != 30 or len(n) != 30 or set(o.index) != set(n.index):
                    continue
                n_eval += 1
                n_hash += int(any(str(o.loc[k, 'pred_sha']) != str(n.loc[k, 'pred_sha']) for k in o.index))
                n_score += int(abs(o.ba.mean() - n.ba.mean()) * 100 >= 0.5)
            est = n_eval > 0
            out.append(dict(edge=f'{a}->{b}', estimator=algo, n_datasets_evaluable=n_eval,
                            datasets_any_hash_change=n_hash, datasets_abs_dBA_ge_0_5pp=n_score,
                            H=(n_hash > 0) if est else None, S1=(n_score > 0) if est else None,
                            S2=(n_score / n_eval >= 0.10) if est else None,
                            status='MEASURED' if est else 'NOT_ESTIMABLE'))
    return pd.DataFrame(out)


def confusion(joined):
    res = {}
    valid = joined[joined.status.eq('MEASURED')]
    for outcome in ('H', 'S1', 'S2'):
        actual = valid[outcome].astype(bool)
        res[outcome] = dict(prevalence=float(actual.mean()) if len(valid) else None)
        for rule, cats in RULES.items():
            pred = valid.predicted_impact_category.isin(cats)
            tp, fp = int((pred & actual).sum()), int((pred & ~actual).sum())
            fn, tn = int((~pred & actual).sum()), int((~pred & ~actual).sum())
            res[outcome][rule] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                                      precision=tp / (tp + fp) if tp + fp else None,
                                      recall=tp / (tp + fn) if tp + fn else None,
                                      predicted_positive_share=float(pred.mean()) if len(valid) else None)
    res['valid_opportunities'] = int(len(valid))
    res['not_estimable'] = int(joined.status.eq('NOT_ESTIMABLE').sum())
    return res


def write_once(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            sys.exit('STOP: existing output differs, preserved: ' + str(path))
        return
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='/mnt/c/VersionShift_v2_2_5')
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    ap.add_argument('--inp', default=None, help='default: STAGE_B_RUNS/LONGITUDINAL14_D_E_HOLDOUT47/CROSS_RELEASE_FOLDS.csv.gz')
    ap.add_argument('--partial-test', action='store_true', help='testing only: evaluate edges present in --inp, skip 78-row check, no write-once')
    a = ap.parse_args()
    pr, dat = Path(a.project), Path(a.data)
    frz = dat / 'AMENDMENT_V226_FREEZE.json'
    if not frz.is_file():
        sys.exit('STOP: amendment v2.2.6 not frozen; run addons/vs_freeze_amendment.py first')
    oracle_path = pr / 'ORACLE_FRAME_REVIEWED.csv'
    if sha(oracle_path) != EXPECTED_ORACLE:
        sys.exit('STOP: oracle frame hash differs from frozen reference')
    oracle = pd.read_csv(oracle_path, keep_default_na=False)
    inp = Path(a.inp) if a.inp else dat / 'STAGE_B_RUNS/LONGITUDINAL14_D_E_HOLDOUT47/CROSS_RELEASE_FOLDS.csv.gz'
    rows = pd.read_csv(inp, dtype={'release': str, 'dataset': str})
    if a.partial_test:
        rels = [r for r in LONG if r in set(rows.release)]
        datasets = sorted(rows.dataset.unique())
        edges = [(x, y) for x, y in zip(rels, rels[1:])]
    else:
        lock = json.loads((inp.parent / 'RUN_LOCK.json').read_text())
        sel = json.loads((dat / 'STAGE_C_HOLDOUT47_SELECTION.json').read_text())
        datasets = sorted(sel['stage_b_datasets'])
        if set(lock['releases']) != set(LONG) or set(lock['selected_datasets']) != set(datasets) or len(datasets) != 47:
            sys.exit('STOP: longitudinal run lock does not match 14 releases / 47 holdout datasets')
        edges = list(zip(LONG, LONG[1:]))
    obs = outcomes(rows, datasets, edges)
    joined = oracle.merge(obs, on=['edge', 'estimator'], how='inner' if a.partial_test else 'outer',
                          validate='one_to_one', indicator=True)
    if not a.partial_test and (len(joined) != 78 or not joined._merge.eq('both').all()):
        sys.exit('STOP: observed opportunities do not align with the 78-row oracle')
    joined = joined.drop(columns='_merge')
    report = dict(amendment='v2.2.6 A3', amendment_sha256=json.loads(frz.read_text())['amendment_sha256'],
                  oracle_frame_sha256=EXPECTED_ORACLE, input=str(inp), input_sha256=sha(inp),
                  population=f'{len(datasets)} datasets, policy D', outcomes=dict(
                      H='any frozen fold prediction-hash change on >=1 evaluable dataset (kit definition)',
                      S1='|dBA| >= 0.5 pp (mean of 30 folds) on >=1 evaluable dataset',
                      S2='|dBA| >= 0.5 pp on >=10% of evaluable datasets'),
                  rules={k: sorted(v) for k, v in RULES.items()}, measures=confusion(joined),
                  role='exploratory RQ3/RQ4; descriptive', partial_test=bool(a.partial_test))
    if a.partial_test:
        print(json.dumps(report['measures'], indent=1))
        print(joined[['edge', 'estimator', 'predicted_impact_category', 'n_datasets_evaluable',
                      'datasets_any_hash_change', 'datasets_abs_dBA_ge_0_5pp', 'H', 'S1', 'S2']].to_string(index=False))
        return
    out = dat / 'STAGE_C_CPU_COMPLETION/longitudinal'
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / 'ORACLE_SCORE_OUTCOMES_V226.csv', joined.to_csv(index=False))
    write_once(out / 'ORACLE_SCORE_OUTCOMES_V226.json', json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps(report['measures'], indent=1))
    print('WROTE', out / 'ORACLE_SCORE_OUTCOMES_V226.json')


if __name__ == '__main__':
    main()
