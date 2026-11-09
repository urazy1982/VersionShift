#!/usr/bin/env python3
"""VersionShift G3-A/B: read-only, dataset-cluster uncertainty and near-tie sensitivity.

Only reads the original frozen replication archive. Does not fit models or
modify the archived dataset, folds, worker, results or amendment.
"""
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import random
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

ALGOS = ('LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB')
EDGES = {
    'POS': ('0.21.3', '0.22.2.post1'),
    'NEG': ('1.8.0', '1.9.1'),
}
SOURCE_TEMPLATE = 'VersionShift/results/STAGE_C_CPU_COMPLETION/merged/{}.csv.gz'
BASELINE_SHA = '5eb716b64c21ded6f979b0b1610bb68aee46ad2f24889a0d3b133bd430c6ff7d'


def canonical_fold(s):
    x = float(s)
    if not math.isfinite(x) or not x.is_integer():
        raise ValueError('Non-integral rep/fold: ' + repr(s))
    return int(x)


def read_edge(z, edge):
    name = SOURCE_TEMPLATE.format(edge)
    old, new = EDGES[edge]
    cells = defaultdict(dict)
    counts = defaultdict(int)
    with gzip.GzipFile(fileobj=z.open(name)) as stream:
        r = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8', newline=''))
        for row in r:
            if row['policy'] != 'D':
                continue
            if row['status'] != 'ok':
                raise ValueError('Non-OK default policy: ' + str(row))
            d, rel, algo = row['dataset'], row['release'], row['algo']
            if rel not in (old, new) or algo not in ALGOS:
                raise ValueError('Unexpected default-policy identity: ' + str((d, rel, algo)))
            fold = (canonical_fold(row['rep']), canonical_fold(row['fold']))
            score = float(row['ba'])
            if not 0 <= score <= 1 or not math.isfinite(score):
                raise ValueError('Invalid score: ' + str((d, rel, algo, fold, score)))
            key = (d, rel, algo)
            if fold in cells[key]:
                raise ValueError('Duplicate fold: ' + str((key, fold)))
            cells[key][fold] = score
            counts[(d, rel)] += 1
    datasets = sorted({x[0] for x in cells})
    if len(datasets) != 63 or len(cells) != 63 * 2 * 6:
        raise ValueError('Frame/cell coverage is not 63 x 2 x 6: %s %s' % (len(datasets), len(cells)))
    expected_keys = {(r, f) for r in range(6) for f in range(5)}
    for key, folds in cells.items():
        if set(folds) != expected_keys:
            raise ValueError('Missing or extra folds: ' + str(key))
    means = {k: math.fsum(v[t] for t in sorted(v)) / 30.0 for k, v in cells.items()}
    return datasets, means


def signed_gap(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def evaluate(datasets, means, edge):
    old, new = EDGES[edge]
    rows = []
    pairs = list(itertools.combinations(ALGOS, 2))
    for d in datasets:
        so = {a: means[(d, old, a)] * 100 for a in ALGOS}
        sn = {a: means[(d, new, a)] * 100 for a in ALGOS}
        def rank(scores):
            order = sorted(ALGOS, key=lambda a: (-scores[a], ALGOS.index(a)))
            return order[0], scores[order[0]] - scores[order[1]]
        wo, go = rank(so)
        wn, gn = rank(sn)
        shifts = []
        flips = strict = material = 0
        for a, b in pairs:
            x = so[a] - so[b]
            y = sn[a] - sn[b]
            shifts.append(abs(y - x))
            sx, sy = signed_gap(x), signed_gap(y)
            if sx != sy:
                flips += 1
            if sx * sy == -1:
                strict += 1
            if sx * sy == -1 and min(abs(x), abs(y)) >= 1.0:
                material += 1
        rows.append(dict(edge=edge, dataset=d, M_pp=math.fsum(shifts)/len(pairs),
                         winner_old=wo, winner_new=wn,
                         winner_changed=int(wo != wn),
                         old_lead_margin_pp=go, new_lead_margin_pp=gn,
                         sign_changes=flips, strict_reversals=strict,
                         material_reversals=material))
    return rows


def bootstrap_mean(vals, nboot, seed):
    r = random.Random(seed)
    n = len(vals)
    draws = sorted(math.fsum(vals[r.randrange(n)] for _ in range(n)) / n for _ in range(nboot))
    def percentile(p):
        q = (len(draws)-1)*p
        lo = math.floor(q)
        hi = math.ceil(q)
        return draws[lo]*(hi-q)+draws[hi]*(q-lo)
    return dict(ci90_percentile_pp=[percentile(.05),percentile(.95)],
                ci95_percentile_pp=[percentile(.025),percentile(.975)],
                bootstrap_replicates=nboot, bootstrap_seed=seed)


def aggregate(rows, nboot, seed):
    vals = [r['M_pp'] for r in rows]
    k = len(rows)
    ordered = sorted(vals)
    remove = max(1, math.ceil(k * .05))
    mean = statistics.mean(vals)
    influence = [(sum(vals)-v)/(k-1) for v in vals]
    robustness = dict(mean_M_pp=mean, median_M_pp=statistics.median(vals),
                      max_M_pp=max(vals), min_M_pp=min(vals),
                      mean_after_dropping_largest_5pct_pp=statistics.mean(ordered[:-remove]),
                      dropped_count=remove,
                      leave_one_out_mean_M_range_pp=[min(influence),max(influence)],
                      affected_gt_0_25=sum(v > .25 for v in vals),
                      affected_gt_0_5=sum(v > .5 for v in vals),
                      affected_gt_1_0=sum(v > 1 for v in vals),
                      winners_changed=sum(r['winner_changed'] for r in rows),
                      sign_changes=sum(r['sign_changes'] for r in rows),
                      strict_reversals=sum(r['strict_reversals'] for r in rows),
                      material_reversals=sum(r['material_reversals'] for r in rows))
    sensitivity = []
    for threshold in (0, .25, .5, 1.0):
        significant = [r for r in rows if r['winner_changed'] and
                       min(r['old_lead_margin_pp'], r['new_lead_margin_pp']) >= threshold]
        sensitivity.append(dict(lead_margin_threshold_pp=threshold,
                                changed_winners_with_both_leads_ge_threshold=len(significant),
                                eligible_tasks_with_both_leads_ge_threshold=sum(
                                    min(r['old_lead_margin_pp'],r['new_lead_margin_pp']) >= threshold
                                    for r in rows),
                                changed_datasets=[r['dataset'] for r in significant]))
    return dict(K=k, summary=robustness, bootstrap=bootstrap_mean(vals,nboot,seed),
                winner_sensitivity=sensitivity)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bootstraps', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261108)
    p.add_argument('--allow-archive-sha-mismatch', action='store_true')
    a = p.parse_args()
    if a.bootstraps < 200:
        p.error('at least 200 bootstrap draws required')
    archive = Path(a.zip)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if sha != BASELINE_SHA and not a.allow_archive_sha_mismatch:
        raise SystemExit('FAIL: source archive SHA-256 differs from frozen reference: ' + sha)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit('FAIL: output directory is not empty (do not overwrite prior evidence): ' + str(out))
    summaries = {}
    allrows = []
    with zipfile.ZipFile(archive) as z:
        for i, edge in enumerate(('POS','NEG')):
            datasets, means = read_edge(z, edge)
            rows = evaluate(datasets, means, edge)
            allrows.extend(rows)
            summaries[edge] = aggregate(rows, a.bootstraps, a.seed+i)
    # Grounding checks against independently verified G0-D point estimates.
    pos, neg = summaries['POS']['summary'], summaries['NEG']['summary']
    expected = [(pos['mean_M_pp'],5.006712451138973),
                (neg['mean_M_pp'],.0513097803725926)]
    for observed, old in expected:
        if abs(observed-old)>1e-7:
            raise SystemExit('FAIL: mean M does not match G0-D; inspect score aggregation: %r vs %r' % (observed,old))
    if (pos['winners_changed'],pos['affected_gt_0_5'],neg['winners_changed']) != (25,59,0):
        raise SystemExit('FAIL: G0-D anchor counts differ')
    out.mkdir(parents=True)
    with (out/'G3_AB_dataset_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]))
        w.writeheader();w.writerows(allrows)
    with (out/'G3_AB_winner_thresholds.csv').open('w',newline='') as f:
        fields=['edge','threshold_pp','eligible_tasks','changed_winners']
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for edge in ('POS','NEG'):
            for row in summaries[edge]['winner_sensitivity']:
                w.writerow(dict(edge=edge,threshold_pp=row['lead_margin_threshold_pp'],
                                eligible_tasks=row['eligible_tasks_with_both_leads_ge_threshold'],
                                changed_winners=row['changed_winners_with_both_leads_ge_threshold']))
    report=dict(status='PASS',source_sha256=sha,scope='frozen six-estimator 63-dataset original D-policy analysis',
                procedure='dataset-level cluster bootstrap percentile intervals; strictly descriptive margin sensitivity',
                interpretation_warning='Bootstrap treats observed 63 tasks as resampled units; inferential generalization beyond finite frame requires assumptions. Winner threshold requires both release-specific first-vs-second gaps to meet threshold.',
                edges=summaries)
    (out/'G3_AB_results.json').write_text(json.dumps(report,indent=2))
    with (out/'G3_AB_summary.txt').open('w') as f:
        f.write('VersionShift G3-A/B: independent read-only robustness assessment\n')
        f.write('Archive sha256: '+sha+'\n')
        for edge in ('POS','NEG'):
            s=summaries[edge]
            f.write('%s: K=%s M=%.8f pp 90%% bootstrap CI=[%.4f,%.4f] 95%% bootstrap CI=[%.4f,%.4f]\n' %
                    (edge,s['K'],s['summary']['mean_M_pp'],*s['bootstrap']['ci90_percentile_pp'],*s['bootstrap']['ci95_percentile_pp']))
            f.write('  median=%.4f; mean dropping top %d datasets=%.4f; winners=%d; flips=%d strict=%d\n'%
                    (s['summary']['median_M_pp'],s['summary']['dropped_count'],
                     s['summary']['mean_after_dropping_largest_5pct_pp'],
                     s['summary']['winners_changed'],s['summary']['sign_changes'],s['summary']['strict_reversals']))
            for row in s['winner_sensitivity']:
                f.write('  winner threshold %.2f pp: changed=%d eligible=%d\n'%
                        (row['lead_margin_threshold_pp'],row['changed_winners_with_both_leads_ge_threshold'],
                         row['eligible_tasks_with_both_leads_ge_threshold']))
    print((out/'G3_AB_summary.txt').read_text())

if __name__=='__main__':
    main()
