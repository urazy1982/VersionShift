"""VersionShift v2.1 changelog-as-oracle frame (review section 6).

The denominator is every estimator x adjacent-release-edge OPPORTUNITY, fixed before any
cross-release outcome is seen, including explicit no-change opportunities. Each opportunity
gets a prediction from the coded changelog under the study pipeline's conditions
(float64, dense, no sample_weight, n <= 5000, NaN only if imputed).

Prediction sources, kept separate:
  changelog_prediction   from manually coded release-note entries (code/changelog_oracle.csv)
  auto_default_diff      objective: resolved-constructor-default diffs from `controller.py introspect`
Coding status per edge: coded_complete | coded_partial | not_yet_coded. The frame must be
complete (no not_yet_coded rows) and its SHA-256 registered before outcomes are collected.

  python3 oracle_frame.py build --out ORACLE_FRAME.csv [--introspect INTROSPECT.json]
  python3 oracle_frame.py evaluate --frame ORACLE_FRAME.csv --observed OBSERVED.csv
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RELEASES = ['0.21.3', '0.22.2.post1', '0.23.2', '0.24.2', '1.0.2', '1.1.3', '1.2.2', '1.3.2', '1.4.2', '1.5.2',
            '1.6.1', '1.7.2', '1.8.0', '1.9.1']
ESTIMATORS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
# Release notes read in full for these six estimators in this project (others: verify before freezing).
EDGE_CODING = {'1.7.2->1.8.0': 'coded_complete', '1.8.0->1.9.1': 'coded_complete', '0.21.3->0.22.2.post1': 'coded_partial',
               '0.22.2.post1->0.23.2': 'coded_partial', '1.0.2->1.1.3': 'coded_partial'}
RANK = {'none': 0, 'possible_fp': 1, 'possible': 2, 'certain_if_native_splits': 0, 'structural_failure_for_old_config': 3, 'certain': 3}


def minor(v):
    p = v.split('.'); return '%s.%s' % (p[0], p[1])


def expected_keys():
    return {('%s->%s' % (o, n), e) for o, n in zip(RELEASES, RELEASES[1:]) for e in ESTIMATORS}


def build(args):
    entries = list(csv.DictReader(open(HERE / 'changelog_oracle.csv')))
    auto = {}
    if args.introspect:
        for d in json.loads(Path(args.introspect).read_text())['adjacent_default_diffs']:
            auto.setdefault((d['edge'], d['estimator']), []).append('%s: %s -> %s' % (d['param'], d['old'], d['new']))
    rows = []
    for old, new in zip(RELEASES, RELEASES[1:]):
        edge = '%s->%s' % (old, new)
        for est in ESTIMATORS:
            hits = [e for e in entries if e['release'] == minor(new) and e['estimator'] in (est, 'ALL')]
            worst = max([RANK.get(e['effect_unweighted_dense_float64_nonan_n_lt_10k'], 2) for e in hits], default=0)
            pred = {0: 'no_change_expected', 1: 'fp_level_change_possible', 2: 'change_possible', 3: 'change_expected'}[worst]
            coded = EDGE_CODING.get(edge)
            diffs = auto.get((edge, est), []) if args.introspect else None
            rows.append(dict(edge=edge, estimator=est, coding_status=coded or 'not_yet_coded',
                             n_entries=len(hits), entry_categories=';'.join(sorted({e['category'] for e in hits})),
                             predicted_impact_category=pred if coded else '',
                             trigger_condition='; '.join(e['trigger_condition'] for e in hits) if coded else '',
                             applicable_data_types='dense float64; categorical one-hot; NaN median-imputed' if coded else '',
                             estimated_compatibility=('unknown_run_introspect' if diffs is None else
                                                      ('strict_pin_portable_expected' if not diffs else 'strict_pin_may_fail: ' + '; '.join(diffs))),
                             release_note_reference='; '.join(sorted({e['source'] for e in hits})) if coded else '',
                             coding_uncertainty='' if not coded else ('low' if coded == 'coded_complete' else 'high'),
                             coder='' if not coded else 'draft (assistant); author must confirm',
                             coded_on='',
                             auto_default_diff='; '.join(diffs) if diffs is not None else 'run_introspect',
                             pipeline_conditions='frozen central splits; float64 dense, no sample_weight, n<=5000, median-imputed NaN'))
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    prov = dict(builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                changelog_csv_sha256=hashlib.sha256((HERE / 'changelog_oracle.csv').read_bytes()).hexdigest(),
                introspect_sha256=hashlib.sha256(Path(args.introspect).read_bytes()).hexdigest() if args.introspect else None,
                releases=RELEASES, estimators=ESTIMATORS, expected_opportunities=len(expected_keys()), built_frame_sha256=digest)
    Path(str(args.out) + '.PROVENANCE.json').write_text(json.dumps(prov, indent=1))
    print('opportunities: %d (%d edges x %d estimators); coding status: %s; sha256 %s' % (
        len(df), len(RELEASES) - 1, len(ESTIMATORS), df.coding_status.value_counts().to_dict(), digest))


def evaluate(args):
    """observed CSV: edge, estimator, observed_change (0/1; e.g. any bitwise prediction change under policy D)."""
    f = pd.read_csv(args.frame); o = pd.read_csv(args.observed)
    if (f.coding_status != 'coded_complete').any():
        raise SystemExit('frame incomplete: every opportunity must be coded_complete (and the frame hash registered) before evaluation')
    m = f.merge(o, on=['edge', 'estimator'], how='left', validate='one_to_one')
    if m.observed_change.isna().any():
        raise SystemExit('observed outcome missing for some opportunities')
    out = {}
    for rule, positive in [('strict', {'change_expected'}), ('lenient', {'change_expected', 'change_possible', 'fp_level_change_possible'})]:
        pred = m.predicted_impact_category.isin(positive).astype(int); obs = m.observed_change.astype(int)
        tp = int(((pred == 1) & (obs == 1)).sum()); fp = int(((pred == 1) & (obs == 0)).sum())
        fn = int(((pred == 0) & (obs == 1)).sum()); tn = int(((pred == 0) & (obs == 0)).sum())
        out[rule] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else None,
                         recall=tp / (tp + fn) if tp + fn else None, specificity=tn / (tn + fp) if tn + fp else None,
                         undocumented_changes=m[(pred == 0) & (obs == 1)][['edge', 'estimator']].to_dict('records'))
    print(json.dumps(out, indent=1)); return out


REQUIRED = ['predicted_impact_category', 'trigger_condition', 'applicable_data_types', 'estimated_compatibility',
            'release_note_reference', 'coding_uncertainty', 'coder', 'coded_on']
CATEGORIES = ['no_change_expected', 'fp_level_change_possible', 'change_possible', 'change_expected']


def freeze(args):
    """Gate: every opportunity coded_complete with all fields; writes ORACLE_FROZEN.json (hash + time)."""
    f = pd.read_csv(args.frame, keep_default_na=False)
    keys = list(zip(f.edge, f.estimator)); exp = expected_keys()
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if len(f) != len(exp) or dups or set(keys) != exp:
        raise SystemExit('cannot freeze: frame must hold exactly the %d expected edge x estimator keys once each '
                         '(rows=%d, duplicates=%s, missing=%s, unexpected=%s)' % (len(exp), len(f), dups[:3],
                         sorted(exp - set(keys))[:3], sorted(set(keys) - exp)[:3]))
    prov_path = Path(str(args.frame) + '.PROVENANCE.json')
    if not prov_path.is_file():
        raise SystemExit('cannot freeze: build provenance file missing (%s)' % prov_path.name)
    prov = json.loads(prov_path.read_text())
    if prov.get('releases') != RELEASES or prov.get('estimators') != ESTIMATORS or prov.get('expected_opportunities') != len(exp):
        raise SystemExit('cannot freeze: frame was built for a different release/estimator set')
    if prov.get('changelog_csv_sha256') != hashlib.sha256((HERE / 'changelog_oracle.csv').read_bytes()).hexdigest():
        raise SystemExit('cannot freeze: changelog_oracle.csv changed since the frame was built')
    bad = f[(f.coding_status != 'coded_complete') | (f[REQUIRED] == '').any(axis=1) |
            ~f.predicted_impact_category.isin(CATEGORIES) | f.coder.str.contains('draft')]
    if len(bad):
        raise SystemExit('cannot freeze: %d of %d opportunities incomplete, draft-coded or invalid (first: %s)' % (
            len(bad), len(f), bad[['edge', 'estimator']].head(3).to_dict('records')))
    import time
    out = dict(status='FROZEN_COMPLETE', frame_path=str(Path(args.frame).resolve()),
               frame_sha256=hashlib.sha256(Path(args.frame).read_bytes()).hexdigest(), opportunities=len(f),
               frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), provenance=prov)
    Path(args.out).write_text(json.dumps(out, indent=1)); print(out)


def recode_sample(args):
    """Blinded intra-rater recode sheet: seeded 25% sample, labels removed, earliest recode date = frozen_at + 14 days."""
    import datetime
    f = pd.read_csv(args.frame, keep_default_na=False); fz = json.loads(Path(args.frozen).read_text())
    if hashlib.sha256(Path(args.frame).read_bytes()).hexdigest() != fz['frame_sha256']:
        raise SystemExit('frame changed since freezing')
    n = int(np.ceil(args.frac * len(f)))
    idx = np.sort(np.random.default_rng(args.seed).choice(len(f), n, replace=False))
    sheet = f.iloc[idx][['edge', 'estimator', 'pipeline_conditions']].copy()
    for c in REQUIRED:
        sheet[c] = ''
    sheet.to_csv(args.out, index=False)
    t0 = datetime.datetime.strptime(fz['frozen_at'][:19], '%Y-%m-%dT%H:%M:%S')
    key = dict(frame_sha256=fz['frame_sha256'], sample_rows=idx.tolist(), seed=args.seed, frac=args.frac,
               earliest_recode=(t0 + datetime.timedelta(days=14)).strftime('%Y-%m-%d'),
               note='intra-rater reliability check (same author, labels concealed); not inter-rater validation')
    Path(args.out + '.KEY.json').write_text(json.dumps(key, indent=1)); print('sheet', args.out, 'rows', n, 'recode on/after', key['earliest_recode'])


def cohen_kappa(a, b, cats):
    a, b = list(a), list(b); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(args):
    import datetime
    f = pd.read_csv(args.frame, keep_default_na=False); r = pd.read_csv(args.recode, keep_default_na=False)
    key = json.loads(Path(args.recode + '.KEY.json').read_text())
    if any(d < key['earliest_recode'] for d in r.coded_on):
        raise SystemExit('recode dated before the 14-day blind window ended')
    m = r.merge(f, on=['edge', 'estimator'], suffixes=('_recode', '_original'), validate='one_to_one')
    po, k = cohen_kappa(m.predicted_impact_category_original, m.predicted_impact_category_recode, CATEGORIES)
    dis = m[m.predicted_impact_category_original != m.predicted_impact_category_recode]
    dis[['edge', 'estimator', 'predicted_impact_category_original', 'predicted_impact_category_recode']].assign(
        adjudicated='', rationale='').to_csv(args.out_adjudication, index=False)
    res = dict(n=len(m), raw_agreement=po, cohen_kappa=k, disagreements=len(dis), kind='intra-rater (same author)',
               adjudication_file=args.out_adjudication)
    print(json.dumps(res, indent=1)); return res


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--out', required=True); b.add_argument('--introspect')
    e = sub.add_parser('evaluate'); e.add_argument('--frame', required=True); e.add_argument('--observed', required=True)
    z = sub.add_parser('freeze'); z.add_argument('--frame', required=True); z.add_argument('--out', required=True)
    r = sub.add_parser('recode-sample'); r.add_argument('--frame', required=True); r.add_argument('--frozen', required=True)
    r.add_argument('--out', required=True); r.add_argument('--frac', type=float, default=0.25); r.add_argument('--seed', type=int, default=20261106)
    g = sub.add_parser('agreement'); g.add_argument('--frame', required=True); g.add_argument('--recode', required=True)
    g.add_argument('--out-adjudication', required=True)
    a = ap.parse_args()
    dict(build=build, evaluate=evaluate, freeze=freeze, **{'recode-sample': recode_sample}, agreement=agreement)[a.cmd](a)


if __name__ == '__main__':
    main()
