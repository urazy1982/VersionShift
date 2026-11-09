"""VersionShift v2.2.1 analysis of real cross-release runs (audit corrections; analysis_v22.py kept as record).

  edge         decision rule per edge at all margins; Holm across --primary edges, BH across the rest
  mitigation   three populations kept apart: strictly portable (P ok), translation-required (P fails, PT ok),
               non-transferable (both fail); mitigation estimated separately for P and PT
  attribution  2x2 cells -> Shapley shares; the runtime component is a RUNTIME-STACK effect (scikit-learn +
               numpy/scipy/BLAS/OpenMP + interpreter), not a scikit-learn-only implementation effect
  size         Stage-C K: max of negative-edge precision, positive/change-coded-edge power, prevalence floor

Strict input validation (audit P0): before ANY inference every (dataset, release, policy, algo) job must have
exactly the frozen fold set {0..R-1} x {0..F-1}, identities must be unique, and old/new cells of a pair must
cover identical folds. Violations stop the analysis; nothing is averaged over partial folds.
"""
import argparse, hashlib, itertools, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decision_v221 as DV
import attribution as AT

ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
ID = ['dataset', 'release', 'policy', 'algo', 'rep', 'fold']


class InvalidInput(SystemExit):
    pass


def gate(args):
    if getattr(args, 'unblinded_exploratory', False):
        return dict(blinding='UNBLINDED_EXPLORATORY', note='outcomes revealed before the oracle frame was frozen')
    if not getattr(args, 'oracle_frozen', None):
        raise SystemExit('blinding gate: pass --oracle-frozen ORACLE_FROZEN.json (or --unblinded-exploratory, recorded in the report)')
    fz = json.loads(Path(args.oracle_frozen).read_text()); frame = Path(fz['frame_path'])
    if fz.get('status') != 'FROZEN_COMPLETE' or hashlib.sha256(frame.read_bytes()).hexdigest() != fz['frame_sha256']:
        raise SystemExit('blinding gate: oracle frame missing, changed since freezing, or incomplete')
    return dict(blinding='ORACLE_FROZEN_BEFORE_OUTCOMES', frame_sha256=fz['frame_sha256'], frozen_at=fz['frozen_at'])


def validate(df, reps, folds):
    """Strict fold matching. Returns the ok rows; raises InvalidInput with a precise reason."""
    ok = df[df.status == 'ok'].copy()
    if ok[['rep', 'fold']].isna().any().any():
        raise InvalidInput('ok rows without rep/fold identity')
    ok['rep'] = ok.rep.astype(int); ok['fold'] = ok.fold.astype(int)
    dup = ok.duplicated(ID, keep=False)
    if dup.any():
        raise InvalidInput('duplicate fold identities, e.g. %s' % ok[dup][ID].head(2).to_dict('records'))
    expected = {(r, f) for r in range(reps) for f in range(folds)}
    for key, s in ok.groupby(['dataset', 'release', 'policy', 'algo']):
        got = set(zip(s.rep, s.fold))
        if got != expected:
            raise InvalidInput('job %s has %d/%d frozen folds (missing %s, unexpected %s)' % (
                key, len(got & expected), len(expected), sorted(expected - got)[:3], sorted(got - expected)[:3]))
    if not np.isfinite(ok.ba).all() or ((ok.ba < 0) | (ok.ba > 1)).any():
        raise InvalidInput('scores outside [0,1] or non-finite')
    return ok


G2_MAX_FAILED_SHARE = 0.05


def planned_datasets(plan_dir, selection):
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / 'PLAN.json').read_text())
    expected = {d['name'] for d in plan['datasets']}
    if selection:
        sel = json.loads(Path(selection).read_text())
        if sel.get('plan_sha256') != hashlib.sha256((plan_dir / 'PLAN.json').read_bytes()).hexdigest():
            raise InvalidInput('selection belongs to another frozen plan')
        expected = set(sel['stage_b_datasets'])
        if len(expected) != len(sel['stage_b_datasets']) or not expected:
            raise InvalidInput('invalid or duplicate selection dataset IDs')
    return expected


def validate_plan_coverage(df, plan_dir, selection, combos, policy=None, algos=ALGOS, d_policy='D'):
    """v2.2.3: one status record must exist for EVERY planned (dataset, release, policy, algo) job named in
    `combos` (a set of (release, policy)); an absent record is an integrity failure, never a silent drop.
    Records with status not_portable / fit_failed are outcomes, not absences:
      * pinned policies (P, PT): not_portable is a portability outcome (failure-to-transfer)
      * default policy D: a failed job excludes that DATASET from the edge analysis; exclusions are ledgered.
        G2 is defined as a job-level success gate (>=95% planned D jobs succeed), NOT a dataset-level
        exclusion gate. Dataset retention is reported separately for scientific interpretation."""
    if policy is not None:      # v2.2.2 call form: (edges ['old:new', ...], policy)
        combos = {(r, policy) for e in combos for r in e.split(':')}; d_policy = policy
    expected = planned_datasets(plan_dir, selection)
    present = set(df.dataset.astype(str))
    if present != expected:
        raise InvalidInput('missing or extra dataset in fold table: missing=%s extra=%s' %
                           (sorted(expected - present), sorted(present - expected)))
    required = {(d, r, pol, a) for d in expected for (r, pol) in combos for a in algos}
    actual = set(map(tuple, df[['dataset', 'release', 'policy', 'algo']].astype(str).drop_duplicates().itertuples(index=False, name=None)))
    missing = required - actual
    if missing:
        raise InvalidInput('%d planned jobs entirely absent (no status record), e.g. %s' % (len(missing), sorted(missing)[:3]))
    d_releases = {r for (r, pol) in combos if pol == d_policy}
    bad = df[(df.policy == d_policy) & df.release.isin(d_releases) & (df.status != 'ok')]
    excluded = sorted(set(bad.dataset.astype(str)))
    ledger = bad.drop_duplicates(['dataset', 'release', 'algo'])[['dataset', 'release', 'algo', 'status']].to_dict('records')
    planned_default_jobs = len(expected) * len(d_releases) * len(algos)
    failed_default_jobs = len({(str(row.dataset), str(row.release), str(row.algo))
                               for row in bad[['dataset', 'release', 'algo']].itertuples(index=False)})
    job_failure_share = failed_default_jobs / planned_default_jobs
    excluded_share = len(excluded) / len(expected)
    if job_failure_share > G2_MAX_FAILED_SHARE + 1e-12:
        raise InvalidInput('G2 failed: %d of %d planned %s jobs (%.2f%%) failed (limit %.0f%%)' % (
            failed_default_jobs, planned_default_jobs, d_policy, 100 * job_failure_share, 100 * G2_MAX_FAILED_SHARE))
    return dict(status='PASS', planned_datasets=len(expected), required_jobs=len(required),
                planned_default_jobs=planned_default_jobs, failed_default_jobs=failed_default_jobs,
                default_job_failure_share=job_failure_share, default_job_success_share=1-job_failure_share,
                excluded_datasets_failed_default_jobs=excluded, failure_ledger=ledger,
                excluded_share=excluded_share, retained_datasets=len(expected)-len(excluded),
                rule='G2 requires >=95%% default-policy job success; any failed default job excludes its dataset from the analysed contrast; dataset retention reported separately')


def coverage_or_exploratory(args, df, combos):
    if getattr(args, 'plan_dir', None):
        return validate_plan_coverage(df, args.plan_dir, getattr(args, 'selection', None), combos)
    if args.unblinded_exploratory:
        return dict(status='NOT_CHECKED_EXPLORATORY', warning='No frozen plan; cannot certify complete job universe',
                    excluded_datasets_failed_default_jobs=[])
    raise InvalidInput('--plan-dir is mandatory for any frozen-oracle or confirmatory analysis')


def pair_deltas(ok, release, policy, algos):
    """delta (pp) per dataset x pair; present only if both algorithms have the complete frozen fold set."""
    sub = ok[(ok.release == release) & (ok.policy == policy)]
    if sub.empty:
        return pd.DataFrame()
    m = sub.pivot(index=['dataset', 'rep', 'fold'], columns='algo', values='ba')   # pivot raises on duplicates
    out = {}
    for a, b in itertools.combinations([x for x in algos if x in m.columns], 2):
        d = (m[a] - m[b])
        full = d.notna().groupby(level='dataset').all()
        out['%s-%s' % (a, b)] = (d.groupby(level='dataset').mean() * 100).where(full)
    return pd.DataFrame(out)


def shift_matrix(ok, old, new, policy_old, policy_new, algos=ALGOS):
    o, n = pair_deltas(ok, old, policy_old, algos), pair_deltas(ok, new, policy_new, algos)
    pairs = ['%s-%s' % (a, b) for a, b in itertools.combinations(algos, 2) if '%s-%s' % (a, b) in o.columns or '%s-%s' % (a, b) in n.columns]
    ds = sorted(set(o.index) | set(n.index))
    o = o.reindex(index=ds, columns=pairs); n = n.reindex(index=ds, columns=pairs)
    # identical paired folds are guaranteed by validate(): every job carries exactly the full frozen fold set
    return n - o, pairs, ds


def weights_for(ds, selection):
    if not selection:
        return None
    sel = json.loads(Path(selection).read_text())
    w = sel.get('stratum_weights_by_dataset')
    if not w:
        raise SystemExit('selection file has no stratum_weights_by_dataset (re-run stage_b.py select v2.2.1)')
    return np.array([w[d] for d in ds], float)


def cmd_edge(args):
    head = gate(args); df = pd.concat([pd.read_csv(p) for p in args.inp], ignore_index=True)
    ok = validate(df, args.reps, args.folds)
    combos = {(r, args.policy) for e in args.edges.split(',') for r in e.split(':')}
    coverage = coverage_or_exploratory(args, df, combos)
    ok = ok[~ok.dataset.astype(str).isin(coverage['excluded_datasets_failed_default_jobs'])]
    primary = args.primary.split(',') if args.primary else []
    edges = args.edges.split(',')
    if not set(primary) <= set(edges):
        raise SystemExit('primary edges must be among --edges')
    out = dict(header=head, population=args.population, plan_coverage=coverage, primary_family=primary, margins={})
    for m in DV.MARGINS:
        res = {}
        for e in edges:
            old, new = e.split(':')
            S, pairs, ds = shift_matrix(ok, old, new, args.policy, args.policy)
            r = DV.decide(S.to_numpy(), m, pair_names=pairs, population=args.population, weights=weights_for(ds, args.selection))
            r.update(edge=e, datasets=ds, evidence='emulation_only' if e in (args.emulation_only or '').split(',') else 'observed_cross_release')
            res[e] = r
        out['margins'][str(m)] = DV.label_edges(res, primary)
    return out


def coverage3(df, release):
    jobs = df[(df.release == release) & df.policy.isin(['P', 'PT'])].drop_duplicates(['dataset', 'algo', 'policy'])
    st = jobs.pivot(index=['dataset', 'algo'], columns='policy', values='status')
    p = st.get('P', pd.Series('missing', index=st.index)); pt = st.get('PT', pd.Series('missing', index=st.index))
    cls = np.where(p == 'ok', 'strictly_portable', np.where(pt == 'ok', 'translation_required', 'non_transferable'))
    c = pd.Series(cls, index=st.index, name='class')
    by = c.groupby(level='algo').value_counts().unstack(fill_value=0)
    return c, by.reset_index().to_dict('records'), c.value_counts().to_dict()


def cmd_mitigation(args):
    head = gate(args); df = pd.concat([pd.read_csv(p) for p in args.inp], ignore_index=True)
    ok = validate(df, args.reps, args.folds); old, new = args.edge.split(':')
    pinned = ['P'] + ([] if args.no_pt else ['PT'])
    coverage = coverage_or_exploratory(args, df, {(old, 'D'), (new, 'D')} | {(new, p) for p in pinned})
    ok = ok[~ok.dataset.astype(str).isin(coverage['excluded_datasets_failed_default_jobs'])]
    df = df[~df.dataset.astype(str).isin(coverage['excluded_datasets_failed_default_jobs'])]
    cls, by_algo, totals = coverage3(df, new)
    tot, pairs, ds = shift_matrix(ok, old, new, 'D', 'D')
    rng = np.random.default_rng(args.seed)
    severity = {'strictly_portable': 0, 'translation_required': 1, 'non_transferable': 2}
    def unit_class(policy_ok_class):
        # a pair unit belongs to the MORE SEVERE class of its two algorithms (e.g. LR portable + RF needing
        # translation -> translation_required); populations are disjoint and never pooled
        mask = pd.DataFrame(False, index=tot.index, columns=tot.columns)
        for d in tot.index:
            for p in pairs:
                a, b = p.split('-')
                ca, cb = cls.get((d, a), 'non_transferable'), cls.get((d, b), 'non_transferable')
                mask.loc[d, p] = max(severity[ca], severity[cb]) == severity[policy_ok_class]
        return mask
    def ci(x):
        if len(x) < 2: return [None, None]
        b = x[rng.integers(0, len(x), (4999, len(x)))].mean(1); return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    res = dict(header=head, edge=args.edge, plan_coverage=coverage, failure_to_transfer=dict(by_estimator=by_algo, totals=totals,
               note='strictly portable, translation-required and non-transferable configurations are never pooled'))
    Mk = lambda X: np.nanmean(np.abs(X.to_numpy()), axis=1)
    res['selection_check'] = dict(M_total_all_units=float(np.nanmean(Mk(tot))))
    for pol, klass in [('P', 'strictly_portable')] + ([] if args.no_pt else [('PT', 'translation_required')]):
        mask = unit_class(klass)
        pinned, _, _ = shift_matrix(ok, old, new, 'D', pol)
        pinned = pinned.reindex(index=tot.index, columns=tot.columns).where(mask)
        t_c = tot.where(mask)
        a, b = Mk(t_c), Mk(pinned); keep = np.isfinite(a) & np.isfinite(b)
        diff = a[keep] - b[keep]
        res['mitigation_' + pol] = dict(population=klass, pair_units=int(mask.to_numpy().sum()), datasets=int(keep.sum()),
                                        M_total_in_population=float(a[keep].mean()) if keep.any() else None,
                                        M_pinned=float(b[keep].mean()) if keep.any() else None,
                                        reduction_pp=float(diff.mean()) if keep.any() else None, reduction_ci95=ci(diff),
                                        scope='valid only for %s configurations' % klass.replace('_', ' '))
        res['selection_check']['M_total_' + klass] = float(a[keep].mean()) if keep.any() else None
    return res


def cmd_attribution(args):
    head = gate(args); rows = []; coverages = {}
    for spec in args.cell:
        cell, rest = spec.split('=', 1); path, rp = rest.split('@'); rel, pol = rp.split(':')
        raw = pd.read_csv(path)
        ok = validate(raw, args.reps, args.folds)
        cov = coverage_or_exploratory(args, raw, {(rel, pol)} | ({(rel, 'D')} if pol != 'D' else set()))
        coverages[cell] = cov
        ok = ok[~ok.dataset.astype(str).isin(cov['excluded_datasets_failed_default_jobs'])]
        d = pair_deltas(ok, rel, pol, ALGOS)
        for ds, r in d.iterrows():
            for pair, v in r.items():
                rows.append(dict(dataset=ds, pair=pair, cell=cell, theta=v if np.isfinite(v) else None))
    dec = AT.decompose_table(pd.DataFrame(rows))
    return dict(header=head, plan_coverage=coverages, identifiability_counts=dec.identifiability.value_counts().to_dict(), summary=AT.summarise(dec),
                component_labels=dict(defaults='default-configuration effect',
                                      runtime_stack='runtime-stack effect: scikit-learn implementation + numpy/scipy/BLAS/OpenMP + interpreter '
                                                    '(not attributable to scikit-learn alone unless the stack is held fixed)'))


def cmd_size(args):
    """Stage-C K = max(negative-edge precision, positive-edge power, prevalence floor 29, coverage floor 48)."""
    need = {'coverage_floor': 48, 'prevalence_floor': DV.min_k_zero_events()}
    def kfind(cond):
        k = 3
        while k < 5000 and not cond(k):
            k += 1
        return k
    for e in args.negative:
        r = json.loads(Path(e).read_text())['margins']['0.5']; d = list(r.values())[0]; sd = d['q_mean']['M_k_sd']
        need['negative_precision:' + d['edge']] = kfind(lambda k: stats.t.ppf(.975, k - 1) * sd / math.sqrt(k) <= args.half_width)
    for e in args.positive:
        r = json.loads(Path(e).read_text())['margins']['0.5']; d = list(r.values())[0]
        M, sd = d['q_mean']['M_pp'], d['q_mean']['M_k_sd']
        if M is None or M <= 0.5 or sd == 0:
            need['positive_power:' + d['edge']] = 'not expected to exceed the margin (M=%s); report precision instead' % M
            need['positive_precision:' + d['edge']] = kfind(lambda k: stats.t.ppf(.975, k - 1) * max(sd, 1e-9) / math.sqrt(k) <= max(abs(0.5 - (M or 0)), args.half_width))
        else:
            need['positive_power90:' + d['edge']] = kfind(lambda k: (M - 0.5) * math.sqrt(k) / sd >= stats.t.ppf(.95, k - 1) + stats.t.ppf(.90, k - 1))
        pv = d['q_prev']['primary_avg_pair']
        if pv and pv['estimate'] and pv['estimate'] > DV.PREV_LIMIT:
            pi = pv['estimate']
            need['positive_prevalence:' + d['edge']] = kfind(lambda k: DV.cp_lower(int(round(pi * k)), k) > DV.PREV_LIMIT)
    numeric = [v for v in need.values() if isinstance(v, int)]
    K = max(numeric)
    return dict(requirements=need, K_required=K, eligible=args.eligible, stage_b_used=args.stage_b_used,
                decision=('exclude Stage-B tasks' if K <= args.eligible - args.stage_b_used else
                          're-use Stage-B tasks (flagged) with a sensitivity analysis excluding them' if K <= args.eligible else
                          'eligible frame too small for K=%d: restrict confirmatory claims to K=%d precision' % (K, args.eligible)))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    def common(p):
        p.add_argument('--out', required=True); p.add_argument('--oracle-frozen'); p.add_argument('--unblinded-exploratory', action='store_true')
        p.add_argument('--reps', type=int, default=6); p.add_argument('--folds', type=int, default=5)
    e = sub.add_parser('edge'); common(e); e.add_argument('--inp', nargs='+', required=True); e.add_argument('--edges', required=True)
    e.add_argument('--primary', default=''); e.add_argument('--policy', default='D'); e.add_argument('--emulation-only', default='')
    e.add_argument('--population', default='superpopulation', choices=['superpopulation', 'frame_census', 'stratified'])
    e.add_argument('--selection', default=None, help='STAGE_B_SELECTION.json (required for --population stratified)')
    e.add_argument('--plan-dir', help='Frozen plan; mandatory unless explicitly unblinded-exploratory')
    m = sub.add_parser('mitigation'); common(m); m.add_argument('--inp', nargs='+', required=True); m.add_argument('--edge', required=True)
    m.add_argument('--seed', type=int, default=7); m.add_argument('--plan-dir'); m.add_argument('--selection')
    m.add_argument('--no-pt', action='store_true', help='declare that no PT arm was run (no translations file)')
    t = sub.add_parser('attribution'); common(t); t.add_argument('--cell', action='append', required=True)
    t.add_argument('--plan-dir'); t.add_argument('--selection')
    z = sub.add_parser('size'); z.add_argument('--out', required=True); z.add_argument('--negative', nargs='+', required=True)
    z.add_argument('--positive', nargs='+', required=True); z.add_argument('--eligible', type=int, required=True)
    z.add_argument('--stage-b-used', type=int, required=True); z.add_argument('--half-width', type=float, default=0.25)
    a = ap.parse_args()
    if a.cmd == 'edge' and a.population == 'stratified' and not a.selection:
        raise SystemExit('--population stratified requires --selection')
    rep = dict(edge=cmd_edge, mitigation=cmd_mitigation, attribution=cmd_attribution, size=cmd_size)[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(json.dumps(rep, default=str)[:1200])


if __name__ == '__main__':
    main()
