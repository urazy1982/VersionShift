"""VersionShift v2.2 analysis of real cross-release runs (frozen for Stage B/C).

Subcommands
  edge         decision rule (Q-MEAN, Q-PREV) for one or more edges under policy D, all margins
  mitigation   strict-pin coverage, mitigation CONDITIONAL on compatibility, selection check,
               and failure-to-transfer as an outcome (protocol amendment A-SM)
  attribution  2x2 cells from (run CSV, release, policy) specs -> Shapley + interaction + identifiability

Blinding gate (protocol amendment A-BL): every subcommand that reveals cross-release outcomes
requires --oracle-frozen pointing at ORACLE_FROZEN.json (complete 78-case frame, hash recorded)
unless --unblinded-exploratory is given; that flag is written into the report header.
Edges built from emulation only must be passed with --emulation-only and are labelled so.
"""
import argparse, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_v22 import decide_all_margins, decide
import attribution as AT

ALGOS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']


def gate(args):
    if getattr(args, 'unblinded_exploratory', False):
        return dict(blinding='UNBLINDED_EXPLORATORY', note='outcomes revealed before the oracle frame was frozen')
    if not args.oracle_frozen:
        raise SystemExit('blinding gate: pass --oracle-frozen ORACLE_FROZEN.json (or --unblinded-exploratory, recorded in the report)')
    fz = json.loads(Path(args.oracle_frozen).read_text())
    frame = Path(fz['frame_path'])
    if hashlib.sha256(frame.read_bytes()).hexdigest() != fz['frame_sha256'] or fz.get('status') != 'FROZEN_COMPLETE':
        raise SystemExit('blinding gate: oracle frame missing, changed since freezing, or incomplete')
    return dict(blinding='ORACLE_FROZEN_BEFORE_OUTCOMES', frame_sha256=fz['frame_sha256'], frozen_at=fz['frozen_at'])


def load(paths):
    return pd.concat([pd.read_csv(p).assign(source=str(p)) for p in paths], ignore_index=True)


def pair_deltas(df, release, policy, algos):
    """delta (pp) per dataset x ordered pair for one (release, policy): mean over all folds of s_A - s_B.
    A pair is present for a dataset only if both algorithms completed every fold."""
    sub = df[(df.release == release) & (df.policy == policy) & (df.status == 'ok')]
    m = sub.pivot_table(index=['dataset', 'rep', 'fold'], columns='algo', values='ba', aggfunc='first')
    out = {}
    for a, b in itertools.combinations([x for x in algos if x in m.columns], 2):
        d = (m[a] - m[b]).groupby(level='dataset')
        ok = d.apply(lambda s: s.notna().all() and len(s) > 0)
        out['%s-%s' % (a, b)] = (d.mean() * 100).where(ok)
    return pd.DataFrame(out)


def shift_matrix(old_d, new_d):
    pairs = sorted(set(old_d.columns) & set(new_d.columns), key=lambda p: (ALGOS.index(p.split('-')[0]), ALGOS.index(p.split('-')[1])))
    ds = sorted(set(old_d.index) & set(new_d.index))
    return (new_d.loc[ds, pairs] - old_d.loc[ds, pairs]), pairs, ds


def cmd_edge(args):
    head = gate(args); df = load(args.inp); rep = dict(header=head, edges=[])
    for e in args.edges.split(','):
        old, new = e.split(':')
        S, pairs, ds = shift_matrix(pair_deltas(df, old, args.policy, ALGOS), pair_deltas(df, new, args.policy, ALGOS))
        rep['edges'].append(dict(edge=e, evidence='emulation_only' if args.emulation_only else 'observed_cross_release',
                                 policy=args.policy, datasets=ds, decisions=decide_all_margins(S.to_numpy(), pairs)))
    return rep


def coverage(df, release, policy='P'):
    jobs = df[(df.release == release) & (df.policy == policy)].drop_duplicates(['dataset', 'algo'])
    by = jobs.groupby('algo').apply(lambda s: pd.Series(dict(
        planned=len(s), portable=int((s.status == 'ok').sum()), not_portable=int((s.status == 'not_portable').sum()),
        fit_failed=int((s.status == 'fit_failed').sum()),
        statuses=s.portability.fillna('none').value_counts().to_dict())), include_groups=False)
    return by.reset_index().to_dict('records'), float((jobs.status == 'ok').mean()) if len(jobs) else None


def cmd_mitigation(args):
    """Edge old->new. Total upgrade effect: D@old -> D@new. Pinned effect: D@old -> P@new (P pinned from old).
    Mitigation is estimated ONLY on (dataset, pair) units where both algorithms were portable."""
    head = gate(args); df = load(args.inp); old, new = args.edge.split(':')
    cov, overall = coverage(df, new, 'P')
    dD_old = pair_deltas(df, old, 'D', ALGOS); dD_new = pair_deltas(df, new, 'D', ALGOS); dP_new = pair_deltas(df, new, 'P', ALGOS)
    tot_all, pairs, ds = shift_matrix(dD_old, dD_new)
    pin, _, _ = shift_matrix(dD_old, dP_new.reindex(index=dD_old.index, columns=dD_old.columns))
    pin = pin.reindex(index=tot_all.index, columns=tot_all.columns)
    compatible = pin.notna()
    tot_c = tot_all.where(compatible)
    rng = np.random.default_rng(args.seed)
    Mk = lambda X: np.nanmean(np.abs(X.to_numpy()), axis=1)
    total_all_k, total_c_k, pinned_k = Mk(tot_all), Mk(tot_c), Mk(pin)
    keep = np.isfinite(total_c_k) & np.isfinite(pinned_k)
    diff = total_c_k[keep] - pinned_k[keep]
    def ci(x):
        if len(x) < 2: return [None, None]
        b = x[rng.integers(0, len(x), (4999, len(x)))].mean(1); return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    rep = dict(header=head, edge=args.edge, evidence='emulation_only' if args.emulation_only else 'observed_cross_release',
               failure_to_transfer=dict(by_estimator=cov, overall_portable_share=overall,
                                        note='reported as a practical outcome in its own right'),
               compatibility_coverage=dict(pair_units=int(tot_all.notna().sum().sum()), compatible_pair_units=int(compatible.sum().sum()),
                                           datasets_with_any_compatible_pair=int(keep.sum()), datasets=len(ds)),
               selection_check=dict(M_total_all_units=float(np.nanmean(total_all_k)), M_total_compatible_units=float(np.nanmean(total_c_k[keep])) if keep.any() else None,
                                    note='a gap between these two shows that the compatible subset is not representative'),
               mitigation_conditional_on_compatibility=dict(M_total=float(total_c_k[keep].mean()) if keep.any() else None,
                                                            M_pinned=float(pinned_k[keep].mean()) if keep.any() else None,
                                                            reduction_pp=float(diff.mean()) if len(diff) else None, reduction_ci95=ci(diff),
                                                            scope='valid only for portable configurations; not an estimate for all upgrades'),
               pinned_decision=decide(pin.to_numpy(), 0.5, pair_names=pairs) if keep.any() else None)
    return rep


def cmd_attribution(args):
    """--cell A=CSV@release:policy etc. A = old defaults / old runtime, B = old defaults / new runtime,
    C = new defaults / old runtime, D = new defaults / new runtime."""
    head = gate(args); rows = []
    for spec in args.cell:
        cell, rest = spec.split('=', 1); path, rp = rest.split('@'); rel, pol = rp.split(':')
        d = pair_deltas(pd.read_csv(path), rel, pol, ALGOS)
        for ds, r in d.iterrows():
            for pair, v in r.items():
                rows.append(dict(dataset=ds, pair=pair, cell=cell, theta=v if np.isfinite(v) else None))
    t = pd.DataFrame(rows)
    dec = AT.decompose_table(t)
    return dict(header=head, evidence='emulation_only' if args.emulation_only else 'observed_cross_release',
                identifiability_counts=dec.identifiability.value_counts().to_dict(), summary=AT.summarise(dec))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    def common(p):
        p.add_argument('--out', required=True); p.add_argument('--oracle-frozen'); p.add_argument('--unblinded-exploratory', action='store_true')
        p.add_argument('--emulation-only', action='store_true')
    e = sub.add_parser('edge'); common(e); e.add_argument('--inp', nargs='+', required=True); e.add_argument('--edges', required=True); e.add_argument('--policy', default='D')
    m = sub.add_parser('mitigation'); common(m); m.add_argument('--inp', nargs='+', required=True); m.add_argument('--edge', required=True); m.add_argument('--seed', type=int, default=7)
    t = sub.add_parser('attribution'); common(t); t.add_argument('--cell', action='append', required=True)
    a = ap.parse_args()
    rep = dict(edge=cmd_edge, mitigation=cmd_mitigation, attribution=cmd_attribution)[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(json.dumps(rep, default=str)[:1500])


if __name__ == '__main__':
    main()
