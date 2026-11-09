"""Stage B helpers (pre-registered rules, written before any Stage B outcome).

  select  seeded, stratified choice of Stage-B tasks from the frozen plan's eligible datasets:
          strata = {binary, multiclass} x {n below / above the eligible median}; equal allocation
  size    Stage-C dataset count from Stage-B heterogeneity on the NEGATIVE-CONTROL edge (1.8.0->1.9.1),
          where an equivalence claim needs precision: K = ceil((t_{0.975,K-1} * SD(M_k) / h)^2), h = 0.25 pp.
          (On a positive edge only the lower bound vs the margin matters, so it does not drive K.)
          floor 48 (coverage under sparse effects, v2.1 calibration) and floor 29 (prevalence bound),
          capped at the eligible count not used in Stage B (else Stage-B tasks are re-used and flagged)
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy import stats


def select(args):
    plan = json.loads((Path(args.plan_dir) / 'PLAN.json').read_text())
    ds = plan['datasets']; med = float(np.median([d['n'] for d in ds]))
    strata = {}
    for d in ds:
        strata.setdefault(('binary' if d['k'] == 2 else 'multiclass', 'small' if d['n'] <= med else 'large'), []).append(d['name'])
    if args.n > len(ds) or args.n <= 0:
        raise SystemExit('selection size must be between 1 and the eligible dataset count')
    rng = np.random.default_rng(args.seed); per = args.n // len(strata); chosen = []
    for key in sorted(strata):
        names = sorted(strata[key]); take = min(per, len(names))
        chosen += sorted(rng.choice(names, take, replace=False).tolist())
    # After equal-stratum allocation, fill remaining slots by explicit known inclusion strata:
    # don't pretend that a pooled remainder still has the original equal-probability design.
    if len(chosen) != args.n:
        raise SystemExit('equal-stratum sample cannot allocate %d tasks: selected %d; '
                         'change the pre-registered allocation, not silently fill from other strata' % (args.n, len(chosen)))
    stratum_of = {name: '%s/%s' % key for key, names in strata.items() for name in names}
    n_h = {}
    for c in chosen:
        n_h[stratum_of[c]] = n_h.get(stratum_of[c], 0) + 1
    N_h = {'%s/%s' % k: len(v) for k, v in strata.items()}
    weights = {c: N_h[stratum_of[c]] / n_h[stratum_of[c]] for c in chosen}   # inverse inclusion probability
    out = dict(rule=__doc__.split('\n')[3].strip(), seed=args.seed, n=len(chosen), median_n=med,
               stratum_of_dataset={c: stratum_of[c] for c in sorted(chosen)}, stratum_weights_by_dataset=weights,
               weights_note='inverse inclusion probabilities within selection strata; use analysis_v221 --population stratified',
               strata={'%s/%s' % k: len(v) for k, v in sorted(strata.items())}, stage_b_datasets=sorted(chosen),
               plan_sha256=hashlib.sha256((Path(args.plan_dir) / 'PLAN.json').read_bytes()).hexdigest())
    Path(args.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def size(args):
    rep = json.loads(Path(args.stage_b_edge).read_text())
    e = rep['edges'][0]; d = e['decisions']['0.5']
    K = d['K']; sd = d['q_mean']['M_k_sd']
    k = 2
    while k < 10000 and stats.t.ppf(0.975, k - 1) * sd / math.sqrt(k) > args.half_width:
        k += 1
    need = max(k, 48, 29)
    out = dict(stage_b_K=K, sd_M_k_pp=sd, target_half_width_pp=args.half_width, k_for_precision=k, k_required=need,
               eligible_total=args.eligible, stage_b_used=args.stage_b_used,
               feasible_without_reuse=need <= args.eligible - args.stage_b_used,
               decision=('exclude Stage-B tasks from the confirmatory frame' if need <= args.eligible - args.stage_b_used else
                         're-use Stage-B tasks (flagged) and report a sensitivity analysis excluding them' if need <= args.eligible else
                         'eligible frame too small: confirmatory claims limited to precision achievable at K=%d' % args.eligible))
    Path(args.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select'); s.add_argument('--plan-dir', required=True); s.add_argument('--n', type=int, default=16)
    s.add_argument('--seed', type=int, default=20261106); s.add_argument('--out', required=True)
    z = sub.add_parser('size'); z.add_argument('--stage-b-edge', required=True); z.add_argument('--eligible', type=int, required=True)
    z.add_argument('--stage-b-used', type=int, required=True); z.add_argument('--half-width', type=float, default=0.25); z.add_argument('--out', required=True)
    a = ap.parse_args(); dict(select=select, size=size)[a.cmd](a)


if __name__ == '__main__':
    main()
