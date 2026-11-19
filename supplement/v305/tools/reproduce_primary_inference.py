#!/usr/bin/env python3
"""Recompute the registered primary family from the published 48-task modern aggregates.

Recomputes, for E_LINEAR_DUAL and E_ADA_SAMME (evaluation frame, P1, D0, eight estimators):
M = mean task-level M_t, the one-sided t test of H0: E[M_t] <= 0.5 pp, its one-sided 95% lower
bound, the Holm adjustment across the two edges and the 95% task-bootstrap interval
(20,000 resamples, numpy RandomState seed 20261116, as in analysis_code/versionshift_v3/statistics.py).
Results are compared with results/primary_two_edges.csv. Requires numpy and scipy; no model fitting.
"""
import csv, math, pathlib, sys
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARGIN, REPS, SEED = 0.5, 20000, 20261116

def rows(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    task = rows(ROOT / 'results/secondary/modern_D0_task_level_conclusions.csv')
    reported = {r['edge_id']: r for r in rows(ROOT / 'results/primary_two_edges.csv')}
    out = {}
    for edge in ('E_LINEAR_DUAL', 'E_ADA_SAMME'):
        v = np.array([float(r['M_pp']) for r in task
                      if r['edge'] == edge and r['frame'] == 'evaluation' and r['preprocessing'] == 'P1'])
        n = len(v); assert n == 48, (edge, n)
        m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
        p = float(1 - stats.t.cdf((m - MARGIN) / se, n - 1))
        lower = float(m - stats.t.ppf(0.95, n - 1) * se)
        draws = v[np.random.RandomState(SEED).randint(0, n, size=(REPS, n))].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[edge] = dict(M=m, p=p, lower=lower, lo=lo, hi=hi)
    ps = sorted(out, key=lambda e: out[e]['p'])
    out[ps[0]]['holm'] = min(1.0, 2 * out[ps[0]]['p'])
    out[ps[1]]['holm'] = min(1.0, max(out[ps[0]]['holm'], out[ps[1]]['p']))
    ok = True
    for edge, r in out.items():
        rep = reported[edge]
        pairs = [('M', 'M_pp'), ('p', 'p_one_sided'), ('holm', 'p_holm'), ('lower', 'lower_one_sided_pp'),
                 ('lo', 'bootstrap95_low_pp'), ('hi', 'bootstrap95_high_pp')]
        for k, col in pairs:
            same = math.isclose(r[k], float(rep[col]), rel_tol=1e-9, abs_tol=1e-12)
            ok &= same
        print(f"{edge}: M={r['M']:.6f} pp  95% CI {r['lo']:.6f}-{r['hi']:.6f}  lower={r['lower']:.6f}  "
              f"p={r['p']:.3g}  Holm p={r['holm']:.3g}  {'MATCH' if all(math.isclose(r[k], float(rep[c]), rel_tol=1e-9, abs_tol=1e-12) for k, c in pairs) else 'MISMATCH'}")
    print('PRIMARY_INFERENCE_REPRODUCED' if ok else 'PRIMARY_INFERENCE_MISMATCH')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
