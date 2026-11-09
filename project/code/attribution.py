"""VersionShift v2.1 — counterfactual attribution over a 2x2 design (review P0-2).

Cells, for one unit (dataset x algorithm pair, or dataset x algorithm):

                    old runtime   new runtime
    old defaults        A             B         B = old release's resolved defaults, strictly pinned in the new runtime
    new defaults        C             D         C = new release's resolved defaults, strictly pinned in the old runtime

theta = the outcome in a cell (e.g. paired score difference delta, in pp).
Total change  T = D - A.

  runtime-first path : (B - A) runtime, then (D - B) defaults
  defaults-first path: (C - A) defaults, then (D - C) runtime
  Shapley (symmetric): phi_runtime_stack = ((B - A) + (D - C)) / 2   [scikit-learn + numpy/scipy/BLAS/interpreter]
                       phi_defaults = ((C - A) + (D - B)) / 2       phi_runtime + phi_defaults == T exactly
  interaction        : I = D - B - C + A  (the two paths differ by I; reported, never hidden)

Identifiability per unit:
  'full'           all four cells           -> Shapley shares + interaction
  'runtime_first'  A, B, D (C not portable) -> one ordered path; interaction NOT identifiable
  'defaults_first' A, C, D (B not portable) -> one ordered path; interaction NOT identifiable
  'total_only'     A, D                     -> no attribution; reported as non-identifiable
Shares are never computed from squared shifts (not additive); magnitudes are summarised
from the signed components afterwards.
"""
import numpy as np
import pandas as pd


def decompose_unit(cells):
    """cells: dict with keys among 'A','B','C','D' -> float (missing / None = not portable)."""
    g = {k: cells.get(k) for k in 'ABCD'}
    have = {k for k, v in g.items() if v is not None and np.isfinite(v)}
    if not {'A', 'D'} <= have:
        return dict(identifiability='no_total')
    T = g['D'] - g['A']
    out = dict(total=T)
    if have == set('ABCD'):
        out.update(identifiability='full',
                   runtime_stack=0.5 * ((g['B'] - g['A']) + (g['D'] - g['C'])),
                   defaults=0.5 * ((g['C'] - g['A']) + (g['D'] - g['B'])),
                   interaction=g['D'] - g['B'] - g['C'] + g['A'],
                   path_runtime_first=(g['B'] - g['A'], g['D'] - g['B']),
                   path_defaults_first=(g['C'] - g['A'], g['D'] - g['C']))
    elif 'B' in have:
        out.update(identifiability='runtime_first', runtime_stack=g['B'] - g['A'], defaults=g['D'] - g['B'])
    elif 'C' in have:
        out.update(identifiability='defaults_first', defaults=g['C'] - g['A'], runtime_stack=g['D'] - g['C'])
    else:
        out.update(identifiability='total_only')
    return out


def decompose_table(df, unit_cols=('dataset', 'pair'), cell_col='cell', value_col='theta'):
    rows = []
    for key, s in df.groupby(list(unit_cols)):
        cells = dict(zip(s[cell_col], s[value_col]))
        r = decompose_unit(cells)
        rows.append(dict(zip(unit_cols, key if isinstance(key, tuple) else (key,)), **{k: v for k, v in r.items() if not k.startswith('path')}))
    return pd.DataFrame(rows)


def summarise(dec, B=4000, seed=5):
    """Dataset-clustered bootstrap of mean components, separately per identifiability class.
    Shares are reported only for 'full' units: share_defaults = sum|phi_def| / (sum|phi_def| + sum|phi_rt|)."""
    rng = np.random.default_rng(seed)
    out = {}
    for ident, s in dec.groupby('identifiability'):
        res = dict(units=int(len(s)), datasets=int(s.dataset.nunique()))
        if ident in ('no_total', 'total_only'):
            res['note'] = 'non-identifiable: no numerical attribution reported'
            if 'total' in s:
                res['mean_total'] = float(s.total.mean())
            out[ident] = res; continue
        comps = [c for c in ('total', 'defaults', 'runtime_stack', 'interaction') if c in s and s[c].notna().any()]
        per = s.groupby('dataset')[comps].mean()
        per_abs = s.assign(**{c + '_abs': s[c].abs() for c in comps}).groupby('dataset')[[c + '_abs' for c in comps]].mean()
        ids = per.index.to_numpy(); idx = rng.integers(0, len(ids), (B, len(ids)))
        for c in comps:
            v = per[c].to_numpy(); va = per_abs[c + '_abs'].to_numpy()
            bs = v[idx].mean(1); bsa = va[idx].mean(1)
            res[c] = dict(mean=float(v.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                          mean_abs=float(va.mean()), mean_abs_ci95=[float(np.quantile(bsa, .025)), float(np.quantile(bsa, .975))])
        if ident == 'full':
            d, r = per_abs['defaults_abs'].to_numpy(), per_abs['runtime_stack_abs'].to_numpy()
            share = d[idx].sum(1) / np.maximum(d[idx].sum(1) + r[idx].sum(1), 1e-15)
            res['share_defaults_of_abs'] = dict(estimate=float(d.sum() / max(d.sum() + r.sum(), 1e-15)),
                                                ci95=[float(np.quantile(share, .025)), float(np.quantile(share, .975))])
            res['interaction_to_total_abs'] = float(s.interaction.abs().mean() / max(s.total.abs().mean(), 1e-15))
        else:
            res['note'] = 'single ordered path; order-dependent; interaction not identifiable'
        out[ident] = res
    return out
