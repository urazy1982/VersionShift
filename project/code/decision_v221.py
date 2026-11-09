"""VersionShift v2.2.1 decision rule (audit corrections to decision_v22.py, which is kept unchanged as the record).

Changes from v2.2
  * P0  explicit p-values per edge for both families (Q-MEAN magnitude and Q-PREV), so the analysis can apply
        Holm across the pre-declared primary edges (labels are assigned only after adjustment: analysis_v221).
  * P1  frozen pair-availability rule: PRIMARY = complete-case datasets (all P pairs present); coverage is reported;
        available-case (nanmean) and missingness bounds are sensitivity analyses, never the primary.
  * P1  prevalence population is explicit:
          'superpopulation'  exact Clopper-Pearson; assumes tasks are an exchangeable sample of a wider population
          'frame_census'     the whole eligible CC18 frame was analysed: descriptive finite-frame proportion, no
                             sampling interval (uncertainty is rerun/measurement only); never generalised beyond the frame
          'stratified'       Stage B design: stratum-weighted estimate with Kish effective n (Korn-Graubard-style CP)
  * new pi_max = P_k( max_p |Delta_kp| > t_max ) as a pre-specified SENSITIVITY outcome (one strongly changed pair).
        t_max = 1.0 pp, not 0.5: the pre-data check (rule_check_v221.py) showed that a maximum over 15 pairs at 0.5 pp
        is triggered by rerun noise alone (12.9% of datasets under realistic noise, 35% under a variance-only
        change), above the 10% limit; at 1.0 pp the null rates are 1.8% and 6.0%, while a single +4 pp pair is
        still detected in 100% of datasets (the average-pair measure detects 1.3%). pi_max at 0.5 pp is
        reported descriptively only.

Unit: dataset k x ordered pair p (frozen order LR, SVC, kNN, RF, GB, HGB) x edge e; Delta in pp.
"""
import numpy as np
from scipy import stats

MARGINS = (0.25, 0.5, 1.0)
PREV_LIMIT = 0.10


def cp_upper(x, n, alpha=0.05):
    return 1.0 if x >= n else float(stats.beta.ppf(1 - alpha, x + 1, n - x))


def cp_lower(x, n, alpha=0.05):
    return 0.0 if x <= 0 else float(stats.beta.ppf(alpha, x, n - x + 1))


def min_k_zero_events(limit=PREV_LIMIT, alpha=0.05):
    k = 1
    while cp_upper(0, k, alpha) >= limit:
        k += 1
    return k


def t_bounds(x, alpha):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2 or np.all(x == x[0]):
        return m, m
    q = stats.t.ppf(1 - alpha, n - 1); se = x.std(ddof=1) / np.sqrt(n)
    return m - q * se, m + q * se


def p_mean_exceeds(x, margin):
    """One-sided t test, H0: E[x] <= margin."""
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return 1.0
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0 if x.mean() > margin else 1.0
    return float(stats.t.sf((x.mean() - margin) / (sd / np.sqrt(n)), n - 1))


def p_prev_exceeds(x, n, limit=PREV_LIMIT):
    """Exact binomial test, H0: pi <= limit."""
    return float(stats.binom.sf(x - 1, n, limit)) if x > 0 else 1.0


def prevalence(affected, population, weights=None, alpha=0.05):
    a = np.asarray(affected, bool); K = len(a); x = int(a.sum())
    if population == 'superpopulation':
        return dict(population=population, estimate=x / K if K else None, affected=x, K=K,
                    lower=cp_lower(x, K, alpha), upper=cp_upper(x, K, alpha), p_exceeds_limit=p_prev_exceeds(x, K),
                    interpretation='assumes tasks are an exchangeable sample of a wider task population')
    if population == 'frame_census':
        return dict(population=population, estimate=x / K if K else None, affected=x, K=K, lower=None, upper=None,
                    p_exceeds_limit=None,
                    interpretation='descriptive proportion of the finite eligible frame; no sampling interval; not generalised')
    if population == 'stratified':
        w = np.asarray(weights, float); w = w / w.sum()
        est = float((w * a).sum()); n_eff = 1.0 / float((w ** 2).sum())
        xe = est * n_eff
        lo = 0.0 if xe <= 0 else float(stats.beta.ppf(alpha, xe, n_eff - xe + 1))
        hi = 1.0 if xe >= n_eff else float(stats.beta.ppf(1 - alpha, xe + 1, n_eff - xe))
        return dict(population=population, estimate=est, affected=x, K=K, n_effective=n_eff, lower=lo, upper=hi,
                    p_exceeds_limit=None,
                    interpretation='design-weighted (selection strata) estimate for the eligible frame; Kish effective n')
    raise ValueError('unknown population')


def decide(pair_shift, margin=0.5, alpha=0.05, pair_names=None, population='superpopulation', weights=None, max_threshold=1.0):
    """pair_shift (K x P), NaN = pair unavailable. Returns quantities + UNADJUSTED p-values; final labels for
    CHANGED_MEAN / WIDESPREAD are set by the caller after Holm (primary edges) or BH (secondary edges)."""
    D = np.asarray(pair_shift, float); K, P = D.shape
    complete = np.isfinite(D).all(axis=1)
    Dc = D[complete]; Kc = int(complete.sum())
    w_c = None if weights is None else np.asarray(weights, float)[complete]
    absD = np.abs(Dc)
    Mk = absD.mean(axis=1) if Kc else np.array([])
    maxk = absD.max(axis=1) if Kc else np.array([])
    lo1, hi1 = t_bounds(Mk, alpha) if Kc else (np.nan, np.nan)
    pairs = []
    for j in range(P):
        col = Dc[:, j]
        lo, hi = t_bounds(col, alpha) if Kc else (np.nan, np.nan)
        pairs.append(dict(pair=pair_names[j] if pair_names else j, n=Kc, mean_signed_pp=float(col.mean()) if Kc else None,
                          ci90=[float(lo), float(hi)], tost_equivalent=bool(Kc > 1 and lo > -margin and hi < margin)))
    prev_avg = prevalence(Mk > margin, population, w_c, alpha) if Kc else None
    prev_max = prevalence(maxk > max(margin, max_threshold), population, w_c, alpha) if Kc else None
    prev_max_at_margin = float(np.mean(maxk > margin)) if Kc else None
    # sensitivity: available-case and missingness bounds (missing |Delta| -> 0 or -> counted as affected)
    Mk_av = np.nanmean(np.abs(D), axis=1)
    any_missing = ~complete
    sens = dict(available_case_M_pp=float(np.nanmean(Mk_av)) if K else None,
                prevalence_bounds_missing_as_unaffected=float(np.mean(np.where(complete, np.nan_to_num(Mk_av) > margin, False))) if K else None,
                prevalence_bounds_missing_as_affected=float(np.mean(np.where(complete, np.nan_to_num(Mk_av) > margin, True))) if K else None)
    equiv_prev_ok = prev_avg is not None and prev_avg['upper'] is not None and prev_avg['upper'] < PREV_LIMIT
    if population == 'frame_census' and prev_avg is not None:
        equiv_prev_ok = prev_avg['estimate'] < PREV_LIMIT
    all_pairs_equiv = Kc > 1 and all(p['tost_equivalent'] for p in pairs)
    return dict(margin_pp=margin, K=int(K), K_complete_case=Kc, complete_case_coverage=Kc / K if K else 0.0,
                pair_availability={(pair_names[j] if pair_names else j): int(np.isfinite(D[:, j]).sum()) for j in range(P)},
                q_mean=dict(M_pp=float(Mk.mean()) if Kc else None, M_k_sd=float(Mk.std(ddof=1)) if Kc > 1 else 0.0,
                            M_one_sided_lower95=float(lo1), M_one_sided_upper95=float(hi1),
                            p_changed_unadjusted=p_mean_exceeds(Mk, margin) if Kc else 1.0,
                            pairs_all_tost_equivalent=bool(all_pairs_equiv), per_pair=pairs),
                q_prev=dict(primary_avg_pair=prev_avg, sensitivity_any_pair_pi_max=dict(prev_max or {}, threshold_pp=max(margin, max_threshold)),
                            descriptive_pi_max_at_margin=prev_max_at_margin, limit=PREV_LIMIT),
                equivalence_inputs=dict(all_pairs_tost=bool(all_pairs_equiv), prevalence_ok=bool(equiv_prev_ok),
                                        M_upper_below_margin=bool(Kc > 1 and hi1 < margin)),
                missingness_sensitivity=sens)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 1.0
    for i in range(m - 1, -1, -1):
        j = order[i]; run = min(run, p[j] * m / (i + 1)); adj[j] = run
    return adj


def label_edges(results, primary, alpha=0.05):
    """results: {edge: decide(...)}; primary: list of edges in the pre-declared primary family.
    Holm within each question family (Q-MEAN, Q-PREV) across primary edges; BH across secondary edges.
    Writes adjusted p-values and the final labels into each result."""
    for fam_key, getp in (('mean', lambda r: r['q_mean']['p_changed_unadjusted']),
                          ('prev', lambda r: (r['q_prev']['primary_avg_pair'] or {}).get('p_exceeds_limit'))):
        for group, adjust in ((list(primary), holm), ([e for e in results if e not in primary], bh)):
            ps = [getp(results[e]) for e in group]
            valid = [i for i, p in enumerate(ps) if p is not None]
            adj = adjust([ps[i] for i in valid]) if valid else []
            for i, a in zip(valid, adj):
                results[group[i]].setdefault('adjusted', {})[fam_key] = dict(
                    p_adjusted=float(a), method='holm' if adjust is holm else 'bh', family='primary' if adjust is holm else 'secondary')
    for e, r in results.items():
        adj = r.get('adjusted', {}); labels = []
        if adj.get('mean', {}).get('p_adjusted', 1.0) < alpha:
            labels.append('CHANGED_MEAN')
        pv = r['q_prev']['primary_avg_pair']
        if pv and pv['population'] == 'frame_census':
            if pv['estimate'] is not None and pv['estimate'] > PREV_LIMIT:
                labels.append('WIDESPREAD_IN_FRAME')
        elif adj.get('prev', {}).get('p_adjusted', 1.0) < alpha:
            labels.append('WIDESPREAD')
        ei = r['equivalence_inputs']
        if not labels and ei['all_pairs_tost'] and ei['prevalence_ok']:
            labels.append('UNIFORMLY_SMALL' if ei['M_upper_below_margin'] else 'EQUIVALENT_BALANCED')
        if not labels:
            labels.append('INCONCLUSIVE')
        r['labels'] = labels; r['family'] = 'primary' if e in primary else 'secondary'
        pm = r['q_prev']['sensitivity_any_pair_pi_max']; flags = []
        if pm.get('population') == 'frame_census':
            if pm.get('estimate') is not None and pm['estimate'] > PREV_LIMIT:
                flags.append('ANY_PAIR_WIDESPREAD_IN_FRAME')
        elif pm.get('lower') is not None and pm['lower'] > PREV_LIMIT:
            flags.append('ANY_PAIR_WIDESPREAD')
        r['sensitivity_flags'] = flags          # pre-specified sensitivity; never changes the primary labels
    return results
