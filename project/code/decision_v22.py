"""VersionShift v2.2 — frozen decision rule (reviewer's second-round decisions, with one correction).

Unit: dataset k, ordered algorithm pair p = (A, B) with A before B in the frozen algorithm list,
release edge e. Upgrade-induced change in the paired comparison (pp):

    Delta_kpe = (s_A,new - s_B,new) - (s_A,old - s_B,old)

Two different scientific questions are answered separately (review round 2):

  Q-MEAN  Does the average algorithm comparison change?
          (a) direction : mean signed shift per pair, mu_p = E_k[Delta_kp]
          (b) magnitude : M = E_k[ mean_p |Delta_kp| ]            (no cancellation)
  Q-PREV  Does the upgrade change a substantial FRACTION of datasets?
          pi = P_k( M_k > m ),  M_k = mean_p |Delta_kp|, with an exact Clopper-Pearson bound

Correction to the review's wording: a "mean signed effect" pooled over pairs is meaningless,
because pair orientation is arbitrary (A-B vs B-A flips the sign). Direction is therefore
tested PER PAIR, and equivalence requires every pair (intersection-union test: no
multiplicity penalty is needed, because all must pass).

Decision labels per edge (margin m = 0.5 pp primary; 0.25 and 1.0 pp sensitivity):
  CHANGED_MEAN          one-sided 95% lower bound of M > m               (primary; Holm across primary edges)
  WIDESPREAD            Clopper-Pearson one-sided 95% lower bound of pi > 10%
  EQUIVALENT_BALANCED   every pair's 90% CI for mu_p inside (-m, +m) [TOST]
                        AND one-sided 95% upper bound of pi < 10%
  UNIFORMLY_SMALL       EQUIVALENT_BALANCED AND one-sided 95% upper bound of M < m
  INCONCLUSIVE          none of the above
CHANGED_MEAN and WIDESPREAD can both hold; neither is compatible with the EQUIVALENT_* labels.
"""
import numpy as np
from scipy import stats

MARGINS = (0.25, 0.5, 1.0)
PREV_LIMIT = 0.10


def clopper_pearson_upper(x, n, alpha=0.05):
    return 1.0 if x >= n else float(stats.beta.ppf(1 - alpha, x + 1, n - x))


def clopper_pearson_lower(x, n, alpha=0.05):
    return 0.0 if x <= 0 else float(stats.beta.ppf(alpha, x, n - x + 1))


def min_k_zero_events(limit=PREV_LIMIT, alpha=0.05):
    """Smallest K with zero affected datasets whose exact upper bound is below `limit`."""
    k = 1
    while clopper_pearson_upper(0, k, alpha) >= limit:
        k += 1
    return k


def t_bounds(x, alpha):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2 or np.all(x == x[0]):
        return m, m
    q = stats.t.ppf(1 - alpha, n - 1); se = x.std(ddof=1) / np.sqrt(n)
    return m - q * se, m + q * se


def decide(pair_shift, margin=0.5, alpha=0.05, pair_names=None):
    """pair_shift: (K datasets, P pairs) array of Delta in pp (NaN = pair not available for that dataset)."""
    D = np.asarray(pair_shift, float); K, P = D.shape
    Mk = np.nanmean(np.abs(D), axis=1)
    lo1, hi1 = t_bounds(Mk, alpha)                      # one-sided 95% bounds for M (each side)
    affected = int(np.sum(Mk > margin))
    pi_lo, pi_hi = clopper_pearson_lower(affected, K, alpha), clopper_pearson_upper(affected, K, alpha)
    pairs = []
    for j in range(P):
        col = D[:, j][np.isfinite(D[:, j])]
        if len(col) == 0:
            pairs.append(dict(pair=pair_names[j] if pair_names else j, n=0, tost_equivalent=False)); continue
        lo, hi = t_bounds(col, alpha)                   # 90% two-sided = TOST at alpha 0.05
        pairs.append(dict(pair=pair_names[j] if pair_names else j, n=int(len(col)), mean_signed_pp=float(col.mean()),
                          ci90=[float(lo), float(hi)], tost_equivalent=bool(lo > -margin and hi < margin)))
    all_pairs_equiv = all(p['tost_equivalent'] for p in pairs)
    labels = []
    if lo1 > margin:
        labels.append('CHANGED_MEAN')
    if pi_lo > PREV_LIMIT:
        labels.append('WIDESPREAD')
    if not labels and all_pairs_equiv and pi_hi < PREV_LIMIT:
        labels.append('UNIFORMLY_SMALL' if hi1 < margin else 'EQUIVALENT_BALANCED')
    if not labels:
        labels.append('INCONCLUSIVE')
    return dict(margin_pp=margin, K=int(K), labels=labels,
                q_mean=dict(M_pp=float(Mk.mean()), M_k_sd=float(Mk.std(ddof=1)) if K > 1 else 0.0, M_one_sided_lower95=float(lo1), M_one_sided_upper95=float(hi1),
                            pairs_all_tost_equivalent=all_pairs_equiv, per_pair=pairs),
                q_prev=dict(affected=affected, pi=affected / K, cp_one_sided_lower95=pi_lo, cp_one_sided_upper95=pi_hi,
                            limit=PREV_LIMIT))


def decide_all_margins(pair_shift, pair_names=None):
    return {str(m): decide(pair_shift, m, pair_names=pair_names) for m in MARGINS}
