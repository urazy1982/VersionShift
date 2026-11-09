"""VersionShift v2.1 estimand hierarchy (review P0-3, P1-5, section 5). One numpy core,
used both by the real-data analysis and by the null-calibration simulation.

Input array S: shape (K datasets, 2 versions [old,new], A algorithms, R reruns, F folds),
balanced accuracy in [0,1]; versions are evaluated on EXACTLY paired splits (and seeds).

PRIMARY   M_k  = mean over pairs |delta_new - delta_old| (pp), delta = mean paired score
                 difference over all R*F folds. No noise denominator. Inference: dataset-
                 clustered percentile bootstrap; confirmatory test H0: E[M] <= margin_pp.
CO-PRIMARY E_k = decision-change excess: P(decision differs old-half vs new-other-half)
                 - P(decision differs between the two halves of the same version), with the
                 corrected repeated-CV t test (Nadeau-Bengio) as the per-dataset decision rule.
SECONDARY  VNR (|shift| / rerun SD), S_k log-ratio, signed shift; diagnostics only.
"""
import itertools
import numpy as np
from scipy import stats


def pair_index(A):
    p = list(itertools.combinations(range(A), 2))
    return np.array([a for a, _ in p]), np.array([b for _, b in p])


def corrected_states(d, folds_per_rep, alpha=0.05):
    """d: (..., n) per-fold differences -> state in {-1, 0, +1} (corrected repeated-CV t)."""
    n = d.shape[-1]; m = d.mean(-1); v = d.var(-1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = m / np.sqrt((1.0 / n + 1.0 / (folds_per_rep - 1)) * v)
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    sig = np.where(v > 0, p < alpha, m != 0)
    return np.where(sig, np.sign(m), 0).astype(int)


def dataset_level(S):
    K, V, A, R, F = S.shape
    ia, ib = pair_index(A)
    d = S[:, :, ia] - S[:, :, ib]                                   # (K,2,P,R,F)
    delta = d.mean((3, 4))                                          # (K,2,P)
    shift = (delta[:, 1] - delta[:, 0]) * 100                       # pp, (K,P)
    M = np.abs(shift).mean(1)                                       # primary
    signed = shift.mean(1)
    h = R // 2
    halves = [d[:, :, :, :h].reshape(K, V, len(ia), -1), d[:, :, :, h:2 * h].reshape(K, V, len(ia), -1)]
    st = [corrected_states(x, F) for x in halves]                   # each (K,2,P)
    cross = 0.5 * ((st[0][:, 0] != st[1][:, 1]).astype(float) + (st[1][:, 0] != st[0][:, 1]).astype(float))
    within = 0.5 * ((st[0][:, 0] != st[1][:, 0]).astype(float) + (st[0][:, 1] != st[1][:, 1]).astype(float))
    E = (cross - within).mean(1)                                    # co-primary
    per_rep = d.mean(4)                                             # (K,2,P,R)
    sd = per_rep[:, 1].std(-1, ddof=1) * 100
    with np.errstate(divide='ignore', invalid='ignore'):
        vnr = np.where(sd > 0, np.abs(shift) / sd, np.where(np.abs(shift) > 0, np.inf, 0.0))
    VNR = np.median(vnr, 1)
    if R >= 6:
        blk = np.stack([d[:, :, :, 2 * i:2 * i + 2].mean((3, 4)) for i in range(3)], -1)   # (K,2,P,3)
        cr = (blk[:, 1, :, 1] - blk[:, 0, :, 0]) ** 2; nu = (blk[:, 0, :, 1] - blk[:, 0, :, 2]) ** 2
        Sk = np.log(cr.sum(1) + 1e-18) - np.log(nu.sum(1) + 1e-18)
    else:
        Sk = np.full(K, np.nan)
    return dict(M=M, E=E, signed=signed, VNR=VNR, S=Sk, pair_shift=shift)


def boot(x, B=999, rng=None, stat=np.mean):
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), (B, len(x)))
    return stat(x[idx], axis=1)


def bca_ci(x, bs, level=0.95):
    """Bias-corrected and accelerated percentile interval for the mean of x."""
    x = np.asarray(x, float); n = len(x); theta = x.mean()
    if np.all(x == x[0]):
        return [float(theta), float(theta)]
    z0 = stats.norm.ppf(np.clip((np.sum(bs < theta) + 0.5 * np.sum(bs == theta)) / len(bs), 1e-6, 1 - 1e-6))
    jk = (x.sum() - x) / (n - 1); d = jk.mean() - jk
    acc = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5) if np.any(d != 0) else 0.0
    out = []
    for q in ((1 - level) / 2, 1 - (1 - level) / 2):
        z = stats.norm.ppf(q); adj = stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        out.append(float(np.quantile(bs, np.clip(adj, 0, 1))))
    return out


def t_ci(x, level=0.95):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    return [float(m - q * se), float(m + q * se)]


def infer(dl, margin_pp=0.5, B=999, rng=None, alpha=0.05):
    """Dataset-clustered inference for one contrast (one release edge)."""
    rng = rng or np.random.default_rng(1)
    out = {}
    bM = boot(dl['M'], B, rng); bE = boot(dl['E'], B, rng)
    out['primary'] = dict(estimate_pp=float(dl['M'].mean()), ci95=[float(np.quantile(bM, .025)), float(np.quantile(bM, .975))],
                          ci95_bca=bca_ci(dl['M'], bM), ci95_t=t_ci(dl['M']),
                          one_sided_lower95_bca=bca_ci(dl['M'], bM, level=1 - 2 * alpha)[0],
                          one_sided_lower95=float(np.quantile(bM, alpha)), margin_pp=margin_pp,
                          p_vs_margin=float((np.sum(bM <= margin_pp) + 1) / (B + 1)),
                          exceeds_margin=bool(np.quantile(bM, alpha) > margin_pp),
                          below_margin_equivalence=bool(np.quantile(bM, 1 - alpha) < margin_pp))
    nz = dl['E'][dl['E'] != 0]
    wp = float(stats.wilcoxon(dl['E'], zero_method='zsplit').pvalue) if len(nz) else 1.0
    out['co_primary'] = dict(estimate=float(dl['E'].mean()), ci95=[float(np.quantile(bE, .025)), float(np.quantile(bE, .975))],
                             ci95_bca=bca_ci(dl['E'], bE), ci95_t=t_ci(dl['E']) if np.any(dl['E'] != dl['E'][0]) else [float(dl['E'][0])] * 2,
                             p_boot_two_sided=float(min(1.0, 2 * min((np.sum(bE <= 0) + 1) / (B + 1), (np.sum(bE >= 0) + 1) / (B + 1)))),
                             p_wilcoxon=wp)
    fin = dl['S'][np.isfinite(dl['S'])]
    out['secondary'] = dict(median_vnr=float(np.median(dl['VNR'][np.isfinite(dl['VNR'])])) if np.isfinite(dl['VNR']).any() else None,
                            share_vnr_gt1=float(np.mean(dl['VNR'] > 1)),
                            S_mean=float(fin.mean()) if len(fin) else None,
                            S_p_wilcoxon=float(stats.wilcoxon(fin).pvalue) if len(fin) > 1 and np.any(fin != 0) else None,
                            signed_shift_pp=float(dl['signed'].mean()))
    return out


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 1.0
    for i in range(m - 1, -1, -1):
        j = o[i]; run = min(run, p[j] * m / (i + 1)); adj[j] = run
    return adj


def array_from_long(df, old, new, algos, condition_col='condition', reps=6, folds=5):
    """Long table (dataset, condition, algo, rep, fold, ba) -> S array; returns (S, datasets).
    Datasets missing any required cell are dropped and listed (never imputed)."""
    sub = df[df[condition_col].isin([old, new]) & df.algo.isin(algos)]
    piv = sub.pivot_table(index='dataset', columns=[condition_col, 'algo', 'rep', 'fold'], values='ba', aggfunc='first')
    cols = [(c, a, r, f) for c in (old, new) for a in algos for r in range(reps) for f in range(folds)]
    piv = piv.reindex(columns=cols)
    complete = piv.notna().all(1)
    S = piv[complete].to_numpy().reshape(-1, 2, len(algos), reps, folds)
    return S, list(piv.index[complete]), list(piv.index[~complete])
