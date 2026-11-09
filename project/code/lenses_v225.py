"""VersionShift v2.2.5 — secondary validation lenses (pre-registered before any Stage B outcome).

The frozen primary analysis (decision_v221: dataset-cluster t bounds, TOST, Clopper-Pearson, Holm) is
unchanged. These lenses answer questions the primary cannot, and each was calibrated on the v2.1
known-truth generator before being applied to data (calibrate_lenses_v225.py).

Input for every lens: X[k, p, r] = upgrade-induced shift of the paired comparison for dataset k,
ordered pair p, rerun r (pp; mean over the folds of rerun r), on exactly paired splits and seeds.

Lens B  heterogeneity / prediction
        Per pair: random-effects meta-analysis across datasets (REML tau^2, Hartung-Knapp-Sidik-Jonkman
        CI for the mean shift, prediction interval mu +/- t_{K-2} sqrt(tau^2 + SE^2) for the TRUE shift
        on a new dataset). Answers: "what conclusion shift should a practitioner expect on their data?"
Lens C  threshold-free discovery
        Per dataset x pair: rerun-calibrated t test of mean shift = 0 over the R reruns; pairs combined by
        the Cauchy combination test (valid under arbitrary dependence, Liu & Xie 2020) -> one p per
        dataset; Benjamini-Hochberg at q = 0.10 -> list of changed datasets; Storey (lambda = 0.5)
        pi1 = 1 - pi0 -> share of datasets with ANY real change, with no 0.5-pp threshold.
Lens D  anytime-valid sequential test of H0: E[M_k] <= margin
        Betting e-process W_n = prod(1 + lambda_i (M_i - m)), lambda_i predictable (aGRAPA plug-in),
        0 <= lambda_i <= 0.5/m. Valid for NONNEGATIVE M_k with no upper bound (E[1 + lambda(M - m)] <= 1
        under H0); reject when W_n >= 1/alpha (Ville). Datasets enter in a pre-registered random order;
        Stage-B datasets can continue into Stage C without a reuse penalty. Related tool: seqbench (R, 2026)
        applies betting confidence sequences to algorithm comparisons; here the target is upgrade-induced
        conclusion change.
"""
import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------------------- lens B
def reml_tau2(y, v, iters=200, tol=1e-10):
    y = np.asarray(y, float); v = np.maximum(np.asarray(v, float), 1e-8)
    t2 = max(0.0, np.var(y, ddof=1) - v.mean()) if len(y) > 1 else 0.0
    for _ in range(iters):
        w = 1.0 / (v + t2); mu = (w * y).sum() / w.sum()
        # REML Fisher-scoring step
        num = (w ** 2 * ((y - mu) ** 2 - v)).sum() + (w ** 2).sum() / w.sum()
        new = max(0.0, num / (w ** 2).sum())
        if abs(new - t2) < tol:
            t2 = new; break
        t2 = new
    return t2


def random_effects(y, v, level=0.95):
    y = np.asarray(y, float); v = np.maximum(np.asarray(v, float), 1e-8); K = len(y)
    t2 = reml_tau2(y, v); w = 1.0 / (v + t2); mu = (w * y).sum() / w.sum()
    q = (w * (y - mu) ** 2).sum() / (K - 1) if K > 1 else 1.0
    se_hk = np.sqrt(max(q, 1e-12) / w.sum())                 # HKSJ
    a = 1 - (1 - level) / 2
    ci = [mu - stats.t.ppf(a, K - 1) * se_hk, mu + stats.t.ppf(a, K - 1) * se_hk]
    pi_half = stats.t.ppf(a, max(K - 2, 1)) * np.sqrt(t2 + se_hk ** 2)
    i2 = max(0.0, 1 - (K - 1) / max(((1 / v) * (y - (y / v).sum() / (1 / v).sum()) ** 2).sum(), 1e-12)) if K > 1 else 0.0
    return dict(mu=float(mu), ci=[float(ci[0]), float(ci[1])], tau2=float(t2), tau=float(np.sqrt(t2)),
                pi=[float(mu - pi_half), float(mu + pi_half)], I2=float(i2), K=int(K))


def lens_b(X, pair_names=None):
    K, P, R = X.shape
    y = X.mean(2); v = X.var(2, ddof=1) / R
    out = {}
    for j in range(P):
        name = pair_names[j] if pair_names else j
        out[name] = random_effects(y[:, j], v[:, j])
    return out


# ----------------------------------------------------------------------------------------- lens C
def rerun_t_pvalues(X):
    """X (K,P,R) -> two-sided p of H0: E[shift]=0 per dataset x pair, over reruns."""
    m = X.mean(2); s = X.std(2, ddof=1); R = X.shape[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = m / (s / np.sqrt(R))
    p = 2 * stats.t.sf(np.abs(t), R - 1)
    p = np.where(s > 0, p, np.where(np.abs(m) > 1e-12, 0.0, 1.0))
    return np.clip(p, 1e-15, 1.0)


def cauchy_combine(p, axis=-1):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    T = np.tan((0.5 - p) * np.pi).mean(axis)
    return np.clip(0.5 - np.arctan(T) / np.pi, 0.0, 1.0)


def bh_reject(p, q=0.10):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = np.nonzero(p[o] <= thresh)[0]
    rej = np.zeros(m, bool)
    if len(passed):
        rej[o[:passed.max() + 1]] = True
    return rej


def storey_pi1(p, lam=0.5):
    p = np.asarray(p, float)
    pi0 = min(1.0, np.mean(p > lam) / (1 - lam))
    return 1.0 - pi0


def combine_dataset(p_row, exact_zero_row, cap=0.99):
    """Cauchy combination over the pairs of ONE dataset. Pairs whose shift is exactly zero in every rerun
    (deterministic, unchanged) carry no evidence and are left out: with p = 1 the Cauchy statistic
    tan((0.5 - p) pi) diverges to -inf and would mask real changes in the other pairs. p is truncated at
    `cap` (truncated Cauchy combination) for robustness near 1."""
    keep = ~exact_zero_row
    if not keep.any():
        return 1.0
    pp = np.clip(p_row[keep], 1e-15, cap)
    return float(np.clip(0.5 - np.arctan(np.tan((0.5 - pp) * np.pi).mean()) / np.pi, 0.0, 1.0))


def lens_c(X, q=0.10):
    p = rerun_t_pvalues(X)
    exact_zero = (np.abs(X) < 1e-12).all(axis=2)
    pk = np.array([combine_dataset(p[k], exact_zero[k]) for k in range(X.shape[0])])
    rej = bh_reject(pk, q)
    return dict(dataset_p=pk.tolist(), discoveries=int(rej.sum()), changed=rej.tolist(), q=q,
                storey_pi1=float(storey_pi1(pk)), K=int(len(pk)))


# ----------------------------------------------------------------------------------------- lens D
def lens_d(M, margin=0.5, alpha=0.05, order_seed=20261106, prior_mean=None):
    """Anytime-valid test of H0: E[M] <= margin for nonnegative M (pp). Returns stopping index or None."""
    M = np.asarray(M, float)
    if np.any(M < 0):
        raise ValueError('lens D requires nonnegative M_k')
    order = np.random.default_rng(order_seed).permutation(len(M)); x = M[order]
    W, path, stop = 1.0, [], None
    mu_hat = margin if prior_mean is None else prior_mean; s2 = (margin ** 2); n = 0
    lam_max = 0.5 / margin
    for i, xi in enumerate(x):
        d = mu_hat - margin
        lam = float(np.clip(d / (s2 + d * d), 0.0, lam_max)) if (s2 + d * d) > 0 else 0.0
        W *= 1.0 + lam * (xi - margin); path.append(W)
        n += 1; delta = xi - mu_hat
        mu_hat = ((n) * mu_hat + xi) / (n + 1)          # shrunk running mean (prior weight 1 at margin)
        s2 = ((n) * s2 + delta ** 2) / (n + 1)
        if stop is None and W >= 1.0 / alpha:
            stop = i + 1
    return dict(rejected=stop is not None, stopping_K=stop, final_wealth=float(W), max_wealth=float(max(path) if path else 1.0),
                K=int(len(M)), margin_pp=margin, alpha=alpha, order_seed=order_seed)
