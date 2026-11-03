"""The design-exact first-order representation of grouped K-fold CV with unequal cluster sizes (proposals/06,
section 6g, draft Proposition 11 (f)), and its two closed forms. No model is fitted here.

Cluster h has size m_h, loss-residual total a_h = L_h(theta*) - m_h mu and influence total beta_h = d' psi_h. Given the
sizes and a grouped partition into folds F_1..F_K (n_j rows in fold j, N_j = n - n_j), keeping only linear terms:
    cluster h's CV loss total   Lhat_h = m_h mu + a_h + m_h B_{T(h)} / N_{(h)}
    fold j's own-row mean       e_j    = mu + A_{F_j}/n_j + B_{T_j}/N_j
    pooled CV                   CV     = mu + (1/n) sum_h (a_h + omega_h beta_h),  omega_h = sum_{j != j(h)} n_j/N_j
    full model's new-row error  Err    = mu + B/n
CV, Err, the unbalanced cluster-robust variance and the Nadeau-Bengio variance (own-row fold means, or each fold's
residual total divided by n/K) are then exact quadratic forms in the totals. Arrays: a, b of shape (D, G) (D draws),
m of shape (G,) or (D, G). Partitions are drawn inside, one per draw: a uniform random order of the clusters dealt into
folds in turn, as in the project's simulation scripts. Used by scripts/sim_cv_unequal.py and tests/test_sim_cv_unequal.py.
"""
import numpy as np

KS_R = {5: 12, 10: 6}                    # K -> R, so that J = R K = 60


def deal(D, G, K, rng):
    """(D, G) fold labels; each row a uniform random grouped partition (random order dealt in turn)."""
    perm = np.argsort(rng.random((D, G)), axis=1)
    lab = np.empty((D, G), int)
    np.put_along_axis(lab, perm, np.broadcast_to(np.arange(G) % K, (D, G)), axis=1)
    return lab


def cv_terms(lab, K, m, a, b):
    """Pooled CV - mu, each cluster's CV loss total minus m_h mu, and the per-fold n_F, A_F, B_T, N_T, and n."""
    masks = np.stack([lab == k for k in range(K)])                      # (K, D, G)
    nF = np.stack([(mk * m).sum(1) for mk in masks], 1)
    AF = np.stack([(mk * a).sum(1) for mk in masks], 1)
    BF = np.stack([(mk * b).sum(1) for mk in masks], 1)
    n = (m * np.ones_like(a)).sum(1); B = b.sum(1)
    NT = n[:, None] - nF; BT = B[:, None] - BF
    Lres = a + m * np.take_along_axis(BT / NT, lab, 1)
    return Lres.sum(1) / n, Lres, nF, AF, BT, NT, n


def linear_stats(m, a, b, rng, K_cv=5, ks_r=KS_R):
    """First-order versions of what sim_cv_unequal.py records, for D draws at once: CV - mu, Err - mu, V_CR (one K_cv
    partition), and for each K the Nadeau-Bengio variances and ratios over R fresh partitions (J = R K)."""
    D, G = a.shape
    m = np.broadcast_to(m, (D, G)).astype(float)
    cv, Lres, nF, AF, BT, NT, n = cv_terms(deal(D, G, K_cv, rng), K_cv, m, a, b)
    R = Lres - m * cv[:, None]
    vcr = G / (G - 1) * (R ** 2).sum(1) / n ** 2
    out = dict(cv=cv, err=b.sum(1) / n, vcr=vcr, n=n)
    for K, Rp in ks_r.items():
        own, res = [], []
        for _ in range(Rp):
            lab = deal(D, G, K, rng)
            cvr, Lr, nFr, AFr, BTr, NTr, _ = cv_terms(lab, K, m, a, b)
            own.append(AFr / nFr + BTr / NTr)
            res.append(np.stack([((Lr - m * cvr[:, None]) * (lab == k)).sum(1) for k in range(K)], 1) * K / n[:, None])
        own, res = np.concatenate(own, 1), np.concatenate(res, 1)
        J = own.shape[1]; fac = (J + K - 1) / (J - 1); c = 1 / J + 1 / (K - 1)
        out[f"vnb_own_K{K}"] = c * own.var(1, ddof=1)
        out[f"vnb_res_K{K}"] = c * res.var(1, ddof=1)
        out[f"ratio_own_K{K}"] = out[f"vnb_own_K{K}"] / (fac * vcr)
        out[f"ratio_res_K{K}"] = out[f"vnb_res_K{K}"] / (fac * vcr)
        out[f"fac_K{K}"] = np.full(D, fac)
    return out


def gaussian_totals(m, row, cc, rng):
    """(a_h, beta_h) ~ N(0, m Sigma_row + m(m-1) Sigma_cc) for sizes m (D, G): a cluster part m U, U ~ N(0, Sigma_cc),
    plus a within part with covariance m (Sigma_row - Sigma_cc). row and cc are 2 x 2."""
    D, G = m.shape
    Lw = np.linalg.cholesky(np.asarray(row) - np.asarray(cc) + 1e-15 * np.eye(2))
    Lb = np.linalg.cholesky(np.asarray(cc) + 1e-15 * np.eye(2))
    t = (rng.standard_normal((D, G, 2)) @ Lw.T * np.sqrt(m)[..., None] + rng.standard_normal((D, G, 2)) @ Lb.T * m[..., None])
    return t[..., 0], t[..., 1]


def p1(row, cc, mtilde, K):
    """The G -> infinity limit (Proposition 11 (c)): Var(a - beta/(K-1)) / Var(a), with per-unit components
    V = Sigma_row + (mtilde - 1) Sigma_cc and mtilde = E[m^2]/E[m]. Returns (ratio, V)."""
    V = np.asarray(row) + (mtilde - 1) * np.asarray(cc)
    return 1 - 2 * V[0, 1] / ((K - 1) * V[0, 0]) + V[1, 1] / ((K - 1) ** 2 * V[0, 0]), V


def surrogate(size_draw, row, cc, rng, D=40000, chunk=2000, K_cv=5, ks_r=KS_R):
    """Monte Carlo of the linear representation with Gaussian totals; size_draw(D, rng) -> (D, G) sizes."""
    parts = []
    for s in range(0, D, chunk):
        d = min(chunk, D - s)
        m = size_draw(d, rng).astype(float)
        a, b = gaussian_totals(m, row, cc, rng)
        parts.append(linear_stats(m, a, b, rng, K_cv, ks_r))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarize(st, ks=tuple(KS_R)):
    """Var(CV - Err), mean V_CR, their ratio, Corr(CV, Err), and per K the mean of the per-draw Nadeau-Bengio ratios
    (what the simulation reports) and the ratio of means."""
    d = st["cv"] - st["err"]
    out = dict(var_diff=d.var(ddof=1), mean_vcr=st["vcr"].mean(), var_ratio=d.var(ddof=1) / st["vcr"].mean(),
               var_err=st["err"].var(ddof=1), corr=np.corrcoef(st["cv"], st["err"])[0, 1] if st["err"].std() > 0 else np.nan)
    for K in ks:
        for v in ("own", "res"):
            out[f"meanratio_{v}_K{K}"] = st[f"ratio_{v}_K{K}"].mean()
            out[f"ratioofmeans_{v}_K{K}"] = st[f"vnb_{v}_K{K}"].mean() / (st[f"fac_K{K}"][0] * st["vcr"].mean())
    return out


def balanced_closed_form(va, vab, vb, G, K_cv, K_nb):
    """Closed form 1 (balanced, K | G) for totals with Var(a) = va, Cov(a, beta) = vab, Var(beta) = vb, when V_CR comes
    from a K_cv-fold run and the Nadeau-Bengio partitions use K_nb folds: returns E[V_CR] / Var(CV - Err) (which involves
    K_cv only) and E[V_NB] / ((J+K_nb-1)/(J-1) E[V_CR]) = Var(a - beta/(K_nb-1)) / (Var a + (Var beta/(K_cv-1) - 2 Cov)/(G-1))."""
    extra = (vb / (K_cv - 1) - 2 * vab) / (G - 1)
    vc = va - 2 * vab / (K_nb - 1) + vb / (K_nb - 1) ** 2
    return 1 + extra / va, vc / (va + extra)


def beta0_closed_form(m, v, K, rng, draws=200000):
    """Closed form 2 (beta = 0, unequal sizes m, Var(a_h) = v_h): E[V_CR] / Var(CV - Err), and the own-row
    E[V_NB] / ((J+K-1)/(J-1) Var(CV - Err)) = n^2 sum v (kappa - lambda^2) / ((K-1) sum v), with kappa_h, lambda_h the
    partition averages of 1{h in F}/n_F^2 and 1{h in F}/n_F (Monte Carlo over partitions). Returns (cr, nb, nb/cr)."""
    m = np.asarray(m, float); v = np.asarray(v, float); G = len(m); n = m.sum()
    lab = deal(draws, G, K, rng)
    nF = np.stack([(m * (lab == k)).sum(1) for k in range(K)], 1)
    nFh = np.take_along_axis(nF, lab, 1)
    kappa, lam = (1 / nFh ** 2).mean(0) / K, (1 / nFh).mean(0) / K
    nb = n ** 2 * np.sum(v * (kappa - lam ** 2)) / ((K - 1) * v.sum())
    cr = G / (G - 1) * (np.sum(v * (1 - 2 * m / n)) + np.sum(m ** 2) * v.sum() / n ** 2) / v.sum()
    return cr, nb, nb / cr


# ---------------------------------------------------------------- second order, log-loss (exploratory)

def second_order_log_stats(m, a, Sv, Hinv, rng, K_cv=5, ks_r=KS_R):
    """Log-loss (d = 0) to second order in theta_hat - theta*: cluster h's CV loss total is
    m_h mu + a_h - S_h' H^-1 S_T / N_T + (1/2) m_h S_T' H^-1 S_T / N_T^2, and Err = mu + (1/2) S' H^-1 S / n^2, with S_h the
    cluster's score total at theta*. a (D, G), Sv (D, G, p), m (D, G). Returns the same keys as linear_stats (own rows only)."""
    D, G = a.shape; n = m.sum(1); Stot = Sv.sum(1)

    def totals(lab, K):
        Lh = np.empty((D, G))
        for k in range(K):
            mk = lab == k
            ST = Stot - (Sv * mk[..., None]).sum(1); NT = n - (m * mk).sum(1)
            delta = np.einsum("ij,dj->di", Hinv, ST) / NT[:, None]
            quad = 0.5 * np.einsum("di,di->d", ST, delta) / NT
            Lh = np.where(mk, a - np.einsum("dgi,di->dg", Sv, delta) + m * quad[:, None], Lh)
        return Lh
    Lh = totals(deal(D, G, K_cv, rng), K_cv); cv = Lh.sum(1) / n
    vcr = G / (G - 1) * ((Lh - m * cv[:, None]) ** 2).sum(1) / n ** 2
    out = dict(cv=cv, err=0.5 * np.einsum("di,ij,dj->d", Stot, Hinv, Stot) / n ** 2, vcr=vcr, n=n)
    for K, Rp in ks_r.items():
        own = []
        for _ in range(Rp):
            lab = deal(D, G, K, rng); Lr = totals(lab, K)
            own.append(np.stack([(Lr * (lab == k)).sum(1) / (m * (lab == k)).sum(1) for k in range(K)], 1))
        own = np.concatenate(own, 1); J = own.shape[1]; fac = (J + K - 1) / (J - 1)
        out[f"vnb_own_K{K}"] = (1 / J + 1 / (K - 1)) * own.var(1, ddof=1)
        out[f"ratio_own_K{K}"] = out[f"vnb_own_K{K}"] / (fac * vcr); out[f"fac_K{K}"] = np.full(D, fac)
    return out


def second_order_surrogate(size_draw, row, cc, Hinv, rng, D=10000, chunk=1000, ks_r=KS_R):
    """Gaussian surrogate for second_order_log_stats: (a_h, S_h) ~ N(0, m Sigma_row + m(m-1) Sigma_cc), with row and cc
    the per-row and same-cluster covariances of (log-loss, score)."""
    p = len(row); Lw = np.linalg.cholesky(row - cc + 1e-12 * np.eye(p)); Lb = np.linalg.cholesky(cc + 1e-12 * np.eye(p)); parts = []
    for s in range(0, D, chunk):
        m = size_draw(min(chunk, D - s), rng).astype(float)
        z = rng.standard_normal(m.shape + (p,)) @ Lw.T * np.sqrt(m)[..., None] + rng.standard_normal(m.shape + (p,)) @ Lb.T * m[..., None]
        parts.append(second_order_log_stats(m, z[..., 0], z[..., 1:], Hinv, rng, ks_r=ks_r))
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
