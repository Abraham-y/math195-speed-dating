"""O1: does the chance of a yes fall with a date's relative position in the evening? (PREREG.md, Part 5)

Input : data/clean.csv
Output: results/order_effects_estimates.csv (every estimate, the permutation test and the balance check),
        results/order_effects_hypotheses.csv
Seeds : permutation test 20263806 (one generator, 1,000 permutations in turn). Nothing else is random.

Model (as registered). Every date with a decision. Within rater i, with T_i dates, the date at position t (`order`)
gets a yes with log-odds alpha_i + gamma t / T_i. Conditioning on each rater's number of yeses k_i removes alpha_i:
    P(y_i | k_i) = exp(gamma sum_j y_ij x_ij) / e_{k_i}(exp(gamma x_i1), ..., exp(gamma x_iT_i)),   x_ij = t_ij / T_i,
where e_k is the elementary symmetric polynomial of degree k, computed (with its first and second derivatives in
the parameters) by the recursion e_k <- e_k + z_j e_{k-1} over the rater's dates. Raters who said yes to all or none
of their dates contribute nothing. Maximum likelihood by Newton's method (the log-likelihood is concave).
SE: the event-clustered sandwich, per-rater scores summed by event, G/(G-1), Hessian bread.
Permutation test: positions shuffled within each rater (1,000 permutations), two-sided p-value
(1 + #{|gamma_perm| >= |gamma|}) / (1 + 1,000); the SD of the permuted gamma against the mean sandwich SE over
the permutations is the SE check.
Balance check: the partner's mean liking received from the other raters (leave this rater out) regressed on the
relative position t / T_i by OLS, event-clustered sandwich SE with G/(G-1); a within-rater version (both sides
demeaned by rater) is reported beside it.
Exploratory: the gender interaction (gamma + delta x man) t / T_i; sensitivity versions without verdicts: positions
ranked 1..T_i within the rater (the 9 men of event 5 have `order` up to 10 with 9 dates, so t / T_i reaches 10/9),
and without wave 12 (the PREREG wave 12 entry).
Run from the repository root: python scripts/order_effects.py [permutations]
"""
import sys, pathlib, time
import numpy as np, pandas as pd

RES = pathlib.Path("results")
PERM_SEED, PERMS, Z90 = 20263806, 1000, 1.6448536269514722


# ---------------------------------------------------------------- data

def rater_arrays(d, cols):
    """Pad each rater's dates into (raters, Tmax) arrays, dates in data order. cols are the covariate columns of the
    conditional logit (one parameter each). Returns X (raters, Tmax, p), y (raters, Tmax), mask, k (yeses), and
    the raters' events, in the order of the sorted rater ids."""
    d = d.sort_values(["rater", "order"], kind="stable")
    r = d.rater.to_numpy(); ids, start, T = np.unique(r, return_index=True, return_counts=True)
    pos = np.arange(len(d)) - np.repeat(start, T)
    ri = np.repeat(np.arange(len(ids)), T)
    X = np.zeros((len(ids), T.max(), len(cols))); y = np.zeros((len(ids), T.max())); mask = np.zeros_like(y, bool)
    X[ri, pos] = d[cols].to_numpy(float); y[ri, pos] = d.dec.to_numpy(float); mask[ri, pos] = True
    ev = d.groupby("rater").event.first().loc[ids].to_numpy()
    return X, y, mask, y.sum(1).astype(int), ev


def relative_position(d, how="order"):
    """t / T_i with t = `order` (as registered) or the rank of the date within the rater (sensitivity)."""
    T = d.groupby("rater").order.transform("size")
    t = d.order if how == "order" else d.groupby("rater").order.rank(method="first")
    return (t / T).to_numpy(float)


# ---------------------------------------------------------------- conditional logit

def esp(z, mask=None):
    """Elementary symmetric polynomials e_0..e_T of the rows of z (raters, T); masked cells count as absent."""
    z = np.where(mask, z, 0.0) if mask is not None else z
    e = np.zeros((z.shape[0], z.shape[1] + 1)); e[:, 0] = 1.0
    for j in range(z.shape[1]):
        e[:, 1:] = e[:, 1:] + z[:, j:j + 1] * e[:, :-1]
    return e


def clogit_parts(beta, X, y, mask, k):
    """Per-rater log-likelihood, score (p) and Hessian (p x p) of the conditional logit at beta, by the
    elementary-symmetric-polynomial recursion carried with its first and second derivatives."""
    n, T, p = X.shape
    z = np.where(mask, np.exp(X @ beta), 0.0)
    e = np.zeros((n, T + 1)); e[:, 0] = 1.0
    de = np.zeros((n, T + 1, p)); d2e = np.zeros((n, T + 1, p, p))
    for j in range(T):
        zj, xj = z[:, j], X[:, j, :]
        dz = zj[:, None] * xj; d2z = zj[:, None, None] * xj[:, :, None] * xj[:, None, :]
        e0, de0, d2e0 = e[:, :-1], de[:, :-1], d2e[:, :-1]                     # e_{k-1} before this date
        d2e[:, 1:] = (d2e[:, 1:] + d2z[:, None] * e0[..., None, None] + dz[:, None, :, None] * de0[:, :, None, :]
                      + de0[:, :, :, None] * dz[:, None, None, :] + zj[:, None, None, None] * d2e0)
        de[:, 1:] = de[:, 1:] + dz[:, None, :] * e0[..., None] + zj[:, None, None] * de0
        e[:, 1:] = e[:, 1:] + zj[:, None] * e0
    idx = np.arange(n)
    E, dE, d2E = e[idx, k], de[idx, k], d2e[idx, k]
    s = np.einsum("nt,ntp->np", y * mask, X)
    m = dE / E[:, None]
    ll = np.einsum("np,p->n", s, beta) - np.log(E)
    score = s - m
    hess = -(d2E / E[:, None, None] - m[:, :, None] * m[:, None, :])
    return ll, score, hess


def clogit_fit(X, y, mask, k, tol=1e-10, iters=100):
    """Maximum likelihood by Newton's method from 0. Returns beta, total log-likelihood, per-rater scores, Hessian."""
    beta = np.zeros(X.shape[2])
    for _ in range(iters):
        ll, sc, h = clogit_parts(beta, X, y, mask, k)
        step = np.linalg.solve(h.sum(0), sc.sum(0))
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    ll, sc, h = clogit_parts(beta, X, y, mask, k)
    assert np.max(np.abs(sc.sum(0))) < 1e-6, "conditional logit did not converge"
    return beta, ll.sum(), sc, h.sum(0)


def cluster_sandwich(scores, hessian, g):
    """Event-clustered sandwich: H^-1 (G/(G-1) sum_g U_g U_g') H^-1, U_g the scores summed within cluster g."""
    U = pd.DataFrame(scores).groupby(np.asarray(g)).sum().to_numpy(); G = len(U)
    B = np.linalg.inv(hessian)
    return B @ (G / (G - 1) * U.T @ U) @ B


def fit_with_se(X, y, mask, k, ev):
    beta, ll, sc, H = clogit_fit(X, y, mask, k)
    return beta, cluster_sandwich(sc, H, ev), ll


def permute_within(X, mask, rng):
    """Shuffle each rater's positions among that rater's dates (padding stays at the end)."""
    keys = np.where(mask, rng.random(mask.shape), np.inf)
    perm = np.argsort(keys, axis=1, kind="stable")
    return np.take_along_axis(X, perm[:, :, None], axis=1)


def permutation_test(X, y, mask, k, ev, B, rng):
    est, se = np.empty(B), np.empty(B)
    for b in range(B):
        beta, V, _ = fit_with_se(permute_within(X, mask, rng), y, mask, k, ev)
        est[b], se[b] = beta[0], np.sqrt(V[0, 0])
    return est, se


# ---------------------------------------------------------------- balance check

def leave_one_out_mean(d, col="like"):
    """For each date (rater i, partner j): the mean of col over the dates on which j was rated by someone other
    than i (missing values skipped)."""
    v = d[col].to_numpy(float); ok = ~np.isnan(v)
    tot = pd.Series(np.where(ok, v, 0.0)).groupby(d.ratee.to_numpy()).transform("sum").to_numpy()
    cnt = pd.Series(ok.astype(float)).groupby(d.ratee.to_numpy()).transform("sum").to_numpy()
    num, den = tot - np.where(ok, v, 0.0), cnt - ok
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def ols_cluster(y, X, g):
    """OLS with the cluster sandwich (bread (X'X)^-1, meat G/(G-1) sum_g (X_g'e_g)(X_g'e_g)')."""
    y, X, g = np.asarray(y, float), np.asarray(X, float), np.asarray(g)
    B = np.linalg.inv(X.T @ X); b = B @ X.T @ y; e = y - X @ b
    U = pd.DataFrame(X * e[:, None]).groupby(g).sum().to_numpy(); G = len(U)
    return b, B @ (G / (G - 1) * U.T @ U) @ B


def balance(d, x):
    """The partner's leave-one-out mean liking on relative position: pooled (with intercept) and within rater."""
    m = leave_one_out_mean(d); ok = ~np.isnan(m)
    dd = pd.DataFrame({"m": m[ok], "x": x[ok], "rater": d.rater.to_numpy()[ok], "event": d.event.to_numpy()[ok]})
    b, V = ols_cluster(dd.m, np.column_stack([np.ones(len(dd)), dd.x]), dd.event)
    mw = dd.m - dd.groupby("rater").m.transform("mean"); xw = dd.x - dd.groupby("rater").x.transform("mean")
    bw, Vw = ols_cluster(mw, xw.to_numpy()[:, None], dd.event)
    sd = dd.m.std()
    return [dict(model="balance: partner's mean liking from other raters on t / T_i (pooled OLS with intercept, as written)", term="slope",
                 estimate=b[1], se_cluster_event=np.sqrt(V[1, 1]), dates=len(dd), outcome_sd=sd),
            dict(model="balance: same, within rater (both sides demeaned by rater; the variation the conditional logit uses)", term="slope",
                 estimate=bw[0], se_cluster_event=np.sqrt(Vw[0, 0]), dates=len(dd), outcome_sd=sd)]


# ---------------------------------------------------------------- main

def row(model, term, est, se, X, k, mask, ev, **kw):
    T = mask.sum(1); informative = (k > 0) & (k < T)
    return dict(model=model, term=term, estimate=est, se_cluster_event=se, ci90_low=est - Z90 * se, ci90_high=est + Z90 * se,
                z=est / se, raters=len(k), informative_raters=int(informative.sum()), dates=int(mask.sum()),
                informative_dates=int(T[informative].sum()), events=len(np.unique(ev)), **kw)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else PERMS
    t0 = time.time()
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    d = clean[clean.dec.notna()].copy()
    d["x"] = relative_position(d); d["x_rank"] = relative_position(d, "rank"); d["x_man"] = d.x * d.gender
    rows = []

    X, y, mask, k, ev = rater_arrays(d, ["x"])
    beta, V, ll = fit_with_se(X, y, mask, k, ev)
    g_hat, g_se = beta[0], np.sqrt(V[0, 0])
    rows.append(row("conditional logit, theta = gamma t / T_i (registered)", "gamma", g_hat, g_se, X, k, mask, ev, loglik=ll))

    rng = np.random.default_rng(PERM_SEED)
    pe, ps = permutation_test(X, y, mask, k, ev, B, rng)
    p_two = (1 + np.sum(np.abs(pe) >= abs(g_hat))) / (1 + B)
    perm = dict(permutations=B, seed=PERM_SEED, p_two_sided=p_two, perm_mean=pe.mean(), perm_sd=pe.std(ddof=1),
                perm_mean_sandwich_se=ps.mean(), se_ratio=ps.mean() / pe.std(ddof=1),
                perm_coverage_90=np.mean(np.abs(pe) <= Z90 * ps))
    rows[-1].update(perm)

    Xg, yg, mg, kg, evg = rater_arrays(d, ["x", "x_man"])
    bg, Vg, llg = fit_with_se(Xg, yg, mg, kg, evg)
    a = np.array([1.0, 1.0])
    for term, est, se in (("gamma (women)", bg[0], np.sqrt(Vg[0, 0])), ("delta (men minus women)", bg[1], np.sqrt(Vg[1, 1])),
                          ("gamma + delta (men)", a @ bg, np.sqrt(a @ Vg @ a))):
        rows.append(row("exploratory: gender interaction, theta = (gamma + delta man) t / T_i", term, est, se, Xg, kg, mg, evg, loglik=llg))

    Xr, yr, mr, kr, evr = rater_arrays(d, ["x_rank"])
    br, Vr, llr = fit_with_se(Xr, yr, mr, kr, evr)
    rows.append(row("sensitivity: t = rank of the date within the rater", "gamma", br[0], np.sqrt(Vr[0, 0]), Xr, kr, mr, evr, loglik=llr))
    d12 = d[d.wave_yes_budget != 1]
    X12, y12, m12, k12, ev12 = rater_arrays(d12, ["x"])
    b12, V12, ll12 = fit_with_se(X12, y12, m12, k12, ev12)
    rows.append(row("sensitivity: without wave 12 (yes budget)", "gamma", b12[0], np.sqrt(V12[0, 0]), X12, k12, m12, ev12, loglik=ll12))

    bal = balance(d, d.x.to_numpy())
    for r in bal:
        r.update(ci90_low=r["estimate"] - Z90 * r["se_cluster_event"], ci90_high=r["estimate"] + Z90 * r["se_cluster_event"],
                 z=r["estimate"] / r["se_cluster_event"], events=d.event.nunique())
    E = pd.DataFrame(rows + bal).assign(fold_scheme="none (all dates)", position="t = order, T_i = the rater's number of dates")
    E.loc[E.model.str.startswith("sensitivity: t = rank"), "position"] = "t = rank of order within the rater"
    E.to_csv(RES / "order_effects_estimates.csv", index=False)

    lo, hi = g_hat - Z90 * g_se, g_hat + Z90 * g_se
    b0 = bal[0]
    H = pd.DataFrame([dict(
        hypothesis="O1", version="all dates, both genders", fold_scheme="none (all dates)",
        statistic=f"gamma {g_hat:.4f} (90% interval {lo:.4f} to {hi:.4f}; event-clustered sandwich SE {g_se:.4f}); "
                  f"permutation p (two-sided, {B} within-rater shuffles) {p_two:.4f}",
        supported=bool(hi < 0),
        se_check=f"permutation SD of gamma {perm['perm_sd']:.4f} against mean sandwich SE {perm['perm_mean_sandwich_se']:.4f} "
                 f"(ratio {perm['se_ratio']:.2f}); 90% coverage of 0 under the permutation null {perm['perm_coverage_90']:.1%} (seed {PERM_SEED})",
        note=f"{int(rows[0]['informative_raters'])} of {len(k)} raters informative; balance check: partner's leave-one-out mean liking on "
             f"t / T_i slope {b0['estimate']:.4f} (SE {b0['se_cluster_event']:.4f}) pooled, {bal[1]['estimate']:.4f} (SE {bal[1]['se_cluster_event']:.4f}) "
             f"within rater (partner-mean SD {b0['outcome_sd']:.3f}); "
             f"exploratory gender interaction delta {bg[1]:.4f} (SE {np.sqrt(Vg[1, 1]):.4f})")])
    H.to_csv(RES / "order_effects_hypotheses.csv", index=False)
    pd.set_option("display.width", 250)
    print(E[["model", "term", "estimate", "se_cluster_event", "ci90_low", "ci90_high", "informative_raters", "dates"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.Series(perm).to_string())
    print(H.drop(columns=["fold_scheme"]).T.to_string(header=False))
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
