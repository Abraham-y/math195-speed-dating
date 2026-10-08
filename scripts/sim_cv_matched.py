"""A2 (PREREG.md, second half and Part 2): coverage of cross-validation intervals in a simulation matched to the
speed-dating data.

Design (fixed in PREREG.md before this ran): the real 4,184 dyads, with their women, men and events. The truth is
a logistic model for the match with the four pre-date predictors chosen most often in Part 1 (age gap, the man's
dating frequency, his self-rated fun, same field; standardized, missing values filled with medians), plus a woman
effect, a man effect and an event effect. Coefficients and the three SDs come from a crossed random-effects
logistic regression fitted to the real matches (statsmodels BinomialBayesMixedGLM, variational Bayes); the
intercept is then set so the simulated match rate equals the real one. Each replicate draws fresh effects and
outcomes and runs, on the same four predictors:
- event 5-fold CV (events dealt into folds at random) and random-row 5-fold CV, each fitting a logistic regression
  in the training folds and predicting a match when the probability exceeds the training match rate; the per-dyad
  loss is 1 for a wrong prediction;
- intervals: random-row error plus or minus 1.645 naive SEs (the conventional interval); event error plus or minus
  1.645 naive SEs; plus or minus 1.645 cluster-robust SEs (events); the cluster-nested and random-row nested (Bates,
  Hastie and Tibshirani) root-MSE estimates of sim_cv_cluster.py; the floor of the cluster-robust and
  cluster-nested SEs; and both nested intervals with the Bates, Hastie and Tibshirani bias adjustment (centred at
  the nested estimate minus (1 + (K - 2)/K)(nested - standard CV), half-width 1.645 sqrt((K - 1)/K) root MSE).
The target is the error of the model fitted to the whole replicate on 300 new events (real events resampled with
fresh effects), and, separately, its average over replicates.
Seed  : 20261906.
Output: results/sim_cv_matched_rows.csv, results/sim_cv_matched_coverage.csv, results/sim_cv_matched_hypotheses.csv
Run from the repository root: python scripts/sim_cv_matched.py [replicates] [nested repetitions]
"""
import sys, pathlib, time
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M

SEED = 20261906
PREDICTORS = ["abs_age_diff", "m_date", "m_self_fun", "same_field"]
NEW_EVENTS, Z = 300, 1.645


def design(pairs):
    X = pairs[PREDICTORS].astype(float)
    X = X.fillna(X.median())
    X = ((X - X.mean()) / X.std()).to_numpy()
    codes = lambda s: pd.factorize(s)[0]
    return X, codes(pairs.f_iid), codes(pairs.m_iid), codes(pairs.event)


def fit_truth(pairs, X, f, m, e):
    """Fixed effects and the woman, man and event SDs from a crossed random-effects logistic regression."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    df = pd.DataFrame(X, columns=PREDICTORS).assign(match=pairs.match.to_numpy(), f=f, m=m, e=e)
    model = BinomialBayesMixedGLM.from_formula("match ~ " + " + ".join(PREDICTORS),
                                               {"woman": "0 + C(f)", "man": "0 + C(m)", "event": "0 + C(e)"}, df)
    r = model.fit_vb()
    beta = np.asarray(r.fe_mean)
    sd = dict(zip(["woman", "man", "event"], np.exp(np.asarray(r.vcp_mean))))
    return beta, sd


def draw(beta0, beta, sd, X, f, m, e, rng):
    eta = beta0 + X @ beta + rng.normal(0, sd["woman"], f.max() + 1)[f] + rng.normal(0, sd["man"], m.max() + 1)[m] + rng.normal(0, sd["event"], e.max() + 1)[e]
    return (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)


def calibrate_intercept(beta0, beta, sd, X, f, m, e, rate, rng, draws=200):
    """Shift the intercept so the average simulated match rate equals the real rate (bisection)."""
    lo, hi = beta0 - 3, beta0 + 3
    eff = [(rng.normal(0, sd["woman"], f.max() + 1)[f] + rng.normal(0, sd["man"], m.max() + 1)[m] + rng.normal(0, sd["event"], e.max() + 1)[e])
           for _ in range(draws)]
    mean_rate = lambda b: np.mean([np.mean(1 / (1 + np.exp(-(b + X @ beta + u)))) for u in eff])
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_rate(mid) < rate else (lo, mid)
    return (lo + hi) / 2


def loss(Xtr, ytr, Xte, yte):
    w, _ = M.fit_logit(Xtr, ytr)
    return ((M.predict_logit(w, Xte) > ytr.mean()).astype(float) != yte).astype(float)


def cv(X, y, folds, K=5):
    e = np.empty(len(y))
    for k in range(K):
        te = folds == k
        e[te] = loss(X[~te], y[~te], X[te], y[te])
    return e


def group_folds(g, K, rng):
    G = g.max() + 1; perm = rng.permutation(G); to = np.empty(G, int); to[perm] = np.arange(G) % K
    return to[g]


def nested_mse(X, y, g, K, rng, grouped, R):
    """Nested CV (Bates, Hastie and Tibshirani 2024): the MSE estimate (mean squared gap between inner CV and outer
    error, minus the outer error's variance, cluster-robust when grouped) and the nested error estimate (mean of the
    inner CV errors). Returns (MSE estimate, nested error estimate)."""
    a, b, ein = [], [], []
    n = len(y)
    for _ in range(R):
        folds = group_folds(g, K, rng) if grouped else rng.permutation(n) % K
        for k in range(K):
            out = folds == k
            Xi, yi, gi = X[~out], y[~out], g[~out]
            fi = group_folds(pd.factorize(gi)[0], K - 1, rng) if grouped else rng.permutation(len(yi)) % (K - 1)
            e_in = cv(Xi, yi, fi, K - 1).mean(); ein.append(e_in)
            eo = loss(Xi, yi, X[out], y[out])
            a.append((e_in - eo.mean()) ** 2)
            b.append(M.cluster_robust_var_of_mean(eo, g[out]) if grouped else eo.var(ddof=1) / out.sum())
    return np.mean(a) - np.mean(b), np.mean(ein)


def bht_interval(err_cv, err_ncv, mse, K=5):
    """Bias-adjusted nested-CV interval (Bates, Hastie and Tibshirani 2024): (centre, SE)."""
    return err_ncv - (1 + (K - 2) / K) * (err_ncv - err_cv), np.sqrt((K - 1) / K) * np.sqrt(max(mse, 0))


def new_events(pairs, X, f, m, e, rng):
    """Covariates of NEW_EVENTS events resampled from the real ones, each with fresh people (new ids)."""
    ev_rows = [np.where(e == k)[0] for k in range(e.max() + 1)]
    pick = rng.integers(0, len(ev_rows), NEW_EVENTS)
    idx = np.concatenate([ev_rows[k] for k in pick])
    block = np.repeat(np.arange(NEW_EVENTS), [len(ev_rows[k]) for k in pick])
    nf = pd.factorize(pd.Series(block).astype(str) + "_" + pd.Series(f[idx]).astype(str))[0]
    nm = pd.factorize(pd.Series(block).astype(str) + "_" + pd.Series(m[idx]).astype(str))[0]
    return X[idx], nf, nm, block


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rng = np.random.default_rng(SEED)
    pairs = pd.read_csv("data/pairs.csv")
    X, f, m, e = design(pairs)
    beta_all, sd = fit_truth(pairs, X, f, m, e)
    beta0, beta = calibrate_intercept(beta_all[0], beta_all[1:], sd, X, f, m, e, pairs.match.mean(), rng), beta_all[1:]
    print(f"truth: intercept {beta0:.3f}, coefficients {dict(zip(PREDICTORS, np.round(beta, 3)))}, SDs {({k: round(v, 3) for k, v in sd.items()})}", flush=True)
    rows, t0 = [], time.time()
    for r in range(reps):
        y = draw(beta0, beta, sd, X, f, m, e, rng)
        Xn, fn, mn, en = new_events(pairs, X, f, m, e, rng)
        yn = draw(beta0, beta, sd, Xn, fn, mn, en, rng)
        err_new = loss(X, y, Xn, yn).mean()
        eg = cv(X, y, group_folds(e, 5, rng)); er = cv(X, y, rng.permutation(len(y)) % 5)
        mse_cn, ncv_cn = nested_mse(X, y, e, 5, rng, True, R); mse_rn, ncv_rn = nested_mse(X, y, e, 5, rng, False, R)
        c_cn, s_cn = bht_interval(eg.mean(), ncv_cn, mse_cn); c_rn, s_rn = bht_interval(er.mean(), ncv_rn, mse_rn)
        rows.append(dict(rep=r, err_new=err_new, match_rate=y.mean(), cv_event=eg.mean(), se_event_naive=np.sqrt(eg.var(ddof=1) / len(eg)),
                         se_event_cr=np.sqrt(M.cluster_robust_var_of_mean(eg, e)), se_event_cluster_nested=np.sqrt(max(mse_cn, 0)),
                         cv_random=er.mean(), se_random_naive=np.sqrt(er.var(ddof=1) / len(er)), se_random_nested=np.sqrt(max(mse_rn, 0)),
                         mse_cluster_nested_raw=mse_cn, mse_random_nested_raw=mse_rn, bht_event_centre=c_cn, bht_event_se=s_cn,
                         bht_random_centre=c_rn, bht_random_se=s_rn))
        if r == 2:
            print(f"about {(time.time() - t0) / 3 * reps / 60:.0f} minutes for {reps} replicates", flush=True)
    D = pd.DataFrame(rows).assign(seed=SEED, nested_repetitions=R)
    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    D.to_csv(out / "sim_cv_matched_rows.csv", index=False)
    D["se_event_floor"] = np.maximum(D.se_event_cr, D.se_event_cluster_nested)
    procs = [("random-row CV, naive SE (conventional)", "cv_random", "se_random_naive"),
             ("random-row CV, nested (Bates et al.)", "cv_random", "se_random_nested"),
             ("event CV, naive SE", "cv_event", "se_event_naive"),
             ("event CV, cluster-robust SE", "cv_event", "se_event_cr"),
             ("event CV, cluster-nested", "cv_event", "se_event_cluster_nested"),
             ("event CV, floor of cluster-robust and cluster-nested", "cv_event", "se_event_floor"),
             ("random-row nested, bias-adjusted (Bates et al.)", "bht_random_centre", "bht_random_se"),
             ("event cluster-nested, bias-adjusted", "bht_event_centre", "bht_event_se")]
    cov = []
    for target, t in (("error on new events (this training set)", D.err_new), ("average error on new events", pd.Series(D.err_new.mean(), index=D.index))):
        for name, est, se in procs:
            cov.append(dict(target=target, procedure=name, coverage_90=np.mean(np.abs(D[est] - t) <= Z * D[se]),
                            mean_half_width=Z * D[se].mean(), bias=(D[est] - t).mean(), reps=reps, seed=SEED))
    C = pd.DataFrame(cov); C.to_csv(out / "sim_cv_matched_coverage.csv", index=False)
    c = C[C.target.str.startswith("error on new events (this")].set_index("procedure").coverage_90
    conv, ecr = c["random-row CV, naive SE (conventional)"], c["event CV, cluster-robust SE"]
    pd.DataFrame([dict(hypothesis="A2", statistic=f"conventional 90% interval covered {conv:.3f}; event CV with cluster-robust SE covered {ecr:.3f} "
                       f"({reps} replicates)", supported=bool(conv < 0.80 and ecr >= 0.85), fold_scheme="random-row and event", seed=SEED)]
                 ).to_csv(out / "sim_cv_matched_hypotheses.csv", index=False)
    print(f"{reps} replicates; true error on new events: mean {D.err_new.mean():.4f}, SD {D.err_new.std():.4f}; "
          f"event CV mean {D.cv_event.mean():.4f}, random-row CV mean {D.cv_random.mean():.4f}")
    print(C.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"A2: conventional {conv:.3f} (needs < 0.80), event cluster-robust {ecr:.3f} (needs >= 0.85)")


if __name__ == "__main__":
    main()
