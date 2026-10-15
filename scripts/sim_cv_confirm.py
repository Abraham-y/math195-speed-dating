"""N6 and N7 (PREREG.md, Part 5): a confirmatory test of the Nadeau-Bengio interval on event folds, and Proposition
6's prediction that on random-row folds it fails under clustering (proposals/06, section 6c).

Input : data/pairs.csv (the A2 truth model and the real events)
Output: results/sim_cv_confirm_rows.csv (one row per data set), results/sim_cv_confirm_coverage.csv,
        results/sim_cv_confirm_hypotheses.csv
Seeds : design d (0 to 11): 20263406 + d.

Designs: the A2 truth with G in {10, 21, 40}, person SD in {0.5, 1.0}, event SD in {0, 0.4}; 200 data sets each,
events redrawn every time as in sim_cv_repair.py; R = 10 repetitions of 5 folds. Target: the error of the
replicate's model on 300 new events. Intervals (90%):
- Nadeau-Bengio t on event folds: the R x K outer-fold mean losses of the grouped nested run, centred at their
  mean, variance (1/J + 1/(K-1)) S^2, t quantile with J - 1 degrees of freedom (N6);
- the same on random-row folds: R random-row partitions into K folds, each fold's model trained on the other
  four (N7);
- the floor (Part 4), the cluster-robust SE, CR2 (each event's residual sum divided by sqrt(1 - n_g/n), t
  quantile on G - 1 degrees of freedom), and the precision-weighted combination of the cluster-robust variance and
  the cluster-nested MSE (lambda = Var(MSE_r) / (Var(VCR_r) + Var(MSE_r)) across the R repetitions, MSE floored at 0).
Also recorded: the Nadeau-Bengio variance divided by (J+K-1)/(J-1) times the mean cluster-robust variance of the R
repetitions' losses, which is the training-set share by Proposition 6 (1 = no training-set variability).
Run from the repository root: python scripts/sim_cv_confirm.py [replicates] [nested repetitions] [workers]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib, time, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "scripts")
import modeling as M
import sim_cv_matched as A2
import sim_cv_designs as SD
import sim_cv_repair as SR

RES = pathlib.Path("results")
SEED, Z, K = 20263406, 1.645, 5
GRID = list(itertools.product((10, 21, 40), (0.5, 1.0), (0.0, 0.4)))
PROCS = [  # (name, centre, SE, quantile column or None for 1.645)
    ("Nadeau-Bengio t, event folds (N6)", "nb_centre", "nb_se", "nb_q"),
    ("Nadeau-Bengio t, random-row folds (N7)", "nbr_centre", "nbr_se", "nbr_q"),
    ("floor: max(cluster-robust, cluster-nested)", "cv_event", "se_floor", None),
    ("event CV, cluster-robust SE", "cv_event", "se_cr", None),
    ("event CV, CR2 with t(G-1)", "cv_event", "se_cr2", "cr2_q"),
    ("precision-weighted cluster-robust and cluster-nested", "cv_event", "se_weighted", None),
    ("event cluster-nested, bias-adjusted", "bht_centre", "bht_se", None),
]


def cr2_var_of_mean(u, g):
    """CR2 (Bell and McCaffrey) variance of a mean: each cluster's residual sum divided by sqrt(1 - n_g/n); equal to
    the G/(G-1) cluster-robust variance when the clusters are balanced."""
    u, g = np.asarray(u, float), np.asarray(g); n = len(u); r = u - u.mean()
    s = pd.Series(r).groupby(g).agg(["sum", "size"])
    return float(np.sum(s["sum"] ** 2 / (1 - s["size"] / n)) / n ** 2)


def precision_weighted(v_cr, mse, v_cr_reps, mse_reps):
    """lambda V_CR + (1 - lambda) max(MSE, 0), lambda = Var(MSE_r) / (Var(V_CR,r) + Var(MSE_r)) across repetitions."""
    a, b = np.var(v_cr_reps, ddof=1), np.var(mse_reps, ddof=1)
    lam = b / (a + b) if a + b > 0 else 0.5
    return lam * v_cr + (1 - lam) * max(mse, 0.0), lam


def nested_grouped(X, y, g, R, rng):
    """Grouped nested CV with per-repetition pieces: (MSE, nested error, outer-fold mean losses, per-repetition
    cluster-robust variances of the outer losses, per-repetition MSE estimates)."""
    a_all, b_all, ein, outer, vcr_r, mse_r = [], [], [], [], [], []
    for _ in range(R):
        folds = A2.group_folds(g, K, rng); a, b, losses = [], [], np.empty(len(y))
        for k in range(K):
            out = folds == k
            Xi, yi, gi = X[~out], y[~out], g[~out]
            e_in = A2.cv(Xi, yi, A2.group_folds(pd.factorize(gi)[0], K - 1, rng), K - 1).mean(); ein.append(e_in)
            eo = A2.loss(Xi, yi, X[out], y[out]); losses[out] = eo
            a.append((e_in - eo.mean()) ** 2); b.append(M.cluster_robust_var_of_mean(eo, g[out])); outer.append(eo.mean())
        a_all += a; b_all += b
        vcr_r.append(M.cluster_robust_var_of_mean(losses, g)); mse_r.append(np.mean(a) - np.mean(b))
    return np.mean(a_all) - np.mean(b_all), np.mean(ein), np.array(outer), np.array(vcr_r), np.array(mse_r)


def row_fold_means(X, y, R, rng):
    out = []
    for _ in range(R):
        f = rng.permutation(len(y)) % K
        out += [A2.loss(X[f != k], y[f != k], X[f == k], y[f == k]).mean() for k in range(K)]
    return np.array(out)


def run_design(args):
    d, reps, R = args
    G, ps, es = GRID[d]; sd = SD.sd_of(ps, es); seed = SEED + d
    rng = np.random.default_rng(seed)
    pairs, X0, f0, m0, e0, beta_all, _ = SR.truth_model()
    beta0 = A2.calibrate_intercept(beta_all[0], beta_all[1:], sd, X0, f0, m0, e0, pairs.match.mean(), rng)
    rows, t0 = [], time.time()
    for r in range(reps):
        X, f, m, e = SR.redraw_events(X0, f0, m0, e0, G, rng)
        y = A2.draw(beta0, beta_all[1:], sd, X, f, m, e, rng)
        Xn, fn, mn, en = A2.new_events(pairs, X0, f0, m0, e0, rng)
        err_new = A2.loss(X, y, Xn, A2.draw(beta0, beta_all[1:], sd, Xn, fn, mn, en, rng)).mean()
        eg = A2.cv(X, y, A2.group_folds(e, K, rng))
        mse, ncv, outer, vcr_r, mse_r = nested_grouped(X, y, e, R, rng)
        nb_c, nb_se, nb_q = SR.nadeau_bengio(outer)
        nbr_c, nbr_se, nbr_q = SR.nadeau_bengio(row_fold_means(X, y, R, rng))
        v_cr = M.cluster_robust_var_of_mean(eg, e); v_w, lam = precision_weighted(v_cr, mse, vcr_r, mse_r)
        c_b, s_b = A2.bht_interval(eg.mean(), ncv, mse, K)
        J = len(outer); factor = (J + K - 1) / (J - 1)
        rows.append(dict(design=d, events=G, person_sd=ps, event_sd=es, rep=r, seed=seed, intercept=beta0, nested_repetitions=R,
                         dyads=len(y), err_new=err_new, cv_event=eg.mean(), se_cr=np.sqrt(v_cr),
                         se_cr2=np.sqrt(cr2_var_of_mean(eg, e)), cr2_q=stats.t.ppf(0.95, e.max()), mse_cn=mse,
                         se_floor=max(np.sqrt(v_cr), np.sqrt(max(mse, 0))), se_weighted=np.sqrt(v_w), lambda_weight=lam,
                         bht_centre=c_b, bht_se=s_b, nb_centre=nb_c, nb_se=nb_se, nb_q=nb_q, nbr_centre=nbr_c, nbr_se=nbr_se, nbr_q=nbr_q,
                         nb_over_scaled_cr=nb_se ** 2 / (factor * np.mean(vcr_r))))
    print(f"design {d} (G {G}, person SD {ps}, event SD {es}): {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def coverage(D):
    out = []
    for d, g in D.groupby("design"):
        for name, centre, se, q in PROCS:
            zq = g[q] if q else Z
            out.append(dict(design=d, events=g.events.iloc[0], person_sd=g.person_sd.iloc[0], event_sd=g.event_sd.iloc[0], procedure=name,
                            coverage_90=float(np.mean(np.abs(g[centre] - g.err_new) <= zq * g[se])),
                            mean_half_width=float(np.mean(zq * g[se])), bias=float((g[centre] - g.err_new).mean()), reps=len(g),
                            nb_over_scaled_cr_median=float(g.nb_over_scaled_cr.median()), seed=int(g.seed.iloc[0])))
    return pd.DataFrame(out)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    with Pool(workers) as pool:
        parts = pool.map(run_design, [(d, reps, R) for d in sorted(range(len(GRID)), key=lambda d: -GRID[d][0])], chunksize=1)
    D = pd.DataFrame([r for p in parts for r in p]).sort_values(["design", "rep"])
    D.to_csv(RES / "sim_cv_confirm_rows.csv", index=False)
    C = coverage(D); C.to_csv(RES / "sim_cv_confirm_coverage.csv", index=False)
    nb = C[C.procedure.str.endswith("(N6)")].set_index("design"); nbr = C[C.procedure.str.endswith("(N7)")].set_index("design")
    fl = C[C.procedure.str.startswith("floor")].set_index("design")
    narrower = nb.mean_half_width <= 0.9 * fl.mean_half_width
    inside = nb.coverage_90.between(0.85, 0.95)
    hyp = pd.DataFrame([
        dict(hypothesis="N6", statistic=f"Nadeau-Bengio on event folds covers {nb.coverage_90.min():.3f} to {nb.coverage_90.max():.3f} over 12 designs "
             f"({int(inside.sum())} of 12 within 0.85 to 0.95); half-width {(nb.mean_half_width / fl.mean_half_width).min():.2f} to "
             f"{(nb.mean_half_width / fl.mean_half_width).max():.2f} times the floor's ({int(narrower.sum())} of 12 at most 0.90)",
             supported=bool(inside.all() and narrower.all())),
        dict(hypothesis="N7", statistic=f"Nadeau-Bengio on random-row folds covers {nbr.coverage_90.min():.3f} to {nbr.coverage_90.max():.3f} over 12 designs; "
             f"{int((nbr.coverage_90 < 0.80).sum())} of 12 below 0.80", supported=bool((nbr.coverage_90 < 0.80).all())),
    ]).assign(seed=f"{SEED} + design index", nested_repetitions=R, fold_scheme="event and random-row")
    hyp.to_csv(RES / "sim_cv_confirm_hypotheses.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(C.pivot_table(index=["events", "person_sd", "event_sd"], columns="procedure", values="coverage_90").round(3).to_string())
        print(C.pivot_table(index=["events", "person_sd", "event_sd"], columns="procedure", values="mean_half_width").round(4).to_string())
        print(C.groupby(["events", "person_sd", "event_sd"]).nb_over_scaled_cr_median.first().round(3).to_string())
        print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
