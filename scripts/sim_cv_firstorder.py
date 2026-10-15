"""N8 to N11 (PREREG.md, Part 6): tests of Proposition 8 (proposals/06, section 6e) on balanced synthetic clusters.

Input : none (synthetic data)
Output: results/sim_cv_firstorder_rows.csv (one row per data set), results/sim_cv_firstorder_summary.csv (per G
        and loss), results/sim_cv_firstorder_hypotheses.csv
Seeds : 20264106 + G for G in {20, 40, 80, 160, 320}.

Each data set has G clusters of 40 rows: x = u_g + v_i (u_g, v_i independent N(0, I_3)),
y ~ Bernoulli(logistic(-1.5 + x'(1, 0.5, 0) + a_g)) with a_g ~ N(0, 1). The model is a logistic regression on x
(ridge 1e-6). Two evaluation losses: 0-1 with the threshold at the training positive rate (the project's rule),
and log-loss (the training criterion, so Proposition 8 (iv) applies).
- One grouped 5-fold partition gives the CV estimate and the cluster-robust variance of its per-row losses
  (balanced clusters, so CR2 equals it; t quantile on G - 1 degrees of freedom).
- R = 10 further grouped partitions give the Nadeau-Bengio variance (J = 50).
- Err_new: the full-data model's mean loss on 2,000 new clusters of 40 rows.
Per G and loss: Var(CV - Err_new) / mean cluster-robust variance (N8), mean Nadeau-Bengio / ((J+K-1)/(J-1) x
cluster-robust) (N9), Corr(CV, Err_new) (N10), coverage of Err_new by the cluster-robust t interval (N11), plus
coverage of the average error over data sets (exploratory).
Part 6b (N8b): `--new-clusters 20000 --seed-base 20264506 --tag b` measures Err_new on 20,000 new clusters, records
its Monte Carlo variance (the cluster-robust variance of the new-sample mean loss), and writes the _b outputs; the
corrected ratio subtracts that variance from Var(CV - Err_new).
Run from the repository root: python scripts/sim_cv_firstorder.py [replicates] [workers] [--new-clusters N] [--seed-base S] [--tag T]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")
SEED, K, R, M_ROWS, NEW_CLUSTERS = 20264106, 5, 10, 40, 2000
GS = (20, 40, 80, 160, 320)
BETA, B0 = np.array([1.0, 0.5, 0.0]), -1.5


def draw(G, rng):
    g = np.repeat(np.arange(G), M_ROWS)
    X = rng.normal(size=(G, 3))[g] + rng.normal(size=(G * M_ROWS, 3))
    eta = B0 + X @ BETA + rng.normal(size=G)[g]
    return X, (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float), g


def losses(w, rate, X, y):
    """Per-row 0-1 loss (predict 1 when the probability exceeds the training positive rate) and log-loss."""
    p = np.clip(M.predict_logit(w, X), 1e-6, 1 - 1e-6)
    return ((p > rate).astype(float) != y).astype(float), -(y * np.log(p) + (1 - y) * np.log(1 - p))


def grouped_folds(G, rng):
    to = np.empty(G, int); to[rng.permutation(G)] = np.arange(G) % K
    return to


def cv_losses(X, y, g, folds_of_cluster):
    f = folds_of_cluster[g]; l01, lll = np.empty(len(y)), np.empty(len(y))
    for k in range(K):
        te = f == k
        w, _ = M.fit_logit(X[~te], y[~te])
        l01[te], lll[te] = losses(w, y[~te].mean(), X[te], y[te])
    return l01, lll


def nb_variance(fold_means):
    J = len(fold_means)
    return (1 / J + 1 / (K - 1)) * np.var(fold_means, ddof=1), np.mean(fold_means)


def one(G, rng, new_clusters=NEW_CLUSTERS):
    X, y, g = draw(G, rng)
    l01, lll = cv_losses(X, y, g, grouped_folds(G, rng))
    fm01, fmll = [], []
    for _ in range(R):
        f = grouped_folds(G, rng)[g]
        for k in range(K):
            te = f == k
            w, _ = M.fit_logit(X[~te], y[~te])
            a, b = losses(w, y[~te].mean(), X[te], y[te]); fm01.append(a.mean()); fmll.append(b.mean())
    w, _ = M.fit_logit(X, y)
    Xn, yn, gn = draw(new_clusters, rng)
    e01, ell = losses(w, y.mean(), Xn, yn)
    out = dict(G=G)
    for name, lv, fm, en in (("01", l01, fm01, e01), ("ll", lll, fmll, ell)):
        nbv, nbc = nb_variance(np.array(fm))
        out.update({f"cv_{name}": lv.mean(), f"vcr_{name}": M.cluster_robust_var_of_mean(lv, g), f"nb_var_{name}": nbv,
                    f"nb_centre_{name}": nbc, f"err_new_{name}": en.mean(), f"v_mc_{name}": M.cluster_robust_var_of_mean(en, gn)})
    return out


def run_G(args):
    G, reps, new_clusters, seed = args
    rng = np.random.default_rng(seed + G); t0 = time.time()
    rows = [dict(one(G, rng, new_clusters), rep=r, seed=seed + G, new_clusters=new_clusters) for r in range(reps)]
    print(f"G {G}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def summarize(D):
    J = R * K; factor = (J + K - 1) / (J - 1); out = []
    for (G, ), d in D.groupby(["G"]):
        q = stats.t.ppf(0.95, G - 1)
        for name, label in (("01", "0-1 at the training rate"), ("ll", "log-loss")):
            cv, t, v, nb = d[f"cv_{name}"], d[f"err_new_{name}"], d[f"vcr_{name}"], d[f"nb_var_{name}"]
            avg = t.mean()
            vmc = d[f"v_mc_{name}"].mean()
            out.append(dict(G=G, loss=label, reps=len(d), ratio_var_diff_to_cr=float((cv - t).var(ddof=1) / v.mean()),
                            ratio_corrected=float(((cv - t).var(ddof=1) - vmc) / v.mean()), mc_var_share=float(vmc / (cv - t).var(ddof=1)),
                            nb_over_scaled_cr=float((nb / (factor * v)).mean()), corr_cv_err_new=float(np.corrcoef(cv, t)[0, 1]),
                            coverage_cr_t=float(np.mean(np.abs(cv - t) <= q * np.sqrt(v))),
                            coverage_cr_z=float(np.mean(np.abs(cv - t) <= 1.645 * np.sqrt(v))),
                            coverage_nb_t=float(np.mean(np.abs(d[f"nb_centre_{name}"] - t) <= stats.t.ppf(0.95, J - 1) * np.sqrt(nb))),
                            coverage_cr_t_average_error=float(np.mean(np.abs(cv - avg) <= q * np.sqrt(v))),
                            sd_cv=float(cv.std(ddof=1)), sd_err_new=float(t.std(ddof=1)), bias=float((cv - t).mean()), seed=int(d.seed.iloc[0])))
    return pd.DataFrame(out)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("reps", nargs="?", type=int, default=400); ap.add_argument("workers", nargs="?", type=int, default=5)
    ap.add_argument("--new-clusters", type=int, default=NEW_CLUSTERS); ap.add_argument("--seed-base", type=int, default=SEED)
    ap.add_argument("--tag", default=""); args = ap.parse_args()
    reps, sfx = args.reps, (f"_{args.tag}" if args.tag else "")
    with Pool(args.workers) as pool:
        parts = pool.map(run_G, [(G, reps, args.new_clusters, args.seed_base) for G in sorted(GS, reverse=True)], chunksize=1)
    D = pd.DataFrame([r for p in parts for r in p]).sort_values(["G", "rep"])
    D.to_csv(RES / f"sim_cv_firstorder_rows{sfx}.csv", index=False)
    S = summarize(D); S.to_csv(RES / f"sim_cv_firstorder_summary{sfx}.csv", index=False)
    s = S.set_index(["loss", "G"])
    big = [80, 160, 320]
    fmt = lambda v: ", ".join(f"{x:.3f}" for x in v)
    losses_ = ("0-1 at the training rate", "log-loss")
    n9_ll = [s.loc[("log-loss", G)].nb_over_scaled_cr for G in (160, 320)]
    n9_01 = [s.loc[("0-1 at the training rate", G)].nb_over_scaled_cr for G in GS]
    n10 = [s.loc[("log-loss", G)].corr_cv_err_new for G in big]
    n11 = list(S.coverage_cr_t)
    rows = []
    if args.tag == "b":
        n8b = [s.loc[(l, G)].ratio_corrected for l in losses_ for G in big]
        rows.append(dict(hypothesis="N8b", statistic=f"corrected Var(CV - Err_new) / mean V_CR at G = 80, 160, 320: 0-1 loss {fmt(n8b[:3])}; "
                         f"log-loss {fmt(n8b[3:])}", supported=bool(all(0.9 <= v <= 1.1 for v in n8b))))
    else:
        n8 = [s.loc[(l, G)].ratio_var_diff_to_cr for l in losses_ for G in big]
        rows.append(dict(hypothesis="N8", statistic=f"Var(CV - Err_new) / mean V_CR at G = 80, 160, 320: 0-1 loss {fmt(n8[:3])}; log-loss {fmt(n8[3:])}",
                         supported=bool(all(0.85 <= v <= 1.15 for v in n8))))
    tag = " (replication, exploratory)" if args.tag == "b" else ""
    rows += [dict(hypothesis="N9" + tag, statistic=f"Nadeau-Bengio / scaled V_CR: log-loss at G = 160, 320 {fmt(n9_ll)}; 0-1 loss at G = 20 to 320 {fmt(n9_01)}",
                  supported=bool(all(0.9 <= v <= 1.1 for v in n9_ll) and all(v > 1.1 for v in n9_01))),
             dict(hypothesis="N10" + tag, statistic=f"log-loss Corr(CV, Err_new) at G = 80, 160, 320: {fmt(n10)}", supported=bool(all(v < 0.2 for v in n10))),
             dict(hypothesis="N11" + tag, statistic=f"cluster-robust t interval coverage of Err_new over 5 G x 2 losses: {min(n11):.3f} to {max(n11):.3f}",
                  supported=bool(all(0.85 <= v <= 0.95 for v in n11)))]
    hyp = pd.DataFrame(rows).assign(reps=reps, seed=f"{args.seed_base} + G", new_clusters=args.new_clusters, fold_scheme="grouped (cluster) 5-fold")
    hyp.to_csv(RES / f"sim_cv_firstorder_hypotheses{sfx}.csv", index=False)
    with pd.option_context("display.width", 250):
        print(S.drop(columns=["seed"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
