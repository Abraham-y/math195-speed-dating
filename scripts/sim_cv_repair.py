"""N3, N4, N5 and the comparison intervals (PREREG.md, Part 4): the error-bar repair, A2 with the events redrawn,
and the two-person design with a target matched to within-rater CV.

Input : data/pairs.csv (the A2 truth model and the real events), data/clean.csv (the two-person design)
Output: results/sim_cv_repair_rows.csv (one row per data set), results/sim_cv_repair_coverage.csv (per design and
        interval), results/sim_cv_repair_dyadic_rows.csv, results/sim_cv_repair_dyadic.csv,
        results/sim_cv_repair_hypotheses.csv
Seeds : grid design d: 20263006 + d; A2 redrawn chunk c: 20263106 + c (its intercept calibrated once with 20263106);
        two-person chunk c: 20263206 + c (its intercept calibrated once with 20263206).

Grid. As sim_cv_designs.py (G in {10, 21, 40}; person SD in {0, 0.5, 1}; event SD in {0, 0.4}; 200 data sets each;
R = 10 nested repetitions), but the events are redrawn in every data set for every G: 10 without replacement, 21
and 40 with replacement, each draw with fresh people. A2 redrawn: the A2 truth (crossed random-effects fit to the
real matches), 21 events drawn with replacement in each of 500 data sets. Target, as in A2: the error on 300 new
events of the model fitted to the data set. Intervals (90%):
- event CV with the naive, cluster-robust, and cluster-nested (root MSE) SEs; the bias-adjusted cluster-nested
  interval; random-row CV with its naive SE (the conventional interval);
- the floor (N3): event CV estimate +/- 1.645 max(cluster-robust SE, cluster-nested root MSE);
- the fallback: the bias-adjusted interval, with the cluster-robust SE when the MSE estimate is not positive;
- the bias-adjusted floor: bias-adjusted centre, SE max(cluster-robust SE, bias-adjusted SE);
- Nadeau-Bengio corrected t on grouped folds: the R x K outer-fold mean losses of the nested run, centre their mean,
  variance (1/(RK) + 1/(K-1)) x their sample variance, t quantile with RK - 1 degrees of freedom;
- cluster bootstrap of grouped CV: B = 50 resamples of events with replacement, copies of an event in one fold,
  SE = SD of the 50 CV estimates, centred at the event CV estimate;
- internal-external CV: leave one event out; per-event error rates with binomial variances (rates clipped to
  [0.5/n_g, 1 - 0.5/n_g]); DerSimonian-Laird random-effects mean and SE;
- grouped nested CV with a row-level variance (as mlr3inferr computes it), bias-adjusted.
The Wald interval of Bayle et al. with events as units is the cluster-robust interval without G/(G-1).
Two-person design: as in sim_cv_designs.py, with the target the expected loss of the CV fold models themselves on
fresh outcomes for their held-out pairs (what within-rater CV estimates); the full-data target is kept alongside.
Run from the repository root: python scripts/sim_cv_repair.py [replicates] [nested repetitions] [workers]
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
import sim_cv_matched as A2
import sim_cv_designs as SD
import yes_extensions as YE

RES = pathlib.Path("results")
GRID_SEED, A2_SEED, DYAD_SEED = 20263006, 20263106, 20263206
Z, K, B_BOOT = 1.645, 5, 50
GRID = SD.GRID
A2_CHUNKS, A2_REPS, DYAD_CHUNKS = 10, 500, 8
PROCS = [  # (name, centre column, SE column, quantile column or None for 1.645)
    ("random-row CV, naive SE (conventional)", "cv_random", "se_random_naive", None),
    ("event CV, naive SE", "cv_event", "se_event_naive", None),
    ("event CV, cluster-robust SE", "cv_event", "se_event_cr", None),
    ("event CV, cluster-nested root MSE", "cv_event", "se_event_cn", None),
    ("event cluster-nested, bias-adjusted", "bht_centre", "bht_se", None),
    ("floor: max(cluster-robust, cluster-nested) (N3)", "cv_event", "se_floor", None),
    ("fallback: bias-adjusted, cluster-robust when MSE <= 0", "bht_centre", "se_fallback", None),
    ("bias-adjusted floor", "bht_centre", "se_bht_floor", None),
    ("Nadeau-Bengio corrected t, grouped folds", "nb_centre", "nb_se", "nb_q"),
    ("cluster bootstrap of grouped CV", "cv_event", "boot_se", None),
    ("internal-external CV, random-effects meta-analysis", "iecv_centre", "iecv_se", None),
    ("grouped nested CV, row-level variance, bias-adjusted", "row_centre", "row_se", None),
]


def redraw_events(X, f, m, e, G, rng):
    """G real events: fewer than the real ones without replacement, otherwise with replacement; fresh people per draw."""
    rows = [np.where(e == k)[0] for k in range(e.max() + 1)]
    pick = rng.choice(len(rows), G, replace=G >= len(rows))
    idx = np.concatenate([rows[k] for k in pick]); block = np.repeat(np.arange(G), [len(rows[k]) for k in pick])
    lab = lambda v: pd.factorize(pd.Series(block).astype(str) + "_" + pd.Series(v[idx]).astype(str))[0]
    return X[idx], lab(f), lab(m), block


def nested_all(X, y, g, R, rng):
    """Grouped nested CV, one pass: the cluster-nested MSE (cluster-robust b_k), the MSE with the row-level b_k, the
    nested error estimate (mean inner CV error), and the R x K outer-fold mean losses (for Nadeau-Bengio)."""
    a, b_cl, b_row, ein, outer = [], [], [], [], []
    for _ in range(R):
        folds = A2.group_folds(g, K, rng)
        for k in range(K):
            out = folds == k
            Xi, yi, gi = X[~out], y[~out], g[~out]
            e_in = A2.cv(Xi, yi, A2.group_folds(pd.factorize(gi)[0], K - 1, rng), K - 1).mean(); ein.append(e_in)
            eo = A2.loss(Xi, yi, X[out], y[out])
            a.append((e_in - eo.mean()) ** 2); outer.append(eo.mean())
            b_cl.append(M.cluster_robust_var_of_mean(eo, g[out])); b_row.append(eo.var(ddof=1) / out.sum())
    return np.mean(a) - np.mean(b_cl), np.mean(a) - np.mean(b_row), np.mean(ein), np.array(outer)


def nadeau_bengio(fold_losses, K=K):
    """Corrected resampled t (Nadeau and Bengio 2003) on J = len(fold_losses) splits with test/train ratio 1/(K-1):
    (centre, SE, t quantile for a two-sided 90% interval)."""
    J = len(fold_losses)
    se = np.sqrt((1 / J + 1 / (K - 1)) * np.var(fold_losses, ddof=1))
    return float(np.mean(fold_losses)), float(se), float(stats.t.ppf(0.95, J - 1))


def bootstrap_folds(orig, K, rng):
    """Folds for a bootstrap sample whose rows come from original events `orig`: all copies of an event in one fold."""
    u = np.unique(orig); perm = rng.permutation(len(u)); to = dict(zip(u[perm], np.arange(len(u)) % K))
    return np.array([to[v] for v in orig])


def cluster_bootstrap_se(X, y, e, B, rng):
    rows = [np.where(e == k)[0] for k in range(e.max() + 1)]
    est = []
    for _ in range(B):
        pick = rng.integers(0, len(rows), len(rows))
        idx = np.concatenate([rows[k] for k in pick]); orig = np.repeat(pick, [len(rows[k]) for k in pick])
        est.append(A2.cv(X[idx], y[idx], bootstrap_folds(orig, K, rng)).mean())
    return float(np.std(est, ddof=1))


def dersimonian_laird(est, var):
    """Random-effects mean and its SE (DerSimonian and Laird 1986), and the between-study variance tau^2."""
    est, var = np.asarray(est, float), np.asarray(var, float)
    w = 1 / var; mean_f = np.sum(w * est) / np.sum(w)
    Q = np.sum(w * (est - mean_f) ** 2)
    tau2 = max(0.0, (Q - (len(est) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (var + tau2)
    return float(np.sum(ws * est) / np.sum(ws)), float(np.sqrt(1 / np.sum(ws))), float(tau2)


def iecv(X, y, e):
    errs, var = [], []
    for k in range(e.max() + 1):
        te = e == k
        rate = A2.loss(X[~te], y[~te], X[te], y[te]).mean(); n = te.sum()
        p = np.clip(rate, 0.5 / n, 1 - 0.5 / n)
        errs.append(rate); var.append(p * (1 - p) / n)
    return dersimonian_laird(errs, var)


def one_dataset(X, y, f, m, e, Xn, yn, R, rng):
    err_new = A2.loss(X, y, Xn, yn).mean()
    eg = A2.cv(X, y, A2.group_folds(e, K, rng)); er = A2.cv(X, y, rng.permutation(len(y)) % K)
    mse_cn, mse_row, ncv, outer = nested_all(X, y, e, R, rng)
    c_cn, s_cn = A2.bht_interval(eg.mean(), ncv, mse_cn, K)
    c_row, s_row = A2.bht_interval(eg.mean(), ncv, mse_row, K)
    se_cr = np.sqrt(M.cluster_robust_var_of_mean(eg, e)); se_cnr = np.sqrt(max(mse_cn, 0))
    nb_c, nb_se, nb_q = nadeau_bengio(outer)
    ie_c, ie_se, tau2 = iecv(X, y, e)
    return dict(dyads=len(y), match_rate=y.mean(), err_new=err_new, cv_event=eg.mean(), cv_random=er.mean(),
                se_event_naive=np.sqrt(eg.var(ddof=1) / len(eg)), se_random_naive=np.sqrt(er.var(ddof=1) / len(er)),
                se_event_cr=se_cr, se_event_cn=se_cnr, mse_cn=mse_cn, mse_row=mse_row, ncv=ncv,
                bht_centre=c_cn, bht_se=s_cn, se_floor=max(se_cr, se_cnr), se_fallback=s_cn if mse_cn > 0 else se_cr,
                se_bht_floor=max(se_cr, s_cn), nb_centre=nb_c, nb_se=nb_se, nb_q=nb_q,
                boot_se=cluster_bootstrap_se(X, y, e, B_BOOT, rng), iecv_centre=ie_c, iecv_se=ie_se, iecv_tau2=tau2,
                row_centre=c_row, row_se=s_row)


def truth_model():
    pairs = pd.read_csv("data/pairs.csv")
    X0, f0, m0, e0 = A2.design(pairs)
    beta_all, sd_fit = A2.fit_truth(pairs, X0, f0, m0, e0)
    return pairs, X0, f0, m0, e0, beta_all, sd_fit


def run_task(task):
    kind, idx, reps, R = task
    t0 = time.time()
    if kind == "dyadic":
        return kind, dyadic_chunk(idx, reps)
    pairs, X0, f0, m0, e0, beta_all, sd_fit = truth_model()
    if kind == "grid":
        G, ps, es = GRID[idx]; sd = SD.sd_of(ps, es); seed = GRID_SEED + idx
        rng = np.random.default_rng(seed)
        beta0 = A2.calibrate_intercept(beta_all[0], beta_all[1:], sd, X0, f0, m0, e0, pairs.match.mean(), rng)
        label = dict(design=f"grid-{idx}", events=G, person_sd=ps, event_sd=es)
    else:
        G, sd, seed = 21, sd_fit, A2_SEED + idx
        beta0 = A2.calibrate_intercept(beta_all[0], beta_all[1:], sd, X0, f0, m0, e0, pairs.match.mean(), np.random.default_rng(A2_SEED))
        rng = np.random.default_rng(seed)
        label = dict(design="A2 redrawn", events=21, person_sd=np.nan, event_sd=np.nan, chunk=idx)
    rows = []
    for r in range(reps):
        X, f, m, e = redraw_events(X0, f0, m0, e0, G, rng)
        y = A2.draw(beta0, beta_all[1:], sd, X, f, m, e, rng)
        Xn, fn, mn, en = A2.new_events(pairs, X0, f0, m0, e0, rng)
        yn = A2.draw(beta0, beta_all[1:], sd, Xn, fn, mn, en, rng)
        rows.append(dict(label, rep=r, intercept=beta0, seed=seed, nested_repetitions=R, **one_dataset(X, y, f, m, e, Xn, yn, R, rng)))
    print(f"{kind} {idx}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return kind, rows


def dyadic_chunk(c, reps):
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    F = SD.dyadic_frame(clean)
    people = np.unique(np.r_[F.rater, F.ratee]); pi = {p: k for k, p in enumerate(people)}
    ra, rb = F.rater.map(pi).to_numpy(), F.ratee.map(pi).to_numpy()
    dy = pd.factorize(pd.Series(np.minimum(ra, rb)).astype(str) + "_" + pd.Series(np.maximum(ra, rb)).astype(str))[0]
    X = YE.indicator_design(F.rater.to_numpy(), F.ratee.to_numpy())
    effects = lambda g: g.normal(size=len(people))[ra] + g.normal(size=len(people))[rb] + g.normal(size=dy.max() + 1)[dy]
    cal = np.random.default_rng(DYAD_SEED); draws = [effects(cal) for _ in range(50)]
    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.mean([np.mean(1 / (1 + np.exp(-(mid + u)))) for u in draws]) < clean.dec.mean() else (lo, mid)
    icpt, rng, rows, t0 = (lo + hi) / 2, np.random.default_rng(DYAD_SEED + c), [], time.time()
    for r in range(reps):
        p = 1 / (1 + np.exp(-(icpt + effects(rng))))
        y = (rng.random(len(p)) < p).astype(float)
        folds = YE.within_rater_folds(F.rater.to_numpy(), K, rng)
        lossv, yhat_cv = np.empty(len(y)), np.empty(len(y))
        for k in range(K):
            te = folds == k
            w, _ = M.fit_logit(X[~te], y[~te], ridge=1.0, iters=200)
            yhat_cv[te] = (M.predict_logit(w, X[te]) > y[~te].mean()).astype(float)
            lossv[te] = (yhat_cv[te] != y[te]).astype(float)
        w, _ = M.fit_logit(X, y, ridge=1.0, iters=200)
        yhat = (M.predict_logit(w, X) > y.mean()).astype(float)
        rows.append(dict(chunk=c, rep=r, seed=DYAD_SEED + c, intercept=icpt, rows=len(y), yes_rate=y.mean(), cv=lossv.mean(),
                         target_matched=np.mean(yhat_cv * (1 - p) + (1 - yhat_cv) * p), target_full=np.mean(yhat * (1 - p) + (1 - yhat) * p),
                         se_naive=np.sqrt(lossv.var(ddof=1) / len(y)), se_rater=np.sqrt(M.cluster_robust_var_of_mean(lossv, ra)),
                         se_ratee=np.sqrt(M.cluster_robust_var_of_mean(lossv, rb)), se_two_way=np.sqrt(max(SD.two_way_var_of_mean(lossv, ra, rb), 0)),
                         se_dyadic=np.sqrt(max(SD.dyadic_var_of_mean(lossv, ra, rb), 0))))
    print(f"dyadic {c}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def coverage(D):
    out = []
    for d, g in D.groupby("design", sort=False):
        for name, centre, se, q in PROCS:
            zq = g[q] if q else Z
            out.append(dict(design=d, events=g.events.iloc[0], person_sd=g.person_sd.iloc[0], event_sd=g.event_sd.iloc[0], procedure=name,
                            coverage_90=float(np.mean(np.abs(g[centre] - g.err_new) <= zq * g[se])), mean_half_width=float(np.mean(zq * g[se])),
                            bias=float((g[centre] - g.err_new).mean()), reps=len(g), share_mse_not_positive=float((g.mse_cn <= 0).mean())))
    return pd.DataFrame(out)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, (os.cpu_count() or 2) - 1)
    a2_per = max(1, reps * A2_REPS // 200 // A2_CHUNKS)
    tasks = ([("dyadic", c, max(1, reps // DYAD_CHUNKS), R) for c in range(DYAD_CHUNKS)]
             + [("grid", d, reps, R) for d in sorted(range(len(GRID)), key=lambda d: -GRID[d][0])]
             + [("a2", c, a2_per, R) for c in range(A2_CHUNKS)])
    with Pool(workers) as pool:
        res = pool.map(run_task, tasks, chunksize=1)
    D = pd.DataFrame([r for kind, rows in res if kind != "dyadic" for r in rows])
    DD = pd.DataFrame([r for kind, rows in res if kind == "dyadic" for r in rows])
    D.to_csv(RES / "sim_cv_repair_rows.csv", index=False); DD.to_csv(RES / "sim_cv_repair_dyadic_rows.csv", index=False)
    C = coverage(D); C.to_csv(RES / "sim_cv_repair_coverage.csv", index=False)
    dyad = []
    for tname, tcol in (("matched: the CV fold models' own expected loss", "target_matched"), ("full-data model's expected loss", "target_full")):
        for name, col in (("naive", "se_naive"), ("rater-clustered", "se_rater"), ("ratee-clustered", "se_ratee"),
                          ("two-way (rater, ratee)", "se_two_way"), ("dyadic-robust", "se_dyadic")):
            dyad.append(dict(target=tname, procedure=f"within-rater CV, {name} SE", coverage_90=float(np.mean(np.abs(DD.cv - DD[tcol]) <= Z * DD[col])),
                             mean_se=float(DD[col].mean()), sd_cv_minus_target=float((DD.cv - DD[tcol]).std(ddof=1)), bias=float((DD.cv - DD[tcol]).mean()),
                             reps=len(DD), fold_scheme="within-rater random folds"))
    DC = pd.DataFrame(dyad); DC.to_csv(RES / "sim_cv_repair_dyadic.csv", index=False)
    grid = C[C.design.str.startswith("grid") & ((C.person_sd > 0) | (C.event_sd > 0))]
    fl = grid[grid.procedure.str.startswith("floor")]
    a2 = C[C.design == "A2 redrawn"].set_index("procedure").coverage_90
    cr, cn = a2["event CV, cluster-robust SE"], a2["event cluster-nested, bias-adjusted"]
    hyp = pd.DataFrame([
        dict(hypothesis="N3", statistic=f"floor interval coverage {fl.coverage_90.min():.3f} to {fl.coverage_90.max():.3f} over {len(fl)} designs "
             f"with events redrawn; {int((fl.coverage_90 >= 0.85).sum())} of {len(fl)} at least 0.85", supported=bool((fl.coverage_90 >= 0.85).all())),
        dict(hypothesis="N4", statistic=f"A2 with the 21 events redrawn: event CV with the cluster-robust SE covers {cr:.3f} ({len(D[D.design == 'A2 redrawn'])} data sets)",
             supported=bool(0.85 <= cr <= 0.95)),
        dict(hypothesis="N5", statistic=f"A2 with the 21 events redrawn: the cluster-nested, bias-adjusted interval covers {cn:.3f}, against 0.926 in A2",
             supported=bool(cn < 0.926)),
    ]).assign(seed=f"grid {GRID_SEED} + design; A2 {A2_SEED} + chunk", nested_repetitions=R, fold_scheme="event and random-row")
    hyp.to_csv(RES / "sim_cv_repair_hypotheses.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        print(C.pivot_table(index=["design"], columns="procedure", values="coverage_90", sort=False).round(3).T.to_string())
        print(DC.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
