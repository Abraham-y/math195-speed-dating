"""N1, N2 and the dyadic design (PREREG.md, Part 3): coverage of cross-validation intervals across simulated designs.

Input : data/pairs.csv (the A2 truth model and the real events), data/clean.csv (the dyadic design's events)
Output: results/sim_cv_designs_rows.csv (one row per design and replicate), results/sim_cv_designs_coverage.csv,
        results/sim_cv_designs_dyadic.csv, results/sim_cv_designs_dyadic_rows.csv, results/sim_cv_designs_hypotheses.csv,
        results/sim_cv_designs_diagnostics.csv (post hoc, added after the run: see below)
Seeds : design d uses 20262806 + d (d = 0, ..., 17); the dyadic design 20262906.

Design grid. The A2 truth model of sim_cv_matched.py (the four pre-date predictors, coefficients from the crossed
random-effects fit to the real matches) with: G events in {10, 21, 40} (21: the real events; 10: ten real events
drawn without replacement in each replicate; 40: forty drawn with replacement, each draw with fresh people);
person-effect SD (women and men alike) in {0, 0.5, 1.0}; event-effect SD in {0, 0.4}. The intercept is set for
each (person SD, event SD) so the average simulated match rate on the real design equals the real rate. Each
replicate runs exactly the A2 procedures (event and random-row 5-fold CV, the naive, cluster-robust and nested
intervals, both bias adjustments) and the target is the error of the model fitted to the replicate on 300 new
events. N1: the cluster-nested, bias-adjusted interval covers 85% to 95% in all 15 designs with person or event
effects. N2: the Bates, Hastie and Tibshirani bias-adjusted nested interval with random-row folds covers below 80%
in all 12 designs with person SD 0.5 or 1.

Dyadic design (exploratory). The real events with every woman-man pair observed in both directions (one row per
date from one side). yes ~ Bernoulli(logistic(c + rater effect + ratee effect + pair effect)): rater and ratee
effects of SD 1, and a pair effect of SD 1 shared by the two directions of a dyad (latent residual correlation
1 / (1 + pi^2 / 3) = 0.23, near the 0.21 dyadic reciprocity of liking in srm.py); c sets the yes rate to the real
one. Within-rater random 5-fold CV of a logistic regression with rater and ratee indicators (ridge 1, as W1),
predicting yes when the probability exceeds the training yes rate. Target: the full-data model's expected loss on
fresh outcomes for the same pairs (same effects, fresh Bernoulli draws). SEs of the CV error: naive, clustered by
rater, by ratee, two-way (Cameron, Gelbach and Miller: V_rater + V_ratee - V_row), and dyadic-robust (any two
rows that share a person; Aronow, Samii and Assenova 2015).
Diagnostics (post hoc, added 2026-10-15 after N1 failed; no hypothesis rests on them): per design, the share of data
sets whose nested MSE estimate is zero or negative (a zero-width interval), the coverage of the bias-adjusted interval
among the others, the correlation of the CV estimate with the target, and the coverage of the floor
max(cluster-robust, cluster-nested) of proposals/06; for the dyadic design, the spread of the CV estimate across data
sets against each mean SE. `--diagnostics` recomputes them from the saved rows without rerunning the simulation.
Run from the repository root: python scripts/sim_cv_designs.py [replicates] [nested repetitions] [workers]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib, time, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import sim_cv_matched as A2
import yes_extensions as YE

RES = pathlib.Path("results")
SEED, DYADIC_SEED, Z, K = 20262806, 20262906, 1.645, 5
GRID = list(itertools.product((10, 21, 40), (0.0, 0.5, 1.0), (0.0, 0.4)))      # (events, person SD, event SD)
PROCS = [("random-row CV, naive SE (conventional)", "cv_random", "se_random_naive"),
         ("random-row nested (Bates et al.), unadjusted", "cv_random", "se_random_nested"),
         ("random-row nested, bias-adjusted (Bates et al.)", "bht_random_centre", "bht_random_se"),
         ("event CV, naive SE", "cv_event", "se_event_naive"),
         ("event CV, cluster-robust SE", "cv_event", "se_event_cr"),
         ("event CV, cluster-nested", "cv_event", "se_event_cluster_nested"),
         ("event cluster-nested, bias-adjusted", "bht_event_centre", "bht_event_se")]


def sd_of(person, event):
    return {"woman": person, "man": person, "event": event}


def resample_events(X, f, m, e, G, rng):
    """G real events (all 21 when G = 21; drawn without replacement when fewer, with replacement when more), each
    draw with its own people."""
    if G == e.max() + 1:
        return X, f, m, e
    rows = [np.where(e == k)[0] for k in range(e.max() + 1)]
    pick = rng.choice(len(rows), G, replace=G > len(rows))
    idx = np.concatenate([rows[k] for k in pick]); block = np.repeat(np.arange(G), [len(rows[k]) for k in pick])
    lab = lambda v: pd.factorize(pd.Series(block).astype(str) + "_" + pd.Series(v[idx]).astype(str))[0]
    return X[idx], lab(f), lab(m), block


def run_design(args):
    d, reps, R = args
    G, ps, es = GRID[d]
    rng = np.random.default_rng(SEED + d)
    pairs = pd.read_csv("data/pairs.csv")
    X0, f0, m0, e0 = A2.design(pairs)
    beta_all, _ = A2.fit_truth(pairs, X0, f0, m0, e0)
    sd = sd_of(ps, es)
    beta0 = A2.calibrate_intercept(beta_all[0], beta_all[1:], sd, X0, f0, m0, e0, pairs.match.mean(), rng)
    beta, rows, t0 = beta_all[1:], [], time.time()
    for r in range(reps):
        X, f, m, e = resample_events(X0, f0, m0, e0, G, rng)
        y = A2.draw(beta0, beta, sd, X, f, m, e, rng)
        Xn, fn, mn, en = A2.new_events(pairs, X0, f0, m0, e0, rng)
        err_new = A2.loss(X, y, Xn, A2.draw(beta0, beta, sd, Xn, fn, mn, en, rng)).mean()
        eg = A2.cv(X, y, A2.group_folds(e, K, rng)); er = A2.cv(X, y, rng.permutation(len(y)) % K)
        mse_cn, ncv_cn = A2.nested_mse(X, y, e, K, rng, True, R); mse_rn, ncv_rn = A2.nested_mse(X, y, e, K, rng, False, R)
        c_cn, s_cn = A2.bht_interval(eg.mean(), ncv_cn, mse_cn, K); c_rn, s_rn = A2.bht_interval(er.mean(), ncv_rn, mse_rn, K)
        rows.append(dict(design=d, events=G, person_sd=ps, event_sd=es, intercept=beta0, rep=r, dyads=len(y), err_new=err_new,
                         match_rate=y.mean(), cv_event=eg.mean(), se_event_naive=np.sqrt(eg.var(ddof=1) / len(eg)),
                         se_event_cr=np.sqrt(M.cluster_robust_var_of_mean(eg, e)), se_event_cluster_nested=np.sqrt(max(mse_cn, 0)),
                         cv_random=er.mean(), se_random_naive=np.sqrt(er.var(ddof=1) / len(er)), se_random_nested=np.sqrt(max(mse_rn, 0)),
                         bht_event_centre=c_cn, bht_event_se=s_cn, bht_random_centre=c_rn, bht_random_se=s_rn, seed=SEED + d, nested_repetitions=R))
    print(f"design {d} (G {G}, person SD {ps}, event SD {es}): {reps} replicates in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def coverage(D):
    out = []
    for d, g in D.groupby("design"):
        for name, est, se in PROCS:
            out.append(dict(design=d, events=g.events.iloc[0], person_sd=g.person_sd.iloc[0], event_sd=g.event_sd.iloc[0], procedure=name,
                            coverage_90=np.mean(np.abs(g[est] - g.err_new) <= Z * g[se]), mean_half_width=Z * g[se].mean(),
                            bias=(g[est] - g.err_new).mean(), sd_target=g.err_new.std(ddof=1), reps=len(g), seed=g.seed.iloc[0]))
    return pd.DataFrame(out)


# ---- dyadic design --------------------------------------------------------------------------------------------

def two_way_var_of_mean(e, a, b):
    """Cameron, Gelbach and Miller two-way cluster-robust variance of a mean: V_a + V_b - V_(a, b), each term the
    unbalanced cluster-robust variance with its own G / (G - 1)."""
    ab = pd.factorize(pd.Series(a).astype(str) + "|" + pd.Series(b).astype(str))[0]
    return M.cluster_robust_var_of_mean(e, a) + M.cluster_robust_var_of_mean(e, b) - M.cluster_robust_var_of_mean(e, ab)


def dyadic_var_of_mean(e, a, b):
    """Dyadic-robust variance of a mean (Aronow, Samii and Assenova 2015): sum over all pairs of rows (i, j) that share
    a person of r_i r_j, over n^2, times N / (N - 1) for N people; r = e - mean(e). Rows are (a_i, b_i) person pairs;
    pairs sharing both people (the same row, or the two directions of a dyad) are counted once."""
    e, a, b = np.asarray(e, float), np.asarray(a), np.asarray(b)
    r, n = e - e.mean(), len(e)
    per_person = pd.Series(np.r_[r, r]).groupby(np.r_[a, b]).sum()
    dyad = pd.Series(r).groupby([np.minimum(a, b), np.maximum(a, b)]).sum()
    N = per_person.size
    return N / (N - 1) * (np.sum(per_person ** 2) - np.sum(dyad ** 2)) / n ** 2


def dyadic_frame(clean):
    """Every woman-man pair at each real event, in both directions."""
    rows = []
    for e, ev in clean.groupby("event"):
        w, m = np.unique(ev[ev.gender == 0].rater), np.unique(ev[ev.gender == 1].rater)
        W, Mn = np.meshgrid(w, m, indexing="ij")
        W, Mn = W.ravel(), Mn.ravel()
        rows.append(pd.DataFrame({"event": e, "rater": np.r_[W, Mn], "ratee": np.r_[Mn, W]}))
    return pd.concat(rows, ignore_index=True)


def dyadic_design(clean, reps, rng):
    F = dyadic_frame(clean)
    people = np.unique(np.r_[F.rater, F.ratee]); pi = {p: k for k, p in enumerate(people)}
    ra, rb = F.rater.map(pi).to_numpy(), F.ratee.map(pi).to_numpy()
    dy = pd.factorize(pd.Series(np.minimum(ra, rb)).astype(str) + "_" + pd.Series(np.maximum(ra, rb)).astype(str))[0]
    X = YE.indicator_design(F.rater.to_numpy(), F.ratee.to_numpy())
    rate = clean.dec.mean()
    effects = lambda: rng.normal(size=len(people))[ra] + rng.normal(size=len(people))[rb] + rng.normal(size=dy.max() + 1)[dy]
    draws = [effects() for _ in range(50)]
    lo, hi = -5.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.mean([np.mean(1 / (1 + np.exp(-(mid + u)))) for u in draws]) < rate else (lo, mid)
    c, rows, t0 = (lo + hi) / 2, [], time.time()
    for r in range(reps):
        p = 1 / (1 + np.exp(-(c + effects())))
        y = (rng.random(len(p)) < p).astype(float)
        folds = YE.within_rater_folds(F.rater.to_numpy(), K, rng)
        lossv = np.empty(len(y))
        for k in range(K):
            te = folds == k
            w, _ = M.fit_logit(X[~te], y[~te], ridge=1.0, iters=200)
            lossv[te] = ((M.predict_logit(w, X[te]) > y[~te].mean()).astype(float) != y[te]).astype(float)
        w, _ = M.fit_logit(X, y, ridge=1.0, iters=200)
        yhat = (M.predict_logit(w, X) > y.mean()).astype(float)
        target = np.mean(yhat * (1 - p) + (1 - yhat) * p)
        rows.append(dict(rep=r, rows=len(y), yes_rate=y.mean(), intercept=c, target=target, cv=lossv.mean(),
                         se_naive=np.sqrt(lossv.var(ddof=1) / len(y)), se_rater=np.sqrt(M.cluster_robust_var_of_mean(lossv, ra)),
                         se_ratee=np.sqrt(M.cluster_robust_var_of_mean(lossv, rb)), se_two_way=np.sqrt(max(two_way_var_of_mean(lossv, ra, rb), 0)),
                         se_dyadic=np.sqrt(max(dyadic_var_of_mean(lossv, ra, rb), 0)), seed=DYADIC_SEED))
        if r == 1:
            print(f"dyadic design: about {(time.time() - t0) / 2 * reps / 60:.0f} minutes for {reps} replicates", flush=True)
    D = pd.DataFrame(rows)
    C = pd.DataFrame([dict(procedure=f"within-rater CV, {name} SE", coverage_90=np.mean(np.abs(D.cv - D.target) <= Z * D[col]),
                           mean_half_width=Z * D[col].mean(), sd_of_cv_minus_target=(D.cv - D.target).std(ddof=1), bias=(D.cv - D.target).mean(),
                           reps=reps, seed=DYADIC_SEED, fold_scheme="within-rater random folds")
                      for name, col in (("naive", "se_naive"), ("rater-clustered", "se_rater"), ("ratee-clustered", "se_ratee"),
                                        ("two-way (rater, ratee)", "se_two_way"), ("dyadic-robust", "se_dyadic"))])
    return D, C


def diagnostics(D, DD):
    """Post hoc diagnostics of the grid (one row per design) and of the dyadic design (one row)."""
    rows = []
    for (G, ps, es), d in D.groupby(["events", "person_sd", "event_sd"]):
        pos = d.bht_event_se > 0
        hit = np.abs(d.bht_event_centre - d.err_new) <= Z * d.bht_event_se
        floor = np.maximum(d.se_event_cr, d.se_event_cluster_nested)
        rows.append(dict(design=int(d.design.iloc[0]), events=G, person_sd=ps, event_sd=es, reps=len(d),
                         share_zero_width=float((~pos).mean()), coverage_bias_adjusted=float(hit.mean()),
                         coverage_bias_adjusted_given_positive=float(hit[pos].mean()),
                         coverage_floor_cr_cluster_nested=float((np.abs(d.cv_event - d.err_new) <= Z * floor).mean()),
                         corr_cv_target=float(np.corrcoef(d.cv_event, d.err_new)[0, 1]), sd_cv=float(d.cv_event.std(ddof=1)),
                         sd_target=float(d.err_new.std(ddof=1)), sd_cv_minus_target=float((d.cv_event - d.err_new).std(ddof=1)),
                         mean_se_cluster_robust=float(d.se_event_cr.mean()),
                         design_redrawn=bool(G != 21), seed=int(d.seed.iloc[0])))
    dy = dict(design="dyadic", reps=len(DD), sd_cv=float(DD.cv.std(ddof=1)), sd_target=float(DD.target.std(ddof=1)),
              corr_cv_target=float(np.corrcoef(DD.cv, DD.target)[0, 1]), mean_bias=float((DD.cv - DD.target).mean()),
              **{f"mean_{c}": float(DD[c].mean()) for c in ("se_naive", "se_rater", "se_ratee", "se_two_way", "se_dyadic")})
    return pd.DataFrame(rows), pd.DataFrame([dy])


def run_dyadic(reps):
    return dyadic_design(pd.read_csv("data/clean.csv", low_memory=False), reps, np.random.default_rng(DYADIC_SEED))


def write_diagnostics():
    DG, DY = diagnostics(pd.read_csv(RES / "sim_cv_designs_rows.csv"), pd.read_csv(RES / "sim_cv_designs_dyadic_rows.csv"))
    pd.concat([DG, DY], ignore_index=True).assign(post_hoc=True).to_csv(RES / "sim_cv_designs_diagnostics.csv", index=False)
    with pd.option_context("display.width", 250):
        print(DG.drop(columns=["seed"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(DY.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    if "--diagnostics" in sys.argv:
        return write_diagnostics()
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, (os.cpu_count() or 2) - 1)
    with Pool(workers) as pool:
        dyadic = pool.apply_async(run_dyadic, (reps,))                  # runs alongside the grid
        parts = pool.map(run_design, [(d, reps, R) for d in range(len(GRID))], chunksize=1)
        DD, DC = dyadic.get()
    D = pd.DataFrame([row for part in parts for row in part])
    D.to_csv(RES / "sim_cv_designs_rows.csv", index=False)
    C = coverage(D); C.to_csv(RES / "sim_cv_designs_coverage.csv", index=False)
    DD.to_csv(RES / "sim_cv_designs_dyadic_rows.csv", index=False); DC.to_csv(RES / "sim_cv_designs_dyadic.csv", index=False)
    n1 = C[(C.procedure == "event cluster-nested, bias-adjusted") & ((C.person_sd > 0) | (C.event_sd > 0))]
    n2 = C[(C.procedure == "random-row nested, bias-adjusted (Bates et al.)") & (C.person_sd >= 0.5)]
    hyp = pd.DataFrame([
        dict(hypothesis="N1", statistic=f"cluster-nested bias-adjusted coverage {n1.coverage_90.min():.3f} to {n1.coverage_90.max():.3f} over {len(n1)} designs; "
             f"{int(n1.coverage_90.between(0.85, 0.95).sum())} of {len(n1)} within 0.85 to 0.95", supported=bool(n1.coverage_90.between(0.85, 0.95).all())),
        dict(hypothesis="N2", statistic=f"random-row nested bias-adjusted coverage {n2.coverage_90.min():.3f} to {n2.coverage_90.max():.3f} over {len(n2)} designs; "
             f"{int((n2.coverage_90 < 0.80).sum())} of {len(n2)} below 0.80", supported=bool((n2.coverage_90 < 0.80).all()))
    ]).assign(reps=reps, nested_repetitions=R, seed=f"{SEED} + design index", fold_scheme="event and random-row")
    hyp.to_csv(RES / "sim_cv_designs_hypotheses.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(C.pivot_table(index=["events", "person_sd", "event_sd"], columns="procedure", values="coverage_90").round(3).to_string())
        print(DC.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))
    write_diagnostics()


if __name__ == "__main__":
    main()
