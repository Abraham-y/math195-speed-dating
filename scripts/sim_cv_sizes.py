"""N12 and N13 (PREREG.md, Part 7): CV error bars when cluster sizes are very unequal; the wild cluster bootstrap-t.

Input : data/pairs.csv (the A2 truth model and the real events, 25 to 440 dyads each)
Output: results/sim_cv_sizes_rows.csv (one row per data set), results/sim_cv_sizes_coverage.csv,
        results/sim_cv_sizes_hypotheses.csv
Seeds : design d (0 to 7): 20264706 + d.

Designs: the A2 truth with G in {21, 40} events drawn with replacement (fresh people per draw), person SD in
{0.5, 1.0} and event SD in {0, 0.4}; 300 data sets each. The target, Err_new, is the full-data model's error on
2,000 new events (its Monte Carlo variance is recorded). Event 5-fold CV gives the estimate. Intervals (90%):
- the cluster-robust SE with normal quantiles (N13);
- CR2 with t(G - 1), and the cluster-robust SE with t(G* - 1), G* = (sum n_g)^2 / sum n_g^2 (exploratory);
- the wild cluster bootstrap-t (N12): Rademacher signs on each event's residual sum, 999 draws, the cluster-robust t
  recomputed each time, the symmetric interval from the 90% quantile of |t*|.
Run from the repository root: python scripts/sim_cv_sizes.py [replicates] [workers]
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
import sim_cv_confirm as CF

RES = pathlib.Path("results")
SEED, K, Z, B_WILD, NEW_EVENTS = 20264706, 5, 1.645, 999, 2000
GRID = list(itertools.product((21, 40), (0.5, 1.0), (0.0, 0.4)))


def wild_cluster_t_quantile(u, g, B, rng, level=0.90):
    """Wild cluster bootstrap-t for a mean: Rademacher signs on each cluster's residual sum; returns the `level`
    quantile of |t*| with the G/(G-1) cluster-robust SE recomputed in every draw."""
    u, g = np.asarray(u, float), np.asarray(g); n = len(u)
    s = pd.Series(u - u.mean()).groupby(g).agg(["sum", "size"])
    R, ng = s["sum"].to_numpy(), s["size"].to_numpy(float); G = len(R)
    W = rng.choice([-1.0, 1.0], size=(B, G))
    d = (W @ R) / n                                       # ybar* - ybar
    Rs = W * R - ng * d[:, None]                          # residual sums of the bootstrap data
    v = G / (G - 1) * np.sum(Rs ** 2, axis=1) / n ** 2
    return float(np.quantile(np.abs(d) / np.sqrt(v), level))


def new_events(pairs, X, f, m, e, n_events, rng):
    rows = [np.where(e == k)[0] for k in range(e.max() + 1)]
    pick = rng.integers(0, len(rows), n_events)
    idx = np.concatenate([rows[k] for k in pick]); block = np.repeat(np.arange(n_events), [len(rows[k]) for k in pick])
    lab = lambda v: pd.factorize(pd.Series(block).astype(str) + "_" + pd.Series(v[idx]).astype(str))[0]
    return X[idx], lab(f), lab(m), block


def run_design(args):
    d, reps = args
    G, ps, es = GRID[d]; sd = SD.sd_of(ps, es); seed = SEED + d
    rng = np.random.default_rng(seed)
    pairs, X0, f0, m0, e0, beta_all, _ = SR.truth_model()
    beta0 = A2.calibrate_intercept(beta_all[0], beta_all[1:], sd, X0, f0, m0, e0, pairs.match.mean(), rng)
    rows, t0 = [], time.time()
    for r in range(reps):
        X, f, m, e = SR.redraw_events(X0, f0, m0, e0, G, rng)
        y = A2.draw(beta0, beta_all[1:], sd, X, f, m, e, rng)
        Xn, fn, mn, en = new_events(pairs, X0, f0, m0, e0, NEW_EVENTS, rng)
        ln = A2.loss(X, y, Xn, A2.draw(beta0, beta_all[1:], sd, Xn, fn, mn, en, rng))
        eg = A2.cv(X, y, A2.group_folds(e, K, rng))
        ng = np.bincount(e).astype(float); gstar = ng.sum() ** 2 / np.sum(ng ** 2)
        rows.append(dict(design=d, events=G, person_sd=ps, event_sd=es, rep=r, seed=seed, dyads=len(y), err_new=ln.mean(),
                         v_mc=M.cluster_robust_var_of_mean(ln, en), cv_event=eg.mean(), se_cr=np.sqrt(M.cluster_robust_var_of_mean(eg, e)),
                         se_cr2=np.sqrt(CF.cr2_var_of_mean(eg, e)), g_star=gstar, size_cv=ng.std() / ng.mean(),
                         wild_q=wild_cluster_t_quantile(eg, e, B_WILD, rng)))
    print(f"design {d} (G {G}, person SD {ps}, event SD {es}): {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def coverage(D):
    out = []
    for dsg, g in D.groupby("design"):
        miss = np.abs(g.cv_event - g.err_new)
        procs = {"cluster-robust, normal (N13)": Z * g.se_cr, "CR2, t(G-1)": stats.t.ppf(0.95, g.events - 1) * g.se_cr2,
                 "cluster-robust, t(G*-1)": stats.t.ppf(0.95, np.maximum(g.g_star - 1, 1)) * g.se_cr,
                 "wild cluster bootstrap-t (N12)": g.wild_q * g.se_cr}
        v = (g.cv_event - g.err_new).var(ddof=1)
        for name, hw in procs.items():
            hit = miss <= hw
            out.append(dict(design=dsg, events=g.events.iloc[0], person_sd=g.person_sd.iloc[0], event_sd=g.event_sd.iloc[0], procedure=name,
                            coverage_90=float(hit.mean()), mean_half_width=float(hw.mean()), reps=len(g),
                            mc_share=float(g.v_mc.mean() / v), ratio_var_corrected=float((v - g.v_mc.mean()) / (g.se_cr ** 2).mean()),
                            miss_corr_size_cv=float(np.corrcoef((~hit).astype(float), g.size_cv)[0, 1]) if (~hit).any() else np.nan,
                            mean_g_star=float(g.g_star.mean()), seed=int(g.seed.iloc[0])))
    return pd.DataFrame(out)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    with Pool(workers) as pool:
        parts = pool.map(run_design, [(d, reps) for d in sorted(range(len(GRID)), key=lambda d: -GRID[d][0])], chunksize=1)
    D = pd.DataFrame([r for p in parts for r in p]).sort_values(["design", "rep"])
    D.to_csv(RES / "sim_cv_sizes_rows.csv", index=False)
    C = coverage(D); C.to_csv(RES / "sim_cv_sizes_coverage.csv", index=False)
    w = C[C.procedure.str.endswith("(N12)")]; c = C[C.procedure.str.endswith("(N13)") & (C.events == 40)]
    hyp = pd.DataFrame([
        dict(hypothesis="N12", statistic=f"wild cluster bootstrap-t covers {w.coverage_90.min():.3f} to {w.coverage_90.max():.3f} over 8 designs; "
             f"{int(w.coverage_90.between(0.85, 0.95).sum())} of 8 within 0.85 to 0.95", supported=bool(w.coverage_90.between(0.85, 0.95).all())),
        dict(hypothesis="N13", statistic=f"cluster-robust (normal) coverage in the four G = 40 designs: {', '.join(f'{v:.3f}' for v in c.coverage_90)}",
             supported=bool((c.coverage_90 < 0.85).any())),
    ]).assign(reps=reps, seed=f"{SEED} + design index", new_events=NEW_EVENTS, fold_scheme="event 5-fold")
    hyp.to_csv(RES / "sim_cv_sizes_hypotheses.csv", index=False)
    with pd.option_context("display.width", 250):
        print(C.pivot_table(index=["events", "person_sd", "event_sd"], columns="procedure", values="coverage_90").round(3).to_string())
        print(C.groupby(["events", "person_sd", "event_sd"])[["mc_share", "ratio_var_corrected", "mean_g_star"]].first().round(3).to_string())
        print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
