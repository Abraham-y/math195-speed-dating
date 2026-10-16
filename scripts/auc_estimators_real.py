"""Exploratory (PREREG.md, Part 9): the four CV AUC estimates on the real data and in the realistic A2 design.

Input : data/pairs.csv, data/feature_sets.json
Output: results/auc_estimators_real.csv (real data), results/auc_estimators_realistic.csv (A2 design, per design)
Seeds : real-data partitions 20265406; realistic design d (0 to 3) 20265506 + d.

Real data: the full logistic regression and LDA on `pre` and `pre+date` (Part 1 preparation without splines, as in
errorbars_final.py), event 5-fold CV on the headline folds and on R = 20 random event partitions. For each: the
pooled AUC (what the project reports), the fold-averaged AUC, BCWF (pairs from different events within a fold,
averaged over folds) and BCWF-pooled, the pooled between-event AUC, the within-event AUC (pairs from the same event;
the matchmaker's target), and the Obuchowski SE of the pooled scores (Proposition 9, proposals/06 section 6f).

Realistic design: Part 7's A2 truth with G = 21 events redrawn (fresh people), person SD in {0.5, 1.0} and event SD
in {0, 0.4}, 300 data sets each. The target is the full model's pooled AUC on 2,000 new events. Reported: bias and
coverage (t on 20 degrees of freedom, Obuchowski SE) of the pooled and BCWF-pooled estimates.
Run from the repository root: python scripts/auc_estimators_real.py [replicates] [workers]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, pathlib, itertools, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "scripts")
import modeling as M
import errorbars_real as ER
import sim_cv_auc as A
import sim_cv_matched as A2
import sim_cv_designs as SD
import sim_cv_repair as SR
import sim_cv_sizes as SZ

RES = pathlib.Path("results")
SEED_REAL, SEED_SIM, R, NEW_EVENTS = 20265406, 20265506, 20, 2000
GRID = list(itertools.product((0.5, 1.0), (0.0, 0.4)))


def within_event_auc(s, y, g):
    conc, pairs = A.within_cluster_counts(s, y, g)
    return conc.sum() / pairs.sum()


def real():
    rng = np.random.default_rng(SEED_REAL)
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    rows = []
    for fs in ("pre", "pre+date"):
        X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, fs)
        fo = M.event_folds(ev); head = np.array([fo[e] for e in ev]); G = len(np.unique(ev))
        for model in ("logistic", "lda"):
            for r in range(R + 1):
                f = head if r == 0 else ER.deal(ev, 5, rng)
                prob, _ = ER.cv(model, sets, X, y, ids, f)
                e = A.estimates(prob, y, ev, f)
                rows.append(dict(feature_set=fs, model=model, partition="headline" if r == 0 else f"random {r}", events=G, dyads=len(y),
                                 **e, within_event=within_event_auc(prob, y, ev), se_obu=np.sqrt(e["v_obu"]), seed=SEED_REAL))
            d = pd.DataFrame(rows[-(R + 1):])
            print(f"{fs} {model}: headline pooled {d.pooled.iloc[0]:.4f}, BCWF-pooled {d.bcwf_pooled.iloc[0]:.4f}, within-event "
                  f"{d.within_event.iloc[0]:.4f}, SE {d.se_obu.iloc[0]:.4f}; over {R} partitions pooled - BCWF-pooled "
                  f"{(d.pooled - d.bcwf_pooled).iloc[1:].mean():+.4f} (SD {(d.pooled - d.bcwf_pooled).iloc[1:].std():.4f})", flush=True)
    D = pd.DataFrame(rows).assign(fold_scheme="event 5-fold")
    D.to_csv(RES / "auc_estimators_real.csv", index=False)
    return D


def realistic(args):
    d, reps = args
    ps, es = GRID[d]; sd = SD.sd_of(ps, es); seed = SEED_SIM + d; rng = np.random.default_rng(seed)
    pairs, X0, f0, m0, e0, beta_all, _ = SR.truth_model()
    beta0 = A2.calibrate_intercept(beta_all[0], beta_all[1:], sd, X0, f0, m0, e0, pairs.match.mean(), rng)
    out, t0 = [], time.time()
    for r in range(reps):
        X, f, m, e = SR.redraw_events(X0, f0, m0, e0, 21, rng)
        y = A2.draw(beta0, beta_all[1:], sd, X, f, m, e, rng)
        s, fold = A.cv_scores(X, y, e, A.grouped(21, rng))
        est = A.estimates(s, y, e, fold)
        w, _ = M.fit_logit(X, y)
        Xn, fn, mn, en = SZ.new_events(pairs, X0, f0, m0, e0, NEW_EVENTS, rng)
        yn = A2.draw(beta0, beta_all[1:], sd, Xn, fn, mn, en, rng)
        out.append(dict(design=d, person_sd=ps, event_sd=es, rep=r, seed=seed, **est, auc_new=M.auc(Xn @ w[1:], yn),
                        v_mc=M.auc_clustered_var(Xn @ w[1:], yn, en)))
    print(f"realistic design {d} (person SD {ps}, event SD {es}): {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    real()
    with Pool(workers) as pool:
        parts = pool.map(realistic, [(d, reps) for d in range(len(GRID))], chunksize=1)
    D = pd.DataFrame([r for p in parts for r in p])
    q = stats.t.ppf(0.95, 20); rows = []
    for d, x in D.groupby("design"):
        se = np.sqrt(x.v_obu); row = dict(design=d, person_sd=x.person_sd.iloc[0], event_sd=x.event_sd.iloc[0], reps=len(x),
                                         mean_auc_new=x.auc_new.mean(), mean_se_obu=se.mean(), seed=int(x.seed.iloc[0]))
        for est in A.ESTIMATES:
            diff = x[est] - x.auc_new
            row[f"bias_{est}"] = diff.mean(); row[f"bias_se_{est}"] = diff.std(ddof=1) / np.sqrt(len(x))
            row[f"cov_t20_{est}"] = float(np.mean(np.abs(diff) <= q * se))
        row["G_x_mix"] = 21 * (x.pooled_between - x.bcwf_pooled).mean()
        rows.append(row)
    S = pd.DataFrame(rows).assign(events=21, fold_scheme="event 5-fold")
    S.to_csv(RES / "auc_estimators_realistic.csv", index=False)
    with pd.option_context("display.width", 250):
        print(S[["design", "person_sd", "event_sd", "mean_auc_new", "mean_se_obu", "bias_pooled", "bias_bcwf_pooled", "bias_fold_averaged",
                 "cov_t20_pooled", "cov_t20_bcwf_pooled", "G_x_mix"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
