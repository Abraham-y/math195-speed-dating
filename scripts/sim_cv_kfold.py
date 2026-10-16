"""N14 and N15 (PREREG.md, Part 8): an out-of-sample test of Proposition 8 (ii), the Nadeau-Bengio limit, at other
fold counts.

Input : none (Part 6's synthetic generator, scripts/sim_cv_firstorder.py)
Output: results/sim_cv_kfold_rows.csv, results/sim_cv_kfold_summary.csv, results/sim_cv_kfold_hypotheses.csv
Seeds : 20264906 + G for G in {80, 160, 320}.

Proposition 8 (ii): the Nadeau-Bengio variance over R grouped partitions into K folds, divided by (J+K-1)/(J-1) times
the cluster-robust variance, tends to 1 - 2Cov(a, beta)/((K-1)Var(a)) + Var(beta)/((K-1)^2 Var(a)). The components
were estimated before this ran from Part 6b's K = 5 runs (G Var(a) = 0.02266, G Var(beta) = 0.00946,
G Cov(a, beta) = -0.00787; 0-1 loss). The predictions are 2.112, 1.451, 1.200 and 1.082 at K = 2, 3, 5 and 10. For
log-loss beta = 0, so the prediction is 1 at every K. Per data set: V_CR from one grouped 5-fold run; the
Nadeau-Bengio variance for each K from R grouped partitions with J = R x K = 60.
Run from the repository root: python scripts/sim_cv_kfold.py [replicates] [workers]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import sim_cv_firstorder as F

RES = pathlib.Path("results")
SEED, GS, J = 20264906, (80, 160, 320), 60
KS = {2: 30, 3: 20, 5: 12, 10: 6}                         # K -> R, so that J = 60
VA, VB, CAB = 0.02266, 0.00946, -0.00787                  # from Part 6b (PREREG.md, Part 8)
PRED = {K: 1 - 2 * CAB / ((K - 1) * VA) + VB / ((K - 1) ** 2 * VA) for K in KS}


def grouped(G, K, rng):
    to = np.empty(G, int); to[rng.permutation(G)] = np.arange(G) % K
    return to


def one(G, rng):
    X, y, g = F.draw(G, rng)
    l01, lll = F.cv_losses(X, y, g, grouped(G, 5, rng))
    out = dict(G=G, vcr_01=M.cluster_robust_var_of_mean(l01, g), vcr_ll=M.cluster_robust_var_of_mean(lll, g))
    for K, R in KS.items():
        fm01, fmll = [], []
        for _ in range(R):
            f = grouped(G, K, rng)[g]
            for k in range(K):
                te = f == k
                w, _ = M.fit_logit(X[~te], y[~te])
                a, b = F.losses(w, y[~te].mean(), X[te], y[te]); fm01.append(a.mean()); fmll.append(b.mean())
        factor = (J + K - 1) / (J - 1)
        for name, fm in (("01", fm01), ("ll", fmll)):
            nb = (1 / J + 1 / (K - 1)) * np.var(fm, ddof=1)
            out[f"ratio_{name}_K{K}"] = nb / (factor * out[f"vcr_{name}"])
    return out


def run_G(args):
    G, reps = args
    rng = np.random.default_rng(SEED + G); t0 = time.time()
    rows = [dict(one(G, rng), rep=r, seed=SEED + G) for r in range(reps)]
    print(f"G {G}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with Pool(workers) as pool:
        parts = pool.map(run_G, [(G, reps) for G in sorted(GS, reverse=True)], chunksize=1)
    D = pd.DataFrame([r for p in parts for r in p]).sort_values(["G", "rep"])
    D.to_csv(RES / "sim_cv_kfold_rows.csv", index=False)
    rows = []
    for G, d in D.groupby("G"):
        for K in KS:
            for name, label, pred in (("01", "0-1 at the training rate", PRED[K]), ("ll", "log-loss", 1.0)):
                v = d[f"ratio_{name}_K{K}"]
                rows.append(dict(G=G, K=K, loss=label, predicted=pred, observed=float(v.mean()), observed_se=float(v.std(ddof=1) / np.sqrt(len(v))),
                                 difference=float(v.mean() - pred), reps=len(d), J=J, R=KS[K], seed=SEED + G))
    S = pd.DataFrame(rows); S.to_csv(RES / "sim_cv_kfold_summary.csv", index=False)
    s = S.set_index(["loss", "K", "G"])
    n14 = [s.loc[("0-1 at the training rate", K, G)].difference for K in (3, 10) for G in GS]
    n15 = [s.loc[("log-loss", K, G)].difference for K in (3, 10) for G in GS]
    fmt = lambda v: ", ".join(f"{x:+.3f}" for x in v)
    hyp = pd.DataFrame([
        dict(hypothesis="N14", statistic=f"0-1 loss, observed minus predicted (K = 3 at G = 80, 160, 320; then K = 10): {fmt(n14)}",
             supported=bool(all(abs(v) <= 0.07 for v in n14))),
        dict(hypothesis="N15", statistic=f"log-loss, observed minus 1 (K = 3 at G = 80, 160, 320; then K = 10): {fmt(n15)}",
             supported=bool(all(abs(v) <= 0.05 for v in n15))),
    ]).assign(reps=reps, seed=f"{SEED} + G", fold_scheme="grouped (cluster) K-fold")
    hyp.to_csv(RES / "sim_cv_kfold_hypotheses.csv", index=False)
    with pd.option_context("display.width", 200):
        print(S.drop(columns=["seed"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
