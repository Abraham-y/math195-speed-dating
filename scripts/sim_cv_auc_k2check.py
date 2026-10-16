"""Exploratory check of N18's miss (proposals/09, E2): the mixing statistic at K = 2 with many data sets.

Input : none (sim_cv_auc.py's generator). Output: results/sim_cv_auc_k2check.csv (one row per data set) and a summary line.
Seeds : 20265906 + chunk, 8 chunks of 1,250 data sets.

Design 4 of Part 9 (sa 2, su 0), G = 80, grouped 2-fold CV; the statistic is G x (pooled between-cluster AUC -
BCWF-pooled), as in N18. The prediction with the G/(G-1) factor is (C_mix + c2) G/(G-1) from sim_cv_auc_components.csv.
Run from the repository root: python scripts/sim_cv_auc_k2check.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "scripts")
import sim_cv_auc as A

SEED, CHUNKS, PER, G, K, D = 20265906, 8, 1250, 80, 2, 4


def chunk(c):
    rng = np.random.default_rng(SEED + c); out = []
    for r in range(PER):
        X, y, g = A.draw(G, rng, **A.DESIGNS[D]); s, f = A.cv_scores(X, y, g, A.grouped(G, rng, K))
        e = A.estimates(s, y, g, f)
        out.append(dict(chunk=c, rep=r, seed=SEED + c, G_x_mix=G * (e["pooled_between"] - e["bcwf_pooled"])))
    return out


def main():
    with Pool(8) as p:
        D_ = pd.DataFrame([r for part in p.map(chunk, range(CHUNKS)) for r in part])
    D_.to_csv(pathlib.Path("results") / "sim_cv_auc_k2check.csv", index=False)
    v = D_.G_x_mix; pred = pd.read_csv("results/sim_cv_auc_components.csv").set_index("design").loc[D, "pred_G_mix_K2"] * G / (G - 1)
    print(f"{len(v)} data sets: mean {v.mean():.4f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.4f}), median {v.median():.3f}, "
          f"skewness {stats.skew(v):.2f}, SD {v.std(ddof=1):.3f}; predicted {pred:.4f}; shortfall {100 * (1 - v.mean() / pred):.1f}%")


if __name__ == "__main__":
    main()
