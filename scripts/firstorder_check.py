"""Exploratory check of Proposition 8 (proposals/06, section 6e) on the Part 6 simulations: the Nadeau-Bengio limit
predicted from the first-order decomposition against the one observed.

Input : results/sim_cv_firstorder_rows.csv (Part 6) and results/sim_cv_firstorder_rows_b.csv (Part 6b)
Output: results/firstorder_check.csv, results/firstorder_check_pooled.csv
Seed  : 20264606 (bootstrap over data sets).

For 0-1 loss, the first-order theory gives Var(Err_new) = Var(beta)/G (after removing the Monte Carlo variance of
Err_new where it was recorded), Cov(CV, Err_new) = (Cov(a, beta) + Var(beta))/G and Var(CV) = Var(a + beta)/G. Solving
gives Var(a), Var(beta) and Cov(a, beta), and the predicted Nadeau-Bengio ratio Var(a - beta/(K-1)) / Var(a), against
the observed mean of NB / ((J+K-1)/(J-1) V_CR). Intervals: 500 bootstrap resamples of data sets. Also the pooled
corrected variance ratio Var(CV - Err_new) / mean V_CR over G >= 80 and both losses (Part 6b).
Run from the repository root after sim_cv_firstorder.py: python scripts/firstorder_check.py
"""
import pathlib
import numpy as np, pandas as pd

RES = pathlib.Path("results")
SEED, K, J = 20264606, 5, 50


def decompose(d):
    vmc = d.v_mc_01.mean() if "v_mc_01" in d else 0.0
    vb = d.err_new_01.var(ddof=1) - vmc
    cab = np.cov(d.cv_01, d.err_new_01)[0, 1] - vb
    va = d.cv_01.var(ddof=1) - vb - 2 * cab
    return va, vb, cab, (va - 2 * cab / (K - 1) + vb / (K - 1) ** 2) / va


def pooled_ratio(groups):
    out = []
    for d in groups.values():
        for l in ("01", "ll"):
            out.append(((d[f"cv_{l}"] - d[f"err_new_{l}"]).var(ddof=1) - d[f"v_mc_{l}"].mean()) / d[f"vcr_{l}"].mean())
    return float(np.mean(out))


def main():
    rng = np.random.default_rng(SEED); factor = (J + K - 1) / (J - 1); rows = []
    for tag, name in (("", "Part 6 (Err_new on 2,000 clusters, no Monte Carlo correction)"), ("_b", "Part 6b (20,000 clusters, Monte Carlo variance removed)")):
        D = pd.read_csv(RES / f"sim_cv_firstorder_rows{tag}.csv")
        for G, d in D.groupby("G"):
            d = d.reset_index(drop=True)
            va, vb, cab, pred = decompose(d)
            bs = [decompose(d.iloc[rng.integers(0, len(d), len(d))])[3] for _ in range(500)]
            obs = float((d.nb_var_01 / (factor * d.vcr_01)).mean())
            lo, hi = np.quantile(bs, [0.05, 0.95])
            rows.append(dict(run=name, G=G, var_a=va, var_beta=vb, corr_a_beta=cab / np.sqrt(va * vb), predicted_nb_ratio=pred,
                             predicted_ci90_low=lo, predicted_ci90_high=hi, observed_nb_ratio=obs, observed_inside=bool(lo <= obs <= hi), seed=SEED))
    C = pd.DataFrame(rows); C.to_csv(RES / "firstorder_check.csv", index=False)
    B = pd.read_csv(RES / "sim_cv_firstorder_rows_b.csv"); groups = {G: d.reset_index(drop=True) for G, d in B[B.G >= 80].groupby("G")}
    est = pooled_ratio(groups)
    bs = [pooled_ratio({G: d.iloc[rng.integers(0, len(d), len(d))] for G, d in groups.items()}) for _ in range(500)]
    P = pd.DataFrame([dict(quantity="corrected Var(CV - Err_new) / mean V_CR, pooled over G = 80, 160, 320 and both losses (Part 6b)",
                           estimate=est, ci90_low=float(np.quantile(bs, 0.05)), ci90_high=float(np.quantile(bs, 0.95)), seed=SEED)])
    P.to_csv(RES / "firstorder_check_pooled.csv", index=False)
    with pd.option_context("display.width", 250):
        print(C.drop(columns=["seed"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(P.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
