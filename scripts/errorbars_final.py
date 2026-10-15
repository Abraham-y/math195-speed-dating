"""The recommended error bars on the real data (exploratory; proposals/06, "Where the error-bar track stands").

Input : data/pairs.csv, data/feature_sets.json
Output: results/errorbars_final.csv
Seed  : 20264806 (the Nadeau-Bengio partitions and the wild cluster bootstrap); headline event folds 20261006.

For the full logistic regression and LDA, on `pre` and `pre+date` (Part 1 preparation without splines, as in
errorbars_real.py), with the per-dyad loss a wrong prediction at the training match rate:
- event 5-fold CV on the headline folds;
- the cluster-robust SE by event with normal quantiles, and with t quantiles on G - 1 = 20 degrees of freedom;
- the wild cluster bootstrap-t (Rademacher signs on each event's residual sum, 999 draws);
- the Nadeau-Bengio variance from R = 10 random event partitions (J = 50), with its ratio to (J+K-1)/(J-1) times
  the cluster-robust variance. By Proposition 8 that ratio estimates Var(a - beta/(K-1)) / Var(a): above 1, the
  correction is conservative for the full model's new-event error.
- The AUC on the headline folds with the clustered (Obuchowski) SE and t quantiles on 20 degrees of freedom.
Run from the repository root: python scripts/errorbars_final.py
"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "scripts")
import modeling as M
import errorbars_real as ER
import sim_cv_sizes as SZ

RES = pathlib.Path("results")
SEED, K, R = 20264806, 5, 10


def main():
    rng = np.random.default_rng(SEED)
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    rows = []
    for fs in ("pre", "pre+date"):
        X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, fs)
        fo = M.event_folds(ev); fe = np.array([fo[e] for e in ev]); G = len(np.unique(ev))
        for model in ("logistic", "lda"):
            prob, e = ER.cv(model, sets, X, y, ids, fe)
            err, v_cr = e.mean(), M.cluster_robust_var_of_mean(e, ev)
            q_wild = SZ.wild_cluster_t_quantile(e, ev, 999, rng)
            fm = []
            for _ in range(R):
                f = ER.deal(ev, K, rng)
                fm += list(pd.Series(ER.cv(model, sets, X, y, ids, f)[1]).groupby(f).mean())
            fm = np.array(fm); J = len(fm)
            nb_var = (1 / J + 1 / (K - 1)) * fm.var(ddof=1); factor = (J + K - 1) / (J - 1)
            auc = M.auc(prob, y); auc_se = np.sqrt(M.auc_clustered_var(prob, y, ev)); t20 = stats.t.ppf(0.95, G - 1)
            base = dict(feature_set=fs, model=model, dyads=len(y), events=G, error_rate=err, seed=SEED, nb_partitions=R)
            for name, hw in (("cluster-robust, normal", 1.645 * np.sqrt(v_cr)), ("cluster-robust, t(20)", t20 * np.sqrt(v_cr)),
                             ("wild cluster bootstrap-t", q_wild * np.sqrt(v_cr)),
                             ("Nadeau-Bengio, event folds", stats.t.ppf(0.95, J - 1) * np.sqrt(nb_var))):
                rows.append(dict(base, metric="error rate", interval=name, half_width=hw, ci90_low=err - hw, ci90_high=err + hw,
                                 nb_over_scaled_cr=nb_var / (factor * v_cr)))
            rows.append(dict(base, metric="AUC", error_rate=np.nan, interval="clustered (Obuchowski), t(20)", estimate=auc,
                             half_width=t20 * auc_se, ci90_low=auc - t20 * auc_se, ci90_high=auc + t20 * auc_se))
            print(f"{fs} {model}: error {err:.4f}, CR-t half-width {t20 * np.sqrt(v_cr):.4f}, wild {q_wild * np.sqrt(v_cr):.4f}, "
                  f"NB/scaled CR {nb_var / (factor * v_cr):.3f}", flush=True)
    D = pd.DataFrame(rows).assign(fold_scheme="event")
    D.to_csv(RES / "errorbars_final.csv", index=False)
    with pd.option_context("display.width", 220):
        print(D[["feature_set", "model", "metric", "interval", "error_rate", "estimate", "half_width", "ci90_low", "ci90_high", "nb_over_scaled_cr"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
