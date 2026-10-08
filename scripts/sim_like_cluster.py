"""Check the cluster-robust SEs of the like model's held-out R-squared and RMSE (modeling.r_squared_and_rmse) on
simulated data with the like-model design before they are reported (convention 7; PREREG.md, second set of fixes of
2026-10-08).

Design: the 21 real events with their real numbers of dates with `like` recorded. Liking = event effect (SD 0.27) +
rater effect (SD 1.0) + partner effect (SD 0.75) + slope times a predictor + noise, scaled so its SD is about the
real 1.84; raters and partners are drawn within each event (about 15 dates per rater, as in the data). Predictions
know the mean and use a wrong slope, and know none of the effects, like predictions for held-out events. Two designs:
`weak` (slope 0.3, predicted 0.65: R-squared near 0, as in the real like model) and `moderate` (slope 1.0, predicted
0.9: R-squared about 0.3), five seeds each. The target is the same quantity in a population of 4,000 events. The
SE that treats the variance of `like` as fixed (the form used before 2026-10-08) is shown for contrast.
Seeds : 20261506 + i for `weak`, 20261511 + i for `moderate` (i = 0..4).
Output: results/sim_like_cluster.csv (per replicate), results/sim_like_cluster_summary.csv, and coverage_note().
Run from the repository root: python scripts/sim_like_cluster.py [replicates per seed]
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from modeling import r_squared_and_rmse, cluster_robust_var_of_mean

SEED = 20261506
TAU, RATER, PARTNER, SD_Y, DATES_PER_RATER = 0.27, 1.0, 0.75, 1.84, 15
DESIGNS = {"weak": (0.3, 0.65, 0), "moderate": (1.0, 0.9, 5)}            # slope, predicted slope, seed offset
SUMMARY = pathlib.Path("results/sim_like_cluster_summary.csv")


def simulate(sizes, rng, slope, pred_slope):
    noise = np.sqrt(max(SD_Y ** 2 - TAU ** 2 - RATER ** 2 - PARTNER ** 2 - slope ** 2, 0.1))
    g = np.repeat(np.arange(len(sizes)), sizes)
    people = np.maximum(2, np.round(sizes / DATES_PER_RATER).astype(int))
    rater = np.concatenate([rng.integers(0, k, n) for k, n in zip(people, sizes)]) + np.repeat(np.cumsum(people) - people, sizes)
    partner = np.concatenate([rng.integers(0, k, n) for k, n in zip(people, sizes)]) + np.repeat(np.cumsum(people) - people, sizes)
    x = rng.normal(size=len(g))
    y = (6.1 + rng.normal(0, TAU, len(sizes))[g] + rng.normal(0, RATER, people.sum())[rater]
         + rng.normal(0, PARTNER, people.sum())[partner] + slope * x + rng.normal(0, noise, len(g)))
    return y, 6.1 + pred_slope * x, g


def run(sizes, design, seed, reps):
    slope, pred_slope, _ = DESIGNS[design]; rng = np.random.default_rng(seed)
    yb, pb, _ = simulate(rng.choice(sizes, 4000), rng, slope, pred_slope)
    target_r2 = 1 - np.mean((yb - pb) ** 2) / np.var(yb); target_rmse = np.sqrt(np.mean((yb - pb) ** 2))
    rows = []
    for r in range(reps):
        y, pred, g = simulate(sizes, rng, slope, pred_slope)
        m = r_squared_and_rmse(y, pred, g)
        old_se = np.sqrt(cluster_robust_var_of_mean((y - pred) ** 2, g)) / np.var(y)
        rows.append(dict(design=design, seed=seed, rep=r, **m, r_squared_se_fixed_variance=old_se,
                         r_squared_target=target_r2, rmse_target=target_rmse))
    return pd.DataFrame(rows)


def summarize(R):
    z, out = 1.645, []
    for (design, seed), d in R.groupby(["design", "seed"], sort=False):
        for k, se, label in (("r_squared", "r_squared_se", "r_squared"), ("r_squared", "r_squared_se_fixed_variance", "r_squared (variance fixed)"),
                             ("rmse", "rmse_se", "rmse")):
            emp = d[k].std()
            out.append(dict(design=design, seed=seed, metric=label, mean_se=d[se].mean(), empirical_sd=emp,
                            ratio_mean_se_to_sd=d[se].mean() / emp, coverage_90=np.mean(np.abs(d[k] - d[f"{k}_target"]) <= z * d[se]),
                            target=d[f"{k}_target"].iloc[0], reps=len(d)))
    return pd.DataFrame(out)


def coverage_note(path=SUMMARY):
    S = pd.read_csv(path)
    def rng_(metric, col, pct):
        v = S[S.metric == metric][col]
        return f"{v.min():.0%} to {v.max():.0%}" if pct else f"{v.min():.2f} to {v.max():.2f}"
    return (f"SEs checked in scripts/sim_like_cluster.py ({S.groupby('design').seed.nunique().max()} seeds x {S.design.nunique()} designs, 21 events): "
            f"R-squared SE / true SD {rng_('r_squared', 'ratio_mean_se_to_sd', False)}, 90% intervals covered {rng_('r_squared', 'coverage_90', True)}; "
            f"RMSE SE / true SD {rng_('rmse', 'ratio_mean_se_to_sd', False)}, covered {rng_('rmse', 'coverage_90', True)}")


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    sizes = clean[clean.like.notna()].groupby("event").size().to_numpy()
    R = pd.concat([run(sizes, d, SEED + off + i, reps) for d, (_, _, off) in DESIGNS.items() for i in range(5)], ignore_index=True)
    pathlib.Path("results").mkdir(exist_ok=True)
    R.to_csv("results/sim_like_cluster.csv", index=False)
    S = summarize(R); S.to_csv(SUMMARY, index=False)
    print(f"{reps} replicates per seed of 21 events ({sizes.sum()} dates); population R-squared by design: "
          + ", ".join(f"{d} {v:.3f}" for d, v in R.groupby('design', sort=False).r_squared_target.first().items()))
    for (d, k), g in S.groupby(["design", "metric"], sort=False):
        print(f"  {d:9s} {k:27s} SE/SD {g.ratio_mean_se_to_sd.min():.2f} to {g.ratio_mean_se_to_sd.max():.2f}; "
              f"90% coverage {g.coverage_90.min():.3f} to {g.coverage_90.max():.3f}")
    print(coverage_note())
    print("wrote results/sim_like_cluster.csv, results/sim_like_cluster_summary.csv")


if __name__ == "__main__":
    main()
