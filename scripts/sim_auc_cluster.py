"""Check the clustered standard errors of the Part 1 metric table on simulated data with the speed-dating design
before they are reported (convention 7; PREREG.md model building, and the fixes entry of 2026-10-08).

Design: the 21 real event sizes (dyads per event). Each event has a random effect on the match logit and a random
shift in the score, so dyads in an event are correlated; the match rate is close to the real 16.5%. Two designs:
`first` (event SD 0.6 on the logit, the original check) and `calibrated` (event SD 0.4, whose spread of match
rates across events is close to the real one). Each runs five seeds. For each replicate, modeling.metric_table
scores the dyads at a fixed threshold, giving the AUC with its clustered SE (Obuchowski 1997) and sensitivity,
specificity and balanced accuracy with their cluster-robust SEs (linearized ratio form); the AUC also gets the
independent-rows SE (DeLong) for contrast. Targets are the population values of the same process, from 4,000
events. Output: results/sim_auc_cluster.csv (per replicate), results/sim_auc_cluster_summary.csv (per design,
seed and SE), and coverage_note(), the one-line range stored with every results table that reports these SEs.
Run from the repository root: python scripts/sim_auc_cluster.py [replicates per seed]
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from modeling import auc, metric_table, placement_values

SEED, SEEDS = 20261006, 5                                     # seeds SEED, SEED + 1, ..., SEED + 4
TAU_S, BETA, ALPHA = 0.4, 0.8, -1.95                          # event shift in the score, slope, intercept
DESIGNS = {"first": 0.6, "calibrated": 0.4}                   # event SD on the match logit
METRICS = ["auc", "sensitivity", "specificity", "balanced_accuracy"]
SUMMARY = pathlib.Path("results/sim_auc_cluster_summary.csv")


def simulate(sizes, rng, tau_y):
    g = np.repeat(np.arange(len(sizes)), sizes)
    a, b = rng.normal(0, tau_y, len(sizes)), rng.normal(0, TAU_S, len(sizes))
    x = rng.normal(size=len(g))
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-(ALPHA + BETA * x + a[g])))).astype(int)
    return x + b[g], y, g


def population(sizes, rng, tau_y):
    """Population AUC, threshold (the score above which 16.5% of dyads fall) and the metrics at that threshold."""
    s, y, _ = simulate(rng.choice(sizes, 4000), rng, tau_y)
    thr = np.quantile(s, 1 - 0.165)
    sens, spec = (s[y == 1] > thr).mean(), (s[y == 0] <= thr).mean()
    return thr, dict(auc=auc(s, y), sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2)


def run(sizes, design, seed, reps):
    tau_y = DESIGNS[design]; rng = np.random.default_rng(seed)
    thr, target = population(sizes, rng, tau_y)
    rows = []
    for r in range(reps):
        s, y, g = simulate(sizes, rng, tau_y)
        m = metric_table(s, y, np.full(len(y), thr), g)
        v10, v01 = placement_values(s, y)
        rate = pd.Series(y).groupby(g).mean()
        rows.append(dict(design=design, seed=seed, rep=r, match_rate=y.mean(), event_rate_sd=rate.std(),
                         auc_se_independent=np.sqrt(np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)),
                         **{k: m[k][0] for k in METRICS}, **{f"{k}_se": m[k][1] for k in METRICS},
                         **{f"{k}_target": target[k] for k in METRICS}))
    return pd.DataFrame(rows)


def summarize(R):
    z = 1.645; out = []
    for (design, seed), d in R.groupby(["design", "seed"], sort=False):
        for k, se in [(k, f"{k}_se") for k in METRICS] + [("auc", "auc_se_independent")]:
            emp = d[k].std()
            out.append(dict(design=design, seed=seed, metric=k, se_type="independent" if se.endswith("independent") else "clustered",
                            mean_se=d[se].mean(), empirical_sd=emp, ratio_mean_se_to_sd=d[se].mean() / emp,
                            coverage_90=np.mean(np.abs(d[k] - d[f"{k}_target"]) <= z * d[se]),
                            target=d[f"{k}_target"].iloc[0], event_rate_sd=d.event_rate_sd.mean(), reps=len(d)))
    return pd.DataFrame(out)


def coverage_note(path=SUMMARY):
    """One line for every results table: the range of the clustered AUC and balanced-accuracy SE checks."""
    S = pd.read_csv(path); S = S[S.se_type == "clustered"]
    def rng_(metric, col, pct):
        v = S[S.metric == metric][col]
        return f"{v.min():.0%} to {v.max():.0%}" if pct else f"{v.min():.2f} to {v.max():.2f}"
    return (f"SEs checked in scripts/sim_auc_cluster.py ({S.seed.nunique()} seeds x {S.design.nunique()} designs, 21 events): "
            f"AUC SE / true SD {rng_('auc', 'ratio_mean_se_to_sd', False)}, 90% intervals covered {rng_('auc', 'coverage_90', True)}; "
            f"balanced accuracy SE / true SD {rng_('balanced_accuracy', 'ratio_mean_se_to_sd', False)}, "
            f"covered {rng_('balanced_accuracy', 'coverage_90', True)}")


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    pairs = pd.read_csv("data/pairs.csv")
    sizes = pairs.groupby("event").size().to_numpy()
    R = pd.concat([run(sizes, d, SEED + i, reps) for d in DESIGNS for i in range(SEEDS)], ignore_index=True)
    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    R.to_csv(out / "sim_auc_cluster.csv", index=False)
    S = summarize(R); S.to_csv(SUMMARY, index=False)
    real_sd = pairs.groupby("event")["match"].mean().std()
    print(f"seeds {SEED} to {SEED + SEEDS - 1}; {reps} replicates per seed of 21 events ({sizes.sum()} dyads); "
          f"SD of event match rates: real {real_sd:.3f}, " + ", ".join(f"{d} {R[R.design == d].event_rate_sd.mean():.3f}" for d in DESIGNS))
    for (d, k, t), g in S.groupby(["design", "metric", "se_type"], sort=False):
        print(f"  {d:10s} {k:17s} {t:11s} SE/SD {g.ratio_mean_se_to_sd.min():.2f} to {g.ratio_mean_se_to_sd.max():.2f}; "
              f"90% coverage {g.coverage_90.min():.3f} to {g.coverage_90.max():.3f}")
    print(coverage_note())
    print("wrote results/sim_auc_cluster.csv, results/sim_auc_cluster_summary.csv")


if __name__ == "__main__":
    main()
