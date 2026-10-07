"""Check the clustered AUC standard error (Obuchowski 1997, modeling.auc_clustered_var) on simulated data with
the speed-dating design before it is reported (convention 7, PREREG.md model building).

Design: the 21 real event sizes (dyads per event). Each event has a random effect on the match logit and a random
shift in the score, so dyads in an event are correlated; the match rate is close to the real 16.5%. For each
replicate the AUC of the score is computed with three standard errors: independent rows (DeLong), and clustered by
event (Obuchowski). The target is the population AUC of the same process, from a very large number of events.
Output: results/sim_auc_cluster.csv (per replicate) and results/sim_auc_cluster_summary.csv.
Run from the repository root: python scripts/sim_auc_cluster.py [replicates]
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from modeling import auc, auc_clustered_var, placement_values

SEED = 20261006
TAU_Y, TAU_S, BETA, ALPHA = 0.6, 0.4, 0.8, -1.95   # event effect on the logit, event shift in the score, slope, intercept


def simulate(sizes, rng):
    g = np.repeat(np.arange(len(sizes)), sizes)
    a, b = rng.normal(0, TAU_Y, len(sizes)), rng.normal(0, TAU_S, len(sizes))
    x = rng.normal(size=len(g))
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-(ALPHA + BETA * x + a[g])))).astype(int)
    return x + b[g], y, g


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    sizes = pd.read_csv("data/pairs.csv").groupby("event").size().to_numpy()
    rng = np.random.default_rng(SEED)
    big = rng.choice(sizes, 4000)                              # population: many events of the same sizes
    s, y, _ = simulate(big, rng)
    target = auc(s, y)
    rows = []
    for r in range(reps):
        s, y, g = simulate(sizes, rng)
        v10, v01 = placement_values(s, y)
        rows.append(dict(rep=r, auc=auc(s, y), match_rate=y.mean(),
                         se_independent=np.sqrt(np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)),
                         se_clustered=np.sqrt(auc_clustered_var(s, y, g))))
    R = pd.DataFrame(rows); out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    R.assign(seed=SEED).to_csv(out / "sim_auc_cluster.csv", index=False)
    z = 1.645; emp = R.auc.std()
    summ = pd.DataFrame([dict(se_type=k, mean_se=R[k].mean(), empirical_sd=emp, ratio_mean_se_to_sd=R[k].mean() / emp,
                              coverage_90=np.mean(np.abs(R.auc - target) <= z * R[k]))
                         for k in ("se_independent", "se_clustered")]).assign(target_auc=target, reps=reps, seed=SEED)
    summ.to_csv(out / "sim_auc_cluster_summary.csv", index=False)
    print(f"seed {SEED}; {reps} replicates of 21 events ({sizes.sum()} dyads); match rate {R.match_rate.mean():.3f}; "
          f"population AUC {target:.4f}; mean AUC {R.auc.mean():.4f}; empirical SD {emp:.4f}")
    for r in summ.itertuples():
        print(f"  {r.se_type:15s} mean SE {r.mean_se:.4f}  (ratio to SD {r.ratio_mean_se_to_sd:.2f})  90% coverage {r.coverage_90:.3f}")
    print("wrote results/sim_auc_cluster.csv, results/sim_auc_cluster_summary.csv")


if __name__ == "__main__":
    main()
