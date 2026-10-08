"""Check the cluster-robust sandwich SEs of the coefficient table and of the yes model (part1_models.coefficient_table)
on their real designs before they are reported (convention 7; PREREG.md, second set of fixes of 2026-10-08).

Parametric bootstrap: the fitted probabilities of the real fit, shifted on the logit by an event effect of SD 0.4
(the calibrated design of scripts/sim_auc_cluster.py), give new outcomes, and each replicate is refitted by the same
code with the same events as clusters. For each term: the mean of its SEs over the replicates against the spread of
its estimates (a robust SD, 1.4826 times the median absolute deviation, because sparse dummies are sometimes
separated in a replicate), and the share of replicates whose 90% interval covers the median estimate. Terms not
estimable in the real fit are left out.
Designs: the headline coefficient table (`pre` and `pre+date`), and the yes model for women and for men.
Seed  : 20261406 (replicate r of design d uses seed + 1000 d + r).
Output: results/sim_coef_cluster.csv (one row per design and term), results/sim_coef_cluster_summary.csv (per
        design), and coverage_note(), the one-line range stored with the tables these SEs appear in.
Run from the repository root: python scripts/sim_coef_cluster.py [replicates for the coefficient table] [for the yes model]
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

SEED, TAU = 20261406, 0.4
SUMMARY = pathlib.Path("results/sim_coef_cluster_summary.csv")


def bootstrap(D, y, g, reps, seed):
    """Per-term mean SE, robust spread and 90% coverage around the median estimate over parametric replicates."""
    real, p = P1.coefficient_table(D, y, g)
    ok = real.estimable.to_numpy() & ~real.term.str.startswith("sum: ").to_numpy()
    terms = real.term[ok].tolist()
    logit = np.log(p / (1 - p)); ev = np.unique(g); gi = np.searchsorted(ev, g)
    est, ses = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        ys = (rng.random(len(y)) < 1 / (1 + np.exp(-(logit + rng.normal(0, TAU, len(ev))[gi])))).astype(float)
        t = P1.coefficient_table(D, ys, g, path=(1e-6, 1e-6))[0].set_index("term")       # one fit: the SE formula is checked
        est.append(t.coef.reindex(terms).to_numpy()); ses.append(t.se_cluster_event.reindex(terms).to_numpy())
    est, ses = np.array(est), np.array(ses)
    med = np.median(est, axis=0)
    spread = 1.4826 * np.median(np.abs(est - med), axis=0)
    cover = np.mean(np.abs(est - med) <= 1.645 * ses, axis=0)
    return pd.DataFrame({"term": terms, "estimate_real": real.coef[ok].to_numpy(), "median_estimate": med,
                         "mean_se": np.nanmean(ses, axis=0), "median_se": np.median(ses, axis=0), "robust_sd": spread,
                         "ratio_se_to_sd": np.median(ses, axis=0) / spread, "coverage_90": cover, "reps": reps})


def designs():
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    for fs in ("pre", "pre+date"):
        _, D, y, ev, _ = P1.full_fit(pairs, sets, M.HEADLINE, fs)
        yield f"coefficients {fs}", "coefficients", D, y, ev
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    for gnum, name in ((0, "women"), (1, "men")):
        Z, yv, ev, _ = P1.yes_design(clean, gnum)
        yield f"yes model {name}", "yes", Z.reset_index(drop=True), yv, ev


def coverage_note(kind, path=SUMMARY):
    """One line for the tables whose SEs this checks: kind 'coefficients' or 'yes'."""
    S = pd.read_csv(path); S = S[S.kind == kind]
    span = lambda lo, hi, f: f(lo) if f(lo) == f(hi) else f"{f(lo)} to {f(hi)}"
    two, pct = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.0%}")
    if kind == "coefficients":
        return (f"SEs checked in scripts/sim_coef_cluster.py (parametric bootstrap on the real headline designs, event SD {TAU}, "
                f"{int(S.reps.iloc[0])} replicates): SEs too narrow, median SE / spread {span(S.median_ratio.min(), S.median_ratio.max(), two)} "
                f"(10% to 90% of terms {two(S.ratio_p10.min())} to {two(S.ratio_p90.max())}); median 90% coverage "
                f"{span(S.median_coverage.min(), S.median_coverage.max(), pct)} (lowest term {pct(S.coverage_min.min())})")
    return (f"SEs checked in scripts/sim_coef_cluster.py (parametric bootstrap on the real yes-model designs, event SD {TAU}, "
            f"{int(S.reps.iloc[0])} replicates): SE / spread {two(S.ratio_min.min())} to {two(S.ratio_max.max())} across the "
            f"{int(S.terms.sum())} coefficients (intercepts included); 90% coverage {pct(S.coverage_min.min())} to {pct(S.coverage_max.max())}")


def main():
    reps_coef = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    reps_yes = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    out, summ = [], []
    for d, (name, kind, D, y, ev) in enumerate(designs()):
        reps = reps_coef if kind == "coefficients" else reps_yes
        t = bootstrap(D, np.asarray(y, float), np.asarray(ev), reps, SEED + 1000 * d).assign(design=name, kind=kind, seed=SEED + 1000 * d)
        out.append(t)
        summ.append(dict(design=name, kind=kind, terms=len(t), reps=reps, seed=SEED + 1000 * d,
                         median_ratio=t.ratio_se_to_sd.median(), ratio_p10=t.ratio_se_to_sd.quantile(0.1), ratio_p90=t.ratio_se_to_sd.quantile(0.9),
                         ratio_min=t.ratio_se_to_sd.min(), ratio_max=t.ratio_se_to_sd.max(),
                         median_coverage=t.coverage_90.median(), coverage_min=t.coverage_90.min(), coverage_max=t.coverage_90.max()))
        print(f"{name}: {len(t)} terms, {reps} replicates; SE / spread median {t.ratio_se_to_sd.median():.2f} "
              f"(10% to 90% of terms {t.ratio_se_to_sd.quantile(0.1):.2f} to {t.ratio_se_to_sd.quantile(0.9):.2f}); "
              f"90% coverage median {t.coverage_90.median():.3f} (range {t.coverage_90.min():.3f} to {t.coverage_90.max():.3f})", flush=True)
    pathlib.Path("results").mkdir(exist_ok=True)
    pd.concat(out, ignore_index=True).to_csv("results/sim_coef_cluster.csv", index=False)
    pd.DataFrame(summ).to_csv(SUMMARY, index=False)
    print(coverage_note("coefficients")); print(coverage_note("yes"))
    print("wrote results/sim_coef_cluster.csv, results/sim_coef_cluster_summary.csv")


if __name__ == "__main__":
    main()
