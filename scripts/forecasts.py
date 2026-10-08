"""Accuracy of participants' own forecasts (PREREG.md, Part 2: C1, C2; proposals/05, Layer C).

Input : data/clean.csv
Output: results/forecasts_metrics.csv (pooled and by gender), results/forecasts_reliability.csv (observed yes rate
        at each forecast value), results/forecasts_counts.csv (C2), results/forecasts_slope_check.csv (simulation
        check of the slope's error bar), results/forecasts_hypotheses.csv, results/forecasts_reliability.png
Seed  : 20262306 (parametric bootstrap of the calibration slope).

C1. On every scorecard the rater guessed how probable it was that the partner would say yes (prob, 1 to 10; zeros
kept as recorded). The forecast is p = prob / 10, clipped to [0.025, 0.975]; the outcome is the partner's actual
decision on this rater. Reported: AUC (clustered SE, Obuchowski), the calibration slope (the coefficient of logit p
in a logistic regression of the outcome, cluster-sandwich SE by event via part1_models.coefficient_table; below 1
means overconfident: guesses spread out more than outcomes do), and the Brier score with its reliability,
resolution and uncertainty parts over the distinct forecast values (Murphy 1973: Brier = reliability - resolution
+ uncertainty).
C2. Each participant's predicted number of matches (match_es) against the matches they actually got.
Run from the repository root: python scripts/forecasts.py [bootstrap replicates]
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

RES = pathlib.Path("results")
SEED, TAU = 20262306, 0.4


def forecast(prob):
    return np.clip(np.asarray(prob, float) / 10, 0.025, 0.975)


def logit(p):
    return np.log(p / (1 - p))


def murphy(p, y):
    """Brier score and its reliability, resolution and uncertainty parts over the distinct forecast values."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    d = pd.DataFrame({"p": p, "y": y}).groupby("p").y.agg(["mean", "size"])
    obar, n = y.mean(), len(y)
    rel = float((d["size"] * (d.index - d["mean"]) ** 2).sum() / n)
    res = float((d["size"] * (d["mean"] - obar) ** 2).sum() / n)
    return dict(brier=float(np.mean((p - y) ** 2)), reliability=rel, resolution=res, uncertainty=float(obar * (1 - obar)))


def score(p, y, g):
    tab, _ = P1.coefficient_table(pd.DataFrame({"logit_forecast": logit(p)}), y, g)
    t = tab.set_index("term")
    se_brier = np.sqrt(M.cluster_robust_var_of_mean((p - y) ** 2, g))
    return dict(dates=len(y), mean_forecast=p.mean(), yes_rate=y.mean(), auc=M.auc(p, y), auc_se=np.sqrt(M.auc_clustered_var(p, y, g)),
                calibration_slope=t.loc["logit_forecast", "coef"], calibration_slope_se=t.loc["logit_forecast", "se_cluster_event"],
                calibration_intercept=t.loc["const", "coef"], brier_se=se_brier, **murphy(p, y))


def slope_check(p, y, g, reps, rng):
    """Parametric bootstrap on the real design: outcomes from the fitted calibration model plus an event effect."""
    tab, fitted = P1.coefficient_table(pd.DataFrame({"x": logit(p)}), y, g)
    lp = np.log(fitted / (1 - fitted)); ev = np.unique(g); gi = np.searchsorted(ev, g)
    est, se = [], []
    for _ in range(reps):
        ys = (rng.random(len(y)) < 1 / (1 + np.exp(-(lp + rng.normal(0, TAU, len(ev))[gi])))).astype(float)
        t = P1.coefficient_table(pd.DataFrame({"x": logit(p)}), ys, g, path=(1e-6, 1e-6))[0].set_index("term")
        est.append(t.loc["x", "coef"]); se.append(t.loc["x", "se_cluster_event"])
    est, se = np.array(est), np.array(se)
    return dict(reps=reps, seed=SEED, event_sd=TAU, mean_se=se.mean(), empirical_sd=est.std(ddof=1), ratio=se.mean() / est.std(ddof=1),
                coverage_90=np.mean(np.abs(est - np.median(est)) <= 1.645 * se))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    d = clean[clean.prob.notna()].copy()
    d["p"] = forecast(d.prob); d["y"] = d.dec_o.astype(float)
    rows, rel = [], []
    for name, s in (("both", d), ("women", d[d.gender == 0]), ("men", d[d.gender == 1])):
        rows.append(dict(rater_gender=name, **score(s.p.to_numpy(), s.y.to_numpy(), s.event.to_numpy())))
        r = s.groupby("p").agg(observed_yes_rate=("y", "mean"), dates=("y", "size")).reset_index().assign(rater_gender=name)
        rel.append(r)
    met = pd.DataFrame(rows).assign(fold_scheme="none (all dates)", forecast="prob / 10 clipped to [0.025, 0.975]")
    chk = slope_check(d.p.to_numpy(), d.y.to_numpy(), d.event.to_numpy(), reps, np.random.default_rng(SEED))
    pd.DataFrame([chk]).to_csv(RES / "forecasts_slope_check.csv", index=False)
    met = met.assign(se_check=f"slope SE checked by parametric bootstrap (event SD {TAU}, {reps} replicates): SE / spread {chk['ratio']:.2f}, "
                               f"90% coverage {chk['coverage_90']:.0%}; AUC SE as in scripts/sim_auc_cluster.py")
    met.to_csv(RES / "forecasts_metrics.csv", index=False)
    R = pd.concat(rel); R.to_csv(RES / "forecasts_reliability.csv", index=False)

    people = clean.drop_duplicates("rater")[["rater", "event", "gender", "match_es"]]
    realized = clean.groupby("rater").match.sum().rename("realized")
    people = people.join(realized, on="rater").dropna(subset=["match_es"])
    people["over"] = people.match_es - people.realized
    cnt = []
    for name, s in (("both", people), ("women", people[people.gender == 0]), ("men", people[people.gender == 1])):
        cnt.append(dict(gender=name, participants=len(s), mean_predicted=s.match_es.mean(), mean_realized=s.realized.mean(),
                        mean_over=s.over.mean(), mean_over_se=np.sqrt(M.cluster_robust_var_of_mean(s.over.to_numpy(), s.event.to_numpy())),
                        share_over=(s.over > 0).mean(), share_exact=(s.over == 0).mean(),
                        correlation=np.corrcoef(s.match_es, s.realized)[0, 1]))
    C = pd.DataFrame(cnt).assign(fold_scheme="none (all participants)"); C.to_csv(RES / "forecasts_counts.csv", index=False)

    b = met.set_index("rater_gender").loc["both"]; cb = C.set_index("gender").loc["both"]
    hyp = pd.DataFrame([
        dict(hypothesis="C1", version="all dates", statistic=f"AUC {b.auc:.4f} (SE {b.auc_se:.4f}); calibration slope {b.calibration_slope:.3f} "
             f"(SE {b.calibration_slope_se:.3f})", supported=bool(b.auc > 0.6 and b.calibration_slope < 0.8)),
        dict(hypothesis="C2", version="participants with a predicted count", statistic=f"predicted {cb.mean_predicted:.2f} against realized "
             f"{cb.mean_realized:.2f} matches: mean over-prediction {cb.mean_over:+.2f} (SE {cb.mean_over_se:.2f}), {int(cb.participants)} people",
             supported=bool(cb.mean_over > 0))]).assign(fold_scheme="none")
    hyp.to_csv(RES / "forecasts_hypotheses.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
    for name, color in (("women", "#2a78d6"), ("men", "#eb6834")):
        r = R[(R.rater_gender == name) & (R.dates >= 20)]
        ax.plot(r.p, r.observed_yes_rate, "o-", color=color, ms=4, lw=1.5, label=name)
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, zorder=0)
    ax.set_xlabel("Guess that the partner says yes (prob / 10)"); ax.set_ylabel("Share of partners who said yes")
    ax.set_title("Reliability of participants' guesses\n(forecast values with at least 20 dates)", loc="left", fontsize=10)
    ax.legend(frameon=False); ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RES / "forecasts_reliability.png", dpi=200); plt.close(fig)

    print(met.drop(columns=["se_check"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame([chk]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(C.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
