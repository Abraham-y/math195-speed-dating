"""Regression to the mean at the top (PREREG.md, Part 3: R1, R2; proposals/05, Layer D).

Input : data/clean.csv
Output: results/mean_reversion_leaders.csv (each block's odd-half leader), results/mean_reversion_summary.csv,
        results/mean_reversion_hypotheses.csv
No random operations: the halves are fixed by the position of each date in the rater's evening (`order`).

Each participant's received ratings (from the opposite sex at their event) are split into odd-position and
even-position dates. R1: the participant with the highest odd-half mean in each wave-and-gender block is rated
lower on the even half (the winner's curse). R2: shrinking the odd-half means toward the block mean by the
empirical-Bayes factor tau2 / (tau2 + s2_i) predicts the even-half means better than the raw odd-half means.
Liking is primary; attractiveness is reported alongside; the reverse split (even to odd) is reported too.
Run from the repository root: python scripts/mean_reversion.py
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")


def halves(clean, col):
    """Per participant and half: mean, variance and count of the ratings received; participants with at least two
    ratings in each half."""
    d = clean[["ratee", "event", "gender", "order", col]].dropna().copy()
    d["ratee_gender"] = 1 - d.gender
    d["half"] = np.where(d["order"] % 2 == 1, "odd", "even")
    g = d.groupby(["event", "ratee_gender", "ratee", "half"])[col].agg(["mean", "var", "size"]).unstack("half")
    keep = (g[("size", "odd")] >= 2) & (g[("size", "even")] >= 2)
    g = g[keep]
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def shrink(raw, var_within, n):
    """Empirical-Bayes shrinkage of raw means toward their mean: m + tau2 / (tau2 + s2_i) (raw_i - m), with
    s2_i = pooled within-person variance / n_i and tau2 = max(0, var(raw) - mean(s2_i))."""
    raw, var_within, n = (np.asarray(v, float) for v in (raw, var_within, n))
    pooled = np.sum((n - 1) * var_within) / np.sum(n - 1)
    s2 = pooled / n
    m = raw.mean()
    tau2 = max(0.0, raw.var(ddof=1) - s2.mean())
    return m + tau2 / (tau2 + s2) * (raw - m), tau2 / (tau2 + s2)


def analyse(H, train, test):
    leaders, preds = [], []
    for (ev, g), b in H.groupby(["event", "ratee_gender"]):
        b = b.sort_values("ratee")
        i = b[f"mean_{train}"].to_numpy().argmax()
        leaders.append(dict(event=ev, ratee_gender="women" if g == 0 else "men", ratee=int(b.ratee.iloc[i]), participants=len(b),
                            leader_train_mean=b[f"mean_{train}"].iloc[i], leader_test_mean=b[f"mean_{test}"].iloc[i],
                            drop=b[f"mean_{train}"].iloc[i] - b[f"mean_{test}"].iloc[i]))
        sh, keep = shrink(b[f"mean_{train}"], b[f"var_{train}"].fillna(0), b[f"size_{train}"])
        preds.append(pd.DataFrame({"event": ev, "raw": b[f"mean_{train}"].to_numpy(), "shrunk": sh, "target": b[f"mean_{test}"].to_numpy(), "kept": keep}))
    L, P = pd.DataFrame(leaders), pd.concat(preds, ignore_index=True)
    rmse = lambda c: float(np.sqrt(np.mean((P[c] - P.target) ** 2)))
    return L, dict(blocks=len(L), participants=len(P), mean_drop=L["drop"].mean(),
                   mean_drop_se=np.sqrt(M.cluster_robust_var_of_mean(L["drop"].to_numpy(), L.event.to_numpy())),
                   share_leaders_dropping=(L["drop"] > 0).mean(), rmse_raw=rmse("raw"), rmse_shrunk=rmse("shrunk"),
                   median_share_kept=float(np.median(P.kept)))      # 1 = no shrinkage, 0 = all the way to the block mean


def main():
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    rows, lead = [], []
    for col in ("like", "attr"):
        H = halves(clean, col)
        for train, test in (("odd", "even"), ("even", "odd")):
            L, s = analyse(H, train, test)
            rows.append(dict(rating=col, direction=f"{train} to {test}", **s))
            lead.append(L.assign(rating=col, direction=f"{train} to {test}"))
    S = pd.DataFrame(rows).assign(fold_scheme="split by date position (odd/even order)")
    S.to_csv(RES / "mean_reversion_summary.csv", index=False)
    pd.concat(lead).to_csv(RES / "mean_reversion_leaders.csv", index=False)
    p = S[(S.rating == "like") & (S.direction == "odd to even")].iloc[0]
    hyp = pd.DataFrame([
        dict(hypothesis="R1", version="liking, odd to even", statistic=f"leaders drop {p.mean_drop:.3f} points on average (SE {p.mean_drop_se:.3f}) "
             f"over {int(p.blocks)} blocks; {p.share_leaders_dropping:.0%} of leaders drop", supported=bool(p.mean_drop >= 0.5)),
        dict(hypothesis="R2", version="liking, odd to even", statistic=f"RMSE shrunk {p.rmse_shrunk:.4f} against raw {p.rmse_raw:.4f} "
             f"({int(p.participants)} participants)", supported=bool(p.rmse_shrunk < p.rmse_raw))])
    hyp.to_csv(RES / "mean_reversion_hypotheses.csv", index=False)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(hyp.to_string(index=False))


if __name__ == "__main__":
    main()
