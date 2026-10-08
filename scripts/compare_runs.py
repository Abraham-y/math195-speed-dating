"""Earlier results next to the rerun after the fixes of 2026-10-08 (PREREG.md: each changed result is reported next
to the earlier one).

Input : results/earlier/*.csv (copied before the fixes) and the same tables in results/ after the rerun
Output: results/compare_runs_metrics.csv (every model row of Part 1 and the additions, earlier and new),
        results/compare_runs_hypotheses.csv (every hypothesis row, earlier and new verdict),
        results/compare_runs_like.csv (the like model)
No random operations, so no seed.
Run from the repository root after part1_models.py, part1_additions.py and like_models.py: python scripts/compare_runs.py
"""
import pathlib
import numpy as np, pandas as pd

RES, OLD = pathlib.Path("results"), pathlib.Path("results/earlier")
METRICS = ["auc", "balanced_accuracy", "sensitivity", "specificity", "error_rate"]
SES = [f"{m}_se" for m in METRICS]


def side_by_side(old, new, keys, cols):
    m = old[keys + cols].merge(new[keys + cols], on=keys, how="outer", suffixes=("_earlier", "_new"), indicator=True)
    m["row"] = m.pop("_merge").map({"both": "both runs", "left_only": "earlier only", "right_only": "new only"})
    for c in cols:
        if pd.api.types.is_numeric_dtype(m[f"{c}_new"]):
            m[f"{c}_change"] = m[f"{c}_new"] - m[f"{c}_earlier"]
    for c in SES:
        if f"{c}_earlier" in m:
            m[f"{c}_ratio"] = m[f"{c}_new"] / m[f"{c}_earlier"]
    return m


def main():
    p1 = side_by_side(pd.read_csv(OLD / "part1_models_metrics.csv"), pd.read_csv(RES / "part1_models_metrics.csv"),
                      ["spec", "headline", "feature_set", "model"], ["dyads", "matches"] + METRICS + SES).assign(table="part1_models")
    ad = side_by_side(pd.read_csv(OLD / "part1_additions_metrics.csv"), pd.read_csv(RES / "part1_additions_metrics.csv"),
                      ["part", "version", "feature_set", "model"], ["dyads", "matches"] + METRICS + SES).assign(table="part1_additions")
    met = pd.concat([p1, ad], ignore_index=True)
    met.to_csv(RES / "compare_runs_metrics.csv", index=False)

    hyp = []
    for name in ("part1_models", "part1_additions", "like_models"):
        o, n = pd.read_csv(OLD / f"{name}_hypotheses.csv"), pd.read_csv(RES / f"{name}_hypotheses.csv")
        for d in (o, n):
            if "version" not in d:
                d["version"] = "headline"
        hyp.append(side_by_side(o, n, ["hypothesis", "version"], ["statistic", "supported"]).assign(table=name))
    hyp = pd.concat(hyp, ignore_index=True)
    hyp["verdict"] = np.select([hyp.row != "both runs", hyp.supported_earlier == hyp.supported_new],
                               [hyp.row, "unchanged"], "CHANGED")
    hyp.to_csv(RES / "compare_runs_hypotheses.csv", index=False)

    lo, ln = pd.read_csv(OLD / "like_models_metrics.csv").assign(version="headline"), pd.read_csv(RES / "like_models_metrics.csv")
    like = lo.merge(ln, on=["version", "model"], how="outer", suffixes=("_earlier", "_new"))
    like.to_csv(RES / "compare_runs_like.csv", index=False)

    both = met[met.row == "both runs"]
    head = both[(both.table == "part1_models") & both.headline.astype(bool)]
    print("Part 1, headline specification (event 5-fold CV), earlier -> new:")
    for r in head.itertuples():
        print(f"  {r.feature_set:15s} {r.model:18s} AUC {r.auc_earlier:.3f} -> {r.auc_new:.3f}   balanced accuracy {r.balanced_accuracy_earlier:.3f} "
              f"(SE {r.balanced_accuracy_se_earlier:.3f}) -> {r.balanced_accuracy_new:.3f} (SE {r.balanced_accuracy_se_new:.3f})")
    print("\nall rows in both runs (Part 1 grid and additions):")
    for c in METRICS:
        print(f"  {c:18s} largest point change {both[f'{c}_change'].abs().max():.4f}; "
              f"SE new / earlier: median {both[f'{c}_se_ratio'].median():.2f}, range {both[f'{c}_se_ratio'].min():.2f} to {both[f'{c}_se_ratio'].max():.2f}")
    ch = both.loc[both.dyads_change != 0, ["table", "spec", "part", "version", "feature_set", "model", "dyads_earlier", "dyads_new"]]
    if len(ch):
        print("\nrows whose dyads changed:"); print(ch.drop_duplicates(["table", "spec", "part", "version", "dyads_earlier", "dyads_new"]).to_string(index=False))
    print("\nhypotheses (earlier -> new):")
    for r in hyp.itertuples():
        old = "" if pd.isna(r.supported_earlier) else ("supported" if r.supported_earlier else "not supported")
        new = "" if pd.isna(r.supported_new) else ("supported" if r.supported_new else "not supported")
        print(f"  {r.table:16s} {r.hypothesis:4s} {r.version:38s} {old:13s} -> {new:13s} [{r.verdict}]")
    print("\nlike model (held-out R-squared, earlier -> new):")
    for r in like.itertuples():
        print(f"  {r.version:13s} {r.model:24s} {r.r_squared_earlier:+.4f} -> {r.r_squared_new:+.4f}  ({r.dates_new:.0f} dates)")
    print("wrote results/compare_runs_metrics.csv, results/compare_runs_hypotheses.csv, results/compare_runs_like.csv")


if __name__ == "__main__":
    main()
