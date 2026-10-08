"""Earlier results next to the current ones (PREREG.md: each changed result is reported next to the earlier one).

Input : an earlier results folder (results/earlier/ = commit 4948d9f, before the first set of fixes;
        results/earlier_ecfc18e/ = commit ecfc18e, before the second) and the same tables in results/
Output: results/compare_runs_<label>_<table>.csv for every table compared, and results/compare_runs_<label>_summary.md
        (committed: the verdicts, the headline numbers, the largest changes and the checksums of every file compared)
No random operations, so no seed.
Run from the repository root after the analysis scripts:
    python scripts/compare_runs.py --earlier results/earlier_ecfc18e --label ecfc18e
"""
import argparse, hashlib, pathlib
import numpy as np, pandas as pd

RES = pathlib.Path("results")
METRICS = ["auc", "balanced_accuracy", "sensitivity", "specificity", "error_rate"]
SES = [f"{m}_se" for m in METRICS]
# table -> (keys, compared columns)
TABLES = {
    "part1_models_metrics": (["spec", "feature_set", "model"], ["dyads", "matches"] + METRICS + SES),
    "part1_additions_metrics": (["part", "version", "feature_set", "model"], ["dyads", "matches"] + METRICS + SES),
    "like_models_metrics": (["version", "model"], ["dates", "predictors", "r_squared", "r_squared_se", "rmse", "rmse_se"]),
    "part1_models_coefficients": (["spec", "feature_set", "term"], ["coef", "se_cluster_event", "estimable"]),
    "part1_models_vif": (["feature_set", "term"], ["vif"]),
    "part1_models_stated_vs_revealed": (["rater_gender", "attribute"], ["std_coef", "se_cluster_event", "revealed_share", "stated_share",
                                                                        "gap_revealed_minus_stated", "stated_people"]),
    "part1_models_calibration": (["feature_set", "decile"], ["mean_predicted", "observed_rate", "dyads"]),
}
HYPOTHESES = ["part1_models_hypotheses", "part1_additions_hypotheses", "like_models_hypotheses"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


def harmonize(name, d):
    """Earlier runs labelled some rows differently; map them onto the current labels."""
    d = d.copy()
    if name == "part1_additions_metrics":
        d.loc[(d.part != "ladder") & (d.version == "headline"), "version"] = "folds re-drawn"
    if name == "like_models_metrics" and "version" not in d:
        d["version"] = "headline"
    if name == "part1_models_hypotheses" or name == "part1_additions_hypotheses" or name == "like_models_hypotheses":
        if "version" not in d:
            d["version"] = "headline"
        d.loc[d.version.isin(["waves 1-5, 10-21", "without wave 12"]), "version"] += ", folds re-drawn"
    return d


def side_by_side(old, new, keys, cols):
    cols = [c for c in cols if c in old.columns or c in new.columns]
    for d in (old, new):
        for c in cols:
            if c not in d:
                d[c] = np.nan
    m = old[keys + cols].merge(new[keys + cols], on=keys, how="outer", suffixes=("_earlier", "_new"), indicator=True)
    m["row"] = m.pop("_merge").map({"both": "both runs", "left_only": "earlier only", "right_only": "new only"})
    for c in cols:
        a, b = m[f"{c}_earlier"], m[f"{c}_new"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            m[f"{c}_change"] = b - a
            if c.endswith("_se") or c == "se_cluster_event":
                m[f"{c}_ratio"] = b / a
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--earlier", required=True); ap.add_argument("--label", required=True)
    args = ap.parse_args(); old_dir, label = pathlib.Path(args.earlier), args.label
    out, files, lines = {}, [], []
    for name, (keys, cols) in TABLES.items():
        o, n = old_dir / f"{name}.csv", RES / f"{name}.csv"
        files.append((name, sha(o), sha(n)))
        if not (o.exists() and n.exists()):
            continue
        m = side_by_side(harmonize(name, pd.read_csv(o)), harmonize(name, pd.read_csv(n)), keys, cols)
        m.to_csv(RES / f"compare_runs_{label}_{name}.csv", index=False); out[name] = m
    hyp = []
    for name in HYPOTHESES:
        o, n = old_dir / f"{name}.csv", RES / f"{name}.csv"
        files.append((name, sha(o), sha(n)))
        if o.exists() and n.exists():
            hyp.append(side_by_side(harmonize(name, pd.read_csv(o)), harmonize(name, pd.read_csv(n)), ["hypothesis", "version"],
                                    ["statistic", "supported"]).assign(table=name))
    hyp = pd.concat(hyp, ignore_index=True)
    se, sn = hyp.supported_earlier.astype("string"), hyp.supported_new.astype("string")
    same = (se == sn).fillna(False).to_numpy(bool)
    hyp["verdict"] = np.select([(hyp.row != "both runs").to_numpy(), (sn.isna() & se.isna()).to_numpy(bool),
                                (sn.isna() & se.notna()).to_numpy(bool), same],
                               [hyp.row.to_numpy(), "no verdict", "now reported without a verdict", "unchanged"], "CHANGED")
    hyp.to_csv(RES / f"compare_runs_{label}_hypotheses.csv", index=False)

    lines += [f"# Earlier results ({label}) next to the current ones", "",
              f"Generated by scripts/compare_runs.py from `{old_dir}` and `results/`. Tables: `results/compare_runs_{label}_*.csv`.", "",
              "## Hypotheses", "", "| Table | Hypothesis | Version | Earlier | Now | |", "| --- | --- | --- | --- | --- | --- |"]
    word = lambda v: "" if pd.isna(v) else ("supported" if str(v) == "True" else "not supported" if str(v) == "False" else "no verdict")
    for r in hyp.itertuples():
        lines.append(f"| {r.table} | {r.hypothesis} | {r.version} | {word(r.supported_earlier)} | {word(r.supported_new)} | {r.verdict} |")
    changed = int((hyp.verdict == "CHANGED").sum())
    lines += ["", f"Verdicts changed: {changed}.", ""]
    if "part1_models_metrics" in out:
        m = out["part1_models_metrics"]; h = m[(m.row == "both runs") & m.spec.str.startswith("missing=median_flags; fit=B; out_of_range=rule; format=standardized")]
        lines += ["## Headline models (event 5-fold CV)", "", "| Feature set | Model | AUC earlier | AUC now | Balanced accuracy earlier (SE) | Now (SE) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in h.itertuples():
            lines.append(f"| {r.feature_set} | {r.model} | {r.auc_earlier:.4f} | {r.auc_new:.4f} | {r.balanced_accuracy_earlier:.4f} ({r.balanced_accuracy_se_earlier:.4f}) | "
                         f"{r.balanced_accuracy_new:.4f} ({r.balanced_accuracy_se_new:.4f}) |")
        lines.append("")
    lines += ["## Largest changes", ""]
    for name, m in out.items():
        b = m[m.row == "both runs"]
        num = [c for c in b.columns if c.endswith("_change") and not c.startswith(("dyads", "matches", "dates", "predictors"))]
        if len(b) == 0 or not num:
            continue
        top = max(num, key=lambda c: b[c].abs().max())
        i = b[top].abs().idxmax()
        where = ", ".join(f"{k}={b.loc[i, k]}" for k in TABLES[name][0])
        lines.append(f"- {name}: {len(b)} rows in both runs, {int((m.row == 'earlier only').sum())} only earlier, {int((m.row == 'new only').sum())} only now; "
                     f"largest change {top[:-7]} {b.loc[i, top]:+.4f} ({where})")
    lines += ["", "## Files compared (sha256, first 12 hex digits)", "", "| Table | Earlier | Now |", "| --- | --- | --- |"]
    lines += [f"| {n} | {a} | {b} |" for n, a, b in files]
    (RES / f"compare_runs_{label}_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote results/compare_runs_{label}_*.csv and results/compare_runs_{label}_summary.md")


if __name__ == "__main__":
    main()
