"""The additions committed to PREREG.md on 2026-10-07 after the Part 1 run (P3b, P3c, the waves 6 to 9 check), and
the wave 12 sensitivity check of the decisions entry, as set out in the fixes of 2026-10-08 (both sets).

Input : data/pairs.csv, data/feature_sets.json, data/clean.csv (P5 without wave 12)
Output: results/part1_additions_metrics.csv, results/part1_additions_hypotheses.csv,
        results/part1_additions_stated_vs_revealed.csv (the yes model without wave 12)
Seeds : as scripts/part1_models.py (event folds 20261006; inner tuning 20261006 + 1 + fold; random forest
        20261106 + fold; lasso 20261206 + fold).

- Ladder (full logistic regression, headline and complete-case): individual features; plus the six pair features;
  plus the five extended pair features (P3b); plus the three declared terms (P3c). The complete-case version uses
  the dyads complete on every column any step uses (including the shared-interests shares in P3c's terms), so
  every step has the same dyads and none is filled. All 21 events, so the folds are the headline folds.
- Waves 6 to 9 check (P1 to P4 on the 17 waves that split 100 points) and wave 12 check (P1 to P4 without wave 12,
  whose participants could say yes to only half their dates; and P5 without wave 12): every model except stepwise
  (the lasso is reported but, as in Part 1, is not in the best-model pool). Each in three versions: headline
  specification with the remaining events dealt into 5 folds by seed 20261006 (the primary version, as the waves 6
  to 9 entry says); the same with the headline folds minus the excluded waves; and the complete-case version.
Every metrics and hypothesis row carries `se_check`, the range from the simulation that checks its error bars.
Run from the repository root: python scripts/part1_additions.py [--jobs 5]
"""
import sys, json, pathlib, argparse, time
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1
from sim_auc_cluster import coverage_note

RES = pathlib.Path("results")
CAND = ["logistic", "lda", "qda", "rf"]                 # the pre-registered pool without stepwise (not run here)
LADDER = ["pre_individual", "pre", "pre_extended", "pre_interactions"]
CHECKS = {"waves 1-5, 10-21": M.WAVES_RATED_1_TO_10, "without wave 12": (12,)}
VERSIONS = ("folds re-drawn", "headline folds", "complete-case")


def hypotheses(met):
    rows = []
    for version in ("headline", "complete-case"):
        d = met[(met.part == "ladder") & (met.version == version)].set_index("feature_set").auc
        rows.append(dict(hypothesis="P3b", version=version, statistic=f"full logistic AUC with extended pair features {d.pre_extended:.4f} "
                         f"minus with the six pair features {d.pre:.4f} = {d.pre_extended - d.pre:.4f}", supported=bool(d.pre_extended - d.pre < 0.02)))
        rows.append(dict(hypothesis="P3c", version=version, statistic=f"plus the three declared terms {d.pre_interactions:.4f} minus "
                         f"{d.pre_extended:.4f} = {d.pre_interactions - d.pre_extended:.4f}", supported=bool(d.pre_interactions - d.pre_extended < 0.02)))
    for part in CHECKS:
        for version in VERSIONS:
            w = met[(met.part == part) & (met.version == version)]
            if w.empty:
                continue
            best = lambda fs: w[(w.feature_set == fs) & w.model.isin(CAND)].sort_values("auc").iloc[-1]
            bp, bd = best("pre"), best("pre+date")
            lp = w[(w.feature_set == "pre") & (w.model == "logistic")].auc.iloc[0]
            li = w[(w.feature_set == "pre_individual") & (w.model == "logistic")].auc.iloc[0]
            rf = w[(w.feature_set == "pre") & (w.model == "rf")].auc.iloc[0]
            v = f"{part}, {version}"
            rows += [dict(hypothesis="P1", version=v, statistic=f"best `pre` AUC {bp.auc:.4f} ({bp.model})", supported=bool(0.58 <= bp.auc < 0.70)),
                     dict(hypothesis="P2", version=v, statistic=f"best `pre+date` {bd.auc:.4f} ({bd.model}) minus best `pre` {bp.auc:.4f} = {bd.auc - bp.auc:.4f}", supported=bool(bd.auc - bp.auc >= 0.10)),
                     dict(hypothesis="P3", version=v, statistic=f"full logistic `pre` {lp:.4f} minus without pair features {li:.4f} = {lp - li:.4f}", supported=bool(lp - li < 0.02)),
                     dict(hypothesis="P4", version=v, statistic=f"random forest `pre` {rf:.4f} minus full logistic {lp:.4f} = {rf - lp:.4f}", supported=bool(rf - lp < 0.03))]
    return pd.DataFrame(rows).assign(fold_scheme="event")


def main():
    from sim_coef_cluster import coverage_note as coef_note
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); args = ap.parse_args()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    headline_folds = M.event_folds(pairs.event)
    met = []
    every_step = sets["pref_reference"] + sets["extended_pair"] + M.INTERACTIONS     # complete-case: one set of dyads
    for version, spec, extra in (("headline", M.HEADLINE, ()), ("complete-case", M.COMPLETE_CASE, every_step)):
        for fs in LADDER:
            t0 = time.time()
            rows, *_ = P1.cross_validate(pairs, sets, spec, fs, ("logistic",), args.jobs, complete_extra=extra)
            met += [dict(r, part="ladder", version=version) for r in rows]
            P1.log(f"ladder {version} {fs}: AUC {rows[0]['auc']:.4f} on {rows[0]['dyads']} dyads in {time.time() - t0:.0f}s")
    for part, excluded in CHECKS.items():
        for version in VERSIONS:
            spec = M.COMPLETE_CASE if version == "complete-case" else M.HEADLINE
            folds = headline_folds if version == "headline folds" else None
            for fs, models in (("pre", ("lda_qda", "rf", "lasso")), ("pre+date", ("lda_qda", "rf", "lasso")), ("pre_individual", ("logistic",))):
                t0 = time.time()
                rows, *_ = P1.cross_validate(pairs, sets, spec, fs, models, args.jobs, exclude_events=excluded, folds=folds)
                met += [dict(r, part=part, version=version) for r in rows]
                P1.log(f"{part}, {version}, {fs}: best AUC {max(r['auc'] for r in rows):.4f} on {rows[0]['dyads']} dyads in {time.time() - t0:.0f}s")
    met = pd.DataFrame(met).assign(se_check=coverage_note()); met.to_csv(RES / "part1_additions_metrics.csv", index=False)
    lad = met[(met.part == "ladder") & (met.version == "complete-case")].dyads
    assert lad.nunique() == 1, "complete-case ladder steps must share their dyads"
    sr = P1.yes_model(exclude_events=(12,)).assign(se_check=coef_note("yes"))
    sr.to_csv(RES / "part1_additions_stated_vs_revealed.csv", index=False)
    hyp = pd.concat([hypotheses(met).assign(se_check=coverage_note()),
                     pd.DataFrame(P1.p5_rows(sr, "{g}, without wave 12", note="dates with all six ratings, wave 12 left out of the yes model "
                                             "and the stated shares")).assign(se_check=coef_note("yes"))], ignore_index=True)
    hyp.to_csv(RES / "part1_additions_hypotheses.csv", index=False)
    cols = ["part", "version", "feature_set", "model", "auc", "auc_se", "balanced_accuracy", "dyads", "matches"]
    print("\n" + met[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + hyp[["hypothesis", "version", "statistic", "supported"]].to_string(index=False))
    P1.log("wrote results/part1_additions_metrics.csv, results/part1_additions_hypotheses.csv, results/part1_additions_stated_vs_revealed.csv")


if __name__ == "__main__":
    main()
