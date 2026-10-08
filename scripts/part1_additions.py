"""The additions committed to PREREG.md on 2026-10-07 after the Part 1 run (P3b, P3c, the waves 6 to 9 check).

Input : data/pairs.csv, data/feature_sets.json
Output: results/part1_additions_metrics.csv, results/part1_additions_hypotheses.csv
Seeds : as scripts/part1_models.py (event folds 20261006; inner tuning 20261006 + 1 + fold; random forest
        20261106 + fold; lasso 20261206 + fold). Same code and folds as Part 1, so the matching rows reproduce it.

- Ladder (full logistic regression, headline and complete-case): individual features; plus the six pair features;
  plus the five extended pair features (P3b); plus the three declared terms (P3c).
- Waves 6 to 9 check: P1 to P4 under the headline specification on the 17 waves that split 100 points; every model
  except stepwise (the lasso is reported but, as in Part 1, is not in the best-model pool).
Run from the repository root: python scripts/part1_additions.py [--jobs 5]
"""
import sys, json, pathlib, argparse, time
import pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

RES = pathlib.Path("results")
CAND = ["logistic", "lda", "qda", "rf"]                 # the pre-registered pool without stepwise (not run here)
LADDER = ["pre_individual", "pre", "pre_extended", "pre_interactions"]


def hypotheses(met):
    rows = []
    for version in ("headline", "complete-case"):
        d = met[(met.part == "ladder") & (met.version == version)].set_index("feature_set").auc
        rows.append(dict(hypothesis="P3b", version=version, statistic=f"full logistic AUC with extended pair features {d.pre_extended:.3f} "
                         f"minus with the six pair features {d.pre:.3f} = {d.pre_extended - d.pre:.3f}", supported=bool(d.pre_extended - d.pre < 0.02)))
        rows.append(dict(hypothesis="P3c", version=version, statistic=f"plus the three declared terms {d.pre_interactions:.3f} minus "
                         f"{d.pre_extended:.3f} = {d.pre_interactions - d.pre_extended:.3f}", supported=bool(d.pre_interactions - d.pre_extended < 0.02)))
    w = met[met.part == "waves 1-5, 10-21"]
    best = lambda fs: w[(w.feature_set == fs) & w.model.isin(CAND)].sort_values("auc").iloc[-1]
    bp, bd = best("pre"), best("pre+date")
    lp = w[(w.feature_set == "pre") & (w.model == "logistic")].auc.iloc[0]
    li = w[(w.feature_set == "pre_individual") & (w.model == "logistic")].auc.iloc[0]
    rf = w[(w.feature_set == "pre") & (w.model == "rf")].auc.iloc[0]
    v = "waves 1-5, 10-21"
    rows += [dict(hypothesis="P1", version=v, statistic=f"best `pre` AUC {bp.auc:.3f} ({bp.model})", supported=bool(0.58 <= bp.auc < 0.70)),
             dict(hypothesis="P2", version=v, statistic=f"best `pre+date` {bd.auc:.3f} ({bd.model}) minus best `pre` {bp.auc:.3f} = {bd.auc - bp.auc:.3f}", supported=bool(bd.auc - bp.auc >= 0.10)),
             dict(hypothesis="P3", version=v, statistic=f"full logistic `pre` {lp:.3f} minus without pair features {li:.3f} = {lp - li:.3f}", supported=bool(lp - li < 0.02)),
             dict(hypothesis="P4", version=v, statistic=f"random forest `pre` {rf:.3f} minus full logistic {lp:.3f} = {rf - lp:.3f}", supported=bool(rf - lp < 0.03))]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); args = ap.parse_args()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    met = []
    for version, spec in (("headline", M.HEADLINE), ("complete-case", M.COMPLETE_CASE)):
        for fs in LADDER:
            t0 = time.time()
            rows, *_ = P1.cross_validate(pairs, sets, spec, fs, ("logistic",), args.jobs)
            met += [dict(r, part="ladder", version=version) for r in rows]
            P1.log(f"ladder {version} {fs}: AUC {rows[0]['auc']:.3f} in {time.time() - t0:.0f}s")
    for fs, models in (("pre", ("lda_qda", "rf", "lasso")), ("pre+date", ("lda_qda", "rf", "lasso")), ("pre_individual", ("logistic",))):
        t0 = time.time()
        rows, *_ = P1.cross_validate(pairs, sets, M.HEADLINE, fs, models, args.jobs, exclude_events=M.WAVES_1_TO_10)
        met += [dict(r, part="waves 1-5, 10-21", version="headline") for r in rows]
        P1.log(f"waves check {fs}: best AUC {max(r['auc'] for r in rows):.3f} in {time.time() - t0:.0f}s")
    met = pd.DataFrame(met); met.to_csv(RES / "part1_additions_metrics.csv", index=False)
    hyp = hypotheses(met); hyp.to_csv(RES / "part1_additions_hypotheses.csv", index=False)
    cols = ["part", "version", "feature_set", "model", "auc", "auc_se", "balanced_accuracy", "dyads", "matches"]
    print("\n" + met[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + hyp.to_string(index=False))
    P1.log("wrote results/part1_additions_metrics.csv, results/part1_additions_hypotheses.csv")


if __name__ == "__main__":
    main()
