"""How much the headline AUCs move with the fold assignment and the forest's seed (exploratory, not a test;
PREREG.md, second set of fixes of 2026-10-08), to read the P1, P3 and P4 margins against.

Input : data/pairs.csv, data/feature_sets.json
Output: results/fold_seed_noise.csv (one row per run) and results/fold_seed_noise_summary.csv
Seeds : event folds 20261007 to 20261016 for the full logistic regression (`pre` and `pre_individual`); forest
        seeds 20261107 to 20261111 (random_state = seed + fold) on the headline folds (seed 20261006). The headline
        run itself (folds 20261006, forest 20261106) is in results/part1_models_metrics.csv.
Run from the repository root: python scripts/fold_seed_noise.py [--jobs 5]
"""
import sys, json, pathlib, argparse, time
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

FOLD_SEEDS = range(20261007, 20261017)
RF_SEEDS = range(20261107, 20261112)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); args = ap.parse_args()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    rows = []
    for seed in FOLD_SEEDS:
        for fs in ("pre", "pre_individual"):
            r, *_ = P1.cross_validate(pairs, sets, M.HEADLINE, fs, ("logistic",), args.jobs, fold_seed=seed)
            rows.append(dict(kind="fold seed", fold_seed=seed, rf_seed=None, feature_set=fs, model="logistic", auc=r[0]["auc"], auc_se=r[0]["auc_se"]))
        P1.log(f"fold seed {seed}: logistic `pre` {rows[-2]['auc']:.4f}, `pre_individual` {rows[-1]['auc']:.4f}")
    for seed in RF_SEEDS:
        t0 = time.time()
        r, *_ = P1.cross_validate(pairs, sets, M.HEADLINE, "pre", ("rf",), args.jobs, rf_seed=seed)
        for x in r:
            if x["model"] == "rf":
                rows.append(dict(kind="forest seed", fold_seed=M.FOLD_SEED, rf_seed=seed, feature_set="pre", model="rf", auc=x["auc"], auc_se=x["auc_se"]))
        P1.log(f"forest seed {seed}: `pre` {rows[-1]['auc']:.4f} in {time.time() - t0:.0f}s")
    R = pd.DataFrame(rows).assign(fold_scheme="event", spec=M.HEADLINE.label)
    R.to_csv("results/fold_seed_noise.csv", index=False)
    lp = R[(R.kind == "fold seed") & (R.feature_set == "pre")].set_index("fold_seed").auc
    li = R[(R.kind == "fold seed") & (R.feature_set == "pre_individual")].set_index("fold_seed").auc
    rf = R[R.kind == "forest seed"].auc
    S = pd.DataFrame([dict(quantity=q, runs=len(v), mean=v.mean(), sd=v.std(), min=v.min(), max=v.max())
                      for q, v in (("full logistic `pre` AUC over fold seeds", lp), ("full logistic `pre_individual` AUC over fold seeds", li),
                                   ("P3 difference (`pre` minus `pre_individual`) over fold seeds", lp - li),
                                   ("random forest `pre` AUC over forest seeds (headline folds)", rf))])
    S.to_csv("results/fold_seed_noise_summary.csv", index=False)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("wrote results/fold_seed_noise.csv, results/fold_seed_noise_summary.csv")


if __name__ == "__main__":
    main()
