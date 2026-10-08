"""The like model committed to PREREG.md on 2026-10-07 (hypotheses L1 and L2).

Input : data/pairs.csv, data/feature_sets.json, data/clean.csv
Output: results/like_models_metrics.csv, results/like_models_hypotheses.csv
Seed  : event folds 20261006 (modeling.FOLD_SEED), the same folds as the match models.

Linear regression (ISL chapter 3) of each date's overall liking rating (`like`, 0 to 10 as recorded) on pre-date
information, one row per date from one side, dates with `like` recorded. Predictors: the rater's and the partner's
own pre-event answers (the `pre_individual` columns relabelled r_ and p_), the rater's gender, and the eleven pair
features of P3b, prepared exactly as the headline match models (survey-format standardization, recording-error fix,
skewness rule, training-fold medians with missing-data flags, rare codes, standardizing). Event 5-fold CV; held-out
R-squared = 1 - (pooled squared error) / (variance of `like`), and RMSE, with cluster-robust standard errors by event.
Ladder: rater only; plus partner; plus pair features.
Run from the repository root: python scripts/like_models.py [--jobs 5]
"""
import sys, json, pathlib, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")


def dates_table(pairs, sets, clean):
    """One row per date from one side: the rater's answers as r_*, the partner's as p_*, the pair features, the
    rater's gender (r_male) and the like rating. Built from the headline match design, so every preparation
    decision that comes before the training folds (format standardization, recording-error fix) is shared."""
    X, _, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, "pre_extended")
    X = X.assign(event=ev, f_iid=ids.f_iid, m_iid=ids.m_iid)
    person = [c[2:] for c in X.columns if c.startswith("f_") and c != "f_iid"]       # stems shared by f_ and m_
    pair = [{"fit_to_f_prefs": "fit_to_rater_prefs", "fit_to_m_prefs": "fit_to_partner_prefs"}.get(c, c)
            for c in X.columns if not c.startswith(("f_", "m_")) and c != "event"]          # relabelled from the rater's side
    rows = []
    for g, me, other in ((0, "f", "m"), (1, "m", "f")):
        part = X.rename(columns={**{f"{me}_{s}": f"r_{s}" for s in person}, **{f"{other}_{s}": f"p_{s}" for s in person},
                                 f"{me}_iid": "r_iid", f"{other}_iid": "p_iid",
                                 f"fit_to_{me}_prefs": "fit_to_rater_prefs", f"fit_to_{other}_prefs": "fit_to_partner_prefs"})
        rows.append(part.assign(r_male=g))
    D = pd.concat(rows, ignore_index=True)
    like = clean.set_index(["rater", "ratee"]).like
    D["like"] = like.reindex(pd.MultiIndex.from_arrays([D.r_iid, D.p_iid])).to_numpy()
    D = D[D.like.notna()].reset_index(drop=True)
    rater = [f"r_{s}" for s in person] + ["r_male"]
    partner = [f"p_{s}" for s in person]
    return D, rater, partner, pair


def r_squared(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def fit_fold(k, D, cols, folds, sets):
    test = D.event.map(folds).to_numpy() == k; train = ~test
    Xtr, Xte = D.loc[train, cols].reset_index(drop=True), D.loc[test, cols].reset_index(drop=True)
    ids = D.loc[train, ["r_iid", "p_iid"]].reset_index(drop=True)
    stems = {c.split("_", 1)[1] for c in sets["categorical"]}
    cat = [c for c in cols if c.split("_", 1)[-1] in stems]
    binary = [c for c in cols if c in sets["binary"] or c == "r_male" or c.endswith("_pre_missing")]
    person = [c for c in cols if c.startswith(("r_", "p_")) and c not in cat + binary]
    p = M.Prep(categorical=cat, binary=binary, person_cols=person, sides=("r", "p")).fit(Xtr, ids)
    A, B = p.transform(Xtr).to_numpy(), p.transform(Xte).to_numpy()
    A1, B1 = np.column_stack([np.ones(len(A)), A]), np.column_stack([np.ones(len(B)), B])
    beta = np.linalg.lstsq(A1, D.like.to_numpy()[train], rcond=None)[0]
    return np.where(test)[0], B1 @ beta


def cross_validate(D, cols, sets, jobs):
    folds = M.event_folds(D.event)
    res = Parallel(n_jobs=jobs)(delayed(fit_fold)(k, D, cols, folds, sets) for k in range(5))
    pred = np.empty(len(D))
    for idx, p in res:
        pred[idx] = p
    y = D.like.to_numpy(); sq = (y - pred) ** 2
    mse, var = sq.mean(), y.var()
    se_mse = np.sqrt(M.cluster_robust_var_of_mean(sq, D.event.to_numpy()))
    return dict(r_squared=r_squared(y, pred), r_squared_se=se_mse / var, rmse=np.sqrt(mse), rmse_se=se_mse / (2 * np.sqrt(mse)),
                dates=len(y), predictors=len(cols), fold_scheme="event", seed=M.FOLD_SEED)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); args = ap.parse_args()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    D, rater, partner, pair = dates_table(pairs, sets, clean)
    ladder = (("rater only", rater), ("rater + partner", rater + partner), ("rater + partner + pair", rater + partner + pair))
    rows = [dict(model=name, **cross_validate(D, cols, sets, args.jobs)) for name, cols in ladder]
    met = pd.DataFrame(rows); met.to_csv(RES / "like_models_metrics.csv", index=False)
    full, rp = met.set_index("model").loc["rater + partner + pair"], met.set_index("model").loc["rater + partner"]
    hyp = pd.DataFrame([
        dict(hypothesis="L1", statistic=f"held-out R-squared of the full model {full.r_squared:.3f} (SE {full.r_squared_se:.3f})", supported=bool(full.r_squared < 0.10)),
        dict(hypothesis="L2", statistic=f"pair features add {full.r_squared - rp.r_squared:.4f} ({full.r_squared:.3f} minus {rp.r_squared:.3f})",
             supported=bool(full.r_squared - rp.r_squared < 0.01))])
    hyp.to_csv(RES / "like_models_hypotheses.csv", index=False)
    print(f"{len(D)} dates with `like` recorded; mean {D.like.mean():.2f}, SD {D.like.std():.2f}; event 5-fold CV, seed {M.FOLD_SEED}")
    print(met.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(hyp.to_string(index=False))
    print("wrote results/like_models_metrics.csv, results/like_models_hypotheses.csv")


if __name__ == "__main__":
    main()
