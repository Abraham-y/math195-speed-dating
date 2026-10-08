"""Part 2 models (PREREG.md, Part 2, committed before this ran): tree models (T1, T2), the leakage comparison
(A1), the two-decision model (D1, D2) and ensembles (E1, E2).

Input : data/pairs.csv, data/feature_sets.json
Output: results/part2_models_metrics.csv (every model, fold scheme, specification and feature set),
        results/part2_models_hypotheses.csv, results/part2_models_folds.csv (tuned settings per fold),
        results/part2_models_yes_predictions.csv (event-held-out pre-date yes predictions, for W1)
Seeds : event folds 20261006 (random-row folds: the same seed over the dyads); inner tuning 20261006 + 1 + fold;
        random forest 20261106 + fold; lasso 20261206 + fold; classification tree 20261606 + fold; bagging
        20261706 + fold; boosting 20261806 + fold.

Every model is fitted inside each training fold on the Part 1 preparation (part1_models.make_prep; splines only
for the logistic models) and tuned by the inner folds by pooled AUC (part1_models.tune_inner). Random-row CV
treats each dyad as its own unit, so the same code deals rows instead of events into folds, and its error bars
are the independent-rows ones. The two-decision model fits each side's yes and multiplies the probabilities.
The mean ensemble averages six base models; the stacked ensemble is a logistic regression on their logits,
fitted to inner event-CV predictions inside the training fold.
Run from the repository root: python scripts/part2_models.py [--jobs 5]
"""
import sys, json, pathlib, argparse, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1
from sim_auc_cluster import coverage_note

RES = pathlib.Path("results")
TREE_ALPHAS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02]
BAG_LEAVES = P1.RF_LEAVES
BOOST_GRID = [(d, n) for d in (1, 2, 3) for n in (100, 300, 600)]
TREE_SEED, BAG_SEED, BOOST_SEED = 20261606, 20261706, 20261806
STACK_BASE = ["logistic", "lda", "qda", "rf", "lasso", "boosting"]
POOL = ["logistic", "stepwise_forward", "stepwise_backward", "stepwise_both", "lda", "qda", "rf"]
TREES = ["tree", "bagging", "rf", "boosting"]


def make_tree(k):
    return lambda a: DecisionTreeClassifier(criterion="gini", ccp_alpha=a, random_state=TREE_SEED + k)


def make_bag(k):
    return lambda leaf: RandomForestClassifier(n_estimators=P1.RF_TREES, max_features=None, min_samples_leaf=leaf,
                                               random_state=BAG_SEED + k, n_jobs=P1.RF_JOBS)


def make_boost(k):
    return lambda dn: HistGradientBoostingClassifier(learning_rate=0.05, max_depth=dn[0], max_iter=dn[1],
                                                     early_stopping=False, random_state=BOOST_SEED + k)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def part2_fold(k, X, y, ev, ids, sets, folds, models, extras, dec):
    """One outer fold: the Part 1 models (part1_models.run_fold), then the extras asked for."""
    import warnings; warnings.filterwarnings("ignore")
    base = P1.run_fold(k, X, y, ev, ids, sets, folds, models)
    test = np.array([folds[e] == k for e in ev]); train = ~test
    Xtr, Xte, ytr, evtr = X[train].reset_index(drop=True), X[test].reset_index(drop=True), y[train], ev[train]
    idtr = ids[train].reset_index(drop=True)
    prob, info, yes = dict(base["prob"]), dict(base["info"]), {}
    spl = tuple(s for s in str(info["splines"]).split(";") if s and s != "nan")
    pl = P1.make_prep(sets, Xtr).fit(Xtr, idtr)
    Ltr, Lte = pl.transform(Xtr).to_numpy(), pl.transform(Xte).to_numpy()
    if "trees" in extras:
        a = P1.tune_inner(Xtr, ytr, evtr, idtr, sets, k, make_tree(k), TREE_ALPHAS)
        prob["tree"] = P1.predict_serial(make_tree(k)(a).fit(Ltr, ytr), Lte)
        leaf = P1.tune_inner(Xtr, ytr, evtr, idtr, sets, k, make_bag(k), BAG_LEAVES)
        prob["bagging"] = P1.predict_serial(make_bag(k)(leaf).fit(Ltr, ytr), Lte)
        dn = P1.tune_inner(Xtr, ytr, evtr, idtr, sets, k, make_boost(k), BOOST_GRID)
        prob["boosting"] = P1.predict_serial(make_boost(k)(dn).fit(Ltr, ytr), Lte)
        info.update(tree_ccp_alpha=a, bagging_min_samples_leaf=leaf, boosting_max_depth=dn[0], boosting_trees=dn[1],
                    tree_random_state=TREE_SEED + k, bagging_random_state=BAG_SEED + k, boosting_random_state=BOOST_SEED + k)
    if "ensembles" in extras:
        prob["ensemble_mean"] = np.mean([prob[m] for m in STACK_BASE], axis=0)
        inner = M.event_folds(evtr, k=4, seed=M.FOLD_SEED + 1 + k)
        f = np.array([inner[e] for e in evtr])
        Z = np.empty((len(ytr), len(STACK_BASE)))
        for j in range(4):
            a_, b_ = f != j, f == j
            Xa, Xb, ida, ya = Xtr[a_].reset_index(drop=True), Xtr[b_].reset_index(drop=True), idtr[a_].reset_index(drop=True), ytr[a_]
            ps = P1.make_prep(sets, Xa, splines=spl).fit(Xa, ida)
            w, _ = M.fit_logit(ps.transform(Xa).to_numpy(), ya)
            pn = P1.make_prep(sets, Xa).fit(Xa, ida)
            La, Lb = pn.transform(Xa).to_numpy(), pn.transform(Xb).to_numpy()
            cols = {"logistic": M.predict_logit(w, ps.transform(Xb).to_numpy()),
                    "lda": LinearDiscriminantAnalysis().fit(La, ya).predict_proba(Lb)[:, 1],
                    "qda": P1.make_qda(info["qda_reg_param"]).fit(La, ya).predict_proba(Lb)[:, 1],
                    "rf": P1.predict_serial(P1.make_rf(k)(info["rf_min_samples_leaf"]).fit(La, ya), Lb),
                    "lasso": P1.make_lasso(k)(info["lasso_C"]).fit(La, ya).predict_proba(Lb)[:, 1],
                    "boosting": P1.predict_serial(make_boost(k)((info["boosting_max_depth"], info["boosting_trees"])).fit(La, ya), Lb)}
            Z[b_] = logit(np.column_stack([cols[m] for m in STACK_BASE]))
        ws, _ = M.fit_logit(Z, ytr)
        prob["ensemble_stacked"] = M.predict_logit(ws, logit(np.column_stack([prob[m] for m in STACK_BASE])))
        info["stack_weights"] = ";".join(f"{m}={v:.3f}" for m, v in zip(["intercept"] + STACK_BASE, ws))
    if "two_decision" in extras:
        ps = P1.make_prep(sets, Xtr, splines=spl).fit(Xtr, idtr)
        Dtr, Dte = ps.transform(Xtr).to_numpy(), ps.transform(Xte).to_numpy()
        for side in ("f", "m"):
            ys = dec[side][train]
            w, _ = M.fit_logit(Dtr, ys)
            yes[f"{side}_logistic"] = M.predict_logit(w, Dte)
            leaf = P1.tune_inner(Xtr, ys, evtr, idtr, sets, k, P1.make_rf(k), P1.RF_LEAVES)
            yes[f"{side}_rf"] = P1.predict_serial(P1.make_rf(k)(leaf).fit(Ltr, ys), Lte)
            info[f"{side}_yes_threshold"] = float(ys.mean()); info[f"{side}_yes_rf_min_samples_leaf"] = leaf
        prob["two_decision_logistic"] = yes["f_logistic"] * yes["m_logistic"]
        prob["two_decision_rf"] = yes["f_rf"] * yes["m_rf"]
    return dict(test_idx=np.where(test)[0], prob=prob, yes=yes, info=info)


def run(pairs, sets, spec, feature_set, models, extras, jobs, scheme):
    """Cross-validate under one scheme ('event' or 'random-row'); returns metric rows, fold info, yes predictions."""
    data = pairs if scheme == "event" else pairs.assign(event=np.arange(len(pairs)))
    X, y, ev, ids = M.build_frame(data, sets, spec, feature_set)
    key = pd.MultiIndex.from_frame(ids)
    dd = pairs.set_index(["f_iid", "m_iid"]).loc[key]
    dec = {"f": dd.f_dec.to_numpy(float), "m": dd.m_dec.to_numpy(float)}
    folds = M.event_folds(ev)
    res = Parallel(n_jobs=jobs)(delayed(part2_fold)(k, X, y, ev, ids, sets, folds, models, extras, dec) for k in range(5))
    names = list(res[0]["prob"])
    prob = {m: np.empty(len(y)) for m in names}; thr = np.empty(len(y))
    yes = {m: np.empty(len(y)) for m in res[0]["yes"]}; ythr = {s: np.empty(len(y)) for s in ("f", "m")}
    for r in res:
        idx = r["test_idx"]; thr[idx] = r["info"]["threshold"]
        for m in names:
            prob[m][idx] = r["prob"][m]
        for m in yes:
            yes[m][idx] = r["yes"][m]
        for s in ("f", "m"):
            if f"{s}_yes_threshold" in r["info"]:
                ythr[s][idx] = r["info"][f"{s}_yes_threshold"]
    base = dict(scheme=scheme, spec=spec.label, headline=spec == M.HEADLINE, feature_set=feature_set, dyads=len(y),
                fold_scheme="event" if scheme == "event" else "random-row", fold_seed=M.FOLD_SEED)
    rows = []
    for m in names:
        met = M.metric_table(prob[m], y, thr, ev)
        rows.append(dict(base, model=m, target="match", positives=int(y.sum()), **{k: v[0] for k, v in met.items()},
                         **{f"{k}_se": v[1] for k, v in met.items()}))
    for m in yes:
        s = m[0]
        met = M.metric_table(yes[m], dec[s], ythr[s], ev)
        rows.append(dict(base, model=m[2:], target="her yes" if s == "f" else "his yes", positives=int(dec[s].sum()),
                         **{k: v[0] for k, v in met.items()}, **{f"{k}_se": v[1] for k, v in met.items()}))
    finfo = [dict(scheme=scheme, spec=spec.label, feature_set=feature_set, **r["info"]) for r in res]
    yes_frame = (pd.DataFrame({"f_iid": ids.f_iid, "m_iid": ids.m_iid, "event": ev, "f_dec": dec["f"], "m_dec": dec["m"],
                               **{f"pred_{m}": v for m, v in yes.items()}}).assign(spec=spec.label, feature_set=feature_set)
                 if yes else None)
    return rows, finfo, yes_frame


def hypotheses(met):
    ev = met[(met.scheme == "event") & met.headline.astype(bool) & (met.target == "match")]
    rr = met[(met.scheme == "random-row") & (met.target == "match")]
    auc = lambda d, fs, m: d[(d.feature_set == fs) & (d.model == m)].auc.iloc[0]
    rows = []
    for fs in ("pre", "pre+date"):
        best_tree = max(("tree", "bagging", "boosting"), key=lambda m: auc(ev, fs, m))
        d = auc(ev, fs, best_tree) - auc(ev, fs, "rf")
        rows.append(dict(hypothesis="T1", version=f"headline, {fs}", statistic=f"best other tree model ({best_tree}) {auc(ev, fs, best_tree):.4f} "
                         f"minus random forest {auc(ev, fs, 'rf'):.4f} = {d:+.4f}", supported=bool(d < 0.02)))
    t = {m: auc(ev, "pre", m) for m in TREES}
    rows.append(dict(hypothesis="T2", version="headline, pre", statistic="; ".join(f"{m} {v:.4f}" for m, v in t.items()),
                     supported=bool(min(t, key=t.get) == "tree")))
    if len(rr):
        gaps = {(fs, m): auc(rr, fs, m) - auc(ev, fs, m) for fs in ("pre", "pre+date") for m in sorted(set(rr.model))}
        every = all(g > 0 for g in gaps.values())
        trees = all(gaps[(fs, m)] > gaps[(fs, "logistic")] for fs in ("pre", "pre+date") for m in TREES)
        rows.append(dict(hypothesis="A1", version="headline", statistic="random-row minus event AUC: " + "; ".join(
            f"{fs} {m} {g:+.4f}" for (fs, m), g in gaps.items()), supported=bool(every and trees),
            note=f"every model higher under random rows: {every}; every tree model's gap above the full logistic model's: {trees}"))
    pool = lambda d, fs: d[(d.feature_set == fs) & d.model.isin(POOL)].sort_values("auc").iloc[-1]
    for version, d in (("headline", ev), ("complete-case", met[(met.scheme == "event") & (met.spec == M.COMPLETE_CASE.label) & (met.target == "match")])):
        if d.empty or "two_decision_rf" not in set(d.model):
            continue
        bp = pool(d, "pre")
        bt = max(("two_decision_logistic", "two_decision_rf"), key=lambda m: auc(d, "pre", m))
        g = auc(d, "pre", bt) - bp.auc
        rows.append(dict(hypothesis="D1", version=version, statistic=f"{bt} {auc(d, 'pre', bt):.4f} minus best single-stage {bp.model} {bp.auc:.4f} = {g:+.4f}",
                         supported=bool(g < 0.02)))
        y = met[(met.scheme == "event") & (met.spec == d.spec.iloc[0]) & (met.feature_set == "pre") & (met.target != "match")]
        best = {t: y[y.target == t].sort_values("auc").iloc[-1] for t in ("her yes", "his yes")}
        rows.append(dict(hypothesis="D2", version=version, statistic="; ".join(f"{t}: {r.model} {r.auc:.4f} (SE {r.auc_se:.4f})" for t, r in best.items()),
                         supported=bool(all(r.auc >= 0.60 for r in best.values()))))
    for h, fs in (("E1", "pre"), ("E2", "pre+date")):
        bp = pool(ev, fs)
        be = max(("ensemble_mean", "ensemble_stacked"), key=lambda m: auc(ev, fs, m))
        g = auc(ev, fs, be) - bp.auc
        rows.append(dict(hypothesis=h, version=f"headline, {fs}", statistic=f"{be} {auc(ev, fs, be):.4f} minus best single {bp.model} {bp.auc:.4f} = {g:+.4f}",
                         supported=bool(g < 0.01)))
    return pd.DataFrame(rows).assign(fold_scheme="event (A1: random-row against event)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); args = ap.parse_args()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    runs = [("event", M.HEADLINE, "pre", ("trees", "ensembles", "two_decision")),
            ("event", M.HEADLINE, "pre+date", ("trees", "ensembles")),
            ("event", M.COMPLETE_CASE, "pre", ("trees", "ensembles", "two_decision")),
            ("event", M.COMPLETE_CASE, "pre+date", ("trees", "ensembles")),
            ("random-row", M.HEADLINE, "pre", ("trees",)),
            ("random-row", M.HEADLINE, "pre+date", ("trees",))]
    met, finfo, yes = [], [], []
    for scheme, spec, fs, extras in runs:
        t0 = time.time()
        rows, fi, yf = run(pairs, sets, spec, fs, ("stepwise", "lda_qda", "rf", "lasso"), extras, args.jobs, scheme)
        met += rows; finfo += fi
        if yf is not None:
            yes.append(yf)
        best = max((r for r in rows if r["target"] == "match"), key=lambda r: r["auc"])
        P1.log(f"{scheme} [{spec.label}] {fs}: best AUC {best['auc']:.4f} ({best['model']}) in {time.time() - t0:.0f}s")
        pd.DataFrame(met).to_csv(RES / "part2_models_metrics.csv", index=False)
    note = coverage_note()
    met = pd.DataFrame(met).assign(se_check=np.where(pd.DataFrame(met).scheme == "event", note, "random-row: independent-rows error bars"))
    met.to_csv(RES / "part2_models_metrics.csv", index=False)
    pd.DataFrame(finfo).to_csv(RES / "part2_models_folds.csv", index=False)
    pd.concat(yes).to_csv(RES / "part2_models_yes_predictions.csv", index=False)
    hyp = hypotheses(met).assign(se_check=note); hyp.to_csv(RES / "part2_models_hypotheses.csv", index=False)

    # consistency: the Part 1 models under event folds must reproduce results/part1_models_metrics.csv
    p1 = pd.read_csv(RES / "part1_models_metrics.csv")
    chk = met[(met.scheme == "event") & (met.target == "match")].merge(p1, on=["spec", "feature_set", "model"], suffixes=("", "_p1"))
    print(f"\nPart 1 rows reproduced: {len(chk)}; largest AUC difference from results/part1_models_metrics.csv {np.abs(chk.auc - chk.auc_p1).max():.2e}")
    cols = ["scheme", "spec", "feature_set", "target", "model", "auc", "auc_se", "balanced_accuracy", "balanced_accuracy_se"]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(met[cols].assign(spec=met.spec.str.replace("missing=", "").str.slice(0, 13)).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\n" + hyp[["hypothesis", "version", "statistic", "supported"]].to_string(index=False))
    P1.log("wrote results/part2_models_metrics.csv, _hypotheses.csv, _folds.csv, _yes_predictions.csv")


if __name__ == "__main__":
    main()
