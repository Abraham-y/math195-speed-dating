"""Part 1 models for a match, following the model-building entry of PREREG.md (committed before this ran).

Input : data/pairs.csv, data/feature_sets.json, data/clean.csv (for the yes model, P5)
Output: results/part1_models_*.csv and results/part1_models_*.png (see the end of main for the list)
Seeds : event folds 20261006 (modeling.FOLD_SEED); inner tuning folds 20261006 + 1 + fold; random forest
        random_state 20261106 + fold; lasso random_state 20261206 + fold.

For each specification (the headline, then the rest of the robustness grid) and each feature set (`pre`,
`pre+date`, and `pre_individual` for P3), 5-fold cross-validation holding out whole events (`event` scheme):
- logistic regression on the full feature set, with a natural spline replacing a candidate's linear term when it
  lowers BIC on the training fold;
- stepwise logistic regression, forward / backward / both ways by BIC (headline and complete-case only);
- LDA, QDA with reg_param tuned by event-grouped 4-fold CV inside the training fold, and a random forest
  (500 trees, sqrt(p) predictors per split, minimum leaf size tuned the same way; PREREG.md entry of 2026-10-06);
- the lasso: L1 logistic regression with its penalty tuned the same way (PREREG.md entry of 2026-10-06). It is
  reported but kept out of the "best model" of P1 and P2, which were tested before it was added.
Everything is refitted inside each training fold. Thresholds: the training fold's match rate. Error bars: the
cluster-robust estimator with events as clusters; AUC by Obuchowski (1997), which the simulation in
scripts/sim_auc_cluster.py found about 9% too narrow at 21 events (90% intervals covered 83%).
Also: the coefficient table (full model, all dyads, cluster-robust by event), diagnostics, the yes model and the
stated-versus-revealed table (P5), and the hypothesis table.
Run from the repository root: python scripts/part1_models.py [--specs headline|all] [--jobs 5]
"""
import sys, json, pathlib, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")
QDA_REGS = [0.1, 0.3, 0.5, 0.7, 0.9]
RF_LEAVES = [1, 5, 20, 50]
RF_SEED, RF_TREES, RF_JOBS = 20261106, 500, 2
LASSO_SEED, LASSO_STEPS = 20261206, 30
DIRECTIONS = ("forward", "backward", "both")
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
SIM_COVERAGE = "clustered AUC SE checked in scripts/sim_auc_cluster.py: about 9% too narrow, 90% intervals covered 83%"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_prep(sets, X, splines=()):
    person = [c for c in sets["pre_individual"] if c in X.columns and c not in sets["categorical"]]
    return M.Prep(categorical=[c for c in sets["categorical"] if c in X.columns],
                  binary=[c for c in sets["binary"] if c in X.columns], person_cols=person, splines=splines)


def choose_splines(Xtr, ytr, idtr, sets):
    """A candidate gets a natural cubic spline (3 df) when that lowers the full model's BIC on the training fold."""
    base = make_prep(sets, Xtr).fit(Xtr, idtr); D = base.transform(Xtr).to_numpy()
    _, ll = M.fit_logit(D, ytr); b0 = M.bic(ll, D.shape[1] + 1, len(ytr))
    chosen = []
    for c in [c for c in sets["spline_candidates"] if c in Xtr.columns]:
        p = make_prep(sets, Xtr, splines=(c,)).fit(Xtr, idtr)
        if c not in p.knots:
            continue
        Dc = p.transform(Xtr).to_numpy(); _, llc = M.fit_logit(Dc, ytr)
        if M.bic(llc, Dc.shape[1] + 1, len(ytr)) < b0:
            chosen.append(c)
    return tuple(chosen)


def tune_inner(Xtr, ytr, evtr, idtr, sets, k, make_model, grid):
    """The grid value with the best pooled inner AUC over event-grouped 4-fold CV on the training fold
    (preparation refitted on each inner training part; ties go to the earlier grid value)."""
    inner = M.event_folds(evtr, k=4, seed=M.FOLD_SEED + 1 + k)
    f = np.array([inner[e] for e in evtr])
    probs = {r: np.empty(len(ytr)) for r in grid}
    for j in range(4):
        a, b = f != j, f == j
        p = make_prep(sets, Xtr[a]).fit(Xtr[a].reset_index(drop=True), idtr[a].reset_index(drop=True))
        Da, Db = p.transform(Xtr[a].reset_index(drop=True)).to_numpy(), p.transform(Xtr[b].reset_index(drop=True)).to_numpy()
        for r in grid:
            probs[r][b] = make_model(r).fit(Da, ytr[a]).predict_proba(Db)[:, 1]
    return max(grid, key=lambda r: M.auc(probs[r], ytr))


def make_qda(r):
    return QuadraticDiscriminantAnalysis(reg_param=r)


def lasso_grid(D, y, steps=LASSO_STEPS):
    """Inverse penalties C for liblinear's L1 logistic regression, strongest penalty first: from the smallest
    penalty that zeroes every coefficient (C = 1 / max |D'(y - mean y)|) to one thousandth of it."""
    cmin = 1 / np.max(np.abs(D.T @ (y - y.mean())))
    return list(cmin * np.logspace(0, 3, steps))


def make_lasso(k):
    return lambda C: LogisticRegression(l1_ratio=1, solver="liblinear", C=C, intercept_scaling=100, tol=1e-4,
                                        random_state=LASSO_SEED + k)


def make_rf(k):
    return lambda leaf: RandomForestClassifier(n_estimators=RF_TREES, max_features="sqrt", min_samples_leaf=leaf,
                                               random_state=RF_SEED + k, n_jobs=RF_JOBS)


def run_fold(k, X, y, ev, ids, sets, folds, models):
    import warnings; warnings.filterwarnings("ignore")
    test = np.array([folds[e] == k for e in ev]); train = ~test
    Xtr, Xte, ytr = X[train].reset_index(drop=True), X[test].reset_index(drop=True), y[train]
    idtr = ids[train].reset_index(drop=True)
    out, info = {}, dict(fold=k, threshold=float(ytr.mean()), n_train=int(train.sum()), n_test=int(test.sum()))
    spl = choose_splines(Xtr, ytr, idtr, sets)
    p = make_prep(sets, Xtr, splines=spl).fit(Xtr, idtr)
    Dtr, Dte = p.transform(Xtr).to_numpy(), p.transform(Xte).to_numpy()
    w, _ = M.fit_logit(Dtr, ytr)
    out["logistic"] = M.predict_logit(w, Dte)
    info["splines"] = ";".join(spl); info["columns"] = Dtr.shape[1]
    sel = {}
    if "stepwise" in models:
        blocks = p.blocks()
        for d in DIRECTIONS:
            chosen = M.stepwise(Dtr, ytr, blocks, d)
            cols = [j for b in chosen for j in blocks[b]]
            wd, _ = M.fit_logit(Dtr[:, cols], ytr)
            out[f"stepwise_{d}"] = M.predict_logit(wd, Dte[:, cols])
            sel[d] = chosen
    if "lda_qda" in models or "rf" in models or "lasso" in models:
        pl = make_prep(sets, Xtr).fit(Xtr, idtr)
        Ltr, Lte = pl.transform(Xtr).to_numpy(), pl.transform(Xte).to_numpy()
    if "lda_qda" in models:
        out["lda"] = LinearDiscriminantAnalysis().fit(Ltr, ytr).predict_proba(Lte)[:, 1]
        reg = tune_inner(Xtr, ytr, ev[train], idtr, sets, k, make_qda, QDA_REGS)
        out["qda"] = make_qda(reg).fit(Ltr, ytr).predict_proba(Lte)[:, 1]
        info["qda_reg_param"] = reg
    if "rf" in models:
        leaf = tune_inner(Xtr, ytr, ev[train], idtr, sets, k, make_rf(k), RF_LEAVES)
        out["rf"] = make_rf(k)(leaf).fit(Ltr, ytr).predict_proba(Lte)[:, 1]
        info["rf_min_samples_leaf"] = leaf
    if "lasso" in models:
        C = tune_inner(Xtr, ytr, ev[train], idtr, sets, k, make_lasso(k), lasso_grid(Ltr, ytr))
        las = make_lasso(k)(C).fit(Ltr, ytr)
        out["lasso"] = las.predict_proba(Lte)[:, 1]
        nz = np.abs(las.coef_[0]) > 1e-8
        sel["lasso"] = [b for b, cols in pl.blocks().items() if nz[cols].any()]
        info["lasso_C"] = C; info["lasso_nonzero_columns"] = int(nz.sum())
    return dict(test_idx=np.where(test)[0], prob=out, info=info, selected=sel)


def cross_validate(pairs, sets, spec, feature_set, models, jobs):
    X, y, ev, ids = M.build_frame(pairs, sets, spec, feature_set)
    folds = M.event_folds(ev)
    res = Parallel(n_jobs=jobs)(delayed(run_fold)(k, X, y, ev, ids, sets, folds, models) for k in range(5))
    names = list(res[0]["prob"])
    prob = {m: np.empty(len(y)) for m in names}; thr = np.empty(len(y))
    for r in res:
        for m in names:
            prob[m][r["test_idx"]] = r["prob"][m]
        thr[r["test_idx"]] = r["info"]["threshold"]
    rows = []
    for m in names:
        met = M.metric_table(prob[m], y, thr, ev)
        rows.append(dict(spec=spec.label, missing=spec.missing, fit=spec.fit, out_of_range=spec.out_of_range,
                         pref_format=spec.pref_format, headline=spec == M.HEADLINE, feature_set=feature_set, model=m,
                         fold_scheme="event", dyads=len(y), matches=int(y.sum()), seed=M.FOLD_SEED,
                         **{f"{k}": v[0] for k, v in met.items()}, **{f"{k}_se": v[1] for k, v in met.items()}))
    fold_info = [dict(spec=spec.label, feature_set=feature_set, **r["info"]) for r in res]
    selection = [dict(spec=spec.label, feature_set=feature_set, direction=d, fold=r["info"]["fold"], block=b)
                 for r in res for d, bl in r["selected"].items() for b in bl]
    return rows, fold_info, selection, (prob, y, ev, X)


# ---------------------------------------------------------------- full-data fit, coefficient table, diagnostics

def full_fit(pairs, sets, spec, feature_set):
    import statsmodels.api as sm
    X, y, ev, ids = M.build_frame(pairs, sets, spec, feature_set)
    spl = choose_splines(X, y, ids, sets)
    p = make_prep(sets, X, splines=spl).fit(X, ids)
    D = p.transform(X)
    fit = sm.Logit(y, sm.add_constant(D)).fit(disp=0, maxiter=200, method="bfgs", cov_type="cluster", cov_kwds={"groups": ev})
    coef = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "se_cluster_event": fit.bse.values,
                         "z": fit.tvalues.values}).assign(feature_set=feature_set, spec=spec.label, splines=";".join(spl),
                                                          dyads=len(y), note="per SD of the prepared predictor; spline terms (:ns) are read as a block")
    return coef, D, y, ev, fit


def vif(D):
    C = np.corrcoef(D.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(C)
    return pd.DataFrame({"term": D.columns, "vif": np.diag(inv)}).sort_values("vif", ascending=False)


def cooks_distance(D, y, fit):
    Xb = np.column_stack([np.ones(len(y)), D.to_numpy()]); p = fit.predict(np.column_stack([np.ones(len(y)), D.to_numpy()]))
    W = p * (1 - p); XW = Xb * np.sqrt(W)[:, None]
    h = np.einsum("ij,jk,ik->i", XW, np.linalg.pinv(XW.T @ XW), XW)
    r = (y - p) / np.sqrt(W)
    return r ** 2 * h / (Xb.shape[1] * (1 - h) ** 2), h, p


def calibration(prob, y, bins=10):
    q = pd.qcut(prob, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"decile": q, "p": prob, "y": y}).groupby("decile").agg(mean_predicted=("p", "mean"),
                                                                             observed_rate=("y", "mean"), dyads=("y", "size")).reset_index()


def binned_residuals(prob, y, X, cols, bins=10):
    rows = []
    for c in cols:
        v = X[c].to_numpy(float); ok = ~np.isnan(v)
        q = pd.qcut(v[ok], bins, labels=False, duplicates="drop")
        d = pd.DataFrame({"bin": q, "x": v[ok], "res": (y - prob)[ok]}).groupby("bin").agg(x_mean=("x", "mean"), mean_residual=("res", "mean"),
                                                                                         sd=("res", "std"), n=("res", "size")).reset_index()
        d["se"] = d.sd / np.sqrt(d.n)
        rows.append(d.assign(predictor=c))
    return pd.concat(rows, ignore_index=True)


def plot_calibration(cal, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2), layout="constrained")
    for fs, color in (("pre", "#2a78d6"), ("pre+date", "#eb6834")):
        d = cal[cal.feature_set == fs]
        ax.plot(d.mean_predicted, d.observed_rate, "o-", color=color, lw=1.5, ms=5, mec="white", label=f"`{fs}`")
    lim = max(cal.mean_predicted.max(), cal.observed_rate.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#c3c2b7", lw=1, zorder=0)
    ax.set_xlabel("Mean predicted match probability (decile)"); ax.set_ylabel("Observed match rate")
    ax.set_title("Calibration of held-out predictions\n(headline logistic regression, event CV)", loc="left", fontsize=10)
    ax.legend(frameon=False); ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200); plt.close(fig)


def plot_binned(br, path):
    preds = br.predictor.unique()
    n = len(preds); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 2.3 * rows), layout="constrained", squeeze=False)
    for ax, c in zip(axes.flat, preds):
        d = br[br.predictor == c]
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.fill_between(d.x_mean, -2 * d.se, 2 * d.se, color="#e1e0d9", alpha=0.7, lw=0)
        ax.plot(d.x_mean, d.mean_residual, "o-", color="#2a78d6", ms=4, lw=1.5)
        ax.set_title(c, fontsize=9, loc="left"); ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Binned residuals (observed minus predicted) of held-out predictions; band = ±2 SE of zero",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=200); plt.close(fig)


# ---------------------------------------------------------------- the yes model (P5)

def yes_model():
    """Logistic regression of each date's yes on the six in-date ratings (standardized within the rater's gender),
    by gender, on dates with all six ratings; cluster-robust by event. Revealed importance = share of the positive
    standardized coefficients; stated importance = mean own-preference share in the 100-point waves (PREREG.md)."""
    import statsmodels.api as sm
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    rows = []
    for g, name in ((0, "women"), (1, "men")):
        d = clean[(clean.gender == g) & clean[ATTRS].notna().all(axis=1)]
        Z = (d[ATTRS] - d[ATTRS].mean()) / d[ATTRS].std()
        fit = sm.Logit(d.dec.to_numpy(float), sm.add_constant(Z)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.event.to_numpy()})
        coef = fit.params[ATTRS]; pos = coef.clip(lower=0)
        people = clean[(clean.gender == g) & (clean.pref_scale == "100pt")].drop_duplicates("rater")
        stated = people[[f"{a}1_1_share" for a in ATTRS]].mean()
        for a in ATTRS:
            rows.append(dict(rater_gender=name, attribute=a, std_coef=coef[a], se_cluster_event=fit.bse[a],
                             revealed_share=100 * pos[a] / pos.sum(), stated_share=stated[f"{a}1_1_share"],
                             dates=len(d), raters=d.rater.nunique(), stated_people=len(people)))
    t = pd.DataFrame(rows)
    t["gap_revealed_minus_stated"] = t.revealed_share - t.stated_share
    return t


# ---------------------------------------------------------------- hypotheses

def hypotheses(met, sr):
    h = met[met.headline]
    cand = ["logistic", "stepwise_forward", "stepwise_backward", "stepwise_both", "lda", "qda", "rf"]
    def best(df, fs):
        d = df[(df.feature_set == fs) & df.model.isin(cand)]
        return d.loc[d.auc.idxmax()]
    rows = []
    for label, df in (("headline", h), ("complete-case", met[met.spec == M.COMPLETE_CASE.label])):
        if df.empty or "pre" not in set(df.feature_set):
            continue
        bp, bd = best(df, "pre"), best(df, "pre+date")
        rows.append(dict(hypothesis="P1", version=label, statistic=f"best `pre` AUC {bp.auc:.3f} ({bp.model}, SE {bp.auc_se:.3f})",
                         supported=bool(0.58 <= bp.auc < 0.70), note=""))
        gap = bd.auc - bp.auc
        rows.append(dict(hypothesis="P2", version=label, statistic=f"best `pre+date` AUC {bd.auc:.3f} ({bd.model}) minus best `pre` {bp.auc:.3f} = {gap:.3f}",
                         supported=bool(gap >= 0.10), note=""))
        lp = df[(df.feature_set == "pre") & (df.model == "logistic")]; li = df[(df.feature_set == "pre_individual") & (df.model == "logistic")]
        if len(lp) and len(li):
            d3 = lp.auc.iloc[0] - li.auc.iloc[0]
            rows.append(dict(hypothesis="P3", version=label, statistic=f"full logistic `pre` AUC {lp.auc.iloc[0]:.3f} minus without pair features {li.auc.iloc[0]:.3f} = {d3:.3f}",
                             supported=bool(d3 < 0.02), note=""))
        rf = df[(df.feature_set == "pre") & (df.model == "rf")]
        if len(lp) and len(rf):
            d4 = rf.auc.iloc[0] - lp.auc.iloc[0]
            rows.append(dict(hypothesis="P4", version=label, statistic=f"random forest `pre` AUC {rf.auc.iloc[0]:.3f} minus full logistic {lp.auc.iloc[0]:.3f} = {d4:.3f}",
                             supported=bool(d4 < 0.03), note=""))
    for g in ("women", "men"):
        s = sr[sr.rater_gender == g].set_index("attribute")
        top_coef, top_gap = s.std_coef.idxmax(), s.gap_revealed_minus_stated.idxmax()
        rows.append(dict(hypothesis="P5", version=g, statistic=f"largest std. coefficient: {top_coef}; largest revealed-minus-stated gap: {top_gap} "
                         f"({s.gap_revealed_minus_stated[top_gap]:+.1f} points)", supported=bool(top_coef == "attr" and top_gap == "attr"), note="dates with all six ratings"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--specs", default="all"); ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    specs = [M.HEADLINE, M.COMPLETE_CASE] + ([s for s in M.grid_specs() if s not in (M.HEADLINE, M.COMPLETE_CASE)] if args.specs == "all" else [])
    log(f"{len(specs)} specifications; event folds seed {M.FOLD_SEED}: {M.event_folds(pairs.event)}")
    met, finfo, sel, keep = [], [], [], {}
    for i, spec in enumerate(specs):
        stepwise = spec in (M.HEADLINE, M.COMPLETE_CASE)
        for fs in ("pre", "pre+date", "pre_individual"):
            models = ("logistic",) if fs == "pre_individual" else (("stepwise", "lda_qda", "rf", "lasso") if stepwise else ("lda_qda", "rf", "lasso"))
            t0 = time.time()
            rows, fi, se, kept = cross_validate(pairs, sets, spec, fs, models, args.jobs)
            met += rows; finfo += fi; sel += se
            if spec == M.HEADLINE:
                keep[fs] = kept
            best = max(rows, key=lambda r: r["auc"])
            log(f"spec {i + 1}/{len(specs)} [{spec.label}] {fs}: best AUC {best['auc']:.3f} ({best['model']}) in {time.time() - t0:.0f}s")
        pd.DataFrame(met).to_csv(RES / "part1_models_metrics.csv", index=False)   # saved as it goes
        pd.DataFrame(finfo).to_csv(RES / "part1_models_folds.csv", index=False)
    metrics = pd.DataFrame(met)

    selection = pd.DataFrame(sel)
    if len(selection):
        las = selection[selection.direction == "lasso"]
        lf = las.groupby(["spec", "feature_set", "block"]).size().rename("folds_nonzero").reset_index()
        lf.sort_values(["spec", "feature_set", "folds_nonzero"], ascending=[True, True, False]).to_csv(RES / "part1_models_lasso_selection.csv", index=False)
        selection = selection[selection.direction != "lasso"]
        freq = selection.groupby(["spec", "feature_set", "direction", "block"]).size().rename("folds_selected").reset_index()
        freq["share_of_folds"] = freq.folds_selected / 5
        freq.sort_values(["spec", "feature_set", "direction", "folds_selected"], ascending=[True, True, True, False]).to_csv(RES / "part1_models_stepwise_selection.csv", index=False)

    log("full-data fits, coefficient table and diagnostics (headline)")
    coefs, cal, br, infl, vifs = [], [], [], [], []
    for fs in ("pre", "pre+date"):
        coef, D, y, ev, fit = full_fit(pairs, sets, M.HEADLINE, fs)
        coefs.append(coef)
        vifs.append(vif(D).assign(feature_set=fs))
        cd, h, p = cooks_distance(D, y, fit)
        X, _, _, ids = M.build_frame(pairs, sets, M.HEADLINE, fs)
        top = np.argsort(cd)[::-1][:10]
        infl.append(pd.DataFrame({"feature_set": fs, "f_iid": ids.f_iid.to_numpy()[top], "m_iid": ids.m_iid.to_numpy()[top], "event": ev[top],
                                  "match": y[top], "fitted_p": p[top], "cooks_distance": cd[top], "leverage": h[top]}))
        prob, yy, evv, XX = keep[fs]
        cal.append(calibration(prob["logistic"], yy).assign(feature_set=fs))
        cont = [c for c in XX.columns if c not in sets["categorical"] and c not in sets["binary"] and XX[c].nunique() > 2]
        br.append(binned_residuals(prob["logistic"], yy, XX, cont).assign(feature_set=fs))
    pd.concat(coefs).to_csv(RES / "part1_models_coefficients.csv", index=False)
    pd.concat(vifs).to_csv(RES / "part1_models_vif.csv", index=False)
    pd.concat(infl).to_csv(RES / "part1_models_influential.csv", index=False)
    cal = pd.concat(cal); cal.to_csv(RES / "part1_models_calibration.csv", index=False)
    br = pd.concat(br); br.to_csv(RES / "part1_models_binned_residuals.csv", index=False)
    plot_calibration(cal, RES / "part1_models_calibration.png")
    flagged = br[(br.feature_set == "pre") & (br.mean_residual.abs() > 2 * br.se)].predictor.value_counts()
    show = list(dict.fromkeys([c for c in sets["spline_candidates"]] + list(flagged.index[:6])))
    plot_binned(br[(br.feature_set == "pre") & br.predictor.isin(show)], RES / "part1_models_binned_residuals.png")

    log("yes model (P5)")
    sr = yes_model(); sr.to_csv(RES / "part1_models_stated_vs_revealed.csv", index=False)
    hyp = hypotheses(metrics, sr); hyp.to_csv(RES / "part1_models_hypotheses.csv", index=False)

    # printed summary for the weekly report
    show_cols = ["feature_set", "model", "auc", "auc_se", "balanced_accuracy", "balanced_accuracy_se", "sensitivity", "specificity", "error_rate"]
    for lab, spec in (("HEADLINE", M.HEADLINE), ("COMPLETE CASES", M.COMPLETE_CASE)):
        t = metrics[metrics.spec == spec.label][show_cols + ["dyads", "matches"]]
        print(f"\n{lab} ({spec.label}); event 5-fold CV; SEs cluster-robust by event; {SIM_COVERAGE}")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    v = pd.concat(vifs)
    print("\nlargest variance inflation factors (headline, full design):")
    print(v.groupby("feature_set").head(5).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\ncalibration (headline logistic, held-out deciles):")
    print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(flagged):
        print("\npredictors with a binned residual beyond 2 SE in `pre` (count of bins):", flagged.to_dict())
    print("\nstated vs revealed (P5):")
    print(sr[["rater_gender", "attribute", "std_coef", "se_cluster_event", "revealed_share", "stated_share", "gap_revealed_minus_stated"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nhypotheses:"); print(hyp.to_string(index=False))
    if len(selection):
        f = pd.read_csv(RES / "part1_models_stepwise_selection.csv")
        f = f[f.spec == M.HEADLINE.label]
        print("\nstepwise blocks selected in at least 3 of 5 folds (headline):")
        for (fs, d), g in f[f.folds_selected >= 3].groupby(["feature_set", "direction"]):
            print(f"  {fs:9s} {d:8s}: " + ", ".join(f"{b} ({n})" for b, n in zip(g.block, g.folds_selected)))
    lf = pd.read_csv(RES / "part1_models_lasso_selection.csv")
    lf = lf[lf.spec == M.HEADLINE.label]
    folds = pd.read_csv(RES / "part1_models_folds.csv")
    folds = folds[(folds.spec == M.HEADLINE.label) & folds.feature_set.isin(["pre", "pre+date"])]
    print("\nlasso (headline; exploratory, not in the best-model pool of P1 and P2):")
    for fs, g in lf.groupby("feature_set"):
        nzc = folds[folds.feature_set == fs].lasso_nonzero_columns.tolist()
        print(f"  {fs:9s} nonzero columns per fold {nzc}; kept in all 5 folds: " + ", ".join(g[g.folds_nonzero == 5].block))
    log("wrote " + ", ".join(sorted(p.name for p in RES.glob("part1_models_*"))))


if __name__ == "__main__":
    main()
