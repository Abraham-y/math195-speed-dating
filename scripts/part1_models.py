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
cluster-robust estimator with events as clusters (sensitivity, specificity and balanced accuracy in the linearized
ratio form); AUC by Obuchowski (1997). scripts/sim_auc_cluster.py checks these SEs (five seeds, two designs); its
range is stored in the `se_check` column of the metrics table.
Also: the coefficient table (full model, all dyads, headline and complete-case; IRLS with a cluster-robust sandwich
by event; identical columns reported as one term and separated terms as not estimable; its SEs checked in
scripts/sim_coef_cluster.py), diagnostics, the yes model and the stated-versus-revealed table (P5: the signed gap,
with the waves 6 to 9 stated shares reported separately without a verdict), and the hypothesis table.
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
from sim_auc_cluster import coverage_note

RES = pathlib.Path("results")
QDA_REGS = [0.1, 0.3, 0.5, 0.7, 0.9]
RF_LEAVES = [1, 5, 20, 50]
RF_SEED, RF_TREES, RF_JOBS = 20261106, 500, 2
LASSO_SEED, LASSO_STEPS = 20261206, 30
DIRECTIONS = ("forward", "backward", "both")
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]


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
            probs[r][b] = predict_serial(make_model(r).fit(Da, ytr[a]), Db)
    return max(grid, key=lambda r: M.auc(probs[r], ytr))


def predict_serial(model, D):
    """Predicted probabilities with one thread: a forest summing its trees across threads adds them in a varying
    order, so the last digits (and a tuning tie) could change from run to run."""
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    return model.predict_proba(D)[:, 1]


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


def make_rf(k, seed=RF_SEED):
    return lambda leaf: RandomForestClassifier(n_estimators=RF_TREES, max_features="sqrt", min_samples_leaf=leaf,
                                               random_state=seed + k, n_jobs=RF_JOBS)


def run_fold(k, X, y, ev, ids, sets, folds, models, rf_seed=RF_SEED):
    import warnings; warnings.filterwarnings("ignore")
    test = np.array([folds[e] == k for e in ev]); train = ~test
    Xtr, Xte, ytr = X[train].reset_index(drop=True), X[test].reset_index(drop=True), y[train]
    idtr = ids[train].reset_index(drop=True)
    out, info = {}, dict(fold=k, threshold=float(ytr.mean()), n_train=int(train.sum()), n_test=int(test.sum()),
                         inner_seed=M.FOLD_SEED + 1 + k, rf_random_state=rf_seed + k, lasso_random_state=LASSO_SEED + k)
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
        leaf = tune_inner(Xtr, ytr, ev[train], idtr, sets, k, make_rf(k, rf_seed), RF_LEAVES)
        out["rf"] = predict_serial(make_rf(k, rf_seed)(leaf).fit(Ltr, ytr), Lte)
        info["rf_min_samples_leaf"] = leaf
    if "lasso" in models:
        C = tune_inner(Xtr, ytr, ev[train], idtr, sets, k, make_lasso(k), lasso_grid(Ltr, ytr))
        las = make_lasso(k)(C).fit(Ltr, ytr)
        out["lasso"] = las.predict_proba(Lte)[:, 1]
        nz = np.abs(las.coef_[0]) > 1e-8
        sel["lasso"] = [b for b, cols in pl.blocks().items() if nz[cols].any()]
        info["lasso_C"] = C; info["lasso_nonzero_columns"] = int(nz.sum())
    return dict(test_idx=np.where(test)[0], prob=out, info=info, selected=sel)


def cross_validate(pairs, sets, spec, feature_set, models, jobs, exclude_events=(), complete_extra=(), folds=None,
                   fold_seed=M.FOLD_SEED, rf_seed=RF_SEED):
    """Event 5-fold CV of the given models. Folds: by default the events used are dealt into 5 folds by fold_seed;
    `folds` (event -> fold) fixes them instead, e.g. the headline folds with some waves left out."""
    X, y, ev, ids = M.build_frame(pairs, sets, spec, feature_set, exclude_events, complete_extra)
    folds_used = M.event_folds(ev, seed=fold_seed) if folds is None else {e: folds[e] for e in np.unique(ev)}
    fold_label = f"seed {fold_seed} over the events used" if folds is None else "given (headline folds)"
    res = Parallel(n_jobs=jobs)(delayed(run_fold)(k, X, y, ev, ids, sets, folds_used, models, rf_seed) for k in range(5))
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
                         fold_scheme="event", folds=fold_label, dyads=len(y), matches=int(y.sum()), seed=fold_seed,
                         rf_seed=rf_seed, lasso_seed=LASSO_SEED,
                         **{f"{k}": v[0] for k, v in met.items()}, **{f"{k}_se": v[1] for k, v in met.items()}))
    fold_info = [dict(spec=spec.label, feature_set=feature_set, **r["info"]) for r in res]
    selection = [dict(spec=spec.label, feature_set=feature_set, direction=d, fold=r["info"]["fold"], block=b)
                 for r in res for d, bl in r["selected"].items() for b in bl]
    return rows, fold_info, selection, (prob, y, ev, X)


# ---------------------------------------------------------------- full-data fit, coefficient table, diagnostics

def twin_groups(D):
    """Groups of columns that are identical over every row of D, in column order."""
    seen = {}
    for c in D.columns:
        seen.setdefault(D[c].to_numpy(float).tobytes(), []).append(c)
    return [grp for grp in seen.values() if len(grp) > 1]


def coefficient_table(D, y, g, ridge=1e-6, path=(1e-4, 1e-8), tol=0.01):
    """Logistic regression by IRLS (modeling.fit_logit, run to convergence) with the cluster-robust sandwich: bread
    (X'WX + ridge)^-1, meat G/(G-1) times the sum over clusters of the outer products of the score sums.
    - Identical columns: only their combined effect is estimable, so each group is fitted as one column and reported
      as one term, its names joined by ' + '.
    - Separation: a coefficient that moves by more than tol when the ridge goes from path[0] to path[1] has no
      finite estimate (the likelihood keeps rising along some direction, through one column or several together),
      so it is reported as not estimable; where several move and their sum does not, the sum gets its own row. The
      other coefficients and SEs are the limit of the likelihood, since the separated dyads' fitted probabilities
      sit at 0 or 1 and add nothing.
    Returns (table, fitted probabilities)."""
    y, g = np.asarray(y, float), np.asarray(g); n = len(y)
    groups = twin_groups(D)
    R = D.drop(columns=[c for grp in groups for c in grp[1:]])
    names = {grp[0]: " + ".join(grp) for grp in groups}
    terms = np.array(["const"] + [names.get(c, c) for c in R.columns], dtype=object)
    A = R.to_numpy(float)
    fits = {r: M.fit_logit(A, y, ridge=r, iters=500)[0] for r in (ridge, *path)}
    w = fits[ridge]; p = M.predict_logit(w, A)
    X = np.column_stack([np.ones(n), A]); pen = np.r_[0.0, np.full(A.shape[1], ridge)]
    score = np.abs(X.T @ (y - p) - pen * w).max()
    moves = np.abs(fits[path[0]] - fits[path[1]]) > tol
    bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + np.diag(pen))
    S = pd.DataFrame(X * (y - p)[:, None]).groupby(g).sum().to_numpy(); G = len(S)
    cov = bread @ (G / (G - 1) * S.T @ S) @ bread
    coef, se = np.where(moves, np.nan, w), np.where(moves, np.nan, np.sqrt(np.diag(cov)))
    note = ["not estimable: separation (the estimate grows without bound as the ridge shrinks)" if m
            else "identical columns in the data: one combined term" if t in names.values() else "" for t, m in zip(terms, moves)]
    t = pd.DataFrame({"term": terms, "coef": coef, "se_cluster_event": se, "z": coef / se, "estimable": ~moves, "note": note})
    idx = np.where(moves)[0]
    if len(idx) >= 2:
        a = np.zeros(len(w)); a[idx] = 1
        if abs(a @ fits[path[0]] - a @ fits[path[1]]) <= tol:
            v = np.sqrt(a @ cov @ a)
            t = pd.concat([t, pd.DataFrame([dict(term="sum: " + " + ".join(terms[idx]), coef=a @ w, se_cluster_event=v, z=a @ w / v,
                                                 estimable=True, note="sum of the separated terms, which the data do identify")])],
                          ignore_index=True)
    t.attrs["max_abs_score"] = score
    return t, p


def full_fit(pairs, sets, spec, feature_set):
    X, y, ev, ids = M.build_frame(pairs, sets, spec, feature_set)
    spl = choose_splines(X, y, ids, sets)
    p = make_prep(sets, X, splines=spl).fit(X, ids)
    D = p.transform(X)
    coef, fitted = coefficient_table(D, y, ev)
    assert coef.attrs["max_abs_score"] < 1e-6, f"IRLS did not converge ({feature_set}): max |score| {coef.attrs['max_abs_score']:.2e}"
    coef = coef.assign(feature_set=feature_set, spec=spec.label, splines=";".join(spl), dyads=len(y), fold_scheme="none (all dyads)",
                       max_abs_score=coef.attrs["max_abs_score"], scale="per SD of the prepared predictor; spline terms (:ns) are read as a block")
    return coef, D, y, ev, fitted


def vif(D, tol=1e-10):
    """Variance inflation factor of each column: 1 / (1 - R^2) from regressing it on all the others with an
    intercept. A column the others reproduce exactly (residual sum of squares below tol times its total) is
    reported as infinite rather than hidden by a pseudo-inverse."""
    A = D.to_numpy(float); n = len(A); out = []
    for j in range(A.shape[1]):
        v = A[:, j]; X = np.column_stack([np.ones(n), np.delete(A, j, axis=1)])
        rss = np.sum((v - X @ np.linalg.lstsq(X, v, rcond=None)[0]) ** 2); tss = np.sum((v - v.mean()) ** 2)
        out.append(np.nan if tss == 0 else np.inf if rss <= tol * tss else tss / rss)
    return pd.DataFrame({"term": D.columns, "vif": out}).sort_values("vif", ascending=False)


def cooks_distance(D, y, p):
    """Cook's distance and leverage of a logistic fit with fitted probabilities p (Pearson residuals)."""
    Xb = np.column_stack([np.ones(len(y)), D.to_numpy()])
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

def yes_design(clean, g, exclude_events=()):
    """One gender's dates with all six ratings: (standardized ratings, yes, event)."""
    d = clean[(clean.gender == g) & clean[ATTRS].notna().all(axis=1) & ~clean.event.isin(exclude_events)]
    return (d[ATTRS] - d[ATTRS].mean()) / d[ATTRS].std(), d.dec.to_numpy(float), d.event.to_numpy(), d


def yes_model(exclude_events=()):
    """Logistic regression of each date's yes on the six in-date ratings (standardized within the rater's gender),
    by gender, on dates with all six ratings; cluster-robust sandwich by event with G/(G-1) (coefficient_table).
    Revealed importance = share of the positive standardized coefficients; stated importance = mean own-preference
    share in the 100-point waves (PREREG.md), with the waves 6 to 9 shares (rated 1 to 10, converted to shares)
    reported separately next to it. exclude_events leaves waves out of both (the wave 12 check)."""
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    rows = []
    shares = [f"{a}1_1_share" for a in ATTRS]
    for g, name in ((0, "women"), (1, "men")):
        Z, yv, ev, d = yes_design(clean, g, exclude_events)
        tab, _ = coefficient_table(Z, yv, ev)
        coef, se = tab.set_index("term").coef[ATTRS], tab.set_index("term").se_cluster_event[ATTRS]
        pos = coef.clip(lower=0)
        people = clean[(clean.gender == g) & ~clean.event.isin(exclude_events)].drop_duplicates("rater")
        p100 = people[(people.pref_scale == "100pt") & people[shares].notna().all(axis=1)]
        p69 = people[(people.pref_scale == "1-10") & people[shares].notna().all(axis=1)]
        stated, stated69 = p100[shares].mean(), p69[shares].mean()
        for a in ATTRS:
            rows.append(dict(rater_gender=name, attribute=a, std_coef=coef[a], se_cluster_event=se[a],
                             revealed_share=100 * pos[a] / pos.sum(), stated_share=stated[f"{a}1_1_share"],
                             stated_share_waves_6_9=stated69[f"{a}1_1_share"], dates=len(d), raters=d.rater.nunique(),
                             stated_people=len(p100), stated_people_waves_6_9=len(p69),
                             excluded_events=";".join(map(str, exclude_events)), fold_scheme="none (all dates)"))
    t = pd.DataFrame(rows)
    t["gap_revealed_minus_stated"] = t.revealed_share - t.stated_share
    t["gap_revealed_minus_stated_waves_6_9"] = t.revealed_share - t.stated_share_waves_6_9
    return t


# ---------------------------------------------------------------- hypotheses

def p5_rows(sr, version="{g}", col="gap_revealed_minus_stated", verdict=True,
            note="dates with all six ratings; stated shares from the 100-point waves"):
    """P5 as tested since its first run: attractiveness has the largest standardized coefficient and the largest
    signed gap (revealed minus stated). The margin over the next gap and the reading by absolute size are
    exploratory. verdict=False reports a version without a verdict (the waves 6 to 9 stated shares)."""
    rows = []
    for g in ("women", "men"):
        s = sr[sr.rater_gender == g].set_index("attribute")
        top_coef, gaps = s.std_coef.idxmax(), s[col].sort_values(ascending=False)
        size = s[col].abs().sort_values(ascending=False)
        rows.append(dict(hypothesis="P5", version=version.format(g=g), fold_scheme="none (all dates)",
                         statistic=f"largest std. coefficient: {top_coef}; largest revealed-minus-stated gap: {gaps.index[0]} ({gaps.iloc[0]:+.1f} points)",
                         supported=bool(top_coef == "attr" and gaps.index[0] == "attr") if verdict else None,
                         note=f"{note}; exploratory: the largest gap leads the next ({gaps.index[1]}) by {gaps.iloc[0] - gaps.iloc[1]:.2f} points, "
                              f"and by absolute size the largest is {size.index[0]}, ahead of {size.index[1]} by {size.iloc[0] - size.iloc[1]:.2f} points"))
    return rows


def hypotheses(met, sr):
    h = met[met.headline.astype(bool)]
    cand = ["logistic", "stepwise_forward", "stepwise_backward", "stepwise_both", "lda", "qda", "rf"]
    def best(df, fs):
        d = df[(df.feature_set == fs) & df.model.isin(cand)]
        return d.loc[d.auc.idxmax()]
    rows = []
    for label, df in (("headline", h), ("complete-case", met[met.spec == M.COMPLETE_CASE.label])):
        if df.empty or "pre" not in set(df.feature_set):
            continue
        bp, bd = best(df, "pre"), best(df, "pre+date")
        rows.append(dict(hypothesis="P1", version=label, statistic=f"best `pre` AUC {bp.auc:.4f} ({bp.model}, SE {bp.auc_se:.4f})",
                         supported=bool(0.58 <= bp.auc < 0.70), note=""))
        gap = bd.auc - bp.auc
        rows.append(dict(hypothesis="P2", version=label, statistic=f"best `pre+date` AUC {bd.auc:.4f} ({bd.model}) minus best `pre` {bp.auc:.4f} = {gap:.4f}",
                         supported=bool(gap >= 0.10), note=""))
        lp = df[(df.feature_set == "pre") & (df.model == "logistic")]; li = df[(df.feature_set == "pre_individual") & (df.model == "logistic")]
        if len(lp) and len(li):
            d3 = lp.auc.iloc[0] - li.auc.iloc[0]
            rows.append(dict(hypothesis="P3", version=label, statistic=f"full logistic `pre` AUC {lp.auc.iloc[0]:.4f} minus without pair features {li.auc.iloc[0]:.4f} = {d3:.4f}",
                             supported=bool(d3 < 0.02), note=""))
        rf = df[(df.feature_set == "pre") & (df.model == "rf")]
        if len(lp) and len(rf):
            d4 = rf.auc.iloc[0] - lp.auc.iloc[0]
            rows.append(dict(hypothesis="P4", version=label, statistic=f"random forest `pre` AUC {rf.auc.iloc[0]:.4f} minus full logistic {lp.auc.iloc[0]:.4f} = {d4:.4f}",
                             supported=bool(d4 < 0.03), note=""))
    for r in rows:
        r["fold_scheme"] = "event"
    rows += p5_rows(sr)
    if "gap_revealed_minus_stated_waves_6_9" in sr:
        rows += p5_rows(sr, "{g}, waves 6 to 9 stated shares", "gap_revealed_minus_stated_waves_6_9", verdict=False,
                        note="reported separately without a verdict (PREREG.md, survey format); same yes model")
    return pd.DataFrame(rows)


def with_se_check(hyp, auc_note, yes_note):
    """The SE check that applies to each hypothesis row: the AUC simulation for P1 to P4, the yes-model bootstrap for P5."""
    return hyp.assign(se_check=np.where(hyp.hypothesis == "P5", yes_note, auc_note))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--specs", default="all"); ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--hypotheses-only", action="store_true", help="rebuild the hypothesis table from the saved metrics and yes model")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    from sim_coef_cluster import coverage_note as coef_note        # imported here: that script imports this one
    if args.hypotheses_only:
        hyp = with_se_check(hypotheses(pd.read_csv(RES / "part1_models_metrics.csv"), pd.read_csv(RES / "part1_models_stated_vs_revealed.csv")),
                            coverage_note(), coef_note("yes"))
        hyp.to_csv(RES / "part1_models_hypotheses.csv", index=False); print(hyp.to_string(index=False)); return
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    specs = [M.HEADLINE, M.COMPLETE_CASE] + ([s for s in M.grid_specs() if s not in (M.HEADLINE, M.COMPLETE_CASE)] if args.specs == "all" else [])
    log(f"{len(specs)} specifications; event folds seed {M.FOLD_SEED}: {M.event_folds(pairs.event)}")
    note = coverage_note()
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
        pd.DataFrame(met).assign(se_check=note).to_csv(RES / "part1_models_metrics.csv", index=False)   # saved as it goes
        pd.DataFrame(finfo).to_csv(RES / "part1_models_folds.csv", index=False)
    metrics = pd.DataFrame(met).assign(se_check=note)

    selection = pd.DataFrame(sel)
    if len(selection):
        las = selection[selection.direction == "lasso"]
        lf = las.groupby(["spec", "feature_set", "block"]).size().rename("folds_nonzero").reset_index()
        lf.sort_values(["spec", "feature_set", "folds_nonzero"], ascending=[True, True, False]).to_csv(RES / "part1_models_lasso_selection.csv", index=False)
        selection = selection[selection.direction != "lasso"]
        freq = selection.groupby(["spec", "feature_set", "direction", "block"]).size().rename("folds_selected").reset_index()
        freq["share_of_folds"] = freq.folds_selected / 5
        freq.sort_values(["spec", "feature_set", "direction", "folds_selected"], ascending=[True, True, True, False]).to_csv(RES / "part1_models_stepwise_selection.csv", index=False)

    log("full-data fits: coefficient table (headline and complete-case) and diagnostics (headline)")
    coefs, cal, br, infl, vifs = [], [], [], [], []
    for fs in ("pre", "pre+date"):
        coefs.append(full_fit(pairs, sets, M.COMPLETE_CASE, fs)[0])
        coef, D, y, ev, fitted = full_fit(pairs, sets, M.HEADLINE, fs)
        coefs.append(coef)
        vifs.append(vif(D).assign(feature_set=fs))
        cd, h, p = cooks_distance(D, y, fitted)
        X, _, _, ids = M.build_frame(pairs, sets, M.HEADLINE, fs)
        top = np.argsort(cd)[::-1][:10]
        infl.append(pd.DataFrame({"feature_set": fs, "f_iid": ids.f_iid.to_numpy()[top], "m_iid": ids.m_iid.to_numpy()[top], "event": ev[top],
                                  "match": y[top], "fitted_p": p[top], "cooks_distance": cd[top], "leverage": h[top]}))
        prob, yy, evv, XX = keep[fs]
        cal.append(calibration(prob["logistic"], yy).assign(feature_set=fs))
        cont = [c for c in XX.columns if c not in sets["categorical"] and c not in sets["binary"] and XX[c].nunique() > 2]
        br.append(binned_residuals(prob["logistic"], yy, XX, cont).assign(feature_set=fs))
    pd.concat(coefs).assign(se_check=coef_note("coefficients")).to_csv(RES / "part1_models_coefficients.csv", index=False)
    pd.concat(vifs).to_csv(RES / "part1_models_vif.csv", index=False)
    pd.concat(infl).to_csv(RES / "part1_models_influential.csv", index=False)
    cal = pd.concat(cal); cal.to_csv(RES / "part1_models_calibration.csv", index=False)
    br = pd.concat(br); br.to_csv(RES / "part1_models_binned_residuals.csv", index=False)
    plot_calibration(cal, RES / "part1_models_calibration.png")
    flagged = br[(br.feature_set == "pre") & (br.mean_residual.abs() > 2 * br.se)].predictor.value_counts()
    show = list(dict.fromkeys([c for c in sets["spline_candidates"]] + list(flagged.index[:6])))
    plot_binned(br[(br.feature_set == "pre") & br.predictor.isin(show)], RES / "part1_models_binned_residuals.png")

    log("yes model (P5)")
    sr = yes_model().assign(se_check=coef_note("yes")); sr.to_csv(RES / "part1_models_stated_vs_revealed.csv", index=False)
    hyp = with_se_check(hypotheses(metrics, sr), note, coef_note("yes")); hyp.to_csv(RES / "part1_models_hypotheses.csv", index=False)

    # printed summary for the weekly report
    show_cols = ["feature_set", "model", "auc", "auc_se", "balanced_accuracy", "balanced_accuracy_se", "sensitivity", "specificity", "error_rate"]
    for lab, spec in (("HEADLINE", M.HEADLINE), ("COMPLETE CASES", M.COMPLETE_CASE)):
        t = metrics[metrics.spec == spec.label][show_cols + ["dyads", "matches"]]
        print(f"\n{lab} ({spec.label}); event 5-fold CV; SEs cluster-robust by event; {note}")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cf = pd.concat(coefs)
    for (spec, fs), g in cf.groupby(["spec", "feature_set"], sort=False):
        print(f"\ncoefficient table [{spec}] {fs}: {g.dyads.iloc[0]} dyads, IRLS max |score| {g.max_abs_score.iloc[0]:.1e}")
        for r in g[g.note != ""].itertuples():
            print(f"  {r.term}: {r.note}" + ("" if np.isnan(r.coef) else f" ({r.coef:+.3f}, SE {r.se_cluster_event:.3f})"))
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
