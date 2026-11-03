"""Part 10: where the missing accuracy is (PREREG.md, Part 10). Before the date the best model reaches an AUC of about
0.58 on new events; with the in-date ratings, about 0.84. Three analyses ask where the difference comes from.

Input : data/pairs.csv, data/feature_sets.json, data/clean.csv, data/participants.csv; results/part1_models_metrics.csv
        (the learning curve's full-size runs must reproduce it)
Output: results/limits_learning.csv        held-out AUC per model, feature set, training-event count and subset
        results/limits_learning_summary.csv  mean and SD over subsets, mean training dyads, exploratory extrapolations
        results/limits_person.csv          within-event match AUCs (person effects, + in-date ratings, ratings only)
        results/limits_person_scores.csv   the held-out match scores per dyad
        results/limits_questionnaire.csv   held-out R2 of each person's appeal and yes-rate from their own answers,
                                           with the split-half reliability and the corrected R2
        results/limits_hypotheses.csv      L1, P1, P2, Q1, Q2
Seeds : event folds 20261006 (Part 1's, modeling.FOLD_SEED) and Part 1's inner seeds; training-event subsets
        [20266006, fold, m, r]; dyad folds within events 20266106; split halves [20266206, split]; the questionnaire's
        random forest random_state 20266306 + fold.

1. Learning curve (`--learning`). Part 1's headline event folds are kept. In each fold, m of the 16 or 17 training
   events are drawn at random (m = 4, 6, ..., 14; 20 draws each) and the lasso and the full logistic regression
   (part1_models.run_fold) are refitted on them and scored on the fold's held-out events. QDA cannot be fitted on
   the small draws (a class needs more rows than the 90 to 125 columns, and 4 events hold about 130 matches), so
   LDA and QDA are left out; the forest and stepwise selection are left out for time.
   Draw r of every fold together make one cross-validation run, scored by its pooled AUC. The full-size run (m = all)
   reproduces Part 1, which is checked.
2. Person ceiling (`--person`). Inside each event, dyads are dealt at random into 5 folds. For the dyads of fold j, a
   logistic model of each yes, mu + a_rater + b_ratee (yes_extensions' W1, ridge 1), is fitted to the decisions of
   every other dyad, so each person's yes-rate and appeal are estimated from their other dates at the same event and
   never from this pair. The match score is p(her yes) x p(his yes). Variants: + the decision-maker's eight in-date
   ratings of the partner (standardized, median-imputed with a missing flag, on the training dyads); the ratings
   alone without the person terms. AUC with the Obuchowski clustered SE by event; paired differences likewise.
3. Questionnaire (`--questionnaire`). W1 fitted to all 8,368 decisions gives each person's appeal (b, how often
   others say yes to them, adjusted for the raters) and yes-rate (a, adjusted for the partners). Each is predicted
   from the person's own pre-date answers (data/participants.csv, no race variables; gender added) with ridge
   regression and a random forest, tuned by event-grouped 4-fold CV inside Part 1's event folds. Held-out R2 with
   the cluster-robust SE by event (modeling.r_squared_and_rmse). Corrected R2 = R2 / reliability, with the
   reliability from split halves: each event's dyads are halved at random 20 times, W1 is fitted to each half, and
   the Spearman-Brown value 2r / (1 + r) of the half-to-half correlation is averaged. The correction assumes the
   estimation noise is unrelated to the answers.
Figures (exploratory, `--plot`, from the saved tables): results/limits_learning.png, results/limits_ladder.png.
Run from the repository root: python scripts/limits.py [--learning] [--person] [--questionnaire] [--plot] [--jobs 8]
(no flag runs all three analyses)
"""
import sys, json, pathlib, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1
from yes_extensions import indicator_design

RES = pathlib.Path("results")
SUBSET_SEED, DYAD_SEED, HALF_SEED, RF_SEED = 20266006, 20266106, 20266206, 20266306
SIZES, REPS = (4, 6, 8, 10, 12, 14), 20
LC_MODELS = ("logistic", "lasso")
RATINGS = ["attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob"]
W1_RIDGE, W1_ITERS = 1.0, 200
HALVES = 20
RIDGE_ALPHAS = list(np.logspace(-1, 4, 16))
RF_LEAVES = [1, 5, 20, 50]
CATEGORICAL = ["field_cd", "career_c", "goal"]
NUMERIC = ["age", "date", "go_out", "exphappy", "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
           "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga",
           "attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "attr1_1_share", "sinc1_1_share", "intel1_1_share",
           "fun1_1_share", "amb1_1_share", "shar1_1_share"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. learning curve

def subsets(folds, sizes=SIZES, reps=REPS, seed=SUBSET_SEED):
    """(fold, m, r, training events) for every draw; m = 'all' is the fold's full training set (one draw)."""
    out = []
    for k in range(5):
        train = sorted(e for e, f in folds.items() if f != k)
        for m in sizes:
            for r in range(reps):
                rng = np.random.default_rng([seed, k, m, r])
                out.append((k, m, r, tuple(sorted(rng.choice(train, m, replace=False).tolist()))))
        out.append((k, "all", 0, tuple(train)))
    return out


def run_subset(k, X, y, ev, ids, sets, folds, events):
    keep = np.isin(ev, list(events)) | np.array([folds[e] == k for e in ev])
    res = P1.run_fold(k, X[keep].reset_index(drop=True), y[keep], ev[keep], ids[keep].reset_index(drop=True), sets, folds, ("lasso",))
    idx = np.where(keep)[0][res["test_idx"]]
    return idx, {m: res["prob"][m] for m in LC_MODELS}, int(keep.sum() - len(idx))


def learning(pairs, sets, jobs):
    stored = pd.read_csv(RES / "part1_models_metrics.csv")
    stored = stored[(stored.headline == True) & (stored.fold_scheme == "event")].set_index(["feature_set", "model"]).auc
    rows = []
    for fs in ("pre", "pre+date"):
        X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, fs)
        folds = M.event_folds(ev); draws = subsets(folds)
        log(f"learning curve, {fs}: {len(draws)} fits")
        out = Parallel(n_jobs=jobs)(delayed(run_subset)(k, X, y, ev, ids, sets, folds, S) for k, m, r, S in draws)
        runs = {}
        for (k, m, r, S), (idx, prob, ntr) in zip(draws, out):
            run = runs.setdefault((m, r), dict(prob={md: np.full(len(y), np.nan) for md in LC_MODELS}, ntr=[], nev=[]))
            for md in LC_MODELS:
                run["prob"][md][idx] = prob[md]
            run["ntr"].append(ntr); run["nev"].append(len(S))
        for (m, r), run in runs.items():
            for md in LC_MODELS:
                p = run["prob"][md]; assert not np.isnan(p).any()
                rows.append(dict(feature_set=fs, model=md, m=str(m), draw=r, train_events_mean=np.mean(run["nev"]),
                                 train_dyads_mean=np.mean(run["ntr"]), auc=M.auc(p, y),
                                 auc_se=np.sqrt(M.auc_clustered_var(p, y, ev)), seed=f"[{SUBSET_SEED}, fold, m, r]",
                                 fold_scheme="event (Part 1 folds), training events subsampled"))
                if m == "all":
                    assert abs(rows[-1]["auc"] - stored[(fs, md)]) < 1e-9, f"{fs} {md}: {rows[-1]['auc']} vs {stored[(fs, md)]}"
        log(f"  full-size runs reproduce Part 1 for {fs}")
    d = pd.DataFrame(rows); d.to_csv(RES / "limits_learning.csv", index=False)
    return d


def extrapolate(points, c):
    """Least-squares fit of AUC = A_inf - b * n^(-c) to (n, AUC) points; returns A_inf and b."""
    n, a = np.asarray(points, float).T
    Z = np.column_stack([np.ones_like(n), -n ** (-c)])
    (ainf, b), *_ = np.linalg.lstsq(Z, a, rcond=None)
    return ainf, b


def learning_summary(d):
    s = (d.groupby(["feature_set", "model", "m"])
         .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"), draws=("auc", "size"), train_events=("train_events_mean", "mean"),
              train_dyads=("train_dyads_mean", "mean"), auc_se_mean=("auc_se", "mean")).reset_index())
    s["gain_to_all"] = np.nan
    ext = []
    for (fs, md), g in s.groupby(["feature_set", "model"]):
        full = g[g.m == "all"].auc_mean.iloc[0]
        s.loc[g.index, "gain_to_all"] = full - g.auc_mean
        pts = g[["train_dyads", "auc_mean"]].to_numpy()
        for c in (0.5, 1.0):
            ainf, b = extrapolate(pts, c)
            ext.append(dict(feature_set=fs, model=md, exponent=c, auc_limit=ainf, b=b, auc_all=full, role="exploratory"))
    return s, pd.DataFrame(ext)


# ---------------------------------------------------------------- 2. person ceiling

def dyad_folds(dyad, event, k=5, seed=DYAD_SEED):
    """Deal each event's dyads at random into k folds; returns {dyad: fold}."""
    rng = np.random.default_rng(seed); out = {}
    t = pd.DataFrame({"dyad": dyad, "event": event}).drop_duplicates("dyad").sort_values(["event", "dyad"])
    for _, g in t.groupby("event"):
        ids = g.dyad.to_numpy()
        out.update(zip(rng.permutation(ids), np.arange(len(ids)) % k))
    return out


def rating_block(c, train):
    """The eight in-date ratings, standardized with the training rows' medians and SDs, missing set to the median
    (0 after centring) with one flag per rating that is ever missing."""
    R = c[RATINGS].to_numpy(float); cols = []
    for j, name in enumerate(RATINGS):
        x = R[:, j]; tr = x[train]
        med, sd = np.nanmedian(tr), np.nanstd(tr)
        miss = np.isnan(x)
        cols.append(np.where(miss, 0.0, (x - med) / sd))
        if miss.any():
            cols.append(miss.astype(float))
    return np.column_stack(cols)


def person_scores(c, folds, variant):
    """Held-out p(yes) for every decision, from the decisions of the other dyads (variant: 'person', 'person+date',
    'date')."""
    f = np.array([folds[d] for d in c.dyad]); y = c.dec.to_numpy(float)
    I = indicator_design(c.rater.to_numpy(), c.ratee.to_numpy())
    pred = np.empty(len(y))
    for j in range(5):
        te = f == j
        parts = [] if variant == "date" else [I]
        if variant != "person":
            parts.append(rating_block(c, ~te))
        X = np.hstack(parts)
        w, _ = M.fit_logit(X[~te], y[~te], ridge=W1_RIDGE, iters=W1_ITERS)
        pred[te] = M.predict_logit(w, X[te])
    return pred


def match_scores(c, pred):
    """Per dyad: p(her yes) x p(his yes), with the match and the event."""
    t = c[["dyad", "event", "gender", "match"]].assign(p=pred)
    w = t.pivot_table(index=["dyad", "event", "match"], columns="gender", values="p").reset_index()
    return pd.DataFrame({"dyad": w.dyad, "event": w.event, "match": w["match"].astype(int), "score": w[0] * w[1]})


def person(clean):
    c = clean[["rater", "ratee", "event", "dyad", "gender", "dec", "match"] + RATINGS].reset_index(drop=True)
    folds = dyad_folds(c.dyad.to_numpy(), c.event.to_numpy())
    scores = {}
    for v in ("person", "person+date", "date"):
        log(f"person ceiling: {v}")
        s = match_scores(c, person_scores(c, folds, v)); scores[v] = s.set_index("dyad")
    base = scores["person"]
    rows = []
    for v, s in scores.items():
        s = s.loc[base.index]; y, g = s.match.to_numpy(), s.event.to_numpy()
        rows.append(dict(model=v, dyads=len(s), matches=int(y.sum()), auc=M.auc(s.score, y),
                         auc_se=np.sqrt(M.auc_clustered_var(s.score, y, g)), seed=DYAD_SEED,
                         fold_scheme="dyads within events, 5 random folds"))
    for a, b in (("person+date", "person"), ("person", "date"), ("person+date", "date")):
        sa, sb = scores[a].loc[base.index], scores[b].loc[base.index]
        y, g = base.match.to_numpy(), base.event.to_numpy()
        rows.append(dict(model=f"{a} minus {b}", dyads=len(base), matches=int(y.sum()), auc=M.auc(sa.score, y) - M.auc(sb.score, y),
                         auc_se=np.sqrt(M.auc_diff_clustered_var(sa.score, sb.score, y, g)), seed=DYAD_SEED,
                         fold_scheme="dyads within events, 5 random folds"))
    d = pd.DataFrame(rows); d.to_csv(RES / "limits_person.csv", index=False)
    pd.concat({v: s for v, s in scores.items()}, names=["model"]).reset_index().to_csv(RES / "limits_person_scores.csv", index=False)
    return d


# ---------------------------------------------------------------- 3. questionnaire

def person_effects(c, rows=None):
    """W1 on the given decisions: each person's yes-rate a (rater term) and appeal b (ratee term)."""
    c = c if rows is None else c[rows]
    rater, ratee = c.rater.to_numpy(), c.ratee.to_numpy()
    ri, rl = pd.factorize(rater); ci, cl = pd.factorize(ratee)
    w, _ = M.fit_logit(indicator_design(rater, ratee), c.dec.to_numpy(float), ridge=W1_RIDGE, iters=W1_ITERS)
    return pd.Series(w[1:1 + len(rl)], index=rl), pd.Series(w[1 + len(rl):], index=cl)


def spearman_brown(r):
    return 2 * r / (1 + r)


def reliability(c, halves=HALVES, seed=HALF_SEED):
    """Split-half reliability of the yes-rate and appeal estimates (Spearman-Brown, averaged over random halvings
    of each event's dyads)."""
    out = {"yes_rate": [], "appeal": []}
    for s in range(halves):
        h = dyad_folds(c.dyad.to_numpy(), c.event.to_numpy(), k=2, seed=[seed, s])
        f = np.array([h[d] for d in c.dyad])
        (a0, b0), (a1, b1) = person_effects(c, f == 0), person_effects(c, f == 1)
        for key, (u, v) in (("yes_rate", (a0, a1)), ("appeal", (b0, b1))):
            common = u.index.intersection(v.index)
            out[key].append(spearman_brown(np.corrcoef(u[common], v[common])[0, 1]))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in out.items()}


def person_design(train, test):
    """Pre-date answers as a numeric matrix: median imputation with flags, one-hot categories (codes held by fewer
    than 10 training people merged), gender; standardized with the training people's means and SDs."""
    def frame(d):
        return d[NUMERIC + CATEGORICAL + ["gender"]].copy()
    tr, te = frame(train), frame(test)
    cols_tr, cols_te = [], []
    for v in NUMERIC:
        med = tr[v].median()
        cols_tr.append(tr[v].fillna(med)); cols_te.append(te[v].fillna(med))
        if tr[v].isna().any():
            cols_tr.append(tr[v].isna().astype(float).rename(v + "_missing")); cols_te.append(te[v].isna().astype(float).rename(v + "_missing"))
    for v in CATEGORICAL:
        counts = tr[v].value_counts(); keep = counts[counts >= 10].index
        for code in keep:
            cols_tr.append((tr[v] == code).astype(float).rename(f"{v}_{code}")); cols_te.append((te[v] == code).astype(float).rename(f"{v}_{code}"))
    cols_tr.append(tr.gender.astype(float)); cols_te.append(te.gender.astype(float))
    A, B = pd.concat(cols_tr, axis=1).to_numpy(float), pd.concat(cols_te, axis=1).to_numpy(float)
    mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
    return (A - mu) / sd, (B - mu) / sd


def tune(people, y, ev, k, make, grid):
    """Grid value with the smallest pooled inner-fold squared error (event-grouped 4-fold, Part 1's inner seed)."""
    inner = M.event_folds(ev, k=4, seed=M.FOLD_SEED + 1 + k); f = np.array([inner[e] for e in ev])
    err = {g: 0.0 for g in grid}
    for j in range(4):
        A, B = person_design(people[f != j], people[f == j])
        for g in grid:
            err[g] += np.sum((make(g).fit(A, y[f != j]).predict(B) - y[f == j]) ** 2)
    return min(grid, key=lambda g: err[g])


def questionnaire(clean, participants):
    c = clean[["rater", "ratee", "event", "dyad", "dec"]].reset_index(drop=True)
    a, b = person_effects(c)
    people = participants.set_index("iid").loc[a.index].rename_axis("iid").reset_index()
    ev = people.wave.to_numpy()
    assert (pd.Series(clean.drop_duplicates("rater").set_index("rater").event).loc[people.iid].to_numpy() == ev).all()
    targets = {"yes_rate": a.loc[people.iid].to_numpy(), "appeal": b.loc[people.iid].to_numpy()}
    rel = reliability(c); log(f"reliability: {rel}")
    folds = M.event_folds(ev); f = np.array([folds[e] for e in ev])
    learners = {"ridge": (lambda al: Ridge(alpha=al), RIDGE_ALPHAS),
                "forest": (None, RF_LEAVES)}
    rows = []
    for tname, y in targets.items():
        for lname, (make, grid) in learners.items():
            pred, chosen = np.empty(len(y)), []
            for k in range(5):
                tr, te = f != k, f == k
                mk = make if make is not None else (lambda leaf, k=k: RandomForestRegressor(n_estimators=500, max_features=1 / 3,
                                                                                        min_samples_leaf=leaf, random_state=RF_SEED + k, n_jobs=1))
                g = tune(people[tr].reset_index(drop=True), y[tr], ev[tr], k, mk, grid); chosen.append(g)
                A, B = person_design(people[tr], people[te])
                pred[te] = mk(g).fit(A, y[tr]).predict(B)
            r = M.r_squared_and_rmse(y, pred, ev); rel_t, rel_sd = rel[tname]
            rows.append(dict(target=tname, learner=lname, people=len(y), r_squared=r["r_squared"], r_squared_se=r["r_squared_se"],
                             reliability=rel_t, reliability_sd=rel_sd, r_squared_corrected=r["r_squared"] / rel_t,
                             r_squared_corrected_se=r["r_squared_se"] / rel_t, tuning=";".join(map(str, chosen)),
                             fold_scheme="event (Part 1 folds), inner event-grouped 4-fold",
                             seed=f"folds {M.FOLD_SEED}; halves [{HALF_SEED}, split]; forest {RF_SEED} + fold"))
            log(f"  {tname} {lname}: R2 {r['r_squared']:.3f} (SE {r['r_squared_se']:.3f}), corrected {r['r_squared'] / rel_t:.3f}")
    d = pd.DataFrame(rows); d.to_csv(RES / "limits_questionnaire.csv", index=False)
    return d


# ---------------------------------------------------------------- hypotheses

def hypotheses(lc_summary, pers, quest, best_pre=None):
    rows = []
    if lc_summary is not None:
        g = lc_summary.set_index(["feature_set", "model", "m"])
        gain = g.loc[("pre", "lasso", "all"), "auc_mean"] - g.loc[("pre", "lasso", "8"), "auc_mean"]
        rows.append(dict(hypothesis="L1", statistic=f"pre-date lasso: AUC with all training events "
                         f"{g.loc[('pre', 'lasso', 'all'), 'auc_mean']:.4f} minus mean over 20 draws of 8 events "
                         f"{g.loc[('pre', 'lasso', '8'), 'auc_mean']:.4f} = {gain:+.4f} (SD over draws "
                         f"{g.loc[('pre', 'lasso', '8'), 'auc_sd']:.4f})", supported=bool(gain < 0.03)))
    if pers is not None:
        p = pers.set_index("model")
        diff = p.loc["person", "auc"] - best_pre
        rows.append(dict(hypothesis="P1", statistic=f"within-event person-effects match AUC {p.loc['person', 'auc']:.4f} "
                         f"(SE {p.loc['person', 'auc_se']:.4f}) minus the best pre-date AUC on new events {best_pre:.4f} = {diff:+.4f}",
                         supported=bool(diff >= 0.10)))
        d, se = p.loc["person+date minus person", "auc"], p.loc["person+date minus person", "auc_se"]
        rows.append(dict(hypothesis="P2", statistic=f"adding the in-date ratings to the person effects: {d:+.4f} (SE {se:.4f}, "
                         f"90% interval {d - 1.645 * se:+.4f} to {d + 1.645 * se:+.4f})",
                         supported=bool(d >= 0.02 and d - 1.645 * se > 0)))
    if quest is not None:
        q = quest.set_index(["target", "learner"])
        for h, t in (("Q1", "appeal"), ("Q2", "yes_rate")):
            vals = [q.loc[(t, l), "r_squared_corrected"] for l in ("ridge", "forest")]
            rows.append(dict(hypothesis=h, statistic=f"{t}: corrected held-out R2, ridge {vals[0]:.3f}, forest {vals[1]:.3f} "
                             f"(reliability {q.loc[(t, 'ridge'), 'reliability']:.3f})", supported=bool(max(vals) < 0.30)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures (exploratory, from the saved tables)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def plots():
    """results/limits_learning.png: the lasso's held-out AUC against training dyads, before the date and with the
    date ratings (mean over draws, band = SD over draws), with the extrapolated limits for c = 1 and 0.5.
    results/limits_ladder.png: every Part 10 AUC on one scale, with 90% intervals (AUC +/- t(20) x SE)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    t20 = stats.t.ppf(0.95, 20)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": REF, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
    s = pd.read_csv(RES / "limits_learning_summary.csv", dtype={"m": str})
    ext = pd.read_csv(RES / "limits_learning_extrapolation.csv").set_index(["feature_set", "model", "exponent"]).auc_limit
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for fs, col, lab in (("pre", BLUE, "Before the date (pre)"), ("pre+date", ORANGE, "With the date ratings (pre+date)")):
        g = s[(s.feature_set == fs) & (s.model == "lasso")].sort_values("train_dyads")
        sd = g.auc_sd.fillna(0)
        ax.fill_between(g.train_dyads, g.auc_mean - sd, g.auc_mean + sd, color=col, alpha=0.15, linewidth=0)
        ax.plot(g.train_dyads, g.auc_mean, "o-", color=col, ms=4.5, linewidth=2, label=lab)
        lo, hi = ext[(fs, "lasso", 1.0)], ext[(fs, "lasso", 0.5)]
        ax.plot([3700, 3700], [lo, hi], color=col, linewidth=5, solid_capstyle="butt", alpha=0.6)
        ax.text(3780, (lo + hi) / 2, f"unlimited data:\n{lo:.2f} to {hi:.2f}", color=INK, fontsize=7.5, va="center")
    ax.axhline(0.5, color=REF, linestyle="--", linewidth=1); ax.text(300, 0.505, "chance", color=INK2, fontsize=8)
    ax.set_xlim(0, 4500); ax.set_ylim(0.48, 0.92)
    ax.set_xlabel("Training pairs (4 to 17 events, held-out events fixed)"); ax.set_ylabel("Held-out AUC (lasso)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5))
    fig.tight_layout(); fig.savefig(RES / "limits_learning.png", dpi=300, facecolor="white"); plt.close(fig)

    pers = pd.read_csv(RES / "limits_person.csv").set_index("model")
    p1 = pd.read_csv(RES / "part1_models_metrics.csv")
    p1 = p1[(p1.headline == True) & (p1.fold_scheme == "event")].set_index(["feature_set", "model"])
    rows = [("Their answers, new events (lasso)", p1.loc[("pre", "lasso"), "auc"], p1.loc[("pre", "lasso"), "auc_se"], BLUE),
            ("Their answers, unlimited events (extrapolated)", (ext[("pre", "lasso", 1.0)], ext[("pre", "lasso", 0.5)]), None, BLUE),
            ("Appeal and yes-rate from their other dates", pers.loc["person", "auc"], pers.loc["person", "auc_se"], AQUA),
            ("This date's ratings", pers.loc["date", "auc"], pers.loc["date", "auc_se"], ORANGE),
            ("Appeal, yes-rate and this date's ratings", pers.loc["person+date", "auc"], pers.loc["person+date", "auc_se"], ORANGE),
            ("Answers and date ratings, new events (lasso)", p1.loc[("pre+date", "lasso"), "auc"], p1.loc[("pre+date", "lasso"), "auc_se"], ORANGE)]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, (lab, a, se, col) in enumerate(rows):
        y = len(rows) - 1 - i
        if se is None:
            ax.plot(a, [y, y], color=col, linewidth=6, solid_capstyle="butt", alpha=0.6)
            ax.text(a[1] + 0.012, y, f"{a[0]:.2f} to {a[1]:.2f}", va="center", fontsize=8, color=INK)
        else:
            ax.errorbar(a, y, xerr=t20 * se, fmt="o", color=col, ms=6, elinewidth=1.6, capsize=0)
            ax.text(a + t20 * se + 0.012, y, f"{a:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=REF, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0.45, 1.0); ax.set_xlabel("Match AUC, with 90% interval (0.5 is chance)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(RES / "limits_ladder.png", dpi=300, facecolor="white"); plt.close(fig)
    print("wrote results/limits_learning.png and results/limits_ladder.png")


def main():
    ap = argparse.ArgumentParser()
    for a in ("learning", "person", "questionnaire", "plot"):
        ap.add_argument(f"--{a}", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.plot:
        return plots()
    every = not (args.learning or args.person or args.questionnaire)
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    stored = pd.read_csv(RES / "part1_models_metrics.csv")
    best_pre = stored[(stored.headline == True) & (stored.fold_scheme == "event") & (stored.feature_set == "pre")].auc.max()
    s = pers = quest = None
    if every or args.learning:
        pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
        s, ext = learning_summary(learning(pairs, sets, args.jobs))
        s.to_csv(RES / "limits_learning_summary.csv", index=False); ext.to_csv(RES / "limits_learning_extrapolation.csv", index=False)
        print(s.to_string()); print(ext.to_string())
    if every or args.person:
        pers = person(clean); print(pers.to_string())
    if every or args.questionnaire:
        quest = questionnaire(clean, pd.read_csv("data/participants.csv")); print(quest.to_string())
    if s is None and (RES / "limits_learning_summary.csv").exists():            # verdicts from saved runs of the other parts
        s = pd.read_csv(RES / "limits_learning_summary.csv", dtype={"m": str})
    if pers is None and (RES / "limits_person.csv").exists():
        pers = pd.read_csv(RES / "limits_person.csv")
    h = hypotheses(s, pers, quest, best_pre)
    path = RES / "limits_hypotheses.csv"
    if path.exists():
        old = pd.read_csv(path); h = pd.concat([old[~old.hypothesis.isin(h.hypothesis)], h]).sort_values("hypothesis")
    h.to_csv(path, index=False); print(h.to_string())


if __name__ == "__main__":
    main()
