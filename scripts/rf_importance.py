"""Which predictors does the random forest (the best Part 1 model) rely on? Held-out variable importance.

Input : data/pairs.csv, data/feature_sets.json, results/part1_models_folds.csv (the leaf size each fold chose),
        results/part1_models_metrics.csv (the refitted forests must reproduce its AUC)
Output: results/rf_importance.csv (one row per predictor or group of predictors and feature set),
        results/rf_importance_pre_date.png and results/rf_importance_pre.png (top predictors, for the poster and slides)
Seeds : event folds 20261006 (modeling.FOLD_SEED); forest random_state 20261106 + fold (part1_models.py);
        shuffles 20261116.

The forests of scripts/part1_models.py (headline specification, `pre` and `pre+date`) are refitted fold by fold
with the leaf size each fold chose there, and the pooled held-out AUC is checked against the reported one.
Permutation importance (ISL section 8.2.2 describes the Gini version): shuffle one predictor's columns (all one-hot
columns of a categorical together) among the held-out dyads of each fold, predict again, and record the drop in
pooled held-out AUC. The drop is averaged over --repeats shuffles. Its standard error is event-clustered
(modeling.auc_diff_clustered_var), averaged over the shuffles; this SE has not been checked by simulation, so the
intervals are a guide only. Groups of predictors (all in-date ratings, all pre-date answers, ...) are shuffled
together the same way. Also reported: the mean decrease in Gini impurity, summed over a predictor's columns and
averaged over the five fold forests (computed on training data, so it favors predictors with many distinct values).
Run from the repository root: python scripts/rf_importance.py [--repeats 10] [--jobs 5]
"""
import sys, json, pathlib, argparse, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P

RES = pathlib.Path("results")
SHUFFLE_SEED = 20261116
T20 = stats.t.ppf(0.95, 20)                     # 21 events: 90% intervals use t with 20 degrees of freedom
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
ATTR = dict(attr="attractiveness", sinc="sincerity", intel="intelligence", fun="fun", amb="ambition",
            shar="shared interests")
OWN = {"age": "age", "field_cd": "field of study", "career_c": "intended career", "goal": "goal for the event",
       "exphappy": "expected happiness at the event"}
INTEREST = {"tvsports": "watching sports", "tv": "TV", "exercise": "exercise"}
PAIR = {"abs_age_diff": "Age gap", "int_corr": "Interest similarity", "same_field": "Same field of study",
        "same_goal": "Same goal for the event", "fit_to_f_prefs": "His fit to her stated preferences",
        "fit_to_m_prefs": "Her fit to his stated preferences"}


def label(v):
    """Plain-English name of a predictor. f_ is the woman's answer (for ratings, her view of him), m_ the man's."""
    if v in PAIR:
        return PAIR[v]
    side, rest = v.split("_", 1)
    she, her, him = ("she", "Her", "him") if side == "f" else ("he", "His", "her")
    if rest.startswith("rating_"):
        a = rest[7:]
        return f"How much {she} liked {him}" if a == "like" else f"{her} rating of {him}: {ATTR[a]}"
    if rest == "prob":
        return f"{her} guess that {she}'ll be chosen"
    if rest.startswith("self_"):
        return f"{her} self-rating: {ATTR[rest[5:]]}"
    if rest.startswith("pref_"):
        return f"{her} stated weight on {ATTR[rest[5:]]}"
    if rest in OWN:
        return f"{her} {OWN[rest]}"
    if rest == "date":
        return f"How often {she} dates"
    if rest == "go_out":
        return f"How often {she} goes out"
    if rest == "pre_missing":
        return f"{her} sign-up answers missing"
    if rest == "date_missing":
        return f"{her} in-date ratings missing"
    return f"{her} interest in {INTEREST.get(rest, rest)}"


def units_for(feature_set, X, sets):
    """Each predictor alone, then the groups that are shuffled together."""
    cols = list(X.columns)
    units = {v: [v] for v in cols}
    date = [c for c in cols if c in sets["date"] or c in sets["flags_date"]]
    pre = [c for c in cols if c not in date]
    pair = [c for c in cols if c in sets["pre_pair"]]
    groups = {"GROUP: her sign-up answers": [c for c in pre if c.startswith("f_") and c not in pair],
              "GROUP: his sign-up answers": [c for c in pre if c.startswith("m_") and c not in pair],
              "GROUP: the 6 pair features": pair}
    if feature_set == "pre+date":
        groups = {"GROUP: all in-date ratings": date, "GROUP: all sign-up answers": pre,
                  "GROUP: her in-date ratings of him": [c for c in date if c.startswith("f_")],
                  "GROUP: his in-date ratings of her": [c for c in date if c.startswith("m_")], **groups}
    units.update(groups)
    return units


def run_fold(k, X, y, ev, ids, sets, folds, leaf, units, repeats):
    import warnings; warnings.filterwarnings("ignore")
    test = np.array([folds[e] == k for e in ev]); train = ~test
    Xtr, Xte = X[train].reset_index(drop=True), X[test].reset_index(drop=True)
    pl = P.make_prep(sets, Xtr).fit(Xtr, ids[train].reset_index(drop=True))
    Ltr, Lte = pl.transform(Xtr).to_numpy(), pl.transform(Xte).to_numpy()
    rf = P.make_rf(k)(leaf).fit(Ltr, y[train])
    base = P.predict_serial(rf, Lte)
    blocks = pl.blocks()
    gini = {v: float(rf.feature_importances_[cols].sum()) for v, cols in blocks.items()}
    rng = np.random.default_rng([SHUFFLE_SEED, k])
    perm = {}
    for name, members in units.items():
        cols = [j for v in members for j in blocks.get(v, [])]
        out = np.empty((repeats, len(Lte)))
        for r in range(repeats):
            L = Lte.copy()
            L[:, cols] = Lte[rng.permutation(len(Lte))][:, cols]
            out[r] = P.predict_serial(rf, L)
        perm[name] = out
    return dict(test_idx=np.where(test)[0], base=base, perm=perm, gini=gini)


def importance(pairs, sets, feature_set, repeats, jobs):
    X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, feature_set)
    folds = M.event_folds(ev, seed=M.FOLD_SEED)
    fi = pd.read_csv(RES / "part1_models_folds.csv")
    fi = fi[(fi.spec == M.HEADLINE.label) & (fi.feature_set == feature_set)].set_index("fold")
    leaves = {k: int(fi.loc[k, "rf_min_samples_leaf"]) for k in range(5)}
    units = units_for(feature_set, X, sets)
    res = Parallel(n_jobs=jobs)(delayed(run_fold)(k, X, y, ev, ids, sets, folds, leaves[k], units, repeats)
                                for k in range(5))
    base = np.empty(len(y)); perm = {u: np.empty((repeats, len(y))) for u in units}
    for r in res:
        base[r["test_idx"]] = r["base"]
        for u in units:
            perm[u][:, r["test_idx"]] = r["perm"][u]
    met = pd.read_csv(RES / "part1_models_metrics.csv")
    reported = met[(met.spec == M.HEADLINE.label) & (met.feature_set == feature_set) & (met.model == "rf")
                   & (met.dyads == len(y))].auc.iloc[0]
    a0 = M.auc(base, y)
    assert abs(a0 - reported) < 1e-9, f"refitted forest AUC {a0} differs from the reported {reported}"
    gini = pd.DataFrame([r["gini"] for r in res]).mean()
    rows = []
    for u, members in units.items():
        drops = [a0 - M.auc(perm[u][r], y) for r in range(repeats)]
        var = np.mean([M.auc_diff_clustered_var(base, perm[u][r], y, ev) for r in range(repeats)])
        rows.append(dict(feature_set=feature_set, predictor=u, label=u if u.startswith("GROUP") else label(u),
                         kind="group" if u.startswith("GROUP") else ("in-date rating" if u in sets["date"] + sets["flags_date"] else "sign-up answer"),
                         n_columns=len(members), auc_drop=np.mean(drops), auc_drop_se=np.sqrt(var),
                         auc_drop_min=np.min(drops), auc_drop_max=np.max(drops),
                         gini_importance=float(sum(gini.get(m, 0.0) for m in members)),
                         forest_auc=a0, dyads=len(y), repeats=repeats, fold_seed=M.FOLD_SEED, rf_seed=P.RF_SEED,
                         shuffle_seed=SHUFFLE_SEED, leaves=";".join(str(leaves[k]) for k in range(5))))
    out = pd.DataFrame(rows)
    single = out.kind != "group"
    out.loc[single, "rank"] = out[single].auc_drop.rank(ascending=False, method="first")
    out.loc[single, "gini_rank"] = out[single].gini_importance.rank(ascending=False, method="first")
    return out


def plot(df, feature_set, top):
    d = df[df.kind != "group"].sort_values("auc_drop", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * top + 1.3))
    colors = [BLUE if k == "in-date rating" else ORANGE for k in d.kind]
    yy = np.arange(len(d))
    ax.barh(yy, d.auc_drop, color=colors, height=0.62)
    ax.errorbar(d.auc_drop, yy, xerr=T20 * d.auc_drop_se, fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
    ax.set_yticks(yy, d.label, fontsize=9, color=INK)
    ax.axvline(0, color=REF, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    a0 = df.forest_auc.iloc[0]
    ax.set_xlabel(f"Drop in held-out AUC when the predictor is shuffled (forest AUC {a0:.3f})", color=INK2, fontsize=9)
    title = {"pre+date": "What the random forest relies on, with the in-date ratings",
             "pre": "What the random forest relies on, before the date"}[feature_set]
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    if feature_set == "pre+date":
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=BLUE, label="Rating given during the date"),
                           Patch(color=ORANGE, label="Answer given at sign-up")],
                  loc="lower right", frameon=False, fontsize=8.5)
    fig.text(0.01, -0.02, f"Event 5-fold cross-validation, 4,184 dyads; mean of {int(df.repeats.iloc[0])} shuffles; "
             "bars show 90% intervals from event-clustered SEs.\nSource: results/rf_importance.csv "
             "(scripts/rf_importance.py).", fontsize=7.5, color=INK2, va="top")
    name = {"pre+date": "pre_date", "pre": "pre"}[feature_set]
    fig.savefig(RES / f"rf_importance_{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=5); ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    out = []
    for fs in ("pre+date", "pre"):
        t0 = time.time()
        df = importance(pairs, sets, fs, args.repeats, args.jobs)
        print(f"{fs}: forest AUC {df.forest_auc.iloc[0]:.4f} reproduced; {time.time() - t0:.0f}s", flush=True)
        show = df.sort_values(["kind", "auc_drop"], ascending=[True, False])
        print(show[["label", "kind", "auc_drop", "auc_drop_se", "gini_importance", "gini_rank"]]
              .head(20 + 6).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
        plot(df, fs, args.top)
        out.append(df)
    pd.concat(out).to_csv(RES / "rf_importance.csv", index=False)
    print("wrote results/rf_importance.csv, results/rf_importance_pre_date.png, results/rf_importance_pre.png")


if __name__ == "__main__":
    main()
