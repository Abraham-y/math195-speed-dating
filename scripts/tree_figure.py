"""One classification tree, drawn for the poster and slides, with its held-out accuracy next to the forest's.

Input : data/pairs.csv, data/feature_sets.json, results/part1_models_metrics.csv (the forest's AUC, for comparison)
Output: results/tree_figure_pre_date.png and results/tree_figure_pre.png (the depth-3 tree fitted to all 4,184 dyads;
        *_poster.png versions without title or notes),
        results/tree_figure_cv.csv (held-out AUC of single trees of depth 1 to 8, event 5-fold CV)
Seeds : event folds 20261006 (modeling.FOLD_SEED); tree random_state 20261117.

The random forest averages 500 deep trees, so there is no single "best tree" to draw. This script fits one
classification tree (Gini splits, ISL section 8.1.2) of depth 3, small enough to read, to all dyads, on the raw
answers (ratings on their 1-10 scales; categorical codes one-hot), so the split points are in the survey's own units.
Missing answers go down the branch the tree learned for them in training (scikit-learn's native handling).
The depth was fixed at 3 for readability before looking at the results; the CV table shows how single trees of
each depth do on held-out events, which is the fair comparison with the forest.
Run from the repository root: python scripts/tree_figure.py
"""
import sys, json, pathlib, textwrap
import numpy as np, pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, "scripts")
import modeling as M
from rf_importance import label as base_label

RES = pathlib.Path("results")
TREE_SEED, DEPTH, MIN_LEAF = 20261117, 3, 30
T20 = stats.t.ppf(0.95, 20)
INK, INK2, REF = "#0b0b0b", "#52514e", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9cc0ee", "#2a78d6", "#123f7a"])
FIELD = {1: "law", 2: "math", 3: "social science or psychology", 4: "medical science", 5: "engineering",
         6: "English or journalism", 7: "history, religion or philosophy", 8: "business or economics",
         9: "education", 10: "natural sciences", 11: "social work", 12: "undecided", 13: "political science",
         14: "film", 15: "fine arts", 16: "languages", 17: "architecture", 18: "other"}
CAREER = {1: "lawyer", 2: "academic or research", 3: "psychologist", 4: "doctor or medicine", 5: "engineer",
          6: "creative arts", 7: "business or finance", 8: "real estate", 9: "international affairs",
          10: "undecided", 11: "social work", 12: "speech pathology", 13: "politics", 14: "pro sports",
          15: "other", 16: "journalism", 17: "architecture"}
FREQ = {1: "several times a week", 2: "twice a week", 3: "once a week", 4: "twice a month", 5: "once a month",
        6: "several times a year", 7: "almost never"}
GOAL = {1: "a fun night out", 2: "to meet new people", 3: "to get a date", 4: "a serious relationship",
        5: "to say I did it", 6: "other"}


def design(pairs, sets, feature_set):
    """Raw columns of the feature set (the same predictors as the models), categoricals one-hot."""
    X, y, ev, _ = M.build_frame(pairs, sets, M.HEADLINE, feature_set)
    raw = pairs.loc[:, list(X.columns)].reset_index(drop=True)
    cats = [c for c in raw.columns if c in sets["categorical"]]
    D = pd.get_dummies(raw, columns=cats, prefix_sep="=", dtype=float)
    return D, y, ev


def name(col):
    if "=" in col:
        v, code = col.split("="); code = int(float(code))
        side, rest = v.split("_", 1)
        her, she = ("Her", "she") if side == "f" else ("His", "he")
        if rest == "field_cd":
            return f"{her} field is {FIELD.get(code, code)}"
        if rest == "career_c":
            return f"{her} career is {CAREER.get(code, code)}"
        return f"{her} goal is {GOAL.get(code, code)}"
    return base_label(col)


def edge_words(col, t):
    """Branch labels in the survey's units. Dating and going-out frequency are codes 1 (several times a week) to
    7 (almost never), so a lower code means more often."""
    if col.endswith(("_date", "_go_out")):
        return f"{FREQ[int(t)]} or more often", f"{FREQ[int(t) + 1]} or less often"
    return f"≤ {fmt(t)}", f"> {fmt(t)}"


def fmt(t):
    """Split points: ratings split at quarter points, shown exactly; preference weights to one decimal."""
    return f"{t:g}" if abs(4 * t - round(4 * t)) < 1e-9 else f"{t:.1f}"


def question(col, t):
    """Poster node text: the split as a yes/no question (yes = the right branch)."""
    if col.endswith(("_date", "_go_out")):
        return f"{name(col)}: {FREQ[int(t) + 1]} or less often?"
    if "=" in col or col.startswith(("same_", "f_pre_", "m_pre_", "f_date_m", "m_date_m")):
        return f"{name(col)}?"
    return f"{name(col)} > {fmt(t)}?"


def held_out(D, y, ev, depth):
    folds = M.event_folds(ev, seed=M.FOLD_SEED)
    f = np.array([folds[e] for e in ev]); p = np.empty(len(y))
    for k in range(5):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=TREE_SEED)
        p[f == k] = t.fit(D[f != k], y[f != k]).predict_proba(D[f == k])[:, 1]
    return M.auc(p, y), np.sqrt(M.auc_clustered_var(p, y, ev))


def draw(tree, cols, y, feature_set, cv_auc, rf_auc, poster=False):
    """poster=True: no title or notes (the poster caption carries them), wrapped labels and larger relative text."""
    T = tree.tree_
    leaves, pos = [], {}
    def walk(n, d):
        if T.children_left[n] == -1:
            pos[n] = (len(leaves), d); leaves.append(n); return pos[n][0]
        a, b = walk(T.children_left[n], d + 1), walk(T.children_right[n], d + 1)
        pos[n] = ((a + b) / 2, d); return pos[n][0]
    walk(0, 0)
    nl, depth = len(leaves), max(d for _, d in pos.values())
    fs_node, fs_leaf, fs_edge = (10, 10, 9.5) if poster else (8.5, 9, 8.5)
    if poster:
        fig, ax = plt.subplots(figsize=(1.05 * nl + 0.2, 1.1 * depth + 0.9))
        ax.set_xlim(-0.55, nl - 0.45); ax.set_ylim(-(depth + 0.38), 0.3)
    else:
        fig, ax = plt.subplots(figsize=(1.75 * nl + 0.6, 1.35 * depth + 1.6))
        ax.set_xlim(-0.6, nl - 0.4); ax.set_ylim(-(depth + 0.4), 0.75)
    ax.axis("off")
    base = y.mean()
    for n, (x, d) in pos.items():
        rate, cnt = T.value[n][0][1], int(T.n_node_samples[n])
        if T.children_left[n] != -1:
            for child, word in zip((T.children_left[n], T.children_right[n]), edge_words(cols[T.feature[n]], T.threshold[n])):
                cx, cd = pos[child]
                ax.plot([x, cx], [-d - 0.2, -cd + 0.2], color=REF, linewidth=1.2, zorder=1)
                c = cols[T.feature[n]]
                if poster or "=" in c or c.startswith(("same_", "f_pre_", "m_pre_", "f_date_m", "m_date_m")):
                    word = "no" if child == T.children_left[n] else "yes"
                ax.text((x + cx) / 2, (-d - 0.2 - cd + 0.2) / 2, word, ha="center", va="center", fontsize=fs_edge,
                        color=INK2, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=3)
            text = name(cols[T.feature[n]])
            text = textwrap.fill(question(cols[T.feature[n]], T.threshold[n]), 22 if d < depth - 1 else 18) if poster else f"{text}\n{cnt:,} pairs, {rate:.0%} match"
            ax.text(x, -d, text, ha="center", va="center", fontsize=fs_node,
                    color=INK, zorder=2, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=REF, lw=1))
        else:
            fc = SEQ(min(rate / 0.6, 1.0))
            tc = "white" if rate > 0.3 else INK
            ax.text(x, -d, f"{rate:.0%}\n{cnt:,} pairs" if poster else f"{rate:.0%}\nmatch\n{cnt:,} pairs", ha="center",
                    va="center", fontsize=fs_leaf, color=tc,
                    fontweight="bold", zorder=2, bbox=dict(boxstyle=f"round,pad={0.28 if poster else 0.4}", fc=fc, ec="none"))
    fn = {"pre+date": "pre_date", "pre": "pre"}[feature_set]
    if poster:
        fig.savefig(RES / f"tree_figure_{fn}_poster.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig); return
    title = {"pre+date": "One decision tree, using the in-date ratings",
             "pre": "One decision tree, using only sign-up answers"}[feature_set]
    ax.text(-0.6, 0.72, title, fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(-0.6, 0.38, f"Depth 3, fitted to all 4,184 pairs (overall match rate {base:.1%}). Held-out AUC of this kind of "
            f"tree: {cv_auc[0]:.3f} (SE {cv_auc[1]:.3f});\nthe random forest, which averages 500 deeper trees: {rf_auc:.3f}.",
            fontsize=8.5, color=INK2, va="top")
    units = {"pre+date": "Ratings and liking are on 1-10 scales.",
             "pre": "Interests and self-ratings are on 1-10 scales; stated weights are points out of 100."}[feature_set]
    fig.text(0.01, 0.02, f"{units} A missing answer follows the branch learned for it in training. "
             "Source: scripts/tree_figure.py.", fontsize=7.5, color=INK2, va="top")
    fig.savefig(RES / f"tree_figure_{fn}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    met = pd.read_csv(RES / "part1_models_metrics.csv")
    rows = []
    for fs in ("pre+date", "pre"):
        D, y, ev = design(pairs, sets, fs)
        for depth in range(1, 9):
            a, se = held_out(D, y, ev, depth)
            rows.append(dict(feature_set=fs, depth=depth, min_samples_leaf=MIN_LEAF, auc=a, auc_se=se, dyads=len(y),
                             fold_scheme="event", fold_seed=M.FOLD_SEED, tree_seed=TREE_SEED))
        cv = next((r["auc"], r["auc_se"]) for r in rows if r["feature_set"] == fs and r["depth"] == DEPTH)
        rf = met[(met.spec == M.HEADLINE.label) & (met.feature_set == fs) & (met.model == "rf") & (met.dyads == len(y))].auc.iloc[0]
        tree = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=TREE_SEED).fit(D, y)
        draw(tree, list(D.columns), y, fs, cv, rf)
        draw(tree, list(D.columns), y, fs, cv, rf, poster=True)
        T = tree.tree_
        print(f"{fs}: depth-{DEPTH} tree, held-out AUC {cv[0]:.4f} (SE {cv[1]:.4f}); forest {rf:.4f}")
        for n in range(T.node_count):
            if T.children_left[n] != -1:
                print(f"  node {n}: {name(D.columns[T.feature[n]])} <= {T.threshold[n]:g}  ({T.n_node_samples[n]} pairs, "
                      f"{T.value[n][0][1]:.3f} match; missing go {'left' if T.missing_go_to_left[n] else 'right'})")
            else:
                print(f"  leaf {n}: {T.n_node_samples[n]} pairs, {T.value[n][0][1]:.3f} match")
    cvt = pd.DataFrame(rows); cvt.to_csv(RES / "tree_figure_cv.csv", index=False)
    print(cvt[["feature_set", "depth", "auc", "auc_se"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("wrote results/tree_figure_pre_date.png, results/tree_figure_pre.png, results/tree_figure_cv.csv")


if __name__ == "__main__":
    main()
