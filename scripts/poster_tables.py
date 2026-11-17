"""The two results tables for the final poster and slides: model comparison and the forest's top predictors.

Input : results/part1_models_metrics.csv, results/tree_figure_cv.csv (scripts/tree_figure.py),
        results/rf_importance.csv (scripts/rf_importance.py)
Output: results/poster_tables_models.png / .csv, results/poster_tables_predictors.png / .csv
Seeds : none here (the inputs record theirs).

Model table: held-out AUC (event 5-fold CV, 4,184 dyads, headline specification) with event-clustered SEs, before
the date (`pre`) and with the in-date ratings (`pre+date`); the single tree is the depth-3 tree drawn by
tree_figure.py (depth fixed for readability, not tuned). Predictor table: the random forest's top predictors by the
drop in held-out AUC when each is shuffled, for both feature sets, plus the grouped shuffles.
Run from the repository root: python scripts/poster_tables.py
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")
INK, INK2, GRID, HL = "#0b0b0b", "#52514e", "#e4e3df", "#e8f0fb"
MODELS = [("logistic", "Logistic regression (with splines)"), ("stepwise_forward", "Stepwise logistic (forward)"),
          ("lda", "LDA"), ("qda", "QDA (regularized)"), ("lasso", "Lasso (L1 logistic)"),
          ("tree", "Single decision tree (depth 3)"), ("rf", "Random forest (500 trees)")]


def model_table():
    met = pd.read_csv(RES / "part1_models_metrics.csv")
    met = met[(met.spec == M.HEADLINE.label) & (met.dyads == 4184) & met.feature_set.isin(["pre", "pre+date"])]
    tree = pd.read_csv(RES / "tree_figure_cv.csv").query("depth == 3").assign(model="tree")
    both = pd.concat([met[["model", "feature_set", "auc", "auc_se"]], tree[["model", "feature_set", "auc", "auc_se"]]])
    t = both.pivot(index="model", columns="feature_set", values=["auc", "auc_se"])
    rows = [dict(model=lab, auc_pre=t.loc[m, ("auc", "pre")], se_pre=t.loc[m, ("auc_se", "pre")],
                 auc_pre_date=t.loc[m, ("auc", "pre+date")], se_pre_date=t.loc[m, ("auc_se", "pre+date")])
            for m, lab in MODELS]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "poster_tables_models.csv", index=False)
    cells = [[r.model, f"{r.auc_pre:.3f} ({r.se_pre:.3f})", f"{r.auc_pre_date:.3f} ({r.se_pre_date:.3f})"]
             for r in df.itertuples()]
    draw(cells, ["Model", "Before\nthe date", "With in-date\nratings"], "poster_tables_models",
         "Held-out AUC (standard error) of each model",
         "Event 5-fold cross-validation: whole events held out, 4,184 pairs. 0.5 = chance, 1 = perfect. The forest's\n"
         "lead over the next models is within the error bars. Source: scripts/poster_tables.py.",
         highlight=len(cells) - 1, widths=(0.56, 0.22, 0.22))
    return df


def predictor_table(top=8):
    imp = pd.read_csv(RES / "rf_importance.csv")
    out, cells = [], []
    for fs, head in (("pre+date", "With in-date ratings"), ("pre", "Before the date (sign-up answers only)")):
        d = imp[(imp.feature_set == fs) & (imp.kind != "group")].sort_values("auc_drop", ascending=False).head(top)
        out.append(d.assign(table_rank=range(1, len(d) + 1)))
        a0 = d.forest_auc.iloc[0]
        cells.append([f"{head}: forest AUC {a0:.3f}", "", ""])
        cells += [[f"{i}. {r.label}", f"{r.auc_drop:.3f}", f"{r.auc_drop_se:.3f}"] for i, r in enumerate(d.itertuples(), 1)]
    pd.concat(out).to_csv(RES / "poster_tables_predictors.csv", index=False)
    heads = [i for i, c in enumerate(cells) if c[1] == ""]
    g = imp[imp.kind == "group"].set_index(["feature_set", "predictor"]).auc_drop
    draw(cells, ["Predictor (ranked)", "AUC drop", "SE"], "poster_tables_predictors",
         "What the random forest relies on most",
         "AUC drop: how much held-out AUC falls when that predictor is shuffled (mean of 30 shuffles; event 5-fold\n"
         "cross-validation, 4,184 pairs; event-clustered SE). Her rating = the woman's view of the man, and vice versa.\n"
         f"Related ratings share credit: shuffling all in-date ratings at once drops AUC by "
         f"{g['pre+date', 'GROUP: all in-date ratings']:.2f}, all sign-up answers\n"
         f"by {g['pre+date', 'GROUP: all sign-up answers']:.3f}. Before the date every drop is small and uncertain. "
         "Source: scripts/poster_tables.py.", section_rows=heads, widths=(0.70, 0.15, 0.15))


def draw(cells, header, name, title, note, highlight=None, section_rows=(), widths=(0.5, 0.25, 0.25)):
    n = len(cells)
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * (n + 1) + 1.2))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-(n + 0.3 + 0.45 * (note.count(chr(10)) + 1)), 1.45)
    x = np.cumsum((0,) + tuple(widths))
    ax.text(0, 1.3, title, fontsize=12, fontweight="bold", color=INK, va="bottom")
    for j, h in enumerate(header):
        ax.text(x[j] + (0.01 if j == 0 else widths[j] / 2), 0.42, h, fontsize=9.5, fontweight="bold", color=INK2,
                ha="left" if j == 0 else "center", va="center", linespacing=1.1)
    ax.plot([0, 1], [-0.1, -0.1], color=INK, linewidth=1)
    for i, row in enumerate(cells):
        yy = -i - 0.55
        if i == highlight:
            ax.add_patch(plt.Rectangle((0, yy - 0.45), 1, 0.9, color=HL, linewidth=0, zorder=0))
        if i in section_rows:
            ax.text(0.01, yy, row[0], fontsize=9.5, fontweight="bold", color=INK, va="center")
            continue
        for j, c in enumerate(row):
            ax.text(x[j] + (0.01 if j == 0 else widths[j] / 2), yy, c, fontsize=9.5, color=INK, va="center",
                    ha="left" if j == 0 else "center", fontweight="bold" if i == highlight else "normal")
        ax.plot([0, 1], [yy - 0.5, yy - 0.5], color=GRID, linewidth=0.6)
    ax.plot([0, 1], [-n - 0.05, -n - 0.05], color=INK, linewidth=1)
    ax.text(0, -n - 0.25, note, fontsize=7.5, color=INK2, va="top")
    fig.savefig(RES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print(model_table().to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if (RES / "rf_importance.csv").exists():
        predictor_table()
        print(pd.read_csv(RES / "poster_tables_predictors.csv")[["feature_set", "table_rank", "label", "auc_drop", "auc_drop_se"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("wrote results/poster_tables_models.png/.csv and results/poster_tables_predictors.png/.csv")


if __name__ == "__main__":
    main()
