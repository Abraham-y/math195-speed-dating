"""The two results tables for the final poster and slides: model comparison and the forest's top predictors.

Input : results/part1_models_metrics.csv, results/tree_figure_cv.csv (scripts/tree_figure.py),
        results/rf_importance.csv (scripts/rf_importance.py)
Output: results/poster_tables_models.png / .csv, results/poster_tables_predictors.png / .csv,
        report/poster/middle_column.tex (the poster's middle column, with both tables as native LaTeX)
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
PRO_CON = {"logistic": ("Interpretable coefficients", "Misses interactions"),
           "stepwise_forward": ("Drops weak predictors", "Unstable predictor choice"),
           "lda": ("Simple and stable", "Straight boundary only"),
           "qda": ("Curved boundary", "Many parameters"),
           "lasso": ("Picks predictors itself", "Still a linear model"),
           "tree": ("Easy to read", "Unstable; overfits"),
           "rf": ("Captures interactions", "Hard to interpret")}


def model_table():
    met = pd.read_csv(RES / "part1_models_metrics.csv")
    met = met[(met.spec == M.HEADLINE.label) & (met.dyads == 4184) & met.feature_set.isin(["pre", "pre+date"])]
    tree = pd.read_csv(RES / "tree_figure_cv.csv").query("depth == 3").assign(model="tree")
    both = pd.concat([met[["model", "feature_set", "auc", "auc_se"]], tree[["model", "feature_set", "auc", "auc_se"]]])
    t = both.pivot(index="model", columns="feature_set", values=["auc", "auc_se"])
    rows = [dict(model=lab, pro=PRO_CON[m][0], con=PRO_CON[m][1], auc_pre=t.loc[m, ("auc", "pre")],
                 se_pre=t.loc[m, ("auc_se", "pre")], auc_pre_date=t.loc[m, ("auc", "pre+date")],
                 se_pre_date=t.loc[m, ("auc_se", "pre+date")])
            for m, lab in MODELS]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "poster_tables_models.csv", index=False)
    cells = [[SHORT.get(r.model, r.model), r.pro, r.con, f"{r.auc_pre:.3f} ({r.se_pre:.3f})",
              f"{r.auc_pre_date:.3f} ({r.se_pre_date:.3f})"] for r in df.itertuples()]
    draw(cells, ["Model", "Pro", "Con", "AUC before\nthe date", "AUC with in-date\nratings"], "poster_tables_models",
         "Comparing the models: held-out AUC (standard error)", "",    # no footnote: the slide and poster say how it was tested
         highlight=len(cells) - 1, widths=(0.19, 0.235, 0.215, 0.18, 0.18), left=(0, 1, 2), figwidth=11.2)
    return df


def predictor_table(top=8):
    imp = pd.read_csv(RES / "rf_importance.csv")
    out, panels = [], []
    for fs, head in (("pre+date", "With in-date ratings"), ("pre", "Before the date (sign-up answers only)")):
        d = imp[(imp.feature_set == fs) & (imp.kind != "group")].sort_values("auc_drop", ascending=False).head(top)
        out.append(d.assign(table_rank=range(1, len(d) + 1)))
        cells = [[f"{i}. {r.label}", f"{r.auc_drop:.3f}", f"{r.auc_drop_se:.3f}"] for i, r in enumerate(d.itertuples(), 1)]
        panels.append((f"{head}: forest AUC {d.forest_auc.iloc[0]:.3f}", cells))
    pd.concat(out).to_csv(RES / "poster_tables_predictors.csv", index=False)
    draw_side_by_side(panels, ["Predictor (ranked)", "AUC drop", "SE"], "poster_tables_predictors",
                      "What the random forest relies on most", widths=(0.68, 0.17, 0.15))


def _table(ax, cells, header, widths, left=(0,), highlight=None, section_rows=()):
    """Header, rules and cells of one table on axes with x in [0, 1] and one unit per row."""
    n = len(cells)
    x = np.cumsum((0,) + tuple(widths))
    for j, h in enumerate(header):
        ax.text(x[j] + (0.01 if j in left else widths[j] / 2), 0.42, h, fontsize=9.5, fontweight="bold", color=INK2,
                ha="left" if j in left else "center", va="center", linespacing=1.1)
    ax.plot([0, 1], [-0.1, -0.1], color=INK, linewidth=1)
    for i, row in enumerate(cells):
        yy = -i - 0.55
        if i == highlight:
            ax.add_patch(plt.Rectangle((0, yy - 0.45), 1, 0.9, color=HL, linewidth=0, zorder=0))
        if i in section_rows:
            ax.text(0.01, yy, row[0], fontsize=9.5, fontweight="bold", color=INK, va="center")
            continue
        for j, c in enumerate(row):
            ax.text(x[j] + (0.01 if j in left else widths[j] / 2), yy, c, fontsize=9.5, color=INK, va="center",
                    ha="left" if j in left else "center", fontweight="bold" if i == highlight else "normal")
        if i < n - 1:
            ax.plot([0, 1], [yy - 0.5, yy - 0.5], color=GRID, linewidth=0.6)
    ax.plot([0, 1], [-n - 0.05, -n - 0.05], color=INK, linewidth=1)


def draw(cells, header, name, title, note, highlight=None, section_rows=(), widths=(0.5, 0.25, 0.25), left=(0,),
         figwidth=7.4):
    """A table as a figure; columns in `left` are left-aligned, the rest centred."""
    n = len(cells)
    fig, ax = plt.subplots(figsize=(figwidth, 0.36 * (n + 1) + 1.2))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-(n + 0.3 + (0.45 * (note.count(chr(10)) + 1) if note else 0)), 1.45)
    ax.text(0, 1.3, title, fontsize=12, fontweight="bold", color=INK, va="bottom")
    _table(ax, cells, header, widths, left, highlight, section_rows)
    if note:
        ax.text(0, -n - 0.25, note, fontsize=7.5, color=INK2, va="top")
    fig.savefig(RES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_side_by_side(panels, header, name, title, widths):
    """Several tables of the same columns next to each other (landscape, for a slide), each with a subtitle."""
    n = max(len(c) for _, c in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 0.36 * (n + 2) + 1.1),
                             gridspec_kw=dict(wspace=0.08))
    for k, (ax, (sub, cells)) in enumerate(zip(axes, panels)):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-(n + 0.3), 2.25)
        if k == 0:
            ax.text(0, 2.05, title, fontsize=12, fontweight="bold", color=INK, va="bottom")
        ax.text(0.01, 1.2, sub, fontsize=10, fontweight="bold", color=INK, va="center")
        _table(ax, cells, header, widths)
    fig.savefig(RES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


SHORT = {"Logistic regression (with splines)": "Logistic (with splines)", "Stepwise logistic (forward)": "Stepwise logistic",
         "Lasso (L1 logistic)": "Lasso", "Single decision tree (depth 3)": "Single tree (depth 3)",
         "Random forest (500 trees)": "Random forest"}
COLUMN = r"""%% Middle column of the poster, written by scripts/poster_tables.py (do not edit the numbers by hand).
%% Needs tree_figure_pre_date_poster.png in the Overleaf project.
\begin{column}{\colwidth}

\begin{myblock}{Technical Implementation}
\begin{enumerate}
  \item \textbf{Pre-date and in-date feature sets.} \emph{Pre-date:} sign-up answers plus six pair features
  (age gap, interest similarity, same field, same goal, preference fit each way). \emph{In-date:} adds the
  16 ratings the two gave each other.
  \item \textbf{Nested event-level cross-validation.} Missing-value fills, scaling and tuning use only the
  training events.
  \item \textbf{Event-clustered standard errors.} Pairs at one event share people, so each of the 21 events
  is one cluster.
\end{enumerate}
\end{myblock}

\begin{myblock}{Performance by Model}
Held-out AUC (standard error); 0.5 = chance.
\begin{center}\small
MODELS_TABLE
\end{center}
\begin{itemize}
  \item Before the date every model is near chance (AUC PRE_LO--PRE_HI).
  \item The four-minute date adds about GAIN to the AUC.
  \item The forest leads, but its lead over the runner-up is within the error bars.
\end{itemize}
\end{myblock}

\begin{myblock}{What the Best Model Uses}
\begin{figure}
\centering
\includegraphics[width=0.92\linewidth]{tree_figure_pre_date_poster.png}
\caption{One 3-level tree fitted to all pairs; leaves give the share that matched. Held out it scores AUC
TREE_AUC; the 500-tree forest scores RF_AUC.}
\end{figure}
\vspace{-0.3cm}
Predictors ranked by how far the forest's held-out AUC falls when each is shuffled:
\begin{center}\small
PREDICTORS_TABLE
\end{center}
\vspace{-0.2cm}
Liking matters most, then fun and attractiveness. Shuffling all in-date ratings at once drops the AUC by
GROUP_DATE; all sign-up answers, by GROUP_PRE.
\end{myblock}

\end{column}
"""


def poster_column(models, n_date=5, n_pre=3):
    """Native LaTeX tables and the poster's middle column (report/poster/middle_column.tex)."""
    best = models.auc_pre_date.idxmax()
    rows = []
    for i, r in models.iterrows():
        b = (lambda t: f"\\textbf{{{t}}}") if i == best else (lambda t: t)
        rows.append(f"{b(SHORT.get(r.model, r.model))} & {b(f'{r.auc_pre:.3f}')} ({r.se_pre:.3f}) & "
                    f"{b(f'{r.auc_pre_date:.3f}')} ({r.se_pre_date:.3f}) \\\\")
    mt = ("\\begin{tabular}{lcc}\n\\toprule\nModel & Before the date & With in-date ratings \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    imp = pd.read_csv(RES / "rf_importance.csv")
    single = imp[imp.kind != "group"]
    g = imp[imp.kind == "group"].set_index(["feature_set", "predictor"]).auc_drop
    lines = ["\\begin{tabular}{lc}", "\\toprule", "Predictor & AUC drop (SE) \\\\", "\\midrule"]
    for fs, n, head in (("pre+date", n_date, "With in-date ratings"), ("pre", n_pre, "Before the date")):
        d = single[single.feature_set == fs].sort_values("auc_drop", ascending=False).head(n)
        if fs == "pre":
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{head} (forest AUC {d.forest_auc.iloc[0]:.3f})}}}} \\\\")
        lines += [f"{r.label} & {r.auc_drop:.3f} ({r.auc_drop_se:.3f}) \\\\" for r in d.itertuples()]
    pt = "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])
    tree = pd.read_csv(RES / "tree_figure_cv.csv").query("depth == 3 and feature_set == 'pre+date'").auc.iloc[0]
    rf = models.loc[best]
    fill = dict(MODELS_TABLE=mt, PREDICTORS_TABLE=pt, PRE_LO=f"{models.auc_pre.min():.2f}", PRE_HI=f"{models.auc_pre.max():.2f}",
                GAIN=f"{rf.auc_pre_date - rf.auc_pre:.2f}", TREE_AUC=f"{tree:.3f}", RF_AUC=f"{rf.auc_pre_date:.3f}",
                GROUP_DATE=f"{g['pre+date', 'GROUP: all in-date ratings']:.2f}",
                GROUP_PRE=f"{g['pre+date', 'GROUP: all sign-up answers']:.3f}")
    text = COLUMN
    for k, v in fill.items():
        text = text.replace(k, v)
    out = pathlib.Path("report/poster"); out.mkdir(parents=True, exist_ok=True)
    (out / "middle_column.tex").write_text(text)
    (out / "poster_tables_models.tex").write_text(mt + "\n")
    (out / "poster_tables_predictors.tex").write_text(pt + "\n")


def main():
    models = model_table()
    print(models.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if (RES / "rf_importance.csv").exists():
        predictor_table()
        poster_column(models)
        print("wrote report/poster/middle_column.tex and the two native tables")
        print(pd.read_csv(RES / "poster_tables_predictors.csv")[["feature_set", "table_rank", "label", "auc_drop", "auc_drop_se"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("wrote results/poster_tables_models.png/.csv and results/poster_tables_predictors.png/.csv")


if __name__ == "__main__":
    main()
