"""LaTeX tables for the written report, generated from the results tables (nothing typed by hand).

Input : results/part1_models_metrics.csv, results/part1_additions_metrics.csv,
        results/part1_models_stated_vs_revealed.csv
Output: results/report_tables_<name>.tex, copied to report/tables/<name>.tex for Overleaf:
  models          held-out AUC and balanced accuracy of the Part 1 models, pre and pre+date (event 5-fold CV)
  ladder          the full logistic model's AUC as compatibility features are added
  stated_revealed stated against revealed importance of the six traits, by gender
Every table uses \\begin{table}[h] (the report template's counter macro needs the optional argument) and booktabs.
Nothing is random.
Run from the repository root: python scripts/report_tables.py
"""
import pathlib, shutil
import pandas as pd

RES, OUT = pathlib.Path("results"), pathlib.Path("report/tables")
HEADLINE = "missing=median_flags; fit=B; out_of_range=rule; format=standardized"
NAMES = {"logistic": "Logistic regression", "stepwise_forward": "Stepwise logistic (forward, both)",
         "stepwise_backward": "Stepwise logistic (backward)", "lda": "LDA", "qda": "QDA (regularized)",
         "rf": "Random forest", "lasso": "L1 logistic regression$^\\dagger$"}


def save(name, body):
    path = RES / f"report_tables_{name}.tex"; path.write_text(body)
    OUT.mkdir(parents=True, exist_ok=True); shutil.copy(path, OUT / f"{name}.tex")
    print(f"wrote {path} (copy: {OUT / (name + '.tex')})")


def f3(x):
    return f"{x:.3f}"


def models():
    m = pd.read_csv(RES / "part1_models_metrics.csv")
    m = m[(m.headline == True) & (m.fold_scheme == "event") & (m.spec == HEADLINE)].set_index(["model", "feature_set"])
    rows = []
    for k, label in NAMES.items():                       # stepwise "both" chose the same model as forward in every fold
        cells = []
        for fs in ("pre", "pre+date"):
            r = m.loc[(k, fs)]
            cells += [f"{f3(r.auc)} ({f3(r.auc_se)})", f"{f3(r.balanced_accuracy)} ({f3(r.balanced_accuracy_se)})"]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    best = {fs: m.xs(fs, level="feature_set").auc.idxmax() for fs in ("pre", "pre+date")}
    body = (r"""\begin{table}[h]
\centering
\caption{Held-out performance of the models (event 5-fold cross-validation, 4,184 dyads, headline
specification). Standard errors in parentheses are clustered by event. Balanced accuracy uses the training-fold match
rate as the threshold. The best model on both feature sets is the """ + NAMES[best["pre"]].replace("$^\\dagger$", "").lower() + r""".}
\label{tab:models}
\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{\texttt{pre}} & \multicolumn{2}{c}{\texttt{pre+date}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & AUC & Bal.\ acc. & AUC & Bal.\ acc. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\smallskip
{\footnotesize $^\dagger$Added after the other models had been compared. Source: results/part1\_models\_metrics.csv.}
\end{table}
""")
    assert best["pre"] == best["pre+date"] == "rf"
    save("models", body)


def ladder():
    a = pd.read_csv(RES / "part1_additions_metrics.csv")
    a = a[(a.spec == HEADLINE) & (a.headline == True) & (a.model == "logistic") & (a.dyads == 4184)].set_index("feature_set")
    steps = [("pre_individual", "Each person's own answers", "--"),
             ("pre", "+ 6 pair features (age gap, interest correlation, same field, same goal, two preference fits)", None),
             ("pre_extended", "+ 5 more pair features (same career, dating and going-out gaps, two interest similarities)", None),
             ("pre_interactions", "+ 3 terms (squared age gap, interest correlation $\\times$ shared-interest weight, both ways)", None)]
    rows, prev = [], None
    for fs, label, _ in steps:
        r = a.loc[fs]
        diff = "--" if prev is None else f"${r.auc - prev:+.4f}$"
        rows.append(f"\\texttt{{{fs.replace('_', chr(92) + '_')}}} & {label} & {r.auc:.4f} ({r.auc_se:.4f}) & {diff} \\\\")
        prev = r.auc
    body = (r"""\begin{table}[h]
\centering
\caption{The full logistic model's held-out AUC as pair-level compatibility features are added (event 5-fold
cross-validation, 4,184 dyads). Standard errors in parentheses are clustered by event.}
\label{tab:ladder}
\begin{tabular}{l p{0.48\textwidth} c c}
\toprule
Feature set & What is added & AUC (SE) & Change \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\smallskip
{\footnotesize Source: results/part1\_additions\_metrics.csv.}
\end{table}
""")
    save("ladder", body)


def stated_revealed():
    s = pd.read_csv(RES / "part1_models_stated_vs_revealed.csv").set_index(["rater_gender", "attribute"])
    traits = [("attr", "Attractive"), ("sinc", "Sincere"), ("intel", "Intelligent"), ("fun", "Fun"), ("amb", "Ambitious"),
              ("shar", "Shared interests")]
    rows = []
    for k, label in traits:
        cells = []
        for g in ("women", "men"):
            r = s.loc[(g, k)]
            cells += [f"{r.stated_share:.1f}", f"{r.revealed_share:.1f}", f"${r.gap_revealed_minus_stated:+.1f}$"]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    n = {g: (int(s.loc[(g, 'attr')].dates), int(s.loc[(g, 'attr')].stated_people)) for g in ("women", "men")}
    body = (r"""\begin{table}[h]
\centering
\caption{Stated against revealed importance of six traits, in percentage points. \emph{Stated}: the average share of
100 points given to the trait before the event (waves 1--5 and 10--21). \emph{Revealed}: the trait's share of the
positive standardized coefficients in a logistic model of each yes decision on the six in-date ratings. Attractiveness
has the largest gap for both genders.}
\label{tab:stated}
\begin{tabular}{l ccc ccc}
\toprule
 & \multicolumn{3}{c}{Women} & \multicolumn{3}{c}{Men} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Trait & Stated & Revealed & Gap & Stated & Revealed & Gap \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\smallskip
{\footnotesize Yes model: """ + f"{n['women'][0]:,} dates by women and {n['men'][0]:,} by men; stated shares: {n['women'][1]} women and {n['men'][1]} men." + r""" Source: results/part1\_models\_stated\_vs\_revealed.csv.}
\end{table}
""")
    save("stated_revealed", body)


if __name__ == "__main__":
    models(); ladder(); stated_revealed()
