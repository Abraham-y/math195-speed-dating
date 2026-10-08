"""Write the methods manual: every step from the raw file to each model, metric and check, as a PDF.

Input : data/feature_sets.json, data/pairs.csv, data/clean.csv, the constants in scripts/modeling.py and
        scripts/part1_models.py, results/*_summary.csv and *_hypotheses.csv (for the tables of checks and verdicts)
Output: results/methods_doc.tex and results/methods_doc.pdf (compiled with pdflatex, run twice for the contents)
No random operations, so no seed. Every number in the document is read from those files or constants.
Run from the repository root after the analysis scripts: python scripts/methods_doc.py
"""
import sys, json, pathlib, subprocess, inspect
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import pairs as PR
import part1_models as P1
import like_models as LM

RES = pathlib.Path("results")
OUT = RES / "methods_doc.tex"

# codebook wording (data/speed_dating_key.doc) for the person-level columns
PERSON_DESC = {
    "age": ("Age in years", "years"),
    "field_cd": ("Field of study, coded: 1 Law, 2 Math, 3 Social science/psychologist, 4 Medical science/pharma/biotech, "
                 "5 Engineering, 6 English/creative writing/journalism, 7 History/religion/philosophy, 8 Business/econ/finance, "
                 "9 Education/academia, 10 Biological sciences/chemistry/physics, 11 Social work, 12 Undergrad/undecided, "
                 "13 Political science/international affairs, 14 Film, 15 Fine arts/arts administration, 16 Languages, "
                 "17 Architecture, 18 Other", "code (categorical)"),
    "career_c": ("Intended career, coded: 1 Lawyer, 2 Academic/research, 3 Psychologist, 4 Doctor/medicine, 5 Engineer, "
                 "6 Creative arts/entertainment, 7 Banking/consulting/finance/marketing/business/CEO/entrepreneur/admin, "
                 "8 Real estate, 9 International/humanitarian affairs, 10 Undecided, 11 Social work, 12 Speech pathology, "
                 "13 Politics, 14 Pro sports/athletics, 15 Other, 16 Journalism, 17 Architecture", "code (categorical)"),
    "goal": ("Primary goal in coming: 1 Seemed like a fun night out, 2 To meet new people, 3 To get a date, "
             "4 Looking for a serious relationship, 5 To say I did it, 6 Other", "code (categorical)"),
    "date": ("How often do you go on dates? 1 Several times a week, 2 Twice a week, 3 Once a week, 4 Twice a month, "
             "5 Once a month, 6 Several times a year, 7 Almost never", "1 to 7 (1 = most often)"),
    "go_out": ("How often do you go out (not necessarily on dates)? Same scale as dating frequency", "1 to 7 (1 = most often)"),
    "exphappy": ("How happy do you expect to be with the people you meet at the event?", "1 to 10"),
}
INTEREST_DESC = {"sports": "Playing sports/athletics", "tvsports": "Watching sports", "exercise": "Body building/exercising",
                 "dining": "Dining out", "museums": "Museums/galleries", "art": "Art", "hiking": "Hiking/camping",
                 "gaming": "Gaming", "clubbing": "Dancing/clubbing", "reading": "Reading", "tv": "Watching TV",
                 "theater": "Theater", "movies": "Movies", "concerts": "Going to concerts", "music": "Music",
                 "shopping": "Shopping", "yoga": "Yoga/meditation"}
ATTR_NAME = {"attr": "attractive", "sinc": "sincere", "intel": "intelligent", "fun": "fun", "amb": "ambitious",
             "shar": "shared interests/hobbies"}


def tx(s):
    """Escape text for LaTeX."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "`": ""}
    return "".join(rep.get(ch, ch) for ch in str(s))


def code(s):
    return r"\texttt{" + tx(s) + "}"


def describe(col):
    """Plain description of one f_/m_ column or pair feature."""
    side, stem = (col[:2], col[2:]) if col[:2] in ("f_", "m_") else ("", col)
    who = {"f_": "Woman's", "m_": "Man's"}.get(side, "")
    if stem in PERSON_DESC:
        return f"{who} answer: {PERSON_DESC[stem][0]}", PERSON_DESC[stem][1]
    if stem in INTEREST_DESC:
        return f"{who} interest in {INTEREST_DESC[stem].lower()}", "1 to 10"
    if stem.startswith("self_"):
        return f"{who} self-rating: how {ATTR_NAME[stem[5:]]} am I?", "1 to 10"
    if stem.startswith("pref_"):
        return f"{who} stated importance of {ATTR_NAME[stem[5:]]} in a date, as a share of 100", "share (sums to 100)"
    if stem.startswith("rating_"):
        a = stem[7:]
        q = "overall, how much do you like this person?" if a == "like" else f"how {ATTR_NAME[a]} is this person?"
        return f"{who} scorecard rating of the partner: {q}", "1 to 10"
    if stem == "prob":
        return f"{who} guess: how probable is it that this person will say yes to you?", "1 to 10"
    if stem.endswith("pre_missing"):
        return f"{who} sign-up form has an answer left blank", "0/1"
    if stem.endswith("date_missing"):
        return f"{who} scorecard has a rating left blank", "0/1"
    return PAIR_DESC.get(col, (col, ""))


PAIR_DESC = {
    "abs_age_diff": ("Age gap |man's age - woman's age|", "years"),
    "int_corr": ("Interest similarity: Pearson correlation of the two people's ratings of the 16 interests other than yoga", "-1 to 1"),
    "same_field": ("1 if the two field codes are equal", "0/1"),
    "same_goal": ("1 if the two goal codes are equal", "0/1"),
    "fit_to_f_prefs": ("How well the man's relative strengths fit the woman's priorities (compatibility term, equation 1)", "standardized"),
    "fit_to_m_prefs": ("How well the woman's relative strengths fit the man's priorities", "standardized"),
    "same_career": ("1 if the two career codes are equal", "0/1"),
    "abs_date_diff": ("Gap in dating frequency |f_date - m_date|", "0 to 6"),
    "abs_go_out_diff": ("Gap in going-out frequency |f_go_out - m_go_out|", "0 to 6"),
    "int_mad": ("Mean absolute difference of the 17 interest ratings", "0 to 9"),
    "int_cosine": ("Cosine similarity of the two 17-interest vectors", "0 to 1"),
    "age_diff_sq": ("Squared signed age difference (m_age - f_age)^2", "years^2"),
    "int_x_f_pref_shar": ("Interest similarity times the woman's (standardized) weight on shared interests", ""),
    "int_x_m_pref_shar": ("Interest similarity times the man's (standardized) weight on shared interests", ""),
}


def snippet(obj, caption, start=None, stop=None):
    """The source of a function or class from the scripts, as a listing (optionally the lines between two markers)."""
    src = inspect.getsource(obj)
    if start:
        lines = src.splitlines()
        i = next(k for k, l in enumerate(lines) if start in l)
        j = next((k for k, l in enumerate(lines) if stop and k > i and stop in l), len(lines))
        src = "\n".join(lines[i:j])
    src = src.replace("\u00b1", "+/-").encode("ascii", "replace").decode()
    where = f"{inspect.getsourcefile(obj).split('/')[-2]}/{inspect.getsourcefile(obj).split('/')[-1]}"
    return ("\\begin{lstlisting}[caption={" + tx(caption) + " (" + tx(where) + ")}]\n" + src.rstrip() + "\n\\end{lstlisting}")


def table(header, rows, spec, caption=None, long=False, size=r"\small"):
    env = "longtable" if long else "tabular"
    out = [size]
    if not long:
        out.append(r"\begin{center}")
    out.append(r"\begin{" + env + "}{" + spec + "}")
    out.append(r"\toprule")
    out.append(" & ".join(r"\textbf{" + h + "}" for h in header) + r" \\")
    out.append(r"\midrule")
    if long:
        out.append(r"\endhead")
    out += [" & ".join(r) + r" \\" for r in rows]
    out.append(r"\bottomrule")
    out.append(r"\end{" + env + "}")
    if not long:
        out.append(r"\end{center}")
    if caption:
        out.append(r"\vspace{-0.4em}{\footnotesize " + caption + r"}\par\medskip")
    out.append(r"\normalsize")
    return "\n".join(out)


def main():
    sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    pairs = pd.read_csv("data/pairs.csv")
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    folds = M.event_folds(pairs.event)
    by_fold = {k: sorted(e for e, f in folds.items() if f == k) for k in range(5)}
    n_dyads, n_match = len(pairs), int(pairs.match.sum())
    n_people, n_events = clean.rater.nunique(), clean.event.nunique()
    miss = pairs.isna().sum()
    used_pre = [c for c in sets["pre"] if c not in sets["pref_reference"]]
    raw = pd.read_csv("data/speed_dating.csv", encoding="latin-1")
    filled = clean[clean.pref_blanks_set_to_0 > 0]
    fmt = pd.read_csv(RES / "eda_pref_format.csv").set_index("format").mean_within_person_sd
    fitr2 = pd.read_csv(RES / "eda_pref_fit.csv").query("preferences_of == 'f' and format == 'all'").r_squared.iloc[0]
    fi = pd.read_csv(RES / "part1_models_folds.csv")
    head_cols = fi[fi.spec == M.HEADLINE.label].groupby("feature_set")["columns"].agg(["min", "max"])
    spl = fi.splines.fillna("")
    spl_who = sorted(set(";".join(spl[spl != ""]).split(";")))
    met = pd.read_csv(RES / "part1_models_metrics.csv")
    cc = met[met.spec == M.COMPLETE_CASE.label].iloc[0]
    tokens = {"@@NOPID@@": str(int(raw.pid.isna().sum())), "@@FILLP@@": str(filled.rater.nunique()), "@@FILLR@@": str(len(filled)),
              "@@GAMING@@": str(clean[clean.gaming_recorded == 14].rater.nunique()),
              "@@READING@@": str(clean[clean.reading_recorded == 13].rater.nunique()),
              "@@SD69@@": f"{fmt[fmt.index.str.startswith('1-10')].iloc[0]:.1f}", "@@SD100@@": f"{fmt[fmt.index.str.startswith('100')].iloc[0]:.1f}",
              "@@R2LEVEL@@": f"{100 * fitr2:.0f}", "@@QDAMAXV@@": f"{max(P1.QDA_REGS):g}",
              "@@QDAMAX@@": str(int((fi.qda_reg_param == max(P1.QDA_REGS)).sum())), "@@QDAN@@": str(int(fi.qda_reg_param.notna().sum())),
              "@@CCD@@": f"{int(cc.dyads):,}", "@@CCM@@": str(int(cc.matches))}
    hyp = pd.concat([pd.read_csv(RES / f) for f in ("part1_models_hypotheses.csv", "part1_additions_hypotheses.csv", "like_models_hypotheses.csv")],
                    ignore_index=True)
    s_auc, s_like, s_coef = (pd.read_csv(RES / f) for f in ("sim_auc_cluster_summary.csv", "sim_like_cluster_summary.csv", "sim_coef_cluster_summary.csv"))
    L = []
    add = L.append

    add(r"""\documentclass[11pt]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array,enumitem,xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{lmodern}\usepackage[T1]{fontenc}
\usepackage{listings}
\lstset{language=Python, basicstyle=\ttfamily\scriptsize, breaklines=true, breakatwhitespace=false, columns=fullflexible,
  keepspaces=true, frame=single, rulecolor=\color{black!25}, backgroundcolor=\color{black!3}, showstringspaces=false,
  keywordstyle=\color{blue!55!black}, commentstyle=\color{black!55}\itshape, stringstyle=\color{green!35!black},
  captionpos=t, xleftmargin=2pt, xrightmargin=2pt, aboveskip=0.6em, belowskip=0.6em,
  literate={~}{{$\sim$}}1 {`}{{\textasciigrave}}1}
\renewcommand{\lstlistingname}{Code}
\setlist{itemsep=2pt,topsep=4pt}
\setlength{\parskip}{0.5em}\setlength{\parindent}{0pt}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\title{Predicting mutual interest before two strangers meet\\[0.3em]\large Part 1 methods: data, predictors, preparation, models, metrics and checks}
\author{Math 195 capstone: Sophia Qin, Jenna Yang, Abraham Yeung}
\date{8 October 2026 \quad (code at commit """ + tx(commit) + r""")}
\begin{document}
\maketitle
\begin{abstract}
This manual records every step that turns the raw Columbia speed-dating file into the Part 1 results: how the data
were cleaned, every predictor (including the ones we constructed), how each training fold is prepared, exactly how
each model (logistic regression, stepwise selection, LDA, QDA, random forest, lasso, the yes model and the liking
model) is fitted and tuned, how AUC and the other metrics and their error bars are computed, and how each
pre-registered hypothesis and robustness check is decided. It is generated by \texttt{scripts/methods\_doc.py}
from the code and data, so its lists and constants match what ran. The decisions themselves, with dates, are in
\texttt{PREREG.md}.
\end{abstract}
\tableofcontents
\newpage
""")

    # ------------------------------------------------------------------ 1 pipeline
    add(r"\section{The pipeline at a glance}")
    add("Run every script from the repository root, in this order. Each writes its outputs to \\texttt{data/} or "
        "\\texttt{results/} and prints the numbers the weekly report needs. Every random operation is seeded (Appendix~B).")
    rows = [
        ["1", code("clean.py"), "Cleans the raw file; one row per date from one side", code("data/clean.csv")],
        ["2", code("pairs.py"), "One row per pair (dyad); constructs the pair features and feature sets", code("data/pairs.csv") + ", " + code("feature_sets.json")],
        ["3", code("participants.py"), "One row per participant (catalog)", code("data/participants.csv")],
        ["4", code("eda.py"), "Exploratory tables and figures", code("results/eda_*")],
        ["5", code("data_description.py"), "Report figures and captions", code("results/data_description_*")],
        ["6", code("sim_auc_cluster.py"), "Checks the match-model error bars by simulation", code("results/sim_auc_cluster_*")],
        ["7", code("sim_like_cluster.py"), "Checks the liking-model error bars", code("results/sim_like_cluster_*")],
        ["8", code("sim_coef_cluster.py"), "Checks the coefficient and yes-model error bars", code("results/sim_coef_cluster_*")],
        ["9", code("like_models.py"), "Liking model (L1, L2)", code("results/like_models_*")],
        ["10", code("part1_models.py"), "All match models, 24 specifications, diagnostics, yes model (P1 to P5); about 30 minutes", code("results/part1_models_*")],
        ["11", code("part1_additions.py"), "Pair-feature ladder (P3b, P3c), wave checks, P5 without wave 12", code("results/part1_additions_*")],
        ["12", code("fold_seed_noise.py"), "Fold and forest seed noise (exploratory)", code("results/fold_seed_noise*")],
        ["13", code("compare_runs.py"), "Earlier results next to the current ones", code("results/compare_runs_*")],
    ]
    add(table(["", "Script (in scripts/)", "What it does", "Main output"], rows, "rL{3.7cm}L{6.4cm}L{4.6cm}"))
    add("Tests with known answers live in \\texttt{tests/} (run each file with \\texttt{python tests/test\\_<name>.py}); "
        "every function that computes a variance, a fold assignment or a pair feature has one.")

    # ------------------------------------------------------------------ 2 data
    add(r"\section{Data and unit of analysis}")
    add(r"\subsection{The raw file}")
    add(f"The Kaggle release of the Columbia speed-dating experiment (Fisman, Iyengar, Kamenica and Simonson): "
        f"{n_events} events (waves) between October 2002 and April 2004. Each person met every participant of the opposite "
        f"sex at their event for four minutes, filled in a scorecard after each date (yes or no, six attribute ratings, "
        f"overall liking, a guess of the partner's decision), and had filled in a sign-up survey before the event. Each row "
        f"of the raw file is one date seen from one side. Always read alongside \\texttt{{data/speed\\_dating\\_key.doc}}.")
    add(r"\subsection{Cleaning (\texttt{clean.py})}")
    add(r"""\begin{enumerate}
\item \textbf{Drop unusable rows.} @@NOPID@@ rows have no partner id: in wave 5, ten men each rated a woman (partner
number 7) whose own record is missing from the release. No row lacks a decision.
\item \textbf{Identifiers.} \texttt{rater} (iid), \texttt{ratee} (pid), \texttt{event} (wave), \texttt{dyad} (the
unordered pair) and \texttt{date\_id} (the ordered pair).
\item \textbf{Stated preferences to shares of 100.} Three sign-up blocks ask how important each of six attributes is
(own preferences \texttt{1\_1}, what the opposite sex wants \texttt{2\_1}, what peers want \texttt{4\_1}). Waves
1--5 and 10--21 split 100 points; waves 6--9 rated each attribute from 1 to 10. Every block is rescaled to shares that
sum to 100 (\texttt{<item>\_share}); the survey format is kept in \texttt{pref\_scale}.
\item \textbf{Blank allocations.} In the 100-point waves, where some items of a block were left blank but the
answered items already sum to 100 (within 1 point), the blank items are set to 0, since no points were left to give
(@@FILLP@@ people, @@FILLR@@ dates). Each item left blank on the form is recorded in \texttt{<item>\_was\_blank}.
\item \textbf{Recording errors.} Gaming = 14 (@@GAMING@@ people) and reading = 13 (@@READING@@ people) recur in the same question across
events on a 1--10 scale, so they are set to missing; the recorded values are kept in \texttt{gaming\_recorded} and
\texttt{reading\_recorded} for the robustness grid. Ratings of 0 are kept as deliberate answers.
\item \textbf{Checks}, all zero: partners in a different wave, \texttt{match} $\ne$ both decisions, duplicate dates.
\end{enumerate}""")
    add(r"\subsection{The pair table (\texttt{pairs.py})}")
    add(f"Match prediction uses one row per dyad: one woman and one man who met ({n_dyads:,} dyads, {n_match} matches, "
        f"{100 * n_match / n_dyads:.1f}\\%). The response is \\texttt{{match}} = 1 when both said yes. Each person's own answers appear "
        f"with prefix \\texttt{{f\\_}} (the woman) or \\texttt{{m\\_}} (the man); for the scorecard columns the prefix says whose "
        f"view it is (\\texttt{{f\\_rating\\_attr}} is her rating of his attractiveness). The two decisions "
        f"\\texttt{{f\\_dec}} and \\texttt{{m\\_dec}} define the response and are never predictors. A check rebuilds the "
        f"table with the genders swapped and confirms every symmetric pair feature is unchanged. The {n_people} participants "
        f"each appear in many dyads, which is why the error bars treat events as clusters (Section~\\ref{{sec:metrics}}).")

    # ------------------------------------------------------------------ 3 predictors
    add(r"\section{Predictors}\label{sec:predictors}")
    add(r"\subsection{Each person's sign-up answers}")
    rows = [[code("p_age"), "Age", "years"],
            [code("p_field_cd"), "Field of study (18 codes; Appendix A)", "categorical"],
            [code("p_career_c"), "Intended career (17 codes)", "categorical"],
            [code("p_goal"), "Primary goal in coming (6 codes)", "categorical"],
            [code("p_date"), "How often do you go on dates?", "1--7, 1 = most often"],
            [code("p_go_out"), "How often do you go out?", "1--7, 1 = most often"],
            [code("p_exphappy"), "How happy do you expect to be with the people you meet?", "1--10"],
            [code("p_<interest>"), "Interest in each of 17 activities: " + ", ".join(INTEREST_DESC.values()).lower(), "1--10"],
            [code("p_self_<a>"), "Self-rating on five attributes: attractive, sincere, intelligent, fun, ambitious", "1--10"],
            [code("p_pref_<a>"), "Stated importance of six attributes (the five plus shared interests), as shares of 100", "share"]]
    add(table(["Column (p = f or m)", "Question", "Scale"], rows, "L{3.6cm}L{8.6cm}L{3cm}",
              caption="Each appears twice, once for the woman and once for the man. The share for shared interests is left out of every "
                      "model (\\texttt{pref\\_reference}): the six shares sum to 100, so one is redundant."))
    add(r"\subsection{The scorecard (in-date ratings)}")
    add("Used only in the \\texttt{pre+date} feature set. For each side p: \\texttt{p\\_rating\\_<a>} for the six attributes "
        "(1 = awful, 10 = great), \\texttt{p\\_rating\\_like} (overall, how much do you like this person; 1--10) and "
        "\\texttt{p\\_prob} (how probable is it that this person will say yes to you; 1--10).")

    add(r"\subsection{Constructed pair features}")
    add("These are new columns we built from the two people's answers. The first six are the pair features of P3; the "
        "next five are the extended features of P3b; the last three are the declared terms of P3c. All use only sign-up "
        "answers. Where a definition needs an answer that is missing, the feature is missing and is filled inside each "
        "training fold (Section~\\ref{sec:prep}).")
    rows = []
    for c in sets["pre_pair"] + sets["extended_pair"] + M.INTERACTIONS:
        d, scale = PAIR_DESC[c]
        group = "P3" if c in sets["pre_pair"] else ("P3b" if c in sets["extended_pair"] else "P3c")
        rows.append([code(c), tx(d), group, f"{int(miss.get(c, 0))}" if c in miss else "--"])
    add(table(["Feature", "Definition", "Set", "Missing"], rows, "L{3.4cm}L{9.2cm}cr",
              caption="Missing = dyads where it cannot be computed (of " + f"{n_dyads:,}" + "). The P3c terms are built after the survey-format "
                      "standardization (Section~\\ref{sec:before}), so they use the standardized shares."))
    add(r"""\textbf{Compatibility (preference fit).} With $w_a$ the woman's stated share for attribute $a$, rescaled to sum
to 1 over the five self-rated attributes (shared interests has no self-rating), $s_a$ the man's self-rating and
$\bar s$ his mean self-rating over the five,
\begin{equation}
\texttt{fit\_to\_f\_prefs} = \sum_{a=1}^{5} \Big(w_a - \tfrac15\Big)\big(s_a - \bar s\big).
\end{equation}
It is positive when the attributes she weights above average are the ones he rates himself relatively highest on,
and 0 when either gave all-equal answers. The plain weighted mean $\sum_a w_a s_a$ (option A, kept for the
robustness grid as \texttt{fit\_raw\_to\_f\_prefs}) equals $\bar s$ plus this term: it is @@R2LEVEL@@\% his overall
self-rating, a fact about him alone, so the headline uses only the compatibility term (option B).
\texttt{fit\_to\_m\_prefs} swaps the roles.

\textbf{Interest similarity.} \texttt{int\_corr} is the Pearson correlation of the two people's ratings of the 16
interests other than yoga, over the interests both answered (at least 3), unrounded and after the recording-error
fix. This reproduces the released \texttt{int\_corr} exactly (rounded to 2 decimals) wherever no fix applies.
\texttt{int\_mad} and \texttt{int\_cosine} use all 17 interests, over those both answered (at least 3):
\[
\texttt{int\_mad} = \frac{1}{|S|}\sum_{i\in S} |f_i - m_i|, \qquad
\texttt{int\_cosine} = \frac{\sum_{i\in S} f_i m_i}{\sqrt{\sum_{i\in S} f_i^2 \sum_{i\in S} m_i^2}}.
\]

\textbf{Missing-answer flags.} \texttt{f\_pre\_missing} and \texttt{m\_pre\_missing} are 1 when that person left any
sign-up answer blank on the form (recorded values count, not the recording-error fix, and so do preference items the
blank-allocation rule set to 0). \texttt{f\_date\_missing} and \texttt{m\_date\_missing} are 1 when that person left
any scorecard rating blank. They let a model learn whether leaving answers blank itself predicts a match (pairs with
a blank in-date answer match less often).""")
    add("The pair features are built in \\texttt{pairs.build\\_pairs}, with the helper functions below.")
    add(snippet(PR.build_pairs, "The constructed pair features and flags", start="new = pd.DataFrame(", stop="pairs = pd.concat([pairs, new]"))
    add(snippet(PR.compat_fit, "Compatibility term, equation (1)"))
    add(snippet(PR.row_corr_pairwise, "Interest similarity over the interests both answered"))
    add(snippet(PR.row_mad_pairwise, "Mean absolute difference of interests"))
    add(snippet(PR.row_cosine_pairwise, "Cosine similarity of interests"))
    add(snippet(PR.same_code, "Same field, goal or career"))

    add(r"\subsection{Feature sets}")
    rows = [
        [code("pre"), "Everything known before the date: both people's sign-up answers and the six pair features",
         f"{len(sets['pre'])} ({len(used_pre)} used)"],
        [code("pre+date"), "Adds the 16 scorecard columns", f"{len(sets['pre+date'])} ({len(sets['pre+date']) - 2} used)"],
        [code("pre_individual"), "Both people's sign-up answers only (for P3)", f"{len(sets['pre_individual'])} ({len(sets['pre_individual']) - 2} used)"],
        [code("pre_extended"), code("pre") + " plus the five P3b features", f"{len(sets['pre']) + len(sets['extended_pair']) - 2} used"],
        [code("pre_interactions"), code("pre_extended") + " plus the three P3c terms", f"{len(sets['pre']) + len(sets['extended_pair']) + 3 - 2} used"],
    ]
    add(table(["Set", "Contents", "Raw columns"], rows, "L{3.2cm}L{9.4cm}L{2.6cm}",
              caption="``Used'' drops the two shared-interests shares. The missing-answer flags are added on top (the two sign-up flags; "
                      "for \\texttt{pre+date} also the two scorecard flags), except in the complete-case version."))
    add("\\textbf{Never used as predictors:} race and race-preference variables (a pre-registered decision), "
        "\\texttt{f\\_dec} and \\texttt{m\\_dec} (they define the response), the signed \\texttt{age\\_diff} (a linear combination "
        "of the two ages), \\texttt{int\\_corr17} (a near-duplicate of \\texttt{int\\_corr}), the event-level flags, and the order of "
        "the date in the evening.")

    # ------------------------------------------------------------------ 4 before CV
    add(r"\section{Preparation before cross-validation}\label{sec:before}")
    add(r"""Two steps use no outcomes. They are applied to the whole pair table, in \texttt{modeling.build\_frame}, before
any fold is formed.
\begin{enumerate}
\item \textbf{Survey-format standardization.} Waves 6--9 rated each attribute 1--10, so their converted shares are
flat (a person's six shares have SD @@SD69@@ against @@SD100@@ in the 100-point waves). Each stated-preference share and the two
compatibility terms are rescaled to mean 0 and SD 1 within each format:
\[
z = \frac{x - \mu_{\text{format}}}{\sigma_{\text{format}}},
\]
where for the shares $\mu$ and $\sigma$ come from one row per participant (women and men pooled) in that format, and
for the compatibility terms from one row per dyad in that format. Because only survey answers are used, this stays
valid even when a fold holds out every waves 6--9 event at once.
\item \textbf{Specification choices} (Section~\ref{sec:robust}): which interest values (fixed or as recorded), which
preference fit (B or A), whether to standardize by format, and how to treat missing values.
\end{enumerate}""")
    add(snippet(M.standardize_by_format, "Survey-format standardization"))
    add(snippet(M.build_frame, "The raw design for one specification and feature set"))

    # ------------------------------------------------------------------ 5 CV
    add(r"\section{Cross-validation}\label{sec:cv}")
    add(r"\subsection{Outer folds: whole events held out}")
    add(f"The {n_events} events are put in a random order (seed {M.FOLD_SEED}) and dealt into 5 folds in turn. Each fold "
        "holds out its events; models are fitted on the other four folds and score the held-out pairs. All five folds' "
        "held-out predictions are pooled, and every metric is computed once on the pooled set. Holding out whole events "
        "means a model is never scored on a person it saw in training: random row splits would put the same people on "
        "both sides and overstate accuracy.")
    rows = [[str(k), ", ".join(map(str, by_fold[k])), f"{int(pairs.event.isin(by_fold[k]).sum()):,}",
             f"{int(pairs[pairs.event.isin(by_fold[k])].match.sum())}"] for k in range(5)]
    add(table(["Fold", "Events held out", "Dyads", "Matches"], rows, "clrr"))
    add(r"""\textbf{Threshold.} A pair is predicted to match when its predicted probability exceeds the training fold's
match rate (about 0.165), chosen on training data only. This puts the cut where an uninformed model would sit, since
matches are rare; AUC does not depend on it.""")
    add(snippet(M.event_folds, "Outer event folds"))
    add(r"\subsection{Inner folds: tuning inside a training fold}")
    add(f"QDA, the random forest and the lasso each have one setting to tune. Inside outer training fold $k$, the training "
        f"events are dealt into 4 inner folds (seed {M.FOLD_SEED} + 1 + $k$). For each grid value, the model is fitted on three "
        "inner folds (with the preparation of Section~\\ref{sec:prep} refitted on them) and predicts the fourth; the four "
        "inner held-out sets are pooled and the grid value with the highest pooled AUC is chosen (ties go to the earlier "
        "value). The model is then refitted on the whole training fold with that value. The outer test fold is never touched.")
    add(snippet(P1.tune_inner, "Inner tuning by pooled AUC"))

    # ------------------------------------------------------------------ 6 prep
    add(r"\section{Preparation inside each training fold}\label{sec:prep}")
    add(f"Every quantity below is estimated on the training fold only and then applied unchanged to the held-out events "
        f"(class \\texttt{{modeling.Prep}}), in this order.")
    add(r"""\begin{enumerate}
\item \textbf{Skewness rule.} For each numeric column with more than two distinct values, the sample skewness
$\gamma = \frac1n\sum_i \big((x_i-\bar x)/s\big)^3$ is computed on the training fold (over one value per participant
for a person's own answers, over dyads for pair features). If $\gamma > """ + f"{M.SKEW_LIMIT:g}" + r"""$ the column becomes
$\log(1 + x - \min x)$; if $\gamma < -""" + f"{M.SKEW_LIMIT:g}" + r"""$ it becomes $-\log(1 + \max x - x)$, with the
minimum and maximum from the training fold and held-out values clipped to that range first. Binary columns are never
transformed.
\item \textbf{Missing values.} Each numeric column's missing values are filled with its training-fold median (after
the transform); binary columns with their training median. Under the headline specification the missing-answer
flags (Section~\ref{sec:predictors}) are included as predictors, so the model can tell a filled value from an answered
one.
\item \textbf{Categorical codes.} Field, career and goal are categorical. A field or career code held by fewer than
""" + f"{M.RARE_CODE_MIN}" + r""" participants in the training fold (each person counted once) is merged into ``other''; a
missing code is its own level ``missing''. The most common level in the training fold is the reference; every other
level gets a 0/1 dummy column. A held-out code never seen in training goes to ``other'' (or the reference if there is
no ``other'').
\item \textbf{Splines} (logistic and stepwise only, Section~\ref{sec:spline}). A natural cubic spline with knots at
the training-fold quantiles $0, \tfrac13, \tfrac23, 1$ of the filled column ($K = 4$ knots, 3 degrees of freedom)
replaces a candidate's straight line. With $d_k(x) = \big[(x-\xi_k)_+^3 - (x-\xi_K)_+^3\big]/(\xi_K - \xi_k)$, the
basis is $x$ and $N_{k+2}(x) = d_k(x) - d_{K-1}(x)$ for $k = 1, \dots, K-2$ (Hastie, Tibshirani and Friedman,
\emph{ESL}, 5.2.1).
\item \textbf{Standardizing.} Every numeric column, spline columns included, is rescaled to the training fold's mean
0 and SD 1. Dummies and flags stay 0/1.
\item \textbf{Identical columns.} When two columns are identical in a training fold (for example two missing-code
dummies held by the same people) both are kept: the data cannot divide their combined effect, and the small ridge
in the logistic fit divides it equally.
\end{enumerate}""")
    add(snippet(M.Prep, "Fitting the preparation on a training fold", start="def fit(", stop="@staticmethod"))
    add(snippet(M.Prep, "Building and applying the design", start="def _labels(", stop="def blocks("))
    add(snippet(M.natural_spline_basis, "Natural cubic spline basis"))
    add(f"The prepared design has {head_cols.loc['pre', 'min']} to {head_cols.loc['pre', 'max']} columns for \\texttt{{pre}} and "
        f"{head_cols.loc['pre+date', 'min']} to {head_cols.loc['pre+date', 'max']} for \\texttt{{pre+date}} under the headline "
        "specification, depending on the fold (dummies and splines differ by fold).")

    # ------------------------------------------------------------------ 7 models
    add(r"\section{The models}\label{sec:models}")
    add("Each model below is fitted on the prepared training fold and predicts a match probability for each held-out pair. "
        "Logistic regression and stepwise selection use the design with the splines chosen in that fold; LDA, QDA, the "
        "random forest and the lasso use the same preparation without splines (they can model curvature themselves or "
        "are linear by design).")
    add(snippet(P1.run_fold, "One outer fold: preparation and every model"))
    add(snippet(P1.cross_validate, "Event cross-validation"))
    add(r"\subsection{Logistic regression (full model)}")
    add(r"""The probability of a match is $p_i = 1/(1 + e^{-\eta_i})$ with $\eta_i = \beta_0 + x_i^\top\beta$. The coefficients
maximize the log-likelihood with a tiny ridge that keeps separated fits finite,
\[
\ell(\beta) - \tfrac{\lambda}{2}\sum_{j\ge1}\beta_j^2,\qquad
\ell(\beta) = \sum_i \big[y_i\eta_i - \log(1 + e^{\eta_i})\big],\qquad \lambda = 10^{-6},
\]
by iteratively reweighted least squares (Newton's method): with $W = \operatorname{diag}\{p_i(1-p_i)\}$ and
$\Lambda = \operatorname{diag}(0, \lambda, \dots, \lambda)$,
\[
\beta \leftarrow \beta + (X^\top W X + \Lambda)^{-1}\big(X^\top(y - p) - \Lambda\beta\big),
\]
stopping when no coefficient moves by more than $10^{-8}$ (\texttt{modeling.fit\_logit}). The full model uses every
column of the feature set; with 690 matches and more than 120 columns it overfits, which is why stepwise selection
and the regularized models exist.""")
    add(snippet(M.fit_logit, "Logistic regression by IRLS"))
    add(snippet(M.predict_logit, "Predicted probabilities"))
    add(r"\subsection{Choosing splines by BIC}\label{sec:spline}")
    add(f"In each training fold, the full model is first fitted with straight lines; its BIC is $-2\\ell + k\\log n$ with $k$ "
        f"the number of columns plus the intercept. Then each candidate ({', '.join(code(c) for c in sets['spline_candidates'])}) "
        "in turn gets a natural spline; a candidate whose spline lowers BIC is kept. The final design has all kept splines "
        f"at once. Across the {len(fi)} fits in the grid, a spline was chosen {int((spl != '').sum())} times, for "
        f"{', '.join(code(c) for c in spl_who)}, so straight lines are adequate.")
    add(snippet(P1.choose_splines, "Choosing splines by BIC"))
    add(snippet(M.bic, "BIC"))
    add(r"\subsection{Stepwise selection by BIC}")
    add(r"""Selection works on \emph{blocks}: each numeric predictor with its spline columns, each binary column, and each
categorical variable with all its dummies. \textbf{Forward} starts from the intercept and at each step adds the block
that lowers BIC most; \textbf{backward} starts from all blocks and removes the one whose removal lowers BIC most;
\textbf{both ways} starts from the intercept and considers every addition and every removal at each step. Each stops
when no single move lowers BIC. Each candidate model is a logistic regression fitted as above; the final model is
refitted on the training fold and predicts the held-out pairs. Stepwise runs only under the headline and complete-case
specifications (it is the slowest model).""")
    add(snippet(M.stepwise, "Stepwise selection over blocks"))
    add(snippet(M.Prep.blocks, "The blocks stepwise selects"))
    add(r"\subsection{Linear discriminant analysis (LDA)}")
    add(r"""LDA models the predictors within each class (match, no match) as multivariate normal with class means
$\mu_1, \mu_0$ and a \emph{shared} covariance matrix $\Sigma$, estimated from the training fold; the class priors
$\pi_1, \pi_0$ are the training class frequencies. By Bayes' rule
\[
P(\text{match}\mid x) = \frac{\pi_1 \phi(x;\mu_1,\Sigma)}{\pi_1 \phi(x;\mu_1,\Sigma) + \pi_0 \phi(x;\mu_0,\Sigma)},
\]
which is a logistic function of a linear score, so LDA is a linear classifier fitted differently from logistic
regression. We use scikit-learn's \texttt{LinearDiscriminantAnalysis} with its defaults (singular value
decomposition solver); there is nothing to tune.""")
    add(r"\subsection{Quadratic discriminant analysis (QDA)}")
    add(r"""QDA lets each class have its own covariance matrix $\Sigma_k$, so the boundary is quadratic. With more than 100
columns and 550 or so matches per training fold, the match class's covariance is poorly estimated, so it is
regularized towards the identity: scikit-learn's \texttt{reg\_param} $r$ replaces each class covariance (in its scaled
form) by $(1-r)\Sigma_k + r I$. $r$ is tuned over """ + "$\\{" + ", ".join(f"{v:g}" for v in P1.QDA_REGS) + r"\}$" + r""" by the
inner folds (Section~\ref{sec:cv}). The strongest smoothing (@@QDAMAXV@@) was chosen in @@QDAMAX@@ of @@QDAN@@ fits, typical of a weak
signal.""")
    add(snippet(P1.make_qda, "QDA"))
    add(r"\subsection{Random forest}")
    add(f"An average of {P1.RF_TREES} classification trees, each grown on a bootstrap sample of the training fold; at each split "
        "a random $\\sqrt{p}$ of the $p$ columns are candidates (Gini impurity). The predicted probability is the average over "
        "trees of the share of matches in the leaf the pair falls into. The minimum leaf size is tuned over "
        "$\\{" + ", ".join(map(str, P1.RF_LEAVES)) + "\\}$ by the inner folds. Seeds: \\texttt{random\\_state} = "
        f"{P1.RF_SEED} + fold. Predictions are computed on one thread so they reproduce bit for bit.")
    add(snippet(P1.make_rf, "Random forest"))
    add(snippet(P1.predict_serial, "Predictions on one thread"))
    add(r"\subsection{Lasso (L1-penalized logistic regression)}")
    add(r"""The lasso minimizes $\|\beta\|_1 + C \sum_i \text{log-loss}_i$ (scikit-learn \texttt{LogisticRegression} with
\texttt{l1\_ratio=1}, solver \texttt{liblinear}, \texttt{intercept\_scaling=100} so the intercept is effectively
unpenalized). Small $C$ sets many coefficients exactly to zero. The grid has """ + f"{P1.LASSO_STEPS}" + r""" values of $C$,
log-spaced from $C_{\min} = 1/\max_j |\sum_i D_{ij}(y_i - \bar y)|$ (the smallest penalty that zeroes every
coefficient) to $1000\,C_{\min}$, tuned by the inner folds. The lasso was added after P1 and P2 had been tested, so it
is reported but kept out of the ``best model'' they compare. Seeds: \texttt{random\_state} = """ + f"{P1.LASSO_SEED}" + r""" + fold.""")
    add(snippet(P1.lasso_grid, "The lasso penalty grid"))
    add(snippet(P1.make_lasso, "Lasso"))
    add(r"\subsection{The yes model (P5)}")
    add(r"""One row per date from one side, separately for women and men, on dates where the rater gave all six attribute
ratings. Each rating is standardized within the rater's gender, and a logistic regression of the rater's yes on the
six ratings is fitted on all those dates (no cross-validation; the coefficients are the result). Revealed importance
of attribute $a$ is its share of the positive standardized coefficients,
\[
\text{revealed}_a = 100 \cdot \frac{\max(\hat\beta_a, 0)}{\sum_b \max(\hat\beta_b, 0)},
\]
and stated importance is the mean own-preference share (\texttt{<a>1\_1\_share}) over participants of that gender in
the 100-point waves who gave all six shares. The gap is revealed minus stated. Waves 6--9's stated shares are
reported separately, without a verdict. Error bars: the cluster sandwich below, with events as clusters.""")
    add(snippet(P1.yes_design, "The yes-model design"))
    add(snippet(P1.yes_model, "The yes model and stated-versus-revealed table"))
    add(r"\subsection{The liking model (L1, L2)}")
    add(r"""One row per date from one side, on dates with a liking rating (1--10). Predictors: the rater's and the partner's
sign-up answers (the \texttt{pre\_individual} columns relabelled rater \texttt{r\_} and partner \texttt{p\_}), the
rater's gender, and the eleven pair features of P3b, with the compatibility terms relabelled as fit to the rater's
and to the partner's priorities. Preparation is the same as for the match models (no splines). The model is ordinary
least squares, fitted in each training fold under the same event folds. Ladder: a baseline that predicts each
held-out date by its training fold's mean liking; rater only; plus partner; plus pair features. A complete-case
version uses only dates complete on every predictor.""")
    add(snippet(LM.dates_table, "One row per date from one side"))
    add(snippet(LM.fit_fold, "The liking model in one fold"))

    # ------------------------------------------------------------------ 8 metrics
    add(r"\section{Metrics and error bars}\label{sec:metrics}")
    add(r"\subsection{The metrics}")
    add(r"""With $\hat y_i = 1$ when $\hat p_i$ exceeds its fold's threshold, $c_i = 1$ when $\hat y_i = y_i$, $n$ pairs,
$n_1$ matches and $n_0$ non-matches:
\begin{align*}
\text{error rate} &= \tfrac1n \sum_i (1 - c_i), &
\text{sensitivity} &= \tfrac{1}{n_1}\sum_{i: y_i = 1} c_i, \\
\text{specificity} &= \tfrac{1}{n_0}\sum_{i: y_i = 0} c_i, &
\text{balanced accuracy} &= \tfrac12(\text{sensitivity} + \text{specificity}).
\end{align*}
Since 83.5\% of pairs do not match, the error rate alone rewards predicting ``no'' for everyone; balanced accuracy
and AUC do not.

\textbf{AUC} is the probability that a randomly chosen match gets a higher predicted probability than a randomly
chosen non-match (ties count half), computed from ranks (the Mann--Whitney statistic): with $R_i$ the rank of
$\hat p_i$ among all $n$ pairs,
\[
\text{AUC} = \frac{\sum_{i: y_i=1} R_i - n_1(n_1+1)/2}{n_1 n_0}.
\]
0.5 is chance and 1 is perfect. It uses the predicted probabilities directly, so it needs no threshold.""")
    add(snippet(M.auc, "AUC from ranks"))
    add(snippet(M.metric_table, "The metric table with its error bars"))
    add(r"\subsection{Cluster-robust error bars}")
    add(r"""Pairs are not independent: the same people appear in many pairs, and pairs at an event share an evening. All
error bars treat the """ + f"$G = {n_events}$" + r""" events as clusters. A metric written as a mean of per-pair contributions
$u_i$ has variance
\[
\widehat{\operatorname{Var}}(\bar u) = \frac{G}{G-1}\cdot\frac{1}{n^2}\sum_{g=1}^{G}\Big(\sum_{i \in g}(u_i - \bar u)\Big)^2
\]
(convention 6; \texttt{modeling.cluster\_robust\_var\_of\_mean}). The error rate is a plain mean ($u_i = 1 - c_i$).
Sensitivity, specificity and balanced accuracy are ratios, so they use linearized contributions: for sensitivity a
matched pair contributes $\widehat{\text{Se}} + (c_i - \widehat{\text{Se}})\,n/n_1$ and an unmatched pair
$\widehat{\text{Se}}$; specificity likewise with $n_0$; balanced accuracy averages the two residuals.

\textbf{AUC} uses Obuchowski's (1997) clustered variance. Each match $i$ gets the placement value $V_{10,i}$, the share
of non-matches it outranks, and each non-match $j$ gets $V_{01,j}$, the share of matches that outrank it (ties half);
the AUC $A$ is the mean of either. With cluster sums $a_g = \sum_{i \in g, y_i=1}(V_{10,i} - A)$ and
$b_g = \sum_{j \in g, y_j=0}(V_{01,j} - A)$,
\[
\widehat{\operatorname{Var}}(\text{AUC}) = \frac{G}{G-1}\sum_{g}\Big(\frac{a_g}{n_1} + \frac{b_g}{n_0}\Big)^2 .
\]
\textbf{Liking R-squared} $= 1 - A/B$, with $A$ the mean squared error and $B$ the mean squared deviation of liking
from its mean, uses the linearized contribution $-(e_i - A)/B + A(d_i - B)/B^2$ ($e_i$ the squared error, $d_i$ the
squared deviation); RMSE $= \sqrt A$ uses the delta method.

\textbf{Coefficients} (yes model and coefficient table) use the cluster sandwich
$\hat V = B\,\big(\tfrac{G}{G-1}\sum_g s_g s_g^\top\big)\,B$ with $B = (X^\top W X + \Lambda)^{-1}$ and $s_g$ the sum of
the score contributions $x_i(y_i - \hat p_i)$ over cluster $g$.

Every one of these error bars was checked by simulation before being reported (Section~\ref{sec:sims}).""")
    add(snippet(M.cluster_robust_var_of_mean, "Cluster-robust variance of a mean (convention 6)"))
    add(snippet(M.placement_values, "Placement values"))
    add(snippet(M.auc_clustered_var, "Clustered AUC variance (Obuchowski 1997)"))
    add(snippet(M.r_squared_and_rmse, "Liking R-squared and RMSE with their error bars"))

    # ------------------------------------------------------------------ 9 hypotheses
    add(r"\section{How each hypothesis is decided}")
    add("The ``best model'' for P1 and P2 is the highest AUC among the pre-registered pool: full logistic regression, the "
        "three stepwise directions, LDA, QDA and the random forest (the lasso is excluded, and stepwise is not run in the "
        "wave checks). All AUCs come from event cross-validation under the stated specification.")
    rows = [["P1", "Best AUC on \\texttt{pre}", "between 0.58 and 0.70"],
            ["P2", "Best AUC on \\texttt{pre+date} minus best on \\texttt{pre}", "at least 0.10"],
            ["P3", "Full logistic AUC on \\texttt{pre} minus on \\texttt{pre\\_individual}", "less than 0.02"],
            ["P3b", "Full logistic AUC on \\texttt{pre\\_extended} minus on \\texttt{pre}", "less than 0.02"],
            ["P3c", "Full logistic AUC on \\texttt{pre\\_interactions} minus on \\texttt{pre\\_extended}", "less than 0.02"],
            ["P4", "Random forest AUC on \\texttt{pre} minus full logistic on \\texttt{pre}", "less than 0.03"],
            ["P5", "Attractiveness has the largest standardized coefficient and the largest gap (revealed minus stated), for each gender", "both true"],
            ["L1", "Held-out R-squared of the full liking model", "less than 0.10"],
            ["L2", "Full liking model minus rater + partner, in held-out R-squared", "less than 0.01"]]
    add(table(["", "Statistic", "Supported when"], rows, "lL{10cm}L{3.6cm}"))
    pick = lambda h, v: hyp[(hyp.hypothesis == h) & (hyp.version == v)].iloc[0]
    rows = []
    for h, v in [("P1", "headline"), ("P2", "headline"), ("P3", "headline"), ("P3b", "headline"), ("P3c", "headline"),
                 ("P4", "headline"), ("P5", "women"), ("P5", "men"), ("L1", "headline"), ("L2", "headline")]:
        r = pick(h, v)
        rows.append([h, tx(v), tx(r.statistic), "supported" if str(r.supported) == "True" else "not supported"])
    add(r"\textbf{Current results} (read from the hypothesis tables):")
    add(table(["", "Version", "Statistic", "Verdict"], rows, "llL{9.6cm}l", size=r"\footnotesize"))

    # ------------------------------------------------------------------ 10 robustness
    add(r"\section{Robustness and sensitivity checks}\label{sec:robust}")
    rows = [["Missing values", "median fill plus all flags (headline); median fill with sign-up flags only; complete cases (dyads complete on every \\texttt{pre+date} column, no filling, no flags)"],
            ["Preference fit", "compatibility term B (headline); plain weighted mean A"],
            ["Interest values", "gaming 14 and reading 13 missing (headline); as recorded, with the released \\texttt{int\\_corr}"],
            ["Survey format", "shares standardized within format (headline); unstandardized"]]
    add(r"\textbf{The 24-specification grid.} Every combination of these choices ($3 \times 2 \times 2 \times 2$) is run for every model "
        r"(stepwise only under the headline and complete-case ones) and reported in one table; the headline was fixed in "
        r"\texttt{PREREG.md} before any model ran.")
    add(table(["Decision", "Choices"], rows, "L{3.2cm}L{12cm}"))
    add(r"""\textbf{Complete cases.} Every results table has a complete-case version: only the dyads with every column of
the feature set recorded (@@CCD@@ dyads, @@CCM@@ matches), nothing filled, no flags. The pair-feature ladder's complete-case
version uses the dyads complete on every column any step uses, so all steps share the same dyads.

\textbf{Wave checks.} P1 to P4 are recomputed without waves 6--9 (the 1--10 survey format) and without wave 12 (yes
limited to half one's dates), every model except stepwise, in three versions: the remaining events dealt into 5 new
folds by the same seed (the pre-registered version), the headline folds minus the dropped waves, and complete cases.
P5 is also recomputed without wave 12.

\textbf{Fold and seed noise} (exploratory). The headline full logistic model is rerun with 10 other fold seeds and the
headline forest with 5 other forest seeds, to show how much the AUCs move for reasons unrelated to the data.""")

    # ------------------------------------------------------------------ 11 diagnostics
    add(r"\section{Diagnostics}")
    add(r"""\begin{itemize}
\item \textbf{Coefficient table} (full logistic model on all dyads, headline and complete-case). Identical columns are
reported as one combined term. A coefficient that moves by more than 0.01 when the ridge goes from $10^{-4}$ to
$10^{-8}$ has no finite estimate (separation: some combination of columns occurs only with non-matches) and is reported
as not estimable, with the sum of the separated terms where that sum is stable. SEs are the cluster sandwich; a
bootstrap shows they are about 20\% too narrow on this design, so read the z-values with care.
\item \textbf{Variance inflation factors}: $1/(1 - R_j^2)$ from regressing each column on all the others; an exactly
reproduced column is reported as infinite.
\item \textbf{Calibration}: held-out predictions in ten equal-sized groups by predicted probability, mean prediction
against observed match rate.
\item \textbf{Binned residuals}: for each continuous predictor, held-out residuals (observed minus predicted) averaged
in ten bins, with $\pm 2$ SE bands, to spot curvature.
\item \textbf{Cook's distance and leverage} of the full-data fit, the ten most influential dyads listed.
\end{itemize}""")

    # ------------------------------------------------------------------ 12 sims
    add(r"\section{Simulation checks of the error bars}\label{sec:sims}")
    add("Each error-bar formula was checked on simulated data with the real design before being reported (convention 7). "
        "The ratio of the average SE to the true spread of the estimate should be near 1, and 90\\% intervals should cover "
        "the true value about 90\\% of the time.")
    cl = s_auc[s_auc.se_type == "clustered"]
    fmt_r = lambda d: f"{d.ratio_mean_se_to_sd.min():.2f} to {d.ratio_mean_se_to_sd.max():.2f}"
    fmt_c = lambda d: f"{d.coverage_90.min():.0%} to {d.coverage_90.max():.0%}".replace("%", r"\%")
    rows = [["AUC (match models)", fmt_r(cl[cl.metric == 'auc']), fmt_c(cl[cl.metric == 'auc'])],
            ["Sensitivity, specificity, balanced accuracy", fmt_r(cl[cl.metric != 'auc']), fmt_c(cl[cl.metric != 'auc'])],
            ["Liking R-squared", fmt_r(s_like[s_like.metric == 'r_squared']), fmt_c(s_like[s_like.metric == 'r_squared'])],
            ["Liking RMSE", fmt_r(s_like[s_like.metric == 'rmse']), fmt_c(s_like[s_like.metric == 'rmse'])]]
    ys, cs = s_coef[s_coef.kind == "yes"], s_coef[s_coef.kind == "coefficients"]
    rows.append(["Yes-model coefficients", f"{ys.ratio_min.min():.2f} to {ys.ratio_max.max():.2f}",
                 (f"{ys.coverage_min.min():.0%} to {ys.coverage_max.max():.0%}").replace("%", r"\%")])
    rows.append(["Coefficient table", f"median {cs.median_ratio.median():.2f}", (f"median {cs.median_coverage.median():.0%}").replace("%", r"\%")])
    add(table(["Error bar", "SE / true spread", "90\\% coverage"], rows, "L{7cm}ll"))
    add(r"""Designs: for the match metrics, the 21 real event sizes with an event effect on the match probability (SD 0.6, and
0.4 to match the real spread of match rates across events) and a score correlated within events, five seeds each
(\texttt{sim\_auc\_cluster.py}); for liking, the real event sizes with event, rater and partner effects
(\texttt{sim\_like\_cluster.py}); for coefficients, a parametric bootstrap on the real designs with an event effect of
SD 0.4 (\texttt{sim\_coef\_cluster.py}). With 21 events the match-model error bars run slightly narrow: read a 90\%
interval as roughly an 83\% one.""")

    # ------------------------------------------------------------------ 13 reproduce
    add(r"\section{Reproducing the results}")
    add(r"""\begin{enumerate}
\item Create the environment: \texttt{python3 -m venv .venv}, then \texttt{.venv/bin/pip install -r requirements.txt}.
\item Run the scripts in the order of Section 1 from the repository root. \texttt{part1\_models.py} takes about 30
minutes with \texttt{-{}-jobs 5}; set \texttt{OMP\_NUM\_THREADS=1} so parallel folds do not compete for threads.
\item Run the tests: \texttt{for t in tests/test\_*.py; do .venv/bin/python \$t; done}.
\item Regenerate this document: \texttt{python scripts/methods\_doc.py}.
\end{enumerate}
The same code and seeds reproduce every number to the last digit (checked by two independent reviews).""")

    # ------------------------------------------------------------------ appendices
    add(r"\appendix")
    add(r"\section{Every predictor}")
    add(r"Every column of \texttt{pre} and \texttt{pre+date}, as listed in \texttt{data/feature\_sets.json}, and the four missing-answer flags, with the number of dyads where each is missing "
        r"before filling (of " + f"{n_dyads:,}" + r"). Columns marked * are left out of every model (redundant shares).")
    rows = []
    for c in sets["pre"] + sets["date"] + sets["flags_pre"] + sets["flags_date"]:
        d, scale = describe(c)
        star = "*" if c in sets["pref_reference"] else ""
        rows.append([code(c) + star, tx(d), tx(scale), str(int(miss.get(c, 0)))])
    add(table(["Column", "Meaning", "Scale", "Missing"], rows, "L{3.3cm}L{8.2cm}L{2.4cm}r", long=True, size=r"\footnotesize"))
    add(r"\section{Seeds}")
    rows = [["Outer event folds", str(M.FOLD_SEED)],
            ["Inner tuning folds", f"{M.FOLD_SEED} + 1 + outer fold"],
            ["Random forest", f"{P1.RF_SEED} + outer fold"],
            ["Lasso", f"{P1.LASSO_SEED} + outer fold"],
            ["AUC simulation", "20261006 to 20261015 (five per design)"],
            ["Liking simulation", "20261506 to 20261515"],
            ["Coefficient bootstrap", "20261406 + 1000 x design + replicate"],
            ["Fold and seed noise", "fold seeds 20261007 to 20261016; forest seeds 20261107 to 20261111"]]
    add(table(["Random operation", "Seed"], rows, "L{5cm}L{9cm}"))
    add(r"\section{More code}")
    add("The rest of the code that decides the reported results, as it ran.")
    add(snippet(P1.hypotheses, "How P1 to P4 are decided"))
    add(snippet(P1.p5_rows, "How P5 is decided"))
    add(snippet(P1.twin_groups, "Identical columns"))
    add(snippet(P1.coefficient_table, "The coefficient table"))
    add(snippet(P1.vif, "Variance inflation factors"))
    add(snippet(P1.calibration, "Calibration by decile"))
    add(r"\end{document}")

    text = "\n\n".join(L)
    for k, v in tokens.items():
        text = text.replace(k, v)
    assert "@@" not in text, "unfilled token"
    OUT.write_text(text)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", OUT.name], cwd=RES, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:]); raise SystemExit("pdflatex failed")
    for ext in ("aux", "log", "out", "toc"):
        (RES / f"methods_doc.{ext}").unlink(missing_ok=True)
    print(f"wrote {OUT} and {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
