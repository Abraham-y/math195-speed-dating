"""Figures for the written report and the slides, drawn from the results tables (nothing typed by hand).

Input : results/part2_models_metrics.csv, results/part1_models_stepwise_selection.csv,
        results/part1_models_coefficients.csv, data/pairs.csv, results/sim_cv_repair_coverage.csv, results/sim_cv_sizes_coverage.csv,
        results/sim_cv_kfold_summary.csv, results/sim_cv_auc_summary.csv, results/sim_cv_auc_components.csv,
        results/errorbars_real.csv, results/errorbars_final.csv
Output: results/report_figures_<name>.png (300 dpi, 6.5 inches wide), copied to report/figures/<name>.png for Overleaf:
  auc_by_model    held-out AUC of every headline model, pre and pre+date, with 90% intervals (event 5-fold)
  auc_part1       the same for the pre-registered (Part 1) models only, matching the report's Results text
  events          dyads and match rate per event (data/pairs.csv)
  cv_design       which events each of the five headline rounds holds out (modeling.event_folds, seed 20261006)
  stepwise_kept   how often stepwise selection kept each pre-date predictor, coloured by the coefficient's direction
  qda_shrink      correlations among 8 predictors in the matched pairs, before and after QDA's shrinkage (r = 0.9)
  eq_bic, eq_shrink  the BIC and shrinkage equations as images (equations())
  leakage         event against random-row CV AUC for the same models (the cost of letting people repeat)
  coverage        simulated coverage of 90% intervals for the new-event error, by method (two grids)
  nb_theory       Proposition 8's predicted Nadeau-Bengio ratio against the observed one, by fold count
  auc_bias        Proposition 9: bias of the pooled CV AUC and of BCWF-pooled against G, with the prediction
  real_intervals  90% intervals for the full logistic model's error rate on the real data, by method
Colors: the first three slots of the dataviz skill's validated categorical palette (blue, orange, aqua), with
direct labels for aqua (below 3:1 contrast). Nothing is random.
With --paper: untitled versions for LaTeX (the caption carries the message), named paper_<name>.png.
With --slides: the same figures for the deck (fonts 1.5x, no titles or footnotes; the slide carries them), named
slide_<name>.png.
Run from the repository root: python scripts/report_figures.py [--paper | --slides]
"""
import sys, pathlib, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import stats

RES = pathlib.Path("results")
OUT = pathlib.Path("report/figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
T20 = stats.t.ppf(0.95, 20)
NAMES = {"logistic": "Logistic", "lda": "LDA", "qda": "QDA (regularized)", "rf": "Random forest", "lasso": "L1 logistic",
         "stepwise_forward": "Stepwise (forward)", "stepwise_backward": "Stepwise (backward)", "stepwise_both": "Stepwise (both)",
         "tree": "Pruned tree", "bagging": "Bagging", "boosting": "Boosting", "ensemble_mean": "Mean ensemble",
         "ensemble_stacked": "Stacked ensemble", "two_decision_logistic": "Two-decision (logistic)", "two_decision_rf": "Two-decision (forest)"}

PAPER = "--paper" in __import__("sys").argv
SLIDE = "--slides" in __import__("sys").argv or PAPER   # both drop titles and footnotes
SC = 1.5 if SLIDE and not PAPER else 1.0
F = lambda v: v * SC
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": F(9), "axes.edgecolor": REF, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": F(10), "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 100})


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def save(fig, name):
    name = f"paper_{name}" if PAPER else (f"slide_{name}" if SLIDE else name)
    fig.savefig(RES / f"report_figures_{name}.png", dpi=200 if SLIDE and not PAPER else 300, bbox_inches="tight", facecolor="white")
    OUT.mkdir(exist_ok=True); shutil.copy(RES / f"report_figures_{name}.png", OUT / f"{name}.png")       # for Overleaf / the deck
    plt.close(fig); print(f"wrote results/report_figures_{name}.png (copy: report/figures/{name}.png)")


def headline_match(scheme):
    d = pd.read_csv(RES / "part2_models_metrics.csv")
    return d[(d.target == "match") & (d.headline == True) & (d.scheme == scheme)]


def auc_by_model():
    d = headline_match("event")
    order = d[d.feature_set == "pre"].sort_values("auc").model.tolist()
    order += [m for m in d[d.feature_set == "pre+date"].sort_values("auc").model if m not in order]
    y = {m: i for i, m in enumerate(order)}
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for fs, col, off, mk in (("pre", BLUE, 0.17, "o"), ("pre+date", ORANGE, -0.17, "s")):
        x = d[d.feature_set == fs]
        yy = np.array([y[m] for m in x.model]) + off
        ax.errorbar(x.auc, yy, xerr=T20 * x.auc_se, fmt=mk, color=col, ms=4.5, elinewidth=1.4, capsize=0, label=f"{fs}")
    ax.axvline(0.5, color=REF, linewidth=1, linestyle="--"); ax.text(0.502, len(order) - 0.4, "chance", color=INK2, fontsize=F(8), va="top")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES.get(m, m) for m in order])
    ax.set_xlabel("Held-out AUC (event 5-fold CV), with 90% interval"); ax.set_xlim(0.45, 0.9)
    best = d.sort_values("auc").groupby("feature_set").tail(1).set_index("feature_set")
    for fs, col in (("pre", BLUE), ("pre+date", ORANGE)):
        r = best.loc[fs]; ax.annotate(f"{r.auc:.3f}", (r.auc + T20 * r.auc_se + 0.005, y[r.model] + (0.17 if fs == "pre" else -0.17)),
                                      color=INK, fontsize=F(8), va="center")
    style(ax); ax.legend(loc="lower right", title="Feature set", title_fontsize=F(8))
    ax.set_title("" if SLIDE else "Before the date every model is near chance; the four-minute date adds most of the signal")
    if not SLIDE: fig.text(0, -0.02, "4,184 dyads, 21 events. Intervals: AUC plus or minus t(20) times the event-clustered (Obuchowski) SE. "
             "Source: results/part2_models_metrics.csv.", fontsize=F(7), color=INK2)
    save(fig, "auc_by_model")


def auc_part1():
    """The pre-registered (Part 1) models only, from the Part 1 table, so the figure matches the report's Results text."""
    m = pd.read_csv(RES / "part1_models_metrics.csv")
    d = m[(m.headline == True) & (m.fold_scheme == "event") & (m.feature_set.isin(["pre", "pre+date"])) & (m.model != "stepwise_both")
          & (m.spec == "missing=median_flags; fit=B; out_of_range=rule; format=standardized")]
    order = d[d.feature_set == "pre"].sort_values("auc").model.tolist(); y = {k: i for i, k in enumerate(order)}
    names = dict(NAMES, stepwise_forward="Stepwise (forward, both)")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for fs, col, off, mk in (("pre", BLUE, 0.15, "o"), ("pre+date", ORANGE, -0.15, "s")):
        x = d[d.feature_set == fs]
        ax.errorbar(x.auc, np.array([y[k] for k in x.model]) + off, xerr=T20 * x.auc_se, fmt=mk, color=col, ms=4.5, elinewidth=1.4,
                    capsize=0, label=fs)
    ax.axvline(0.5, color=REF, linewidth=1, linestyle="--"); ax.text(0.503, -0.62, "chance", color=INK2, fontsize=F(8), va="bottom")
    ax.set_ylim(-0.7, len(order) - 0.45)                                   # room for the "chance" label under the bottom row
    for fs, off in (("pre", 0.15), ("pre+date", -0.15)):
        r = d[d.feature_set == fs].sort_values("auc").iloc[-1]
        ax.annotate(f"{r.auc:.3f}", (r.auc + T20 * r.auc_se + 0.005, y[r.model] + off), color=INK, fontsize=F(8), va="center")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names.get(k, k) for k in order]); ax.set_xlim(0.45, 0.9)
    ax.set_xlabel("Held-out AUC (event 5-fold CV), with 90% interval"); style(ax)
    ax.legend(loc="center", bbox_to_anchor=(0.53, 0.5), title="Feature set", title_fontsize=F(8))   # in the empty band between the two columns
    ax.set_title("" if SLIDE else "Before the date every model is near chance; the four-minute date adds most of the signal")
    if not SLIDE: fig.text(0, -0.03, "Pre-registered models; 4,184 dyads, 21 events. Intervals: AUC plus or minus t(20) times the event-clustered SE. "
             "Source: results/part1_models_metrics.csv.", fontsize=F(7), color=INK2)
    save(fig, "auc_part1")


def events():
    """Event sizes and match rates: why the folds hold out whole events."""
    p = pd.read_csv("data/pairs.csv"); g = p.groupby("event").agg(n=("match", "size"), rate=("match", "mean")).sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharex=True)
    ax = axes[0]; ax.bar(g.index, g.n, color=BLUE, width=0.7); style(ax, "y")
    ax.set_title("(a) Dyads per event", fontsize=F(8.5)); ax.set_xlabel("Event (wave)"); ax.set_xticks([1, 5, 10, 15, 21])
    ax = axes[1]; ax.bar(g.index, 100 * g.rate, color=BLUE, width=0.7); ax.axhline(100 * p.match.mean(), color=REF, linestyle="--", linewidth=1)
    ax.set_title(f"(b) Match rate per event (%); dashed: all, {100 * p.match.mean():.1f}%", fontsize=F(8.5)); ax.set_xlabel("Event (wave)"); style(ax, "y")
    if not SLIDE: fig.suptitle("Events differ in size and in how often people match", x=0.01, ha="left", fontsize=F(10), fontweight="bold", color=INK, y=1.04)
    if not SLIDE: fig.text(0.01, -0.06, f"{len(p):,} dyads in {len(g)} events; match rates range from {100 * g.rate.min():.0f}% to {100 * g.rate.max():.0f}%. "
             "Source: data/pairs.csv.", fontsize=F(7), color=INK2)
    fig.tight_layout(); save(fig, "events")


def cv_design():
    """The headline event 5-fold design: which events are held out in each round (modeling.event_folds, seed 20261006)."""
    sys.path.insert(0, "scripts"); import modeling as M
    p = pd.read_csv("data/pairs.csv"); n = p.groupby("event").size()
    folds = M.event_folds(p.event); order = sorted(folds, key=lambda e: (folds[e], e))
    fig, ax = plt.subplots(figsize=(6.5, 2.5) if not SLIDE or PAPER else (8.0, 2.6))
    for k in range(5):
        y = 4 - k
        for j, e in enumerate(order):
            out = folds[e] == k
            ax.add_patch(plt.Rectangle((j + 0.06, y + 0.12), 0.88, 0.76, color=ORANGE if out else BLUE, alpha=1 if out else 0.35, linewidth=0))
        held = [e for e in order if folds[e] == k]
        ax.text(len(order) + 0.3, y + 0.5, f"{n[held].sum():,} dyads", va="center", fontsize=F(7), color=INK2)
    for j, e in enumerate(order):
        ax.text(j + 0.5, 5.15, str(e), ha="center", va="bottom", fontsize=F(6) if not SLIDE or PAPER else 8.5, color=INK2)
    edges = [j for j in range(1, len(order)) if folds[order[j]] != folds[order[j - 1]]]
    for j in edges:
        ax.plot([j, j], [0.05, 5.0], color="white", linewidth=2.5)
    ax.text(-0.3, 5.15, "Event:", ha="right", va="bottom", fontsize=F(6) if not SLIDE or PAPER else 8.5, color=INK2)
    ax.set_yticks([4.5 - k for k in range(5)]); ax.set_yticklabels([f"Round {k + 1}" for k in range(5)])
    ax.set_xlim(0, len(order)); ax.set_ylim(0, 5.7); ax.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.35), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["training set: the other events (tuning uses inner folds of these)", "test set: held-out events (scored)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=F(7), handlelength=1.2)
    if not SLIDE: ax.set_title("Each round holds out whole events; every event is held out once", pad=F(16))
    fig.tight_layout(); save(fig, "cv_design")


HEADLINE_SPEC = "missing=median_flags; fit=B; out_of_range=rule; format=standardized"
STEP_LABELS = {"m_self_fun": "His self-rated fun", "abs_age_diff": "Age gap", "m_date": "How often he dates",
               "same_field": "Same field of study", "m_pref_intel": "His weight on intelligence",
               "f_clubbing": "Her interest in clubbing", "f_art": "Her interest in art", "f_pref_fun": "Her weight on fun"}
REVERSED = {"f_date", "m_date", "f_go_out", "m_go_out"}   # coded 1 = most often, so "more often" flips the sign


def stepwise_kept(top=8):
    """How often stepwise selection kept each pre-date predictor (3 directions x 5 headline folds = 15 fits), coloured by
    the direction of its coefficient in the full logistic model (results/part1_models_coefficients.csv, all dyads)."""
    s = pd.read_csv(RES / "part1_models_stepwise_selection.csv"); s = s[(s.spec == HEADLINE_SPEC) & (s.feature_set == "pre")]
    tot = s.groupby("block").folds_selected.sum().sort_values(ascending=False, kind="stable").head(top)
    assert set(tot.index) <= set(STEP_LABELS), f"label missing for {set(tot.index) - set(STEP_LABELS)}"
    c = pd.read_csv(RES / "part1_models_coefficients.csv"); c = c[(c.spec == HEADLINE_SPEC) & (c.feature_set == "pre")].set_index("term").coef
    up = {b: (np.sign(c[b] if b in c.index else c["bin:" + b]) * (-1 if b in REVERSED else 1)) > 0 for b in tot.index}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(tot))[::-1]
    ax.barh(y, tot.values, color=[BLUE if up[b] else ORANGE for b in tot.index], height=0.62)
    for yi, (b, v) in zip(y, tot.items()):
        ax.text(v + 0.2, yi, f"{int(v)} / 15", va="center", fontsize=F(7.5), color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([STEP_LABELS[b] for b in tot.index]); ax.set_xlim(0, 16.5); ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Fits that kept it, of 15 (3 directions × 5 folds)"); style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["more of it → more matches", "more of it → fewer matches"], loc="upper center", bbox_to_anchor=(0.32, -0.2), ncol=2, fontsize=F(7), handlelength=1)
    if not SLIDE: ax.set_title("Predictors stepwise selection kept most often (before the date)")
    fig.tight_layout(); save(fig, "stepwise_kept")
    print(pd.DataFrame({"kept_of_15": tot, "more_means_more_matches": pd.Series(up)}))


QDA_VARS = [("f_age", "Her age"), ("m_age", "His age"), ("f_date", "Her dating"), ("f_go_out", "Her going out"),
            ("m_date", "His dating"), ("m_go_out", "His going out"), ("f_self_attr", "Her looks (self)"), ("f_self_fun", "Her fun (self)")]


def qda_shrink(r=0.9):
    """What regularized QDA does to the matches' covariance: correlations among 8 standardized pre-date predictors in the
    matched pairs, as estimated and after (1 - r) S + r I with the r chosen in every fold (0.9). Illustration on 8 of the
    roughly 125 columns; standardized over all pairs, missing values filled with medians, as in the models."""
    import matplotlib.colors as mcolors
    p = pd.read_csv("data/pairs.csv"); X = p[[v for v, _ in QDA_VARS]].astype(float)
    X = X.fillna(X.median()); Z = (X - X.mean()) / X.std(ddof=0)
    S = np.cov(Z[p.match == 1].to_numpy().T, ddof=1); St = (1 - r) * S + r * np.eye(len(S))
    corr = lambda M: M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
    cmap = mcolors.LinearSegmentedColormap.from_list("div", [ORANGE, "#ffffff", BLUE])
    labels = [l for _, l in QDA_VARS]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
    for ax, M, title in ((axes[0], corr(S), f"Estimated from the {int(p.match.sum())} matches"), (axes[1], corr(St), f"Used by QDA after shrinking (r = {r})")):
        ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
        for i in range(len(M)):
            for j in range(len(M)):
                if i != j: ax.text(j, i, f"{(0.0 if abs(M[i, j]) < 0.005 else M[i, j]):.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=F(5.2), color=INK)
        ax.set_xticks(range(len(M))); ax.set_yticks(range(len(M)))
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=F(5.6)); ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=F(5.6))
        ax.set_title(title, fontsize=F(7), loc="center", color=INK, fontweight="bold"); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); save(fig, "qda_shrink")
    print("matches' correlations, estimated:\n", pd.DataFrame(corr(S), index=labels, columns=labels).round(2).to_string())


WOMEN, MEN = "#E0607E", "#2a78d6"     # the deck's pink for women; the validated blue for men


def eda_who():
    """Exploratory: yes rates by gender (data/pairs.csv) and stated preferences by gender (the 100-point waves,
    results/part1_models_stated_vs_revealed.csv)."""
    p = pd.read_csv("data/pairs.csv")
    sr = pd.read_csv(RES / "part1_models_stated_vs_revealed.csv").set_index(["rater_gender", "attribute"]).stated_share
    traits = [("attr", "Attractive"), ("sinc", "Sincere"), ("intel", "Intelligent"), ("fun", "Fun"), ("amb", "Ambitious"), ("shar", "Shared interests")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.9]})
    rates = [100 * p.f_dec.mean(), 100 * p.m_dec.mean()]
    a.bar([0, 1], rates, color=[WOMEN, MEN], width=0.6)
    for x, v in zip([0, 1], rates):
        a.text(x, v + 1.2, f"{v:.0f}%", ha="center", fontsize=F(9), color=INK, fontweight="bold")
    a.set_xticks([0, 1]); a.set_xticklabels(["Women", "Men"]); a.set_ylim(0, 60); a.set_ylabel("Said yes to a partner (%)")
    a.set_title("Women said yes less often", fontsize=F(8.5)); style(a, "y")
    y = np.arange(len(traits))[::-1]
    for off, g, col, lab in ((0.2, "women", WOMEN, "Women"), (-0.2, "men", MEN, "Men")):
        vals = [sr[(g, t)] for t, _ in traits]
        b.barh(y + off, vals, height=0.38, color=col, label=lab)
        for yi, v in zip(y, vals):
            b.text(v + 0.4, yi + off, f"{v:.0f}", va="center", fontsize=F(6.5), color=INK2)
    b.set_yticks(y); b.set_yticklabels([l for _, l in traits]); b.set_xlim(0, 34); b.set_xlabel("Average share of 100 points given before the event")
    b.set_title("What they say they want in a partner", fontsize=F(8.5)); style(b); b.legend(loc="lower right", fontsize=F(7))
    fig.tight_layout(); save(fig, "eda_who")


def eda_pairs():
    """Exploratory: match rate by age gap, same field of study, and how often he dates (data/pairs.csv)."""
    p = pd.read_csv("data/pairs.csv"); overall = 100 * p.match.mean()
    panels = [("Age gap (years)", pd.cut(p.abs_age_diff, [-0.1, 1, 3, 5, 9, 100], labels=["0–1", "2–3", "4–5", "6–9", "10+"])),
              ("Same field of study", p.same_field.map({0.0: "No", 1.0: "Yes"})),
              ("How often he goes on dates", pd.cut(p.m_date, [0, 2, 4, 5, 7], labels=["2+ a week", "weekly to\ntwice a month", "once a\nmonth", "rarely"]))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True, gridspec_kw={"width_ratios": [5, 2.4, 4]})
    for ax, (title, g) in zip(axes, panels):
        t = p.groupby(g, observed=True).match.agg(["mean", "size"])
        x = np.arange(len(t)); v = 100 * t["mean"].to_numpy()
        ax.bar(x, v, color=BLUE, width=0.62)
        for xi, vi, n in zip(x, v, t["size"]):
            ax.text(xi, vi - 0.6, f"{vi:.0f}%", ha="center", va="top", fontsize=F(8), color="white", fontweight="bold")
        ax.axhline(overall, color=REF, linestyle="--", linewidth=1.2)
        ax.set_xticks(x); ax.set_xticklabels([f"{lab}\n(n = {n:,})" for lab, n in zip(t.index, t["size"])], fontsize=F(6.2))
        ax.set_title(title, fontsize=F(8.5)); style(ax, "y")
    axes[0].set_ylabel("Pairs that matched (%)"); axes[0].set_ylim(0, 31)
    axes[-1].text(len(panels[2][1].cat.categories) - 0.55, overall + 0.6, f"all pairs: {overall:.1f}%", ha="right", fontsize=F(6.5), color=INK2)
    fig.tight_layout(); save(fig, "eda_pairs")


SPLINE_LABELS = {"m_age": "His age", "f_age": "Her age", "abs_age_diff": "Age gap", "int_corr": "Interest similarity",
                 "fit_to_f_prefs": "His fit to her preferences", "fit_to_m_prefs": "Her fit to his preferences"}


def spline_bic():
    """Why no spline was chosen: in each headline training fold (pre), how much a natural spline (3 df) for each candidate
    improves the full logistic model's fit (drop in -2 log L), against its BIC price (2 extra coefficients x log n).
    Recomputes part1_models.choose_splines' comparison; a spline is chosen only when the improvement beats the price."""
    import json, warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, "scripts"); import modeling as M, part1_models as P
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, "pre")
    folds = M.event_folds(ev); f = np.array([folds[e] for e in ev]); rows = []
    for k in range(5):
        tr = f != k; Xtr, ytr, idtr = X[tr].reset_index(drop=True), y[tr], ids[tr].reset_index(drop=True)
        D = P.make_prep(sets, Xtr).fit(Xtr, idtr).transform(Xtr).to_numpy(); _, ll = M.fit_logit(D, ytr)
        for c in sets["spline_candidates"]:
            p = P.make_prep(sets, Xtr, splines=(c,)).fit(Xtr, idtr); assert c in p.knots
            Dc = p.transform(Xtr).to_numpy(); _, llc = M.fit_logit(Dc, ytr)
            rows.append(dict(fold=k, cand=c, gain=2 * (llc - ll), price=(Dc.shape[1] - D.shape[1]) * np.log(len(ytr))))
    r = pd.DataFrame(rows); assert (r.gain < r.price).all()       # matches the empty `splines` column of part1_models_folds.csv
    order = r.groupby("cand").gain.max().sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, c in enumerate(order):
        g = r[r.cand == c].gain
        ax.plot([g.min(), g.max()], [i, i], color=GRID, linewidth=F(3), solid_capstyle="round", zorder=1)
        ax.scatter(g, [i] * len(g), s=F(42), color=BLUE, edgecolor="white", linewidth=1.2, zorder=2)
    price = r.price.mean()
    ax.axvline(price, color=ORANGE, linewidth=2); ax.axvspan(price, 20, color=ORANGE, alpha=0.08, linewidth=0)
    ax.text(price + 0.3, len(order) - 0.45, f"price of a curve:\n2 coefficients × log n ≈ {price:.0f}", fontsize=F(7), color=INK, va="top")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([SPLINE_LABELS[c] for c in order]); ax.set_xlim(0, 20); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("How much a curve improved the fit, per training fold (drop in −2 log L)"); style(ax); ax.tick_params(axis="y", length=0)
    if not SLIDE: ax.set_title("A curve never improved the fit by more than BIC charges for it")
    fig.tight_layout(); save(fig, "spline_bic")
    print(r.groupby("cand")[["gain", "price"]].agg(["min", "max"]).round(1))


def toy_cluster():
    """Toy example for the clustered standard error: two events of three pairs, errors leaning the same way within each."""
    a, b = [0.2, 0.3, 0.1], [-0.2, -0.3, -0.1]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.bar([0, 1, 2], a, color="#E0607E", width=0.62); ax.bar([4, 5, 6], b, color="#A9C1E8", width=0.62)
    ax.axhline(0, color=INK2, linewidth=1.2); ax.axvline(3, color=GRID, linewidth=1.5, linestyle="--")
    for x, v in zip([0, 1, 2, 4, 5, 6], a + b):
        ax.text(x, v + (0.015 if v > 0 else -0.015), f"{v:+.1f}".replace("-", "−"), ha="center", va="bottom" if v > 0 else "top", fontsize=F(7.5), color=INK)
    ax.text(1, 0.40, "Event A: all too high", ha="center", fontsize=F(8), fontweight="bold", color=INK)
    ax.text(5, 0.40, "Event B: all too low", ha="center", fontsize=F(8), fontweight="bold", color=INK)
    ax.set_ylim(-0.5, 0.47); ax.set_xlim(-0.6, 6.6); ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"): ax.spines[side].set_visible(False)
    ax.text(-0.5, -0.62, "Usual: 0.2² + 0.3² + 0.1² + 0.2² + 0.3² + 0.1² = 0.28", fontsize=F(7.5), color=INK2, transform=ax.transData, clip_on=False)
    ax.text(-0.5, -0.74, "Clustered: (0.2 + 0.3 + 0.1)² + (−0.2 − 0.3 − 0.1)² = 0.72", fontsize=F(7.5), color=INK, fontweight="bold", transform=ax.transData, clip_on=False)
    fig.tight_layout(); save(fig, "toy_cluster")


def equations():
    """The two equations for the implementation slides, as transparent images (mathtext)."""
    for name, tex in (("eq_bic", r"$\mathrm{BIC} \;=\; -2\,\log L \;+\; k\,\log n$"), ("eq_shrink", r"$\tilde{\Sigma}_k \;=\; (1-r)\,\hat{\Sigma}_k \;+\; r\,I$"),
                      ("eq_se", r"$\mathrm{SE}^{2} \;=\; \dfrac{G}{G-1}\cdot\dfrac{1}{n^{2}}\,\sum_{g=1}^{G}\left(\sum_{i\,\in\,g}(\ell_i-\bar{\ell})\right)^{2}$")):
        fig = plt.figure(figsize=(4, 0.6)); fig.text(0, 0.5, tex, fontsize=26, color="#1B2440", va="center")
        name = f"paper_{name}" if PAPER else (f"slide_{name}" if SLIDE else name)
        fig.savefig(RES / f"report_figures_{name}.png", dpi=200, bbox_inches="tight", transparent=True, pad_inches=0.02)
        OUT.mkdir(exist_ok=True); shutil.copy(RES / f"report_figures_{name}.png", OUT / f"{name}.png"); plt.close(fig)
        print(f"wrote results/report_figures_{name}.png (copy: report/figures/{name}.png)")


def leakage():
    ev, rr =headline_match("event").set_index(["feature_set", "model"]).auc, headline_match("random-row").set_index(["feature_set", "model"]).auc
    keep = ["logistic", "lda", "qda", "lasso", "rf", "boosting"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, fs in zip(axes, ("pre", "pre+date")):
        for i, m in enumerate(keep):
            a, b = ev.get((fs, m)), rr.get((fs, m))
            ax.plot([a, b], [i, i], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
            ax.scatter([a], [i], color=BLUE, s=26, zorder=2, label="event CV (whole events held out)" if i == 0 else None)
            ax.scatter([b], [i], color=ORANGE, marker="s", s=24, zorder=2, label="random-row CV (people repeat)" if i == 0 else None)
            ax.text(b + 0.006, i, f"+{b - a:.3f}", va="center", fontsize=F(7.5), color=INK2)
        ax.set_title(f"{fs}"); style(ax); ax.set_xlabel("Held-out AUC")
        ax.set_xlim(min(ev.loc[fs].min(), rr.loc[fs].min()) - 0.02, max(rr.loc[fs][keep].max(), ev.loc[fs][keep].max()) + 0.06)
    axes[0].set_yticks(range(len(keep))); axes[0].set_yticklabels([NAMES[m] for m in keep])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.24), ncol=2, fontsize=F(8))
    if not SLIDE: fig.suptitle("Letting the same people appear in training and test inflates every model's AUC", x=0.01, ha="left",
                 fontsize=F(10), fontweight="bold", color=INK, y=1.03)
    if not SLIDE: fig.text(0.01, -0.3, "Same models, same features; only the fold assignment differs. Source: results/part2_models_metrics.csv.", fontsize=F(7), color=INK2)
    save(fig, "leakage")


def coverage():
    r = pd.read_csv(RES / "sim_cv_repair_coverage.csv")
    r = r[(r.design == "A2 redrawn") | (r.person_sd > 0) | (r.event_sd > 0)]
    left = [("random-row CV, naive SE (conventional)", "Random-row CV, rows independent"),
            ("event CV, naive SE", "Event CV, rows independent"),
            ("event cluster-nested, bias-adjusted", "Cluster-nested CV (our first fix)"),
            ("event CV, cluster-robust SE", "Event CV, cluster-robust (normal)"),
            ("cluster bootstrap of grouped CV", "Cluster bootstrap"),
            ("floor: max(cluster-robust, cluster-nested) (N3)", "Floor (our repair)"),
            ("Nadeau-Bengio corrected t, grouped folds", "Nadeau-Bengio, event folds")]
    s_ = pd.read_csv(RES / "sim_cv_sizes_coverage.csv")
    right = [("cluster-robust, normal (N13)", "Cluster-robust (normal)"), ("CR2, t(G-1)", "CR2 with t(G-1)"),
             ("cluster-robust, t(G*-1)", "Cluster-robust with t(G*-1)"), ("wild cluster bootstrap-t (N12)", "Wild cluster bootstrap-t")]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[len(left), len(right)], hspace=0.35))
    for ax, (df, items, title) in zip(axes, ((r, left, f"(a) {r.design.nunique()} designs with events redrawn"),
                                             (s_, right, f"(b) {s_.design.nunique()} designs with real event sizes and a precisely measured target"))):
        ax.axvspan(0.85, 0.95, color="#f0efec", zorder=0); ax.axvline(0.90, color=REF, linewidth=1, linestyle="--", zorder=1)
        for i, (proc, lab) in enumerate(items):
            v = df[df.procedure == proc].coverage_90
            good = v.min() >= 0.85 and v.max() <= 0.95
            col = AQUA if good else (BLUE if v.median() >= 0.85 else ORANGE)
            ax.plot([v.min(), v.max()], [i, i], color=col, linewidth=2.4, solid_capstyle="round", zorder=2)
            ax.scatter([v.median()], [i], color=col, s=24, zorder=3, edgecolor="white", linewidth=0.8)
            ax.text(1.02, i, f"{v.min():.0%} to {v.max():.0%}", va="center", fontsize=F(7.5), color=INK2, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(items))); ax.set_yticklabels([lab for _, lab in items], fontsize=F(8))
        ax.set_ylim(-0.6, len(items) - 0.4); ax.set_title(title, fontsize=F(8.5)); style(ax)
    axes[1].set_xlim(0.4, 1.0); axes[1].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].xaxis.set_major_formatter(lambda x, _: f"{x:.0%}"); axes[1].set_xlabel("Coverage of nominal 90% intervals for the new-event error")
    if not SLIDE: fig.suptitle("Only event-aware error bars reach their 90% target; rows-independent ones cover about half the time",
                 x=0.01, ha="left", fontsize=F(10), fontweight="bold", color=INK, y=0.99)
    if not SLIDE: fig.text(0.01, -0.03, "Line: range over designs; dot: median; shaded band: 85% to 95%. Aqua: every design in the band; blue: median in "
             "the band; orange: median below.\nSources: results/sim_cv_repair_coverage.csv, results/sim_cv_sizes_coverage.csv.",
             fontsize=F(7), color=INK2)
    save(fig, "coverage")


def nb_theory():
    k = pd.read_csv(RES / "sim_cv_kfold_summary.csv")
    VA, VB, CAB = 0.02266, 0.00946, -0.00787                       # PREREG.md Part 8 (from Part 6b, K = 5)
    kk = np.linspace(2, 10, 200); pred = 1 - 2 * CAB / ((kk - 1) * VA) + VB / ((kk - 1) ** 2 * VA)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(kk, pred, color=BLUE, linewidth=2, label="0-1 loss: Proposition 8 prediction")
    ax.axhline(1, color=ORANGE, linewidth=2, label="log-loss: prediction (= 1)")
    for loss, col, mk, lab in (("0-1 at the training rate", BLUE, "o", "0-1 loss: observed"), ("log-loss", ORANGE, "s", "log-loss: observed")):
        x = k[k.loss == loss]
        for j, (G, g) in enumerate(x.groupby("G")):
            off = (j - 1) * 0.12
            ax.errorbar(g.K + off, g.observed, yerr=1.96 * g.observed_se, fmt=mk, color=col, ms=4, elinewidth=1, capsize=0,
                        markeredgecolor="white", markeredgewidth=0.6, label=lab if j == 0 else None)
    for K, yv, txt in ((3, 1.452, "predicted 1.45 before the run"), (10, 1.082, "1.08")):
        ax.annotate(txt, (K, yv), xytext=(K + 0.35, yv + 0.18), fontsize=F(7.5), color=INK2, arrowprops=dict(arrowstyle="-", color=REF, lw=0.8))
    ax.set_xticks([2, 3, 5, 10]); ax.set_xlabel("Number of folds K (J = 60 fold errors)")
    ax.set_ylabel("Nadeau-Bengio variance /\n(J+K-1)/(J-1) x cluster-robust"); style(ax, "y"); ax.legend(fontsize=F(7.5), ncol=2, loc="upper right")
    ax.set_title("" if SLIDE else "Proposition 8 predicts how conservative the Nadeau-Bengio correction is, at any fold count")
    if not SLIDE: fig.text(0.01, -0.1, "Synthetic balanced clusters; at each K the points are G = 80, 160, 320 (left to right), 400 data sets each; bars: "
             "95% Monte Carlo intervals.\nK = 5 was used to estimate the components; K = 3 and 10 were predicted before running. "
             "Source: results/sim_cv_kfold_summary.csv.", fontsize=F(7), color=INK2)
    save(fig, "nb_theory")


def auc_bias():
    s = pd.read_csv(RES / "sim_cv_auc_summary.csv"); c = pd.read_csv(RES / "sim_cv_auc_components.csv").set_index("design")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2) if SLIDE and not PAPER else (6.5, 3.0))
    ax = axes[0]; gg = np.geomspace(18, 180, 100)
    for d, col, mk, lab in ((4, BLUE, "o", "cluster SD 2"), (2, ORANGE, "s", "cluster SD 1")):
        x = s[s.design == d]; cc = c.loc[d]
        ax.errorbar(x.G, x.bias_pooled, yerr=1.96 * x.bias_se_pooled, fmt=mk, color=col, ms=4.5, elinewidth=1, capsize=0, label=f"pooled, {lab}")
        pred = (cc.c_mix + cc.c_mix2 / 4) / (gg - 1) + (cc.c_train + cc.c_within) / gg
        ax.plot(gg, pred, color=col, linewidth=1.4, linestyle="--")
        ax.errorbar(x.G * 1.07, x.bias_bcwf_pooled, yerr=1.96 * x.bias_se_bcwf_pooled, fmt="D", color=AQUA, ms=3.5, elinewidth=1,
                    capsize=0, label="BCWF-pooled (both)" if d == 4 else None)
    ax.axhline(0, color=REF, linewidth=0.8); ax.set_xscale("log"); ax.set_xticks([20, 40, 80, 160]); ax.set_xticklabels(["20", "40", "80", "160"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of clusters G"); ax.set_ylabel("Bias against the new-cluster AUC"); style(ax, "y")
    ax.set_title("(a) Bias (dashed: Proposition 9)" if SLIDE else "(a) Bias; dashed: Proposition 9", fontsize=F(8.5))
    if SLIDE and not PAPER:
        ax.legend(fontsize=F(7), loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    else:
        ax.legend(fontsize=F(7), loc="lower right")
    ax = axes[1]
    x = s[s.sa == 2].groupby("G")[["cov_t_pooled", "cov_t_bcwf_pooled"]].mean()
    ax.axhspan(0.85, 0.95, color="#f0efec"); ax.axhline(0.9, color=REF, linestyle="--", linewidth=1)
    ax.plot(x.index, x.cov_t_pooled, marker="o", color=BLUE, linewidth=2, ms=4.5, label="pooled")
    ax.plot(x.index, x.cov_t_bcwf_pooled, marker="D", color=AQUA, linewidth=2, ms=4, label="BCWF-pooled")
    for G in x.index:
        ax.text(G, x.cov_t_pooled[G] - 0.025, f"{x.cov_t_pooled[G]:.0%}", ha="center", fontsize=F(7), color=INK2)
    ax.set_xscale("log"); ax.set_xticks([20, 40, 80, 160]); ax.set_xticklabels(["20", "40", "80", "160"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0.6, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.set_xlabel("Number of clusters G"); style(ax, "y")
    ax.set_title("(b) Coverage, cluster SD 2" if SLIDE else "(b) Coverage of 90% intervals, cluster SD 2", fontsize=F(8.5))
    if SLIDE and not PAPER:
        ax.legend(fontsize=F(7), loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    else:
        ax.legend(fontsize=F(7), loc="lower right")
    if not SLIDE: fig.suptitle("Pooling held-out scores across fold models biases the AUC down under clustering; pairs within folds fix it",
                 x=0.01, ha="left", fontsize=F(10), fontweight="bold", color=INK, y=1.04)
    if not SLIDE: fig.text(0, -0.06, "Synthetic clusters of 40 rows, 400 data sets per point; bars: 95% Monte Carlo intervals; (b) averages the two "
             "cluster-SD-2 designs. Sources: results/sim_cv_auc_summary.csv, sim_cv_auc_components.csv.", fontsize=F(7), color=INK2)
    fig.tight_layout(); save(fig, "auc_bias")


def real_intervals():
    e = pd.read_csv(RES / "errorbars_real.csv"); f = pd.read_csv(RES / "errorbars_final.csv")
    e = e[(e.feature_set == "pre") & (e.model == "logistic") & (e.metric == "error rate")]
    f = f[(f.feature_set == "pre") & (f.model == "logistic") & (f.metric == "error rate")]
    rows = [("Random-row CV, rows independent", e[(e.scheme == "random-row") & e.se_type.str.startswith("naive")]),
            ("Event CV, rows independent", e[(e.scheme == "event") & e.se_type.str.startswith("naive")])]
    items = [(lab, float(x.estimate.iloc[0]), float(x.ci90_low.iloc[0]), float(x.ci90_high.iloc[0]), ORANGE) for lab, x in rows]
    for name, lab in (("cluster-robust, t(20)", "Event CV, cluster-robust t(20)"), ("wild cluster bootstrap-t", "Event CV, wild cluster bootstrap-t"),
                      ("Nadeau-Bengio, event folds", "Event CV, Nadeau-Bengio")):
        x = f[f.interval == name].iloc[0]; items.append((lab, float(x.error_rate), float(x.ci90_low), float(x.ci90_high), BLUE))
    naive_hw = (items[1][3] - items[1][2]) / 2; ratios = [((hi - lo) / 2) / naive_hw for _, _, lo, hi, c in items[2:]]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for i, (lab, est, lo, hi, col) in enumerate(items[::-1]):
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.4, solid_capstyle="round"); ax.scatter([est], [i], color=col, s=22, zorder=3, edgecolor="white")
        ax.text(hi + 0.004, i, f"{est:.3f} ({lo:.3f} to {hi:.3f})", va="center", fontsize=F(7.5), color=INK2)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([it[0] for it in items[::-1]]); ax.set_xlim(0.35, 0.475)
    ax.set_xlabel("Error rate of the full logistic model on `pre` (threshold at the training match rate)"); style(ax)
    ax.set_title("" if SLIDE else f"Real data: event-aware 90% intervals are {min(ratios):.1f} to {max(ratios):.1f} times wider,\nand random-row CV is centred too low")
    if not SLIDE: fig.text(0, -0.1, "Sources: results/errorbars_real.csv (rows-independent intervals), errorbars_final.csv (event-aware intervals).",
             fontsize=F(7), color=INK2)
    save(fig, "real_intervals")


if __name__ == "__main__":
    auc_by_model(); auc_part1(); events(); cv_design(); stepwise_kept(); spline_bic(); qda_shrink(); equations(); toy_cluster(); eda_who(); eda_pairs(); leakage(); coverage(); nb_theory(); auc_bias(); real_intervals()
