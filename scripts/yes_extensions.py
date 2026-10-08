"""Extensions on the yes decisions (PREREG.md, Part 2): W1 (within-event prediction from who rates and who is
rated), S1 (strategic yes-saying), and two exploratory checks (nonlinearity of the yes model; variance components
and halo in the ratings).

Input : data/clean.csv, results/part2_models_yes_predictions.csv (from part2_models.py, for W1's comparison)
Output: results/yes_extensions_within_event.csv, results/yes_extensions_strategic.csv,
        results/yes_extensions_strategic_check.csv, results/yes_extensions_gam.csv,
        results/yes_extensions_ratings_variance.csv, results/yes_extensions_halo.csv, results/yes_extensions_hypotheses.csv
Seeds : within-event folds 20262006; parametric bootstrap for S1 20262406; event folds 20261006 (nonlinearity check).

W1. One row per date from one side. Within each rater, the rater's dates are dealt at random into 5 folds, so every
rater and (almost always) every ratee has dates in each training part. A logistic regression with an indicator per
rater (how choosy they are) and per ratee (how often they are chosen), ridge 1 on the indicators, predicts each
held-out yes. The comparison is the AUC of the pre-date yes predictions for people at new events (part2_models.py,
D2's better learner for each side, both sides pooled).
S1. By gender, dates with all six ratings, liking and the guess; standardized within gender; logistic regression of
the yes on the eight; the guess's coefficient with a 90% cluster-sandwich interval (events, G/(G-1)).
Run from the repository root after part2_models.py: python scripts/yes_extensions.py [bootstrap replicates]
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

RES = pathlib.Path("results")
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
WITHIN_SEED, BOOT_SEED, TAU = 20262006, 20262406, 0.4


def within_rater_folds(rater, k, rng):
    """Deal each rater's dates at random into k folds (fold j gets the rater's j-th, (j+k)-th, ... date)."""
    folds = np.empty(len(rater), int)
    for r in np.unique(rater):
        idx = np.where(rater == r)[0]
        folds[rng.permutation(idx)] = np.arange(len(idx)) % k
    return folds


def indicator_design(rater, ratee):
    ri, ci = pd.factorize(rater)[0], pd.factorize(ratee)[0]
    X = np.zeros((len(rater), ri.max() + 1 + ci.max() + 1))
    X[np.arange(len(rater)), ri] = 1.0
    X[np.arange(len(rater)), ri.max() + 1 + ci] = 1.0
    return X


def within_event(clean, rng):
    d = clean[["rater", "ratee", "event", "gender", "dec"]].reset_index(drop=True)
    X = indicator_design(d.rater.to_numpy(), d.ratee.to_numpy()); y = d.dec.to_numpy(float)
    folds = within_rater_folds(d.rater.to_numpy(), 5, rng)
    pred = np.empty(len(y))
    for k in range(5):
        te = folds == k
        w, _ = M.fit_logit(X[~te], y[~te], ridge=1.0, iters=200)
        pred[te] = M.predict_logit(w, X[te])
    return d.assign(pred=pred, fold=folds)


def pre_date_comparison(yp):
    """Pre-date yes predictions on new events, the better learner per side (by AUC), both sides pooled."""
    h = yp[yp.spec.str.startswith("missing=median_flags") & (yp.feature_set == "pre")]
    out = []
    for s, ycol in (("f", "f_dec"), ("m", "m_dec")):
        best = max(("logistic", "rf"), key=lambda l: M.auc(h[f"pred_{s}_{l}"], h[ycol]))
        out.append(pd.DataFrame({"pred": h[f"pred_{s}_{best}"].to_numpy(), "y": h[ycol].to_numpy(), "event": h.event.to_numpy(),
                                 "side": s, "learner": best}))
    return pd.concat(out, ignore_index=True)


def strategic(clean):
    rows, designs = [], {}
    cols = ATTRS + ["like", "prob"]
    for g, name in ((0, "women"), (1, "men")):
        d = clean[(clean.gender == g) & clean[cols].notna().all(axis=1)]
        Z = ((d[cols] - d[cols].mean()) / d[cols].std()).reset_index(drop=True)
        y, ev = d.dec.to_numpy(float), d.event.to_numpy()
        tab, fitted = P1.coefficient_table(Z, y, ev)
        designs[name] = (Z, y, ev, fitted)
        for r in tab.itertuples():
            rows.append(dict(rater_gender=name, term=r.term, coef=r.coef, se_cluster_event=r.se_cluster_event, z=r.z,
                             ci90_low=r.coef - 1.645 * r.se_cluster_event, ci90_high=r.coef + 1.645 * r.se_cluster_event, dates=len(y)))
    return pd.DataFrame(rows), designs


def strategic_check(designs, reps, rng):
    out = []
    for name, (Z, y, ev, fitted) in designs.items():
        lp = np.log(fitted / (1 - fitted)); evs = np.unique(ev); gi = np.searchsorted(evs, ev)
        est, se = [], []
        for _ in range(reps):
            ys = (rng.random(len(y)) < 1 / (1 + np.exp(-(lp + rng.normal(0, TAU, len(evs))[gi])))).astype(float)
            t = P1.coefficient_table(Z, ys, ev, path=(1e-6, 1e-6))[0].set_index("term")
            est.append(t.loc["prob", "coef"]); se.append(t.loc["prob", "se_cluster_event"])
        est, se = np.array(est), np.array(se)
        out.append(dict(rater_gender=name, term="prob", reps=reps, seed=BOOT_SEED, event_sd=TAU, mean_se=se.mean(),
                        empirical_sd=est.std(ddof=1), ratio=se.mean() / est.std(ddof=1),
                        coverage_90=np.mean(np.abs(est - np.median(est)) <= 1.645 * se)))
    return pd.DataFrame(out)


def gam_check(clean):
    """Exploratory: event-CV AUC and full-data BIC of the yes model with straight lines against natural splines
    (3 df, knots at training quantiles) on each of the six ratings."""
    rows = []
    for g, name in ((0, "women"), (1, "men")):
        d = clean[(clean.gender == g) & clean[ATTRS].notna().all(axis=1)].reset_index(drop=True)
        y, ev = d.dec.to_numpy(float), d.event.to_numpy()
        folds = M.event_folds(ev); f = np.array([folds[e] for e in ev])
        def design(tr, te, spline):
            parts_tr, parts_te = [], []
            for a in ATTRS:
                xtr, xte = d.loc[tr, a].to_numpy(float), d.loc[te, a].to_numpy(float)
                if spline:
                    kn = np.unique(np.quantile(xtr, [0, 1 / 3, 2 / 3, 1]))
                    if len(kn) == 4:
                        btr, bte = M.natural_spline_basis(xtr, kn), M.natural_spline_basis(xte, kn)
                    else:
                        btr, bte = xtr[:, None], xte[:, None]
                else:
                    btr, bte = xtr[:, None], xte[:, None]
                mu, sd = btr.mean(axis=0), btr.std(axis=0) + 1e-12
                parts_tr.append((btr - mu) / sd); parts_te.append((bte - mu) / sd)
            return np.column_stack(parts_tr), np.column_stack(parts_te)
        for spline in (False, True):
            pred = np.empty(len(y))
            for k in range(5):
                te = f == k
                A, Bm = design(~te, te, spline)
                w, _ = M.fit_logit(A, y[~te]); pred[te] = M.predict_logit(w, Bm)
            A, _ = design(np.ones(len(y), bool), np.ones(len(y), bool), spline)
            _, ll = M.fit_logit(A, y)
            rows.append(dict(rater_gender=name, model="natural splines (3 df each)" if spline else "straight lines", columns=A.shape[1],
                             auc=M.auc(pred, y), auc_se=np.sqrt(M.auc_clustered_var(pred, y, ev)), bic=M.bic(ll, A.shape[1] + 1, len(y)),
                             dates=len(y), fold_scheme="event", seed=M.FOLD_SEED))
    return pd.DataFrame(rows)


def two_way_shares(d, col):
    """Exploratory: shares of a rating's variance from the rater, the ratee and the rest (additive fixed effects
    fitted by alternating means, within rater gender)."""
    x = d[col].to_numpy(float); r, c = d.rater.to_numpy(), d.ratee.to_numpy()
    mu = x.mean(); a = pd.Series(0.0, index=np.unique(r)); b = pd.Series(0.0, index=np.unique(c))
    for _ in range(50):
        a = pd.Series(x - mu - b.loc[c].to_numpy()).groupby(r).mean()
        b = pd.Series(x - mu - a.loc[r].to_numpy()).groupby(c).mean()
    fa, fb = a.loc[r].to_numpy(), b.loc[c].to_numpy(); e = x - mu - fa - fb
    tot = x.var()
    return dict(rater_share=fa.var() / tot, ratee_share=fb.var() / tot, residual_share=e.var() / tot), e


def ratings_structure(clean):
    var, halo = [], []
    for g, name in ((0, "women"), (1, "men")):
        d = clean[(clean.gender == g) & clean[ATTRS + ["like"]].notna().all(axis=1)].reset_index(drop=True)
        resid = {}
        for col in ATTRS + ["like"]:
            s, e = two_way_shares(d, col); resid[col] = e
            var.append(dict(rater_gender=name, rating=col, dates=len(d), **s))
        for col in ATTRS[1:] + ["like"]:
            halo.append(dict(rater_gender=name, rating=col, raw_corr_with_attr=np.corrcoef(d.attr, d[col])[0, 1],
                             residual_corr_with_attr=np.corrcoef(resid["attr"], resid[col])[0, 1]))
    return pd.DataFrame(var), pd.DataFrame(halo)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    W = within_event(clean, np.random.default_rng(WITHIN_SEED))
    cmp_ = pre_date_comparison(pd.read_csv(RES / "part2_models_yes_predictions.csv"))
    rows = [dict(prediction="within event: rater and ratee indicators", dates=len(W), auc=M.auc(W.pred, W.dec),
                 auc_se=np.sqrt(M.auc_clustered_var(W.pred, W.dec, W.event)), fold_scheme="within-rater random folds", seed=WITHIN_SEED)]
    for g, name in ((0, "women"), (1, "men")):
        s = W[W.gender == g]
        rows.append(dict(prediction=f"within event, {name}'s decisions", dates=len(s), auc=M.auc(s.pred, s.dec),
                         auc_se=np.sqrt(M.auc_clustered_var(s.pred, s.dec, s.event)), fold_scheme="within-rater random folds", seed=WITHIN_SEED))
    rows.append(dict(prediction="new events: pre-date information (" + ", ".join(f"{s} {l}" for s, l in cmp_.groupby('side').learner.first().items()) + ")",
                     dates=len(cmp_), auc=M.auc(cmp_.pred, cmp_.y), auc_se=np.sqrt(M.auc_clustered_var(cmp_.pred, cmp_.y, cmp_.event)),
                     fold_scheme="event", seed=M.FOLD_SEED))
    WE = pd.DataFrame(rows); WE.to_csv(RES / "yes_extensions_within_event.csv", index=False)
    S, designs = strategic(clean); S.to_csv(RES / "yes_extensions_strategic.csv", index=False)
    SC = strategic_check(designs, reps, np.random.default_rng(BOOT_SEED)); SC.to_csv(RES / "yes_extensions_strategic_check.csv", index=False)
    G = gam_check(clean); G.to_csv(RES / "yes_extensions_gam.csv", index=False)
    V, H = ratings_structure(clean); V.to_csv(RES / "yes_extensions_ratings_variance.csv", index=False); H.to_csv(RES / "yes_extensions_halo.csv", index=False)

    gap = WE.auc.iloc[0] - WE.auc.iloc[-1]
    sp = S[S.term == "prob"].set_index("rater_gender")
    hyp = pd.DataFrame([
        dict(hypothesis="W1", version="all decisions", statistic=f"within-event AUC {WE.auc.iloc[0]:.4f} minus pre-date AUC on new events {WE.auc.iloc[-1]:.4f} = {gap:+.4f}",
             supported=bool(gap >= 0.10)),
        dict(hypothesis="S1", version="both genders", statistic="; ".join(f"{g}: guess coefficient {r.coef:.3f} (90% interval {r.ci90_low:.3f} to {r.ci90_high:.3f})"
                                                                         for g, r in sp.iterrows()),
             supported=bool((sp.ci90_low > 0).all()), se_check="; ".join(f"{r.rater_gender}: SE / spread {r.ratio:.2f}, coverage {r.coverage_90:.0%}" for r in SC.itertuples()))])
    hyp.to_csv(RES / "yes_extensions_hypotheses.csv", index=False)
    for t in (WE, S, SC, G, V, H, hyp):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}")); print()


if __name__ == "__main__":
    main()
