"""S2: do stated preferences tell us how a person weighs a partner's traits? (PREREG.md, Part 5)

Input : data/clean.csv
Output: results/self_knowledge_coefficients.csv (gamma_k, their mean, tau_k, R2_k, by gender, pooled, without wave
        12, and the gender comparison), results/self_knowledge_check.csv (the parametric bootstrap of the SEs, the
        simulation check of tau and R2, and the optimizer check), results/self_knowledge_hypotheses.csv
Seeds : 20263506. Parametric bootstrap replicate r of design d (0 women, 1 men) uses seed + 1000 d + r (as in
        sim_coef_cluster.py); the tau / R2 simulation check uses seed + 1000 d + 500 + r; the variational fits
        start from statsmodels' jittered standard deviations with rng = seed.

Data (as registered): by gender, dates with all six in-date ratings and the rater's six stated shares
(`*1_1_share`). Stated shares are standardized within survey format (`pref_scale`) from one row per participant,
pooling women and men, as modeling.standardize_by_format does for the pair table; the six ratings are standardized
within the rater's gender on the analysis dates.
Model: logistic regression of the yes on rater indicators (reference: the rater with the most dates among those
with both answers; ties to the lowest id), the six ratings and the six products rating_k x stated_k. gamma_k are the
product coefficients. SEs: the event-clustered sandwich of part1_models.coefficient_table (bread (X'WX + ridge)^-1,
meat G/(G-1) sum_g S_g S_g'); the covariance matrix, which coefficient_table does not return, comes from
`logit_sandwich`, the same computation (the SEs are checked equal to coefficient_table's), and gives the SE of the
mean of the six gamma_k.
Check: parametric bootstrap, 200 replicates per gender: new yeses drawn from the fitted probabilities, refitted by
the same code; the spread of each gamma and of their mean against the mean sandwich SE, and the share of 90%
intervals covering the values used to simulate.
tau_k: mixed logistic model (statsmodels BinomialBayesMixedGLM, fit_vb) with fixed effects for the six ratings, the
six stated shares and the six products, a random intercept per rater and an independent random slope per trait
per rater (one variance component each, vc_formulas). Its vcp are log SDs, so tau_k = exp(vcp_k).
R2_k = gamma_k^2 / (gamma_k^2 + tau_k^2), gamma_k from the fixed-effects model.
S2 by gender: the mean gamma's 90% interval lies above 0 and R2_k < 0.10 for all six traits. The settings say
"by gender" and the hypothesis names no pooling, so each gender gets its own verdict and S2 overall is supported
only if it holds for both (as S1, whose settings were also by gender). The model pooling both genders is reported
without a verdict.
Exploratory: the gender comparison (differences of gamma_k and of the mean, with a joint sandwich over events);
R2_k with gamma_k from the mixed model, with the gender's own variance of the standardized stated shares, and with
gamma_k^2 corrected by its sampling variance; a simulation check of the tau and R2 estimators (50 data sets per
gender drawn from the fitted mixed model); the versions without wave 12 (the PREREG wave 12 entry).
Optimizer: fit_vb with L-BFGS-B. statsmodels' default (BFGS) keeps a dense inverse Hessian over ~3,700 parameters
and takes about 11 minutes a fit; `--bfgs-check` refits both genders with the default and records the difference.
Run from the repository root: python scripts/self_knowledge.py [--reps 200] [--tau-reps 50] [--jobs 2] [--bfgs-check]
"""
import sys, pathlib, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

RES = pathlib.Path("results")
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
SHARES = [f"{a}1_1_share" for a in ATTRS]
PROD = [f"{a}_x_stated" for a in ATTRS]
STATED = [f"stated_{a}" for a in ATTRS]
SEED, REPS, TAU_REPS, Z90, R2_LIMIT = 20263506, 200, 50, 1.6448536269514722, 0.10
VB_OPTS = dict(maxiter=20000, maxfun=40000, ftol=1e-13, gtol=1e-7, maxcor=30)
GENDERS = {"women": (0,), "men": (1,), "pooled": (0, 1)}


# ---------------------------------------------------------------- data

def stated_standardized(clean, fmt_col="pref_scale"):
    """Each participant's six stated shares, standardized to mean 0 and SD 1 within survey format over one row per
    participant (women and men pooled), as modeling.standardize_by_format does for the person columns. Uses no
    outcomes. Index: rater id; columns: stated_<attr>."""
    people = clean.drop_duplicates("rater").set_index("rater")
    out = pd.DataFrame(index=people.index)
    for a, s in zip(ATTRS, SHARES):
        stats = people.groupby(fmt_col)[s].agg(["mean", "std"])
        out[f"stated_{a}"] = (people[s] - people[fmt_col].map(stats["mean"])) / people[fmt_col].map(stats["std"])
    return out


def s2_data(clean, genders, exclude_events=()):
    """Dates of the given rater genders with all six ratings and the rater's six stated shares. Returns the dates,
    the ratings standardized within gender (Z), the standardized stated shares (S) and the products (P)."""
    keep = clean.gender.isin(genders) & clean[ATTRS].notna().all(axis=1) & clean[SHARES].notna().all(axis=1) & ~clean.event.isin(exclude_events)
    d = clean[keep].reset_index(drop=True)
    Z = d.groupby("gender")[ATTRS].transform(lambda v: (v - v.mean()) / v.std())
    S = stated_standardized(clean).loc[d.rater.to_numpy()].reset_index(drop=True)
    P = pd.DataFrame({p: Z[a].to_numpy() * S[f"stated_{a}"].to_numpy() for a, p in zip(ATTRS, PROD)})
    return d, Z, S, P


def reference_rater(d):
    """The rater with the most dates among those who said both yes and no (ties: lowest id)."""
    s = d.groupby("rater").dec.agg(["mean", "size"])
    s = s[(s["mean"] > 0) & (s["mean"] < 1)].reset_index().sort_values(["size", "rater"], ascending=[False, True])
    return int(s.rater.iloc[0])


def fe_design(d, Z, P):
    """Rater indicators (all but the reference rater), the six ratings, the six products."""
    R = pd.get_dummies(d.rater, prefix="rater", dtype=float).drop(columns=f"rater_{reference_rater(d)}")
    return pd.concat([R, Z.reset_index(drop=True), P], axis=1)


# ---------------------------------------------------------------- fixed-effects logit with the cluster sandwich

def logit_sandwich(D, y, g, ridge=1e-6):
    """The computation of part1_models.coefficient_table at its headline ridge, returning what it does not: the
    coefficients (intercept first), the sandwich covariance, the fitted probabilities, each cluster's influence row
    S_g B (so cov = G/(G-1) sum_g of their outer products), and the largest absolute score."""
    A = D.to_numpy(float) if hasattr(D, "to_numpy") else np.asarray(D, float)
    y, g = np.asarray(y, float), np.asarray(g)
    w, _ = M.fit_logit(A, y, ridge=ridge, iters=500)
    X = np.column_stack([np.ones(len(y)), A]); p = M.predict_logit(w, A)
    pen = np.r_[0.0, np.full(A.shape[1], ridge)]
    bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + np.diag(pen))
    S = pd.DataFrame(X * (y - p)[:, None]).groupby(g).sum()
    G = len(S)
    infl = pd.DataFrame(S.to_numpy() @ bread, index=S.index)
    cov = G / (G - 1) * infl.to_numpy().T @ infl.to_numpy()
    return dict(w=w, cov=cov, p=p, infl=infl, G=G, max_abs_score=float(np.abs(X.T @ (y - p) - pen * w).max()))


def linear_combination(fit, idx, weights=None):
    """Estimate and sandwich SE of a' w over the coefficient positions idx (default: their mean)."""
    a = np.zeros(len(fit["w"])); a[idx] = 1.0 / len(idx) if weights is None else weights
    return float(a @ fit["w"]), float(np.sqrt(a @ fit["cov"] @ a)), a


def joint_difference_se(fit_a, a_a, fit_b, a_b):
    """SE of a_a' w_a - a_b' w_b for two models fitted on different rows of the same clusters: the sandwich over the
    stacked influence rows, G/(G-1) sum_g (S_ag B_a a_a - S_bg B_b a_b)^2, G counting the clusters of either."""
    ia = pd.Series(fit_a["infl"].to_numpy() @ a_a, index=fit_a["infl"].index)
    ib = pd.Series(fit_b["infl"].to_numpy() @ a_b, index=fit_b["infl"].index)
    u = ia.sub(ib, fill_value=0.0); G = len(u)
    return float(np.sqrt(G / (G - 1) * np.sum(u ** 2)))


def bootstrap(D, y, g, fit, idx, reps, seed):
    """Parametric bootstrap from the fitted probabilities: gamma_k and their mean, with their sandwich SEs."""
    A = D.to_numpy(float); est, ses = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        ys = (rng.random(len(y)) < fit["p"]).astype(float)
        f = logit_sandwich(A, ys, g)
        m, sm, _ = linear_combination(f, idx)
        est.append(np.r_[f["w"][idx], m]); ses.append(np.r_[np.sqrt(np.diag(f["cov"])[idx]), sm])
    return np.array(est), np.array(ses)


def check_rows(name, est, ses, truth, reps, seed):
    med = np.median(est, axis=0)
    out = []
    for j, term in enumerate(PROD + ["mean gamma"]):
        sd = est[:, j].std(ddof=1)
        out.append(dict(check="parametric bootstrap of the sandwich SE", design=name, term=term, truth=truth[j],
                        mean_estimate=est[:, j].mean(), bias=est[:, j].mean() - truth[j],
                        bias_corrected_estimate=2 * truth[j] - est[:, j].mean(), sd_estimates=sd,
                        robust_sd=1.4826 * np.median(np.abs(est[:, j] - med[j])), mean_se=ses[:, j].mean(),
                        ratio_mean_se_to_sd=ses[:, j].mean() / sd,
                        coverage_90_of_truth=np.mean(np.abs(est[:, j] - truth[j]) <= Z90 * ses[:, j]), reps=reps,
                        seed=f"{seed} + replicate"))
    return out


# ---------------------------------------------------------------- mixed model

def mixed_frame(d, Z, S, P):
    """Data, formula and variance-component formulas of the mixed logistic model: fixed effects for the six ratings,
    the six stated shares and the six products; a random intercept per rater and one independent random slope per
    trait per rater."""
    df = pd.concat([Z.reset_index(drop=True), S.reset_index(drop=True), P], axis=1)
    df["dec"] = d.dec.to_numpy(float); df["rater"] = d.rater.astype(int).astype(str).to_numpy()
    formula = "dec ~ " + " + ".join(ATTRS + STATED + PROD)
    vc = {"rater_intercept": "0 + C(rater)", **{f"slope_{a}": f"0 + C(rater):{a}" for a in ATTRS}}
    return df, formula, vc


def fit_vb(model, method="L-BFGS-B", seed=SEED):
    opts = VB_OPTS if method == "L-BFGS-B" else None
    r = model.fit_vb(fit_method=method, minim_opts=opts, rng=seed)
    fe = pd.Series(r.fe_mean, index=model.fep_names); fe_sd = pd.Series(r.fe_sd, index=model.fep_names)
    vcp = pd.Series(r.vcp_mean, index=model.vcp_names); vcp_sd = pd.Series(r.vcp_sd, index=model.vcp_names)
    return dict(fe=fe, fe_sd=fe_sd, vcp=vcp, vcp_sd=vcp_sd, tau=np.exp(vcp), elbo=-float(r.optim_retvals.fun) if hasattr(r, "optim_retvals") and r.optim_retvals is not None else np.nan,
                converged=bool(getattr(r.optim_retvals, "success", True)) if hasattr(r, "optim_retvals") and r.optim_retvals is not None else None)


def build_mixed(d, Z, S, P):
    df, formula, vc = mixed_frame(d, Z, S, P)
    return BinomialBayesMixedGLM.from_formula(formula, vc, df)


def r2(gamma, tau):
    gamma, tau = np.asarray(gamma, float), np.asarray(tau, float)
    return gamma ** 2 / (gamma ** 2 + tau ** 2)


def simulate_mixed(model, vb, rng):
    """New yeses from the fitted mixed model: fixed effects at their posterior means, fresh random intercepts and
    slopes drawn with SDs tau."""
    sd = np.exp(vb["vcp"].to_numpy())[model.ident]
    u = rng.normal(size=model.k_vc) * sd
    eta = model.exog @ vb["fe"].to_numpy() + model.exog_vc @ u
    return (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)


def tau_check_one(model, vb, D, g, idx, seed):
    rng = np.random.default_rng(seed)
    ys = simulate_mixed(model, vb, rng)
    m = BinomialBayesMixedGLM(ys, model.exog, model.exog_vc, model.ident, vcp_p=model.vcp_p, fe_p=model.fe_p,
                              fep_names=model.fep_names, vcp_names=model.vcp_names)
    v = fit_vb(m, seed=seed)
    f = logit_sandwich(D, ys, g)
    tau = v["tau"][[f"slope_{a}" for a in ATTRS]].to_numpy()
    return np.r_[tau, f["w"][idx], r2(f["w"][idx], tau)]


# ---------------------------------------------------------------- main

def analyse(clean, label, genders, exclude_events=(), mixed=True):
    d, Z, S, P = s2_data(clean, genders, exclude_events)
    D = fe_design(d, Z, P); y = d.dec.to_numpy(float); g = d.event.to_numpy()
    fit = logit_sandwich(D, y, g)
    names = np.array(["const"] + list(D.columns)); idx = np.array([list(names).index(p) for p in PROD])
    out = dict(label=label, d=d, Z=Z, S=S, P=P, D=D, y=y, g=g, fit=fit, idx=idx, names=names)
    if mixed:
        out["model"] = build_mixed(d, Z, S, P)
        out["vb"] = fit_vb(out["model"])
    return out


def coefficient_rows(a, version):
    fit, idx, names = a["fit"], a["idx"], a["names"]
    se = np.sqrt(np.diag(fit["cov"]))
    rows = []
    stated_var = a["S"].groupby(a["d"].rater.to_numpy()).first().var()
    for j in [list(names).index(t) for t in ATTRS] + list(idx):
        t = names[j]
        r = dict(version=version, term=t, kind="product rating x stated (gamma)" if t in PROD else "rating",
                 coef=fit["w"][j], se_cluster_event=se[j])
        if t in PROD and "vb" in a:
            k = t.replace("_x_stated", ""); tau = a["vb"]["tau"][f"slope_{k}"]
            g_mm = a["vb"]["fe"][t]; v = stated_var[f"stated_{k}"]
            gc = max(fit["w"][j] ** 2 - se[j] ** 2, 0.0)
            r.update(tau=tau, log_tau_vb_sd=a["vb"]["vcp_sd"][f"slope_{k}"], R2=float(r2(fit["w"][j], tau)),
                     gamma_mixed=g_mm, gamma_mixed_vb_sd=a["vb"]["fe_sd"][t], R2_gamma_mixed=float(r2(g_mm, tau)),
                     stated_var_raters=v, R2_own_variance=fit["w"][j] ** 2 * v / (fit["w"][j] ** 2 * v + tau ** 2),
                     R2_gamma2_minus_se2=gc / (gc + tau ** 2))
        rows.append(r)
    m, sm, _ = linear_combination(fit, idx)
    rows.append(dict(version=version, term="mean gamma", kind="mean of the six gamma_k", coef=m, se_cluster_event=sm))
    if "vb" in a:
        rows.append(dict(version=version, term="rater intercept SD", kind="mixed model variance component (exp vcp)",
                         tau=a["vb"]["tau"]["rater_intercept"], log_tau_vb_sd=a["vb"]["vcp_sd"]["rater_intercept"]))
    for r in rows:
        if pd.notna(r.get("se_cluster_event", np.nan)):
            r.update(ci90_low=r["coef"] - Z90 * r["se_cluster_event"], ci90_high=r["coef"] + Z90 * r["se_cluster_event"],
                     z=r["coef"] / r["se_cluster_event"])
        r.update(dates=len(a["y"]), raters=a["d"].rater.nunique(), events=a["d"].event.nunique(),
                 separated_raters=int(a["d"].groupby("rater").dec.agg(lambda v: v.min() == v.max()).sum()),
                 reference_rater=reference_rater(a["d"]), max_abs_score=a["fit"]["max_abs_score"],
                 vb_converged=a["vb"]["converged"] if "vb" in a else None, fold_scheme="none (all dates)")
    return rows


def verdict(a):
    m, sm, _ = linear_combination(a["fit"], a["idx"])
    lo, hi = m - Z90 * sm, m + Z90 * sm
    tau = a["vb"]["tau"][[f"slope_{k}" for k in ATTRS]].to_numpy()
    R2 = r2(a["fit"]["w"][a["idx"]], tau)
    ok = bool(lo > 0 and (R2 < R2_LIMIT).all())
    text = (f"mean gamma {m:.4f} (90% interval {lo:.4f} to {hi:.4f}); R2_k " +
            ", ".join(f"{k} {v:.3f}" for k, v in zip(ATTRS, R2)) + f"; {len(a['y'])} dates, {a['d'].rater.nunique()} raters")
    return ok, text, lo > 0, (R2 < R2_LIMIT).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS); ap.add_argument("--tau-reps", type=int, default=TAU_REPS)
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--bfgs-check", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    wave12 = tuple(clean.loc[clean.wave_yes_budget == 1, "event"].unique())
    A = {name: analyse(clean, name, gs) for name, gs in GENDERS.items()}
    A12 = {name: analyse(clean, name, gs, exclude_events=wave12) for name, gs in (("women", (0,)), ("men", (1,)))}
    print(f"[{time.time() - t0:.0f} s] fits done", flush=True)

    # the covariance here reproduces coefficient_table's SEs
    for name in ("women", "men"):
        a = A[name]
        tab, _ = P1.coefficient_table(a["D"], a["y"], a["g"])
        t = tab.set_index("term").loc[ATTRS + PROD]
        assert np.allclose(t.se_cluster_event, np.sqrt(np.diag(a["fit"]["cov"]))[[list(a["names"]).index(c) for c in ATTRS + PROD]], rtol=1e-6), name
        assert np.allclose(t.coef, a["fit"]["w"][[list(a["names"]).index(c) for c in ATTRS + PROD]], rtol=1e-6), name
        assert t.estimable.all() and a["fit"]["max_abs_score"] < 1e-6, name

    rows = []
    for name, a in A.items():
        rows += coefficient_rows(a, name if name != "pooled" else "pooled (both genders; exploratory)")
    for name, a in A12.items():
        rows += coefficient_rows(a, f"{name}, without wave 12 (sensitivity)")
    w, m = A["women"], A["men"]
    for j, term in enumerate(PROD + ["mean gamma"]):
        if term == "mean gamma":
            ew, _, aw = linear_combination(w["fit"], w["idx"]); em, _, am = linear_combination(m["fit"], m["idx"])
        else:
            ew, _, aw = linear_combination(w["fit"], [w["idx"][j]], 1.0); em, _, am = linear_combination(m["fit"], [m["idx"][j]], 1.0)
        se = joint_difference_se(w["fit"], aw, m["fit"], am)
        rows.append(dict(version="gender comparison: women minus men (exploratory)", term=term, kind="difference", coef=ew - em,
                         se_cluster_event=se, ci90_low=ew - em - Z90 * se, ci90_high=ew - em + Z90 * se, z=(ew - em) / se,
                         events=len(set(w["g"]) | set(m["g"])), fold_scheme="none (all dates)"))
    C = pd.DataFrame(rows)
    C.to_csv(RES / "self_knowledge_coefficients.csv", index=False)

    # parametric bootstrap of the sandwich SEs
    checks = []
    for di, name in enumerate(("women", "men")):
        a = A[name]
        est, ses = bootstrap(a["D"], a["y"], a["g"], a["fit"], a["idx"], args.reps, SEED + 1000 * di)
        mg, _, _ = linear_combination(a["fit"], a["idx"])
        checks += check_rows(name, est, ses, np.r_[a["fit"]["w"][a["idx"]], mg], args.reps, SEED + 1000 * di)
    print(f"[{time.time() - t0:.0f} s] bootstrap done", flush=True)

    # simulation check of tau and R2 (exploratory)
    for di, name in enumerate(("women", "men")):
        a = A[name]
        res = np.array(Parallel(n_jobs=args.jobs)(delayed(tau_check_one)(a["model"], a["vb"], a["D"], a["g"], a["idx"], SEED + 1000 * di + 500 + r)
                                                  for r in range(args.tau_reps)))
        tau_true = a["vb"]["tau"][[f"slope_{k}" for k in ATTRS]].to_numpy()
        g_true = a["vb"]["fe"][PROD].to_numpy(); R2_true = r2(g_true, tau_true)
        for k in range(6):
            checks.append(dict(check="simulation from the fitted mixed model: tau and R2", design=name, term=PROD[k],
                               tau_true=tau_true[k], mean_tau_hat=res[:, k].mean(), sd_tau_hat=res[:, k].std(ddof=1),
                               gamma_true=g_true[k], mean_gamma_fe=res[:, 6 + k].mean(), R2_true=R2_true[k],
                               mean_R2_hat=res[:, 12 + k].mean(), share_R2_hat_below_0_10=np.mean(res[:, 12 + k] < R2_LIMIT),
                               reps=args.tau_reps, seed=f"{SEED + 1000 * di + 500} + replicate"))
    print(f"[{time.time() - t0:.0f} s] tau check done", flush=True)

    if args.bfgs_check:
        fits = Parallel(n_jobs=args.jobs)(delayed(fit_vb)(A[n]["model"], "BFGS") for n in ("women", "men"))
        for n, f in zip(("women", "men"), fits):
            v = A[n]["vb"]
            checks.append(dict(check="optimizer: fit_vb with BFGS (statsmodels default) against L-BFGS-B", design=n, term="all",
                               elbo_lbfgsb=v["elbo"], elbo_bfgs=f["elbo"],
                               max_abs_diff_log_tau=float(np.abs(v["vcp"] - f["vcp"]).max()),
                               max_abs_diff_fixed_effects=float(np.abs(v["fe"] - f["fe"]).max()), seed=SEED))
        print(f"[{time.time() - t0:.0f} s] optimizer check done", flush=True)
    K = pd.DataFrame(checks)
    K.to_csv(RES / "self_knowledge_check.csv", index=False)

    boot = K[K.check.str.startswith("parametric")]
    def se_note(name):
        b = boot[boot.design == name].set_index("term")
        r = b.loc["mean gamma"]
        return (f"{name}: mean gamma SE / spread {r.ratio_mean_se_to_sd:.2f}, coverage {r.coverage_90_of_truth:.0%}, bias {r.bias:+.3f}; "
                f"gamma_k SE / spread {b.loc[PROD].ratio_mean_se_to_sd.min():.2f} to {b.loc[PROD].ratio_mean_se_to_sd.max():.2f}, "
                f"coverage {b.loc[PROD].coverage_90_of_truth.min():.0%} to {b.loc[PROD].coverage_90_of_truth.max():.0%} "
                f"({args.reps} replicates, seed {SEED + 1000 * (name == 'men')} + r)")
    hyp = []
    oks = []
    for name in ("women", "men"):
        ok, text, gpart, rpart = verdict(A[name]); oks.append(ok)
        hyp.append(dict(hypothesis="S2", version=name, statistic=text + f"; parts: mean gamma above 0 {bool(gpart)}, every R2_k < 0.10 {bool(rpart)}",
                        supported=ok, se_check=se_note(name), fold_scheme="none (all dates)"))
    hyp.append(dict(hypothesis="S2", version="both genders (the decision: supported only if supported for women and for men)",
                    statistic="; ".join(f"{h['version']}: {'supported' if h['supported'] else 'not supported'}" for h in hyp),
                    supported=bool(all(oks)), se_check="; ".join(se_note(n) for n in ("women", "men")), fold_scheme="none (all dates)"))
    ok, text, gpart, rpart = verdict(A["pooled"])
    hyp.append(dict(hypothesis="S2", version="pooled model, both genders (exploratory, no verdict: the settings say by gender)",
                    statistic=text + f"; would be {'supported' if ok else 'not supported'}", supported=None,
                    se_check="not checked separately (the gender-specific bootstraps apply to the same estimator)", fold_scheme="none (all dates)"))
    for name in ("women", "men"):
        ok, text, gpart, rpart = verdict(A12[name])
        hyp.append(dict(hypothesis="S2", version=f"{name}, without wave 12 (sensitivity, no verdict)", statistic=text + f"; would be {'supported' if ok else 'not supported'}",
                        supported=None, se_check="as the headline version", fold_scheme="none (all dates)"))
    H = pd.DataFrame(hyp)[["hypothesis", "version", "statistic", "supported", "se_check", "fold_scheme"]]
    H.to_csv(RES / "self_knowledge_hypotheses.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 400)
    show = ["version", "term", "coef", "se_cluster_event", "ci90_low", "ci90_high", "tau", "R2", "gamma_mixed", "R2_gamma_mixed", "R2_own_variance", "R2_gamma2_minus_se2"]
    print(C[C.term.isin(PROD + ["mean gamma", "rater intercept SD"])][show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(K.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(H.to_string(index=False))
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
