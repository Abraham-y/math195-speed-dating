"""J1: joint decisions, a bivariate probit on the two sides' held-out pre-date yes predictions (PREREG.md, Part 5).

Input : results/part2_models_yes_predictions.csv (event-held-out yes predictions of part2_models.py)
Output: results/bivariate_probit_estimates.csv (parameters with event-clustered sandwich SEs and 90% intervals),
        results/bivariate_probit_check.csv (parametric bootstrap check of the SEs),
        results/bivariate_probit_match_auc.csv (exploratory: the match AUC from the joint model against D1's products),
        results/bivariate_probit_hypotheses.csv
Seeds : parametric bootstrap 20263706; event folds 20261006 (modeling.FOLD_SEED) for the exploratory cross-fitted AUC.

Data: the headline specification and feature set `pre` (4,184 dyads). For each side the better learner by held-out AUC,
as in D2 (her yes: random forest; his yes: logistic). Indices z = Phi^-1(clip(p, 1e-4, 1 - 1e-4)).
Model: y_f = 1{alpha_f + theta_f z_f + e_f > 0}, y_m = 1{alpha_m + theta_m z_m + e_m > 0}, (e_f, e_m) standard
bivariate normal with correlation rho = tanh(r); maximum likelihood over the four cells of (her yes, his yes). rho is
what the two decisions share beyond what the pre-date predictions explain: positive if a dyad in which she says yes
more than predicted is one in which he does too.
Phi_2 by Owen's T function (Owen 1956): Phi_2(h, k; rho) = Phi(h)/2 + Phi(k)/2 - T(h, a_h) - T(k, a_k) - beta, with
a_h = (k - rho h) / (h sqrt(1 - rho^2)), a_k likewise, beta = 0 if hk > 0 or (hk = 0 and h + k >= 0) and 1/2
otherwise; T(0, a_h) = sign(k)/4 when h = 0, and Phi_2(0, 0; rho) = 1/4 + arcsin(rho) / (2 pi).
SEs: the event-clustered sandwich A^-1 B A^-1, A the observed information (central differences of the analytic score),
B = G/(G-1) times the sum over events of the outer products of the events' summed analytic scores. The 90% interval for
rho is tanh(r +- 1.645 SE(r)) (the delta-method interval rho +- 1.645 (1 - rho^2) SE(r) is reported beside it).
Check: 200 parametric bootstrap data sets, both decisions drawn from the fitted model given the z's, refitted; the SD
of rho-hat against the mean SE, and the coverage of the fitted rho by the 90% intervals.
J1 is supported if the 90% interval for rho lies above 0.
Run from the repository root: python scripts/bivariate_probit.py [bootstrap replicates]
"""
import sys, time, pathlib
import numpy as np, pandas as pd
from scipy.special import owens_t, ndtr, ndtri
from scipy.optimize import minimize
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")
BOOT_SEED, CLIP, Z90 = 20263706, 1e-4, 1.6448536269514722
PARAMS = ["alpha_f", "theta_f", "alpha_m", "theta_m", "r"]


# ---------------------------------------------------------------- the bivariate normal

def phi2(h, k, rho):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho, vectorized (Owen's T formula)."""
    h, k, rho = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(rho, float))
    s = np.sqrt(1 - rho ** 2)
    hz, kz = h == 0, k == 0
    hs, ks = np.where(hz, 1.0, h), np.where(kz, 1.0, k)                 # placeholders where the ratio is undefined
    ah, ak = (k - rho * h) / (hs * s), (h - rho * k) / (ks * s)
    Th = np.where(hz, np.sign(k) / 4, owens_t(h, ah))
    Tk = np.where(kz, np.sign(h) / 4, owens_t(k, ak))
    hk = h * k
    beta = np.where((hk > 0) | ((hk == 0) & (h + k >= 0)), 0.0, 0.5)
    out = ndtr(h) / 2 + ndtr(k) / 2 - Th - Tk - beta
    both = hz & kz
    out = np.where(both, 0.25 + np.arcsin(rho) / (2 * np.pi), out)
    return np.clip(out, 0.0, 1.0)


def phi2_pdf(h, k, rho):
    s2 = 1 - rho ** 2
    return np.exp(-(h ** 2 - 2 * rho * h * k + k ** 2) / (2 * s2)) / (2 * np.pi * np.sqrt(s2))


def cell_probs(h, k, rho):
    """The four cells (her yes, his yes) = (1, 1), (1, 0), (0, 1), (0, 0)."""
    return phi2(h, k, rho), phi2(h, -k, -rho), phi2(-h, k, -rho), phi2(-h, -k, rho)


# ---------------------------------------------------------------- likelihood and scores

def loglik_scores(par, zf, zm, yf, ym):
    """Per-dyad log-likelihood and analytic scores with respect to (alpha_f, theta_f, alpha_m, theta_m, r)."""
    af, tf, am, tm, r = par
    rho = np.tanh(r)
    qf, qm = 2 * yf - 1, 2 * ym - 1
    w1, w2, rs = qf * (af + tf * zf), qm * (am + tm * zm), qf * qm * rho
    P = np.maximum(phi2(w1, w2, rs), 1e-300)
    s = np.sqrt(1 - rho ** 2)
    g1 = np.exp(-w1 ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr((w2 - rs * w1) / s)
    g2 = np.exp(-w2 ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr((w1 - rs * w2) / s)
    gr = phi2_pdf(w1, w2, rs)
    dh, dk, drho = qf * g1 / P, qm * g2 / P, qf * qm * gr / P
    S = np.column_stack([dh, dh * zf, dk, dk * zm, drho * (1 - rho ** 2)])
    return np.log(P), S


def fit(zf, zm, yf, ym, start=None, free=(True,) * 5):
    """Maximum likelihood (BFGS on the analytic gradient). Parameters not free stay at their start values."""
    p0 = np.array([0.0, 1.0, 0.0, 1.0, 0.0]) if start is None else np.array(start, float)
    free = np.array(free)
    def f(q):
        p = p0.copy(); p[free] = q
        ll, S = loglik_scores(p, zf, zm, yf, ym)
        return -ll.sum(), -S.sum(0)[free]
    res = minimize(f, p0[free], jac=True, method="BFGS", options=dict(gtol=1e-7, maxiter=500))
    p = p0.copy(); p[free] = res.x
    return p, -res.fun, bool(res.success or np.max(np.abs(res.jac)) < 1e-4)


def sandwich(par, zf, zm, yf, ym, events, step=1e-5):
    """Event-clustered sandwich and model-based covariances of the parameter estimates."""
    def total_score(p):
        return loglik_scores(p, zf, zm, yf, ym)[1].sum(0)
    H = np.empty((5, 5))
    for j in range(5):
        e = np.zeros(5); e[j] = step
        H[:, j] = (total_score(par + e) - total_score(par - e)) / (2 * step)
    A = -(H + H.T) / 2
    S = loglik_scores(par, zf, zm, yf, ym)[1]
    Sg = pd.DataFrame(S).groupby(np.asarray(events)).sum().to_numpy()
    G = len(Sg)
    B = G / (G - 1) * Sg.T @ Sg
    Ai = np.linalg.inv(A)
    return Ai @ B @ Ai, Ai


def rho_intervals(r, se_r):
    rho = np.tanh(r)
    return (np.tanh(r - Z90 * se_r), np.tanh(r + Z90 * se_r)), (rho - Z90 * (1 - rho ** 2) * se_r, rho + Z90 * (1 - rho ** 2) * se_r)


# ---------------------------------------------------------------- data and checks

def load(path=RES / "part2_models_yes_predictions.csv"):
    """Headline specification, `pre`; the better learner per side by held-out AUC (D2)."""
    yp = pd.read_csv(path)
    h = yp[(yp.spec == M.HEADLINE.label) & (yp.feature_set == "pre")].reset_index(drop=True)
    learners = {s: max(("logistic", "rf"), key=lambda l: M.auc(h[f"pred_{s}_{l}"], h[f"{s}_dec"])) for s in ("f", "m")}
    z = {s: ndtri(np.clip(h[f"pred_{s}_{learners[s]}"].to_numpy(float), CLIP, 1 - CLIP)) for s in ("f", "m")}
    return h, learners, z


def bootstrap_check(par, zf, zm, events, reps, rng):
    """Both decisions drawn from the fitted model given the z's, refitted from the fitted values."""
    af, tf, am, tm, r = par; rho = np.tanh(r)
    h, k = af + tf * zf, am + tm * zm
    out = []
    for _ in range(reps):
        e1 = rng.standard_normal(len(h)); e2 = rho * e1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(h))
        yf, ym = (h + e1 > 0).astype(float), (k + e2 > 0).astype(float)
        p, _, ok = fit(zf, zm, yf, ym, start=par)
        Vc, Vm = sandwich(p, zf, zm, yf, ym, events)
        se = np.sqrt(np.diag(Vc)); (lo, hi), (dlo, dhi) = rho_intervals(p[4], se[4])
        out.append(dict(**{f"{n}_hat": v for n, v in zip(PARAMS, p)}, **{f"{n}_se": v for n, v in zip(PARAMS, se)},
                        rho_hat=np.tanh(p[4]), rho_se=(1 - np.tanh(p[4]) ** 2) * se[4], rho_se_model=(1 - np.tanh(p[4]) ** 2) * np.sqrt(Vm[4, 4]),
                        rho_lo=lo, rho_hi=hi, rho_dlo=dlo, rho_dhi=dhi, converged=ok))
    B = pd.DataFrame(out)
    rows = []
    for n, truth in zip(PARAMS[:4], par[:4]):
        est, se = B[f"{n}_hat"], B[f"{n}_se"]
        rows.append(dict(parameter=n, truth=truth, mean_estimate=est.mean(), sd_estimate=est.std(ddof=1), mean_se_cluster_event=se.mean(),
                         ratio_mean_se_to_sd=se.mean() / est.std(ddof=1), coverage_90=np.mean(np.abs(est - truth) <= Z90 * se)))
    rows.append(dict(parameter="rho", truth=rho, mean_estimate=B.rho_hat.mean(), sd_estimate=B.rho_hat.std(ddof=1),
                     mean_se_cluster_event=B.rho_se.mean(), ratio_mean_se_to_sd=B.rho_se.mean() / B.rho_hat.std(ddof=1),
                     coverage_90=np.mean((B.rho_lo <= rho) & (rho <= B.rho_hi)),
                     coverage_90_delta_interval=np.mean((B.rho_dlo <= rho) & (rho <= B.rho_dhi)),
                     mean_se_model=B.rho_se_model.mean(), share_lower_bound_above_0=np.mean(B.rho_lo > 0)))
    return pd.DataFrame(rows).assign(reps=reps, seed=BOOT_SEED, converged=int(B.converged.sum()),
                                     design="fitted model given the z's; dyads independent (no event effect in the simulated errors)")


def match_auc(h, z, par, par0, events):
    """Exploratory: the match AUC of P(both yes) from the joint model, from the independent probits (rho = 0), and from
    D1's products of the raw yes predictions; paired differences with the clustered variance by event. Also the joint
    model refitted on 4 event folds and applied to the 5th (cross-fitted), since the five parameters are otherwise
    fitted on the dyads they score."""
    y = (h.f_dec * h.m_dec).to_numpy(float)
    joint = phi2(par[0] + par[1] * z["f"], par[2] + par[3] * z["m"], np.tanh(par[4]))
    indep = ndtr(par0[0] + par0[1] * z["f"]) * ndtr(par0[2] + par0[3] * z["m"])
    folds = M.event_folds(events); f = np.array([folds[e] for e in events]); cross = np.empty(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        p, _, _ = fit(z["f"][tr], z["m"][tr], h.f_dec.to_numpy(float)[tr], h.m_dec.to_numpy(float)[tr], start=par)
        cross[te] = phi2(p[0] + p[1] * z["f"][te], p[2] + p[3] * z["m"][te], np.tanh(p[4]))
    scores = {"bivariate probit, P(both yes) = Phi_2 (in-sample calibration)": joint,
              "bivariate probit, cross-fitted on event folds": cross,
              "independent probits, Phi(h) Phi(k) (rho = 0)": indep,
              "D1 product: her rf x his rf (two_decision_rf)": (h.pred_f_rf * h.pred_m_rf).to_numpy(),
              "D1 product: her logistic x his logistic (two_decision_logistic)": (h.pred_f_logistic * h.pred_m_logistic).to_numpy(),
              "product of the learners used here: her rf x his logistic": (h.pred_f_rf * h.pred_m_logistic).to_numpy()}
    ref = scores["D1 product: her rf x his rf (two_decision_rf)"]
    rows = []
    for name, s in scores.items():
        d = M.auc(s, y) - M.auc(ref, y)
        se = np.sqrt(M.auc_diff_clustered_var(s, ref, y, events))
        rows.append(dict(score=name, auc=M.auc(s, y), auc_se=np.sqrt(M.auc_clustered_var(s, y, events)), minus_d1_rf_product=d,
                         diff_se=se, dyads=len(y), matches=int(y.sum()), fold_scheme="event", fold_seed=M.FOLD_SEED))
    return pd.DataFrame(rows)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    t0 = time.time()
    h, learners, z = load()
    yf, ym, ev = h.f_dec.to_numpy(float), h.m_dec.to_numpy(float), h.event.to_numpy()
    assert learners == {"f": "rf", "m": "logistic"}, f"D2's better learners are her rf and his logistic, got {learners}"
    par, ll, ok = fit(z["f"], z["m"], yf, ym)
    Vc, Vm = sandwich(par, z["f"], z["m"], yf, ym, ev)
    par0, ll0, ok0 = fit(z["f"], z["m"], yf, ym, free=(True, True, True, True, False))
    Vc0, Vm0 = sandwich(par0, z["f"], z["m"], yf, ym, ev)
    se, se_m = np.sqrt(np.diag(Vc)), np.sqrt(np.diag(Vm))
    (lo, hi), (dlo, dhi) = rho_intervals(par[4], se[4])
    rho = np.tanh(par[4])
    base = dict(dyads=len(yf), events=len(np.unique(ev)), learner_f=learners["f"], learner_m=learners["m"], spec=M.HEADLINE.label,
                feature_set="pre", fold_scheme="event (held-out yes predictions of part2_models.py)")
    rows = []
    for model, p, V, Vmod, l, conv in (("bivariate probit", par, Vc, Vm, ll, ok), ("independent probits (rho fixed at 0)", par0, Vc0, Vm0, ll0, ok0)):
        s, sm = np.sqrt(np.diag(V)), np.sqrt(np.diag(Vmod))
        for j, n in enumerate(PARAMS[:4]):
            rows.append(dict(model=model, parameter=n, estimate=p[j], se_cluster_event=s[j], se_model=sm[j],
                             ci90_low=p[j] - Z90 * s[j], ci90_high=p[j] + Z90 * s[j], loglik=l, converged=conv, **base))
    rows.append(dict(model="bivariate probit", parameter="r = atanh(rho)", estimate=par[4], se_cluster_event=se[4], se_model=se_m[4],
                     ci90_low=par[4] - Z90 * se[4], ci90_high=par[4] + Z90 * se[4], loglik=ll, converged=ok, **base))
    rows.append(dict(model="bivariate probit", parameter="rho", estimate=rho, se_cluster_event=(1 - rho ** 2) * se[4], se_model=(1 - rho ** 2) * se_m[4],
                     ci90_low=lo, ci90_high=hi, ci90_low_delta=dlo, ci90_high_delta=dhi, loglik=ll, converged=ok,
                     lr_stat_rho0=2 * (ll - ll0), **base))
    E = pd.DataFrame(rows); E.to_csv(RES / "bivariate_probit_estimates.csv", index=False)
    C = bootstrap_check(par, z["f"], z["m"], ev, reps, np.random.default_rng(BOOT_SEED)); C.to_csv(RES / "bivariate_probit_check.csv", index=False)
    A = match_auc(h, z, par, par0, ev); A.to_csv(RES / "bivariate_probit_match_auc.csv", index=False)
    cr = C.set_index("parameter").loc["rho"]
    hyp = pd.DataFrame([dict(hypothesis="J1", version=f"headline, pre (her {learners['f']}, his {learners['m']})",
                             statistic=f"rho {rho:.4f} (SE {(1 - rho ** 2) * se[4]:.4f}; 90% interval {lo:.4f} to {hi:.4f}, delta-method "
                                       f"{dlo:.4f} to {dhi:.4f}); alpha_f {par[0]:.3f}, theta_f {par[1]:.3f}, alpha_m {par[2]:.3f}, theta_m {par[3]:.3f}; "
                                       f"{len(yf)} dyads, {len(np.unique(ev))} events",
                             supported=bool(lo > 0),
                             se_check=f"parametric bootstrap ({reps} replicates, seed {BOOT_SEED}, dyads independent given the z's): SD of rho-hat "
                                      f"{cr.sd_estimate:.4f}, mean clustered SE {cr.mean_se_cluster_event:.4f} (ratio {cr.ratio_mean_se_to_sd:.2f}), "
                                      f"90% coverage of the fitted rho {cr.coverage_90:.0%}",
                             fold_scheme="event (held-out yes predictions); SEs clustered by event")])
    hyp.to_csv(RES / "bivariate_probit_hypotheses.csv", index=False)
    with pd.option_context("display.width", 250):
        print(E.drop(columns=["spec", "fold_scheme", "feature_set", "learner_f", "learner_m"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(C.drop(columns=["design"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(A.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(hyp[["hypothesis", "statistic", "supported", "se_check"]].to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
