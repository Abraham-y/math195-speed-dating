"""The error-bar study on the real data (PREREG.md, Part 2 addendum; proposals/06, section 7b with the final models).

Input : data/pairs.csv, data/feature_sets.json
Output: results/errorbars_real.csv
Seed  : 20262506 (random-row folds and the nested-CV repetitions); event folds 20261006.

For the full logistic regression and LDA under the headline specification, on `pre` and `pre+date` (both on the
Part 1 preparation without the spline step, which nested CV would have to repeat hundreds of times; Part 1 chose a
spline in 16 of 360 fits): the per-dyad loss is a wrong prediction at the training match rate.
- Event 5-fold CV: error rate with naive, event cluster-robust, and two-way (woman and man) cluster-robust SEs
  (Cameron, Gelbach and Miller 2011: V_woman + V_man - V_dyad); AUC with the independent-rows and event-clustered
  SEs; the cluster-nested CV root-MSE estimate (10 repetitions) and its bias-adjusted interval.
- Random-row 5-fold CV: error rate with its naive SE, and the nested interval of Bates, Hastie and Tibshirani (2024),
  unadjusted and bias-adjusted.
Every interval is the estimate plus or minus 1.645 SEs (90%).
Run from the repository root: python scripts/errorbars_real.py [nested repetitions]
"""
import sys, json, pathlib
import numpy as np, pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

RES = pathlib.Path("results")
SEED, K, Z = 20262506, 5, 1.645


def fit_predict(model, sets, Xtr, ytr, idtr, Xte):
    p = P1.make_prep(sets, Xtr).fit(Xtr.reset_index(drop=True), idtr.reset_index(drop=True))
    A, B = p.transform(Xtr.reset_index(drop=True)).to_numpy(), p.transform(Xte.reset_index(drop=True)).to_numpy()
    if model == "logistic":
        w, _ = M.fit_logit(A, ytr)
        return M.predict_logit(w, B)
    return LinearDiscriminantAnalysis().fit(A, ytr).predict_proba(B)[:, 1]


def cv(model, sets, X, y, ids, folds, k=K):
    """Held-out probabilities and per-dyad losses (wrong at the training match rate)."""
    prob, lossv = np.empty(len(y)), np.empty(len(y))
    for j in range(k):
        te = folds == j
        prob[te] = fit_predict(model, sets, X[~te], y[~te], ids[~te], X[te])
        lossv[te] = ((prob[te] > y[~te].mean()).astype(float) != y[te]).astype(float)
    return prob, lossv


def deal(g, k, rng):
    u = np.unique(g); to = dict(zip(u[rng.permutation(len(u))], np.arange(len(u)) % k))
    return np.array([to[v] for v in g])


def nested(model, sets, X, y, ids, g, grouped, R, rng):
    """Nested CV (Bates, Hastie and Tibshirani 2024): MSE estimate and nested error estimate; grouped = by event,
    with the cluster-robust variance of the outer errors."""
    a, b, ein = [], [], []
    for _ in range(R):
        folds = deal(g, K, rng) if grouped else rng.permutation(len(y)) % K
        for j in range(K):
            out = folds == j
            Xi, yi, idi, gi = X[~out].reset_index(drop=True), y[~out], ids[~out].reset_index(drop=True), g[~out]
            fi = deal(gi, K - 1, rng) if grouped else rng.permutation(len(yi)) % (K - 1)
            e_in = cv(model, sets, Xi, yi, idi, fi, K - 1)[1].mean(); ein.append(e_in)
            p = fit_predict(model, sets, Xi, yi, idi, X[out])
            eo = ((p > yi.mean()).astype(float) != y[out]).astype(float)
            a.append((e_in - eo.mean()) ** 2)
            b.append(M.cluster_robust_var_of_mean(eo, g[out]) if grouped else eo.var(ddof=1) / out.sum())
    return np.mean(a) - np.mean(b), np.mean(ein)


def bht(err_cv, err_ncv, mse):
    return err_ncv - (1 + (K - 2) / K) * (err_ncv - err_cv), np.sqrt((K - 1) / K) * np.sqrt(max(mse, 0))


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rng = np.random.default_rng(SEED)
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    rows = []
    for fs in ("pre", "pre+date"):
        X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, fs)
        fo = M.event_folds(ev); fe = np.array([fo[e] for e in ev])
        for model in ("logistic", "lda"):
            base = dict(feature_set=fs, model=model, spec=M.HEADLINE.label, dyads=len(y), seed=SEED, nested_repetitions=R,
                        preparation="Part 1 preparation without the spline step")
            prob, e = cv(model, sets, X, y, ids, fe)
            err = e.mean()
            v_f, v_m = M.cluster_robust_var_of_mean(e, ids.f_iid.to_numpy()), M.cluster_robust_var_of_mean(e, ids.m_iid.to_numpy())
            v_d = M.cluster_robust_var_of_mean(e, np.arange(len(e)))
            mse_g, ncv_g = nested(model, sets, X, y, ids, ev, True, R, rng)
            c_g, s_g = bht(err, ncv_g, mse_g)
            v10, v01 = M.placement_values(prob, y)
            add = lambda **kw: rows.append(dict(base, **kw))
            add(scheme="event", metric="error rate", estimate=err, se_type="naive (independent rows)", se=np.sqrt(e.var(ddof=1) / len(e)))
            add(scheme="event", metric="error rate", estimate=err, se_type="cluster-robust by event", se=np.sqrt(M.cluster_robust_var_of_mean(e, ev)))
            add(scheme="event", metric="error rate", estimate=err, se_type="two-way cluster-robust (woman, man)", se=np.sqrt(max(v_f + v_m - v_d, 0)))
            add(scheme="event", metric="error rate", estimate=err, se_type="cluster-nested root MSE", se=np.sqrt(max(mse_g, 0)))
            add(scheme="event", metric="error rate", estimate=c_g, se_type="cluster-nested, bias-adjusted (centre moved)", se=s_g)
            add(scheme="event", metric="AUC", estimate=M.auc(prob, y), se_type="naive (independent rows, DeLong)",
                se=np.sqrt(np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)))
            add(scheme="event", metric="AUC", estimate=M.auc(prob, y), se_type="clustered by event (Obuchowski)", se=np.sqrt(M.auc_clustered_var(prob, y, ev)))
            fr = rng.permutation(len(y)) % K
            prob_r, er = cv(model, sets, X, y, ids, fr)
            mse_r, ncv_r = nested(model, sets, X, y, ids, ev, False, R, rng)
            c_r, s_r = bht(er.mean(), ncv_r, mse_r)
            add(scheme="random-row", metric="error rate", estimate=er.mean(), se_type="naive (independent rows)", se=np.sqrt(er.var(ddof=1) / len(er)))
            add(scheme="random-row", metric="error rate", estimate=er.mean(), se_type="nested root MSE (Bates et al.)", se=np.sqrt(max(mse_r, 0)))
            add(scheme="random-row", metric="error rate", estimate=c_r, se_type="nested, bias-adjusted (Bates et al.)", se=s_r)
            add(scheme="random-row", metric="AUC", estimate=M.auc(prob_r, y), se_type="naive (independent rows, DeLong)",
                se=np.sqrt(np.var(M.placement_values(prob_r, y)[0], ddof=1) / y.sum() + np.var(M.placement_values(prob_r, y)[1], ddof=1) / (len(y) - y.sum())))
            print(f"{fs} {model}: event error {err:.4f}, random-row error {er.mean():.4f}", flush=True)
    D = pd.DataFrame(rows)
    D["ci90_low"], D["ci90_high"] = D.estimate - Z * D.se, D.estimate + Z * D.se
    naive = D[D.se_type.str.startswith("naive")].set_index(["feature_set", "model", "scheme", "metric"]).se
    D["se_ratio_to_naive"] = [r.se / naive.get((r.feature_set, r.model, r.scheme, r.metric), np.nan) for r in D.itertuples()]
    D.assign(fold_scheme=D.scheme).to_csv(RES / "errorbars_real.csv", index=False)
    with pd.option_context("display.width", 220):
        print(D[["feature_set", "model", "scheme", "metric", "se_type", "estimate", "se", "se_ratio_to_naive"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
