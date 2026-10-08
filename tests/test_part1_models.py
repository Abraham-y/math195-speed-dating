"""Known-answer tests for the helpers in scripts/part1_models.py.
Run from the repository root: python tests/test_part1_models.py   (pytest also collects it)
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import part1_models as P1
import modeling as M


def test_cooks_distance_matches_statsmodels_glm():
    import statsmodels.api as sm
    rng = np.random.default_rng(7)
    D = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = (rng.random(300) < 1 / (1 + np.exp(-(D.a - 0.5 * D.b)))).astype(float)
    fit = sm.Logit(y, sm.add_constant(D)).fit(disp=0)
    cd, h, p = P1.cooks_distance(D, y, fit.predict(sm.add_constant(D)).to_numpy())
    ref = sm.GLM(y, sm.add_constant(D), family=sm.families.Binomial()).fit().get_influence()
    assert np.allclose(h, ref.hat_matrix_diag, atol=1e-6)
    assert np.allclose(cd, ref.cooks_distance[0], rtol=1e-4, atol=1e-8)


def test_coefficient_table_matches_statsmodels_cluster_sandwich():
    import statsmodels.api as sm
    rng = np.random.default_rng(12)
    n, g = 1200, np.repeat(np.arange(15), 80)
    D = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + D.a - 0.5 * D.b + rng.normal(0, 0.5, 15)[g])))).astype(float)
    tab, p = P1.coefficient_table(D, y, g)
    ref = sm.Logit(y, sm.add_constant(D)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g, "use_correction": False})
    assert np.allclose(tab.coef, ref.params, atol=1e-5) and tab.attrs["max_abs_score"] < 1e-8
    assert np.allclose(tab.se_cluster_event, ref.bse * np.sqrt(15 / 14), rtol=1e-4)      # G/(G-1) only (convention 6)


def test_coefficient_table_marks_a_separated_dummy():
    import statsmodels.api as sm
    rng = np.random.default_rng(13)
    n, g = 1500, np.repeat(np.arange(15), 100)
    D = pd.DataFrame({"a": rng.normal(size=n), "d": (rng.random(n) < 0.05).astype(float)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + D.a)))).astype(float)
    y[D.d == 1] = 0                                                              # the dummy's category has no matches
    assert P1.separated_columns(D, y) == ["d"]
    tab, _ = P1.coefficient_table(D, y, g)
    row = tab.set_index("term")
    assert not row.loc["d", "estimable"] and np.isnan(row.loc["d", "coef"]) and np.isnan(row.loc["d", "se_cluster_event"])
    # the other estimates are the limit of the likelihood: the fit without the category's dyads
    keep = (D.d == 0).to_numpy()
    ref = sm.Logit(y[keep], sm.add_constant(D.loc[keep, ["a"]])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g[keep], "use_correction": False})
    assert np.allclose(row.loc[["const", "a"], "coef"], ref.params, atol=1e-4)
    assert np.allclose(row.loc[["const", "a"], "se_cluster_event"], ref.bse * np.sqrt(15 / 14), rtol=1e-3)


def test_vif_known_answers_and_exact_collinearity():
    rng = np.random.default_rng(14)
    D = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
    v = P1.vif(D).set_index("term").vif
    assert np.allclose(v[["a", "b", "c"]], np.diag(np.linalg.inv(np.corrcoef(D.to_numpy(), rowvar=False))))
    D["s"] = D.a + 2 * D.b                                                       # reproduced exactly by a and b
    v = P1.vif(D).set_index("term").vif
    assert np.isinf(v[["a", "b", "s"]]).all() and np.isfinite(v["c"])


def test_predict_serial_matches_threaded_forest():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(15)
    X = rng.normal(size=(400, 4)); y = (X[:, 0] + rng.normal(size=400) > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=2).fit(X, y)
    threaded = rf.predict_proba(X)[:, 1]
    serial = P1.predict_serial(rf, X)
    assert rf.n_jobs == 1 and np.allclose(serial, threaded) and np.array_equal(serial, P1.predict_serial(rf, X))


def test_calibration_deciles():
    prob = np.linspace(0.01, 0.5, 100); y = np.r_[np.zeros(50), np.ones(50)]
    cal = P1.calibration(prob, y)
    assert len(cal) == 10 and cal.dyads.sum() == 100
    assert np.allclose(cal.observed_rate, [0] * 5 + [1] * 5) and np.all(np.diff(cal.mean_predicted) > 0)


def test_binned_residuals_zero_when_predictions_are_rates():
    x = np.repeat(np.arange(10.0), 20); y = np.tile(np.r_[np.ones(5), np.zeros(15)], 10); prob = np.full(200, 0.25)
    br = P1.binned_residuals(prob, y, pd.DataFrame({"x": x}), ["x"])
    assert np.allclose(br.mean_residual, 0) and (br.n == 20).all()


def test_hypotheses_logic():
    base = dict(spec=M.HEADLINE.label, headline=True, auc_se=0.02)
    met = pd.DataFrame([
        dict(base, feature_set="pre", model="logistic", auc=0.60), dict(base, feature_set="pre", model="qda", auc=0.62),
        dict(base, feature_set="pre+date", model="logistic", auc=0.75), dict(base, feature_set="pre_individual", model="logistic", auc=0.59),
        dict(base, feature_set="pre", model="rf", auc=0.635), dict(base, feature_set="pre+date", model="rf", auc=0.70)])
    sr = pd.DataFrame([dict(rater_gender=g, attribute=a, std_coef=c, gap_revealed_minus_stated=gp)
                       for g in ("women", "men") for a, c, gp in (("attr", 1.0, 20.0), ("fun", 0.5, 5.0), ("sinc", 0.1, -10.0))])
    h = P1.hypotheses(met, sr).set_index(["hypothesis", "version"])
    assert h.loc[("P1", "headline"), "supported"]                      # best pre AUC 0.635 (rf) in [0.58, 0.70)
    assert "rf" in h.loc[("P1", "headline"), "statistic"]
    assert h.loc[("P2", "headline"), "supported"]                      # 0.75 - 0.635 = 0.115 >= 0.10
    assert h.loc[("P3", "headline"), "supported"]                      # 0.60 - 0.59 = 0.01 < 0.02
    assert h.loc[("P5", "women"), "supported"] and h.loc[("P5", "men"), "supported"]
    assert not h.loc[("P4", "headline"), "supported"]                  # 0.635 - 0.60 = 0.035, not < 0.03
    assert "leads the next (fun) by 15.00 points" in h.loc[("P5", "women"), "note"]
    assert "by absolute size the largest is attr, ahead of sinc by 10.00 points" in h.loc[("P5", "women"), "note"]


def test_hypotheses_keep_the_lasso_out_of_the_best_model():
    base = dict(spec=M.HEADLINE.label, headline=True, auc_se=0.02)
    met = pd.DataFrame([
        dict(base, feature_set="pre", model="logistic", auc=0.60), dict(base, feature_set="pre", model="lasso", auc=0.71),
        dict(base, feature_set="pre+date", model="logistic", auc=0.75), dict(base, feature_set="pre+date", model="lasso", auc=0.90)])
    sr = pd.DataFrame([dict(rater_gender=g, attribute=a, std_coef=c, gap_revealed_minus_stated=gp, gap_revealed_minus_stated_waves_6_9=g69)
                       for g in ("women", "men") for a, c, gp, g69 in (("attr", 1.0, 20.0, 5.0), ("fun", 0.5, 5.0, 8.0))])
    h = P1.hypotheses(met, sr).set_index(["hypothesis", "version"])
    assert "logistic" in h.loc[("P1", "headline"), "statistic"] and h.loc[("P1", "headline"), "supported"]   # 0.60, not the lasso's 0.71
    assert h.loc[("P2", "headline"), "supported"]                                                            # 0.75 - 0.60, lasso ignored
    assert not h.loc[("P5", "women, waves 6 to 9 stated shares"), "supported"]                              # fun's gap is larger there


def test_tune_inner_picks_the_better_setting():
    from sklearn.dummy import DummyClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.default_rng(8)
    n = 800; ev = np.repeat(np.arange(8), 100)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * X.x1 - 1)))).astype(float)
    ids = pd.DataFrame({"f_iid": np.arange(n), "m_iid": np.arange(n) + n})
    sets = {"pre_individual": [], "categorical": [], "binary": []}
    make = lambda v: DummyClassifier(strategy="prior") if v == "constant" else LinearDiscriminantAnalysis()
    assert P1.tune_inner(X, y, ev, ids, sets, 0, make, ["constant", "lda"]) == "lda"
    assert P1.tune_inner(X, y, ev, ids, sets, 0, make, ["lda", "constant"]) == "lda"


def test_lasso_grid_starts_where_every_coefficient_is_zero():
    rng = np.random.default_rng(9)
    D = rng.normal(size=(1000, 10)); y = (rng.random(1000) < 1 / (1 + np.exp(-(D[:, 0] - 1.5)))).astype(float)
    grid = P1.lasso_grid(D, y)
    assert len(grid) == 30 and np.all(np.diff(grid) > 0) and np.isclose(grid[-1] / grid[0], 1000)
    first = P1.make_lasso(0)(grid[0]).fit(D, y); second = P1.make_lasso(0)(grid[1]).fit(D, y)
    assert np.all(np.abs(first.coef_) < 1e-8) and np.any(np.abs(second.coef_) > 1e-8)


def test_lasso_tuning_keeps_the_signal():
    rng = np.random.default_rng(10)
    n = 2000; ev = np.repeat(np.arange(10), 200)
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"x{j}" for j in range(12)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * X.x0 - 1.2 * X.x3 + 0.8 * X.x7 - 1.5)))).astype(float)
    ids = pd.DataFrame({"f_iid": np.arange(n), "m_iid": np.arange(n) + n})
    sets = {"pre_individual": [], "categorical": [], "binary": []}
    p = P1.make_prep(sets, X).fit(X, ids); D = p.transform(X).to_numpy()
    C = P1.tune_inner(X, y, ev, ids, sets, 0, P1.make_lasso(0), P1.lasso_grid(D, y))
    coef = P1.make_lasso(0)(C).fit(D, y).coef_[0]
    kept = {c for c, b in zip(p.columns, coef) if abs(b) > 1e-8}
    assert {"x0", "x3", "x7"} <= kept and len(kept) <= 9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
