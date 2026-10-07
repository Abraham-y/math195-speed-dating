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
    cd, h, p = P1.cooks_distance(D, y, fit)
    ref = sm.GLM(y, sm.add_constant(D), family=sm.families.Binomial()).fit().get_influence()
    assert np.allclose(h, ref.hat_matrix_diag, atol=1e-6)
    assert np.allclose(cd, ref.cooks_distance[0], rtol=1e-4, atol=1e-8)


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
