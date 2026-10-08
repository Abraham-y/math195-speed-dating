"""Known-answer tests for the Part 2 scripts (part2_models, ranking, forecasts, yes_extensions, sim_cv_matched).
Run from the repository root: python tests/test_part2.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import modeling as M
import part2_models as P2
import ranking as RK
import forecasts as FC
import yes_extensions as YX
import sim_cv_matched as SM


def test_logit_is_clipped():
    assert np.allclose(P2.logit([0.5]), 0) and np.isfinite(P2.logit([0.0, 1.0])).all()
    assert np.isclose(P2.logit([1.0])[0], np.log(0.9999 / 0.0001))


def test_part2_hypothesis_logic():
    rows = []
    for scheme, bump in (("event", 0.0), ("random-row", 0.05)):
        for fs in ("pre", "pre+date"):
            base = 0.58 if fs == "pre" else 0.84
            for m, a in [("logistic", 0.54), ("stepwise_forward", 0.55), ("stepwise_backward", 0.55), ("stepwise_both", 0.55), ("lda", 0.55),
                         ("qda", 0.57), ("rf", 0.58), ("lasso", 0.57), ("tree", 0.53), ("bagging", 0.575), ("boosting", 0.585),
                         ("ensemble_mean", 0.585), ("ensemble_stacked", 0.583), ("two_decision_logistic", 0.586), ("two_decision_rf", 0.57)]:
                gap = bump * (2 if m in P2.TREES else 1)            # tree models leak more under random rows
                rows.append(dict(scheme=scheme, spec=M.HEADLINE.label, headline=True, feature_set=fs, target="match", model=m,
                                 auc=a + (base - 0.58) + gap, auc_se=0.02))
    for t, m, a in (("her yes", "logistic", 0.62), ("her yes", "rf", 0.61), ("his yes", "logistic", 0.59), ("his yes", "rf", 0.605)):
        rows.append(dict(scheme="event", spec=M.HEADLINE.label, headline=True, feature_set="pre", target=t, model=m, auc=a, auc_se=0.02))
    h = P2.hypotheses(pd.DataFrame(rows)).set_index(["hypothesis", "version"])
    assert h.loc[("T1", "headline, pre"), "supported"]                     # boosting +0.005 over the forest
    assert h.loc[("T2", "headline, pre"), "supported"]                     # the single tree is lowest
    assert h.loc[("A1", "headline"), "supported"]
    assert h.loc[("D1", "headline"), "supported"] and "two_decision_logistic" in h.loc[("D1", "headline"), "statistic"]
    assert h.loc[("D2", "headline"), "supported"]                          # 0.62 and 0.605, both at least 0.60
    assert h.loc[("E1", "headline, pre"), "supported"]                     # 0.585 - 0.58 = 0.005 < 0.01


def test_rasch_recovers_effects():
    rng = np.random.default_rng(30)
    J, R = 12, 300
    d, p = rng.normal(0, 1, J), rng.normal(0, 1.5, R)
    ri, ci = np.repeat(np.arange(R), J), np.tile(np.arange(J), R)
    y = (rng.random(len(ri)) < 1 / (1 + np.exp(-(0.2 + d[ci] - p[ri])))).astype(float)
    b0, dh, ph = RK.rasch(ri, ci, y)
    assert np.isclose(dh.mean(), 0) and np.corrcoef(dh, d)[0, 1] > 0.97 and np.corrcoef(ph, p)[0, 1] > 0.8


def test_rank_ranges_limits():
    est = np.arange(6, dtype=float)[::-1] * 10                       # far apart: ranks 1..6 exactly
    boot = est + np.random.default_rng(31).normal(0, 0.1, (500, 6))
    lo, up, c = RK.rank_ranges(est, boot)
    assert (lo == np.arange(1, 7)).all() and (up == np.arange(1, 7)).all()
    noisy = np.random.default_rng(32).normal(0, 5, (500, 6))         # no information: every rank possible
    lo, up, _ = RK.rank_ranges(np.zeros(6) + 1e-3 * np.arange(6), noisy)
    assert (lo == 1).all() and (up == 6).all()


def test_murphy_decomposition_identity():
    rng = np.random.default_rng(33)
    p = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], 2000); y = (rng.random(2000) < p * 0.8).astype(float)
    m = FC.murphy(p, y)
    assert np.isclose(m["brier"], m["reliability"] - m["resolution"] + m["uncertainty"])
    assert np.isclose(FC.forecast([0, 5, 10]), [0.025, 0.5, 0.975]).all()


def test_calibration_slope_known_answer():
    rng = np.random.default_rng(34)
    n, g = 6000, np.repeat(np.arange(20), 300)
    p = rng.choice(np.arange(1, 10) / 10, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-0.5 * np.log(p / (1 - p))))).astype(float)     # true slope 0.5: overconfident
    s = FC.score(p, y, g)
    assert abs(s["calibration_slope"] - 0.5) < 0.1 and s["auc"] > 0.55


def test_within_rater_folds_spread_each_rater():
    rater = np.repeat(np.arange(30), np.random.default_rng(35).integers(5, 22, 30))
    f = YX.within_rater_folds(rater, 5, np.random.default_rng(36))
    for r in range(30):
        c = np.bincount(f[rater == r], minlength=5)
        assert c.min() >= 1 and c.max() - c.min() <= 1


def test_two_way_shares_known_answer():
    rng = np.random.default_rng(37)
    R, C = 60, 60
    a, b = rng.normal(0, 1.0, R), rng.normal(0, 0.5, C)
    rows = [(i, j) for i in range(R) for j in range(C)]
    d = pd.DataFrame(rows, columns=["rater", "ratee"])
    d["x"] = a[d.rater] + b[d.ratee] + rng.normal(0, 1.0, len(d))
    s, _ = YX.two_way_shares(d, "x")
    assert abs(s["rater_share"] - 1 / 2.25) < 0.05 and abs(s["ratee_share"] - 0.25 / 2.25) < 0.04


def test_matched_sim_folds_and_loss():
    g = np.repeat(np.arange(21), 10)
    f = SM.group_folds(g, 5, np.random.default_rng(38))
    assert all(len(set(f[g == k])) == 1 for k in range(21)) and set(f) == set(range(5))
    X = np.column_stack([np.r_[np.zeros(50), np.ones(50)]]); y = np.r_[np.zeros(50), np.ones(50)]
    assert SM.loss(X, y, X, y).sum() == 0                              # separable: no errors at the training rate


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
