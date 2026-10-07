"""Known-answer tests for scripts/modeling.py (convention 7).
Run from the repository root: python tests/test_modeling.py   (pytest also collects it)
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modeling as M
import pairs as P
from test_pairs import toy


# ---------------------------------------------------------------- folds and specifications

def test_event_folds_partition():
    f = M.event_folds(np.repeat(np.arange(1, 22), 3), k=5, seed=1)
    assert sorted(f) == list(range(1, 22))
    sizes = np.bincount(list(f.values()))
    assert sorted(sizes) == [4, 4, 4, 4, 5]
    assert f == M.event_folds(np.arange(1, 22), k=5, seed=1)          # same seed, same assignment
    assert f != M.event_folds(np.arange(1, 22), k=5, seed=2)


def test_grid_has_24_distinct_specs_headline_first():
    g = M.grid_specs()
    assert len(g) == 24 and len(set(g)) == 24 and g[0] == M.HEADLINE


def test_standardize_by_format():
    df = pd.DataFrame({"f_iid": [1, 1, 2, 3], "m_iid": [10, 11, 10, 12], "pref_scale": ["a", "a", "a", "b"],
                       "f_pref_x": [10.0, 10.0, 20.0, 5.0], "m_pref_x": [30.0, 40.0, 30.0, 7.0], "fit": [1.0, 2.0, 3.0, 9.0]})
    df.loc[3, "pref_scale"] = "b"; df = pd.concat([df, df.iloc[[3]].assign(f_iid=4, m_iid=13, f_pref_x=9.0, m_pref_x=11.0, fit=11.0)], ignore_index=True)
    out = M.standardize_by_format(df, ["f_pref_x", "m_pref_x"], ["fit"])
    # format a: people 1 (10), 2 (20), 10 (30), 11 (40) -> mean 25, sd sqrt(166.67)
    sd = np.std([10, 20, 30, 40], ddof=1)
    assert np.isclose(out.f_pref_x[0], (10 - 25) / sd) and np.isclose(out.m_pref_x[1], (40 - 25) / sd)
    # format b: people 3 (5), 4 (9), 12 (7), 13 (11) -> mean 8
    assert np.isclose(out.f_pref_x[3], (5 - 8) / np.std([5, 9, 7, 11], ddof=1))
    # dyad column: format a dyads 1, 2, 3 -> mean 2, sd 1
    assert np.allclose(out.fit[:3], [-1, 0, 1])


def test_build_frame_applies_spec():
    pairs, sets = P.build_pairs(toy()), P.feature_sets()
    X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, "pre+date")
    assert "f_pref_shar" not in X and "m_pref_shar" not in X                       # reference share dropped
    assert set(sets["flags_pre"] + sets["flags_date"]) <= set(X.columns) and len(X) == 2
    Xn, *_ = M.build_frame(pairs, sets, M.Spec(missing="median_no_date_flags"), "pre+date")
    assert set(sets["flags_pre"]) <= set(Xn.columns) and not set(sets["flags_date"]) & set(Xn.columns)
    Xr, *_ = M.build_frame(pairs, sets, M.Spec(out_of_range="recorded", pref_format="unstandardized"), "pre")
    order = pairs.sort_values(["event", "f_iid", "m_iid"]).reset_index(drop=True)
    assert np.allclose(Xr.int_corr, order.int_corr_released) and Xr.m_gaming.iloc[0] == 14
    Xa, *_ = M.build_frame(pairs, sets, M.Spec(fit="A", pref_format="unstandardized"), "pre")
    assert np.allclose(Xa.fit_to_f_prefs, order.fit_raw_to_f_prefs)
    Xc, yc, *_ = M.build_frame(pairs, sets, M.COMPLETE_CASE, "pre")
    assert len(Xc) == 0                                  # both toy dyads miss something (gaming fix, field blank)
    Xi, *_ = M.build_frame(pairs, sets, M.HEADLINE, "pre_individual")
    assert not set(sets["pre_pair"]) & set(Xi.columns)


# ---------------------------------------------------------------- preparation

def test_skew_rule_both_directions_and_binary_untouched():
    rng = np.random.default_rng(0)
    right = rng.exponential(size=400); left = 10 - rng.exponential(size=400); flag = (rng.random(400) < .1).astype(float)
    X = pd.DataFrame({"r": right, "l": left, "flag": flag})
    ids = pd.DataFrame({"f_iid": np.arange(400), "m_iid": np.arange(400)})
    p = M.Prep(categorical=[], binary=["flag"], person_cols=[]).fit(X, ids)
    assert p.tf["r"][0] == "right" and p.tf["l"][0] == "left" and "flag" not in p.tf
    T = p._transform_skew(X)
    assert abs(M.skew(T.r.to_numpy())) < abs(M.skew(right)) and abs(M.skew(T.l.to_numpy())) < abs(M.skew(left))
    assert np.all(np.diff(T.r.to_numpy()[np.argsort(right)]) >= 0)                # order kept: higher stays higher
    assert np.all(np.diff(T.l.to_numpy()[np.argsort(left)]) >= 0)


def test_natural_spline_basis_properties():
    knots = np.array([0.0, 1.0, 2.0, 4.0])
    x = np.linspace(-3, 8, 221)
    B = M.natural_spline_basis(x, knots)
    assert B.shape == (221, 3) and np.allclose(B[:, 0], x)
    assert np.allclose(B[x <= 0, 1:], 0)                                           # nothing below the first knot
    for side in (x <= 0, x >= 4):                                                  # linear beyond the boundary knots
        assert np.allclose(np.diff(B[side], n=2, axis=0), 0, atol=1e-9)


def test_prep_fills_merges_and_standardizes_from_training_only():
    tr = pd.DataFrame({"f_a": [1.0, 2.0, np.nan, 4.0, 3.0] * 4,
                       "f_field_cd": [1, 1, 1, 2, np.nan] * 4, "m_field_cd": [1, 3, 1, 1, 1] * 4})
    ids = pd.DataFrame({"f_iid": np.arange(20), "m_iid": np.arange(100, 120)})
    p = M.Prep(categorical=["f_field_cd", "m_field_cd"], binary=[], person_cols=["f_a"]).fit(tr, ids)
    assert p.medians["f_a"] == 2.5
    rare, keep, ref = p.levels["f_field_cd"]
    assert rare == {2.0, 3.0} and ref == "1"                     # codes 2 and 3 held by 4 participants each (< 10)
    te = pd.DataFrame({"f_a": [np.nan, 10.0], "f_field_cd": [7, 1], "m_field_cd": [3, 1]})
    D = p.transform(te)
    a_tr = tr.f_a.fillna(2.5)
    assert np.isclose(D.f_a[0], (2.5 - a_tr.mean()) / a_tr.std())                    # training median and scale
    assert D["cat:f_field_cd=other"][0] == 1 and D["cat:m_field_cd=other"][0] == 1  # unseen 7 and rare 3 -> other
    assert set(p.blocks()) == {"f_a", "f_field_cd", "m_field_cd"}


# ---------------------------------------------------------------- logistic regression and stepwise

def test_fit_logit_matches_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 4)); y = (rng.random(800) < 1 / (1 + np.exp(-(0.5 + X @ [1, -1, .5, 0])))).astype(float)
    w, ll = M.fit_logit(X, y, ridge=0)
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    assert np.allclose(w, ref.params, atol=1e-6) and np.isclose(ll, ref.llf)
    assert np.isclose(M.bic(ll, 5, 800), ref.bic)


def test_stepwise_recovers_signal_blocks():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3000, 8))
    eta = -1 + 1.2 * X[:, 0] - 1.0 * X[:, 3] + 0.8 * (X[:, 5] + X[:, 6])           # block "c" = columns 5 and 6
    y = (rng.random(3000) < 1 / (1 + np.exp(-eta))).astype(float)
    blocks = {"a": [0], "b": [1], "x": [2], "d": [3], "e": [4], "c": [5, 6], "f": [7]}
    for direction in ("forward", "backward", "both"):
        assert set(M.stepwise(X, y, blocks, direction)) == {"a", "d", "c"}, direction


# ---------------------------------------------------------------- metrics

def test_cluster_robust_known_answer():
    assert np.isclose(M.cluster_robust_var_of_mean([1, 2, 3, 4], ["a", "a", "b", "b"]), 1.0)
    u = np.arange(10.0)
    assert np.isclose(M.cluster_robust_var_of_mean(u, np.arange(10)), 10 / 9 * np.sum((u - u.mean()) ** 2) / 100)


def test_auc_known_answer():
    assert np.isclose(M.auc([.1, .4, .35, .8], [0, 0, 1, 1]), 0.75)
    assert np.isclose(M.auc([.5, .5, .5, .5], [0, 1, 0, 1]), 0.5)                 # all ties
    v10, v01 = M.placement_values(np.array([.1, .4, .35, .8]), np.array([0, 0, 1, 1]))
    # positives .35 and .8 beat 1 and 2 of the negatives; negatives .1 and .4 are beaten by 2 and 1 of the positives
    assert np.allclose(v10, [0.5, 1.0]) and np.allclose(v01, [1.0, 0.5]) and np.isclose(v10.mean(), v01.mean())


def test_auc_clustered_var_singletons_match_delong():
    rng = np.random.default_rng(5)
    y = (rng.random(400) < .3).astype(int); s = rng.normal(size=400) + y
    v10, v01 = M.placement_values(s, y); A = v10.mean()
    delong = np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)
    ob = M.auc_clustered_var(s, y, np.arange(400))
    assert abs(ob / delong - 1) < 0.01                                               # differs only in n vs n-1
    # duplicating every observation into a cluster of two copies leaves the AUC and its variance about the same,
    # where the independent-rows formula would halve the variance
    ob2 = M.auc_clustered_var(np.repeat(s, 2), np.repeat(y, 2), np.repeat(np.arange(400), 2))
    assert abs(ob2 / ob - 1) < 0.02


def test_metric_table_known_answer():
    prob = np.array([.9, .2, .7, .1, .6, .3]); y = np.array([1, 1, 0, 0, 1, 0]); thr = np.full(6, .5)
    m = M.metric_table(prob, y, thr, g=np.array([1, 1, 2, 2, 3, 3]))
    assert np.isclose(m["sensitivity"][0], 2 / 3) and np.isclose(m["specificity"][0], 2 / 3)
    assert np.isclose(m["error_rate"][0], 2 / 6) and np.isclose(m["balanced_accuracy"][0], 2 / 3)
    assert all(se >= 0 for _, se in m.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
