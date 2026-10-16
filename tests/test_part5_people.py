"""Known-answer tests for the Part 5 analyses of people's decisions: order effects (O1, scripts/order_effects.py) and
self-knowledge (S2, scripts/self_knowledge.py).
Run from the repository root: python tests/test_part5_people.py"""
import sys, pathlib, itertools
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import modeling as M
import part1_models as P1
import order_effects as O
import self_knowledge as S


# ---------------------------------------------------------------- O1: conditional logit

def brute_loglik(beta, X, y):
    """One rater's conditional log-likelihood by enumerating every subset with the observed number of yeses."""
    k, T = int(y.sum()), len(y)
    num = np.exp((y @ X) @ beta)
    den = sum(np.exp(X[list(c)].sum(0) @ beta) for c in itertools.combinations(range(T), k))
    return np.log(num / den)


def tiny_raters(rng, sizes=(3, 4, 6, 5), p=2):
    Xs = [rng.normal(size=(T, p)) for T in sizes]
    ys = [np.r_[np.ones(T // 2), np.zeros(T - T // 2)][rng.permutation(T)] for T in sizes]
    Tmax = max(sizes)
    X = np.zeros((len(sizes), Tmax, p)); y = np.zeros((len(sizes), Tmax)); mask = np.zeros_like(y, bool)
    for i, (x, yy) in enumerate(zip(Xs, ys)):
        X[i, :len(yy)] = x; y[i, :len(yy)] = yy; mask[i, :len(yy)] = True
    return X, y, mask, y.sum(1).astype(int), Xs, ys


def test_esp_against_enumeration():
    rng = np.random.default_rng(1)
    z = rng.random((3, 6)) + 0.1
    mask = np.ones_like(z, bool); mask[1, 4:] = False                              # a shorter rater (padding)
    e = O.esp(z, mask)
    for i in range(3):
        zi = z[i][mask[i]]
        for k in range(len(zi) + 1):
            brute = sum(np.prod(zi[list(c)]) for c in itertools.combinations(range(len(zi)), k))
            assert np.isclose(e[i, k], brute)
        assert np.allclose(e[i, len(zi) + 1:], 0)


def test_clogit_loglik_score_hessian_against_enumeration():
    rng = np.random.default_rng(2)
    X, y, mask, k, Xs, ys = tiny_raters(rng)
    beta = np.array([0.7, -0.4])
    ll, score, hess = O.clogit_parts(beta, X, y, mask, k)
    for i in range(len(Xs)):
        assert np.isclose(ll[i], brute_loglik(beta, Xs[i], ys[i]))
        h = 1e-5
        num_grad = [(brute_loglik(beta + h * np.eye(2)[j], Xs[i], ys[i]) - brute_loglik(beta - h * np.eye(2)[j], Xs[i], ys[i])) / (2 * h) for j in range(2)]
        assert np.allclose(score[i], num_grad, atol=1e-6)
        num_hess = np.array([[(brute_loglik(beta + h * (np.eye(2)[a] + np.eye(2)[b]), Xs[i], ys[i]) - brute_loglik(beta + h * (np.eye(2)[a] - np.eye(2)[b]), Xs[i], ys[i])
                               - brute_loglik(beta - h * (np.eye(2)[a] - np.eye(2)[b]), Xs[i], ys[i]) + brute_loglik(beta - h * (np.eye(2)[a] + np.eye(2)[b]), Xs[i], ys[i])) / (4 * h * h)
                              for b in range(2)] for a in range(2)])
        assert np.allclose(hess[i], num_hess, atol=1e-4)


def test_uninformative_raters_contribute_nothing():
    X = np.array([[[0.2], [0.5], [1.0]], [[0.3], [0.6], [0.9]]]); y = np.array([[0, 0, 0], [1, 1, 1.0]]); mask = np.ones((2, 3), bool)
    ll, sc, h = O.clogit_parts(np.array([1.3]), X, y, mask, y.sum(1).astype(int))
    assert np.allclose(ll, 0) and np.allclose(sc, 0) and np.allclose(h, 0)


def test_clogit_mle_known_answer():
    # two raters with the yes on the later date, one with it on the earlier: L = sigma(g)^2 (1 - sigma(g)), max at g = log 2
    X = np.array([[[0.0], [1.0]]] * 3); y = np.array([[0, 1], [0, 1], [1, 0.0]]); mask = np.ones((3, 2), bool)
    beta, ll, sc, H = O.clogit_fit(X, y, mask, y.sum(1).astype(int))
    assert np.isclose(beta[0], np.log(2)) and np.isclose(ll, 2 * np.log(2 / 3) + np.log(1 / 3))
    assert np.isclose(H[0, 0], -3 * (2 / 3) * (1 / 3))                              # minus the sum of p(1 - p)


def test_cluster_sandwich_known_answer():
    scores = np.array([[1.0], [2.0], [-1.0], [0.5]]); g = np.array([1, 1, 2, 3]); H = np.array([[-4.0]])
    V = O.cluster_sandwich(scores, H, g)
    assert np.isclose(V[0, 0], 3 / 2 * (3.0 ** 2 + 1.0 + 0.25) / 16)


def test_permutation_keeps_each_raters_positions():
    rng = np.random.default_rng(3)
    X, y, mask, k, Xs, ys = tiny_raters(rng, p=1)
    Xp = O.permute_within(X, mask, rng)
    for i in range(len(Xs)):
        assert np.allclose(np.sort(Xp[i, mask[i], 0]), np.sort(X[i, mask[i], 0]))
        assert np.allclose(Xp[i, ~mask[i]], 0)


def test_relative_position_and_rater_arrays():
    d = pd.DataFrame(dict(rater=[1, 1, 1, 2, 2], order=[3, 1, 4, 2, 1], dec=[1, 0, 0, 1, 1], event=[5, 5, 5, 6, 6], gender=[1, 1, 1, 0, 0]))
    assert np.allclose(O.relative_position(d), [1.0, 1 / 3, 4 / 3, 1.0, 0.5])              # t = order, T_i = number of dates
    assert np.allclose(O.relative_position(d, "rank"), [2 / 3, 1 / 3, 1.0, 1.0, 0.5])
    d["x"] = O.relative_position(d); d["x_man"] = d.x * d.gender
    X, y, mask, k, ev = O.rater_arrays(d, ["x", "x_man"])
    assert np.allclose(X[0, :, 0], [1 / 3, 1.0, 4 / 3]) and np.allclose(X[0, :, 1], X[0, :, 0])    # rater 1 is a man
    assert np.allclose(X[1, :2, 1], 0) and np.allclose(y[0], [0, 1, 0]) and list(k) == [1, 2] and list(ev) == [5, 6]
    assert mask.sum() == 5 and not mask[1, 2]


def test_gender_interaction_splits_into_the_two_gender_fits():
    """With (gamma + delta man) x, women's raters inform gamma only, so gamma = the women-only fit and gamma + delta =
    the men-only fit."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(80):
        T = rng.integers(4, 9); man = i % 2
        for t in range(1, T + 1):
            rows.append(dict(rater=i, order=t, gender=man, event=i % 7, dec=float(rng.random() < 1 / (1 + np.exp(-(0.5 - (1.0 if man else -0.5) * t / T))))))
    d = pd.DataFrame(rows); d["x"] = O.relative_position(d); d["x_man"] = d.x * d.gender
    b, V, _ = O.fit_with_se(*O.rater_arrays(d, ["x", "x_man"]))
    bw, *_ = O.fit_with_se(*O.rater_arrays(d[d.gender == 0], ["x"]))
    bm, *_ = O.fit_with_se(*O.rater_arrays(d[d.gender == 1], ["x"]))
    assert np.isclose(b[0], bw[0], atol=1e-8) and np.isclose(b[0] + b[1], bm[0], atol=1e-8)


def test_leave_one_out_mean():
    d = pd.DataFrame(dict(rater=[1, 2, 3, 1, 2], ratee=[9, 9, 9, 8, 8], like=[6.0, 8.0, np.nan, 5.0, np.nan]))
    m = O.leave_one_out_mean(d)
    assert np.allclose(m[:3], [8.0, 6.0, 7.0]) and np.isnan(m[3]) and np.isclose(m[4], 5.0)


def test_ols_cluster_known_answers():
    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(9), 6); y = rng.normal(size=54) + rng.normal(size=9)[g]
    b, V = O.ols_cluster(y, np.ones((54, 1)), g)                                           # intercept only: the mean
    assert np.isclose(b[0], y.mean()) and np.isclose(V[0, 0], M.cluster_robust_var_of_mean(y, g))
    x = rng.normal(size=54); X = np.column_stack([np.ones(54), x])
    b, V = O.ols_cluster(y, X, g)
    import statsmodels.api as sm
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})                       # adds (N-1)/(N-K)
    assert np.allclose(b, r.params) and np.allclose(V * (53 / 52), r.cov_params())


# ---------------------------------------------------------------- S2: design, sandwich, mixed model

def test_stated_standardized_matches_standardize_by_format():
    clean = pd.read_csv(ROOT / "data" / "clean.csv", low_memory=False); pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    st = S.stated_standardized(clean)
    person = [f"f_pref_{a}" for a in S.ATTRS] + [f"m_pref_{a}" for a in S.ATTRS]
    ref = M.standardize_by_format(pairs, person, [])
    for side in ("f", "m"):
        r = ref.drop_duplicates(f"{side}_iid").set_index(f"{side}_iid")
        for a in S.ATTRS:
            assert np.allclose(r[f"{side}_pref_{a}"].fillna(99), st.loc[r.index, f"stated_{a}"].fillna(99))


def test_s2_design():
    clean = pd.read_csv(ROOT / "data" / "clean.csv", low_memory=False)
    d, Z, St, P = S.s2_data(clean, (0, 1))
    assert d[S.ATTRS + S.SHARES].notna().all().all()
    for g in (0, 1):                                                                       # ratings standardized within gender
        z = Z[d.gender == g]
        assert np.allclose(z.mean(), 0, atol=1e-12) and np.allclose(z.std(), 1)
    for a, p in zip(S.ATTRS, S.PROD):
        assert np.allclose(P[p], Z[a] * St[f"stated_{a}"])
    assert (St.groupby(d.rater.to_numpy()).nunique() == 1).all().all()                    # stated shares constant within rater
    dw, Zw, Sw, Pw = S.s2_data(clean, (0,))
    D = S.fe_design(dw, Zw, Pw); ref = S.reference_rater(dw)
    s = dw.groupby("rater").dec.agg(["mean", "size"])
    assert 0 < s.loc[ref, "mean"] < 1 and s.loc[ref, "size"] == s[(s["mean"] > 0) & (s["mean"] < 1)]["size"].max()
    assert D.shape[1] == dw.rater.nunique() - 1 + 12 and f"rater_{ref}" not in D.columns
    assert np.allclose(D.filter(like="rater_").sum(axis=1) + (dw.rater == ref), 1)            # one indicator per date, or the reference


def test_logit_sandwich_matches_coefficient_table():
    rng = np.random.default_rng(6)
    n = 400; g = rng.integers(0, 12, n); X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + X.a - 0.5 * X.b + rng.normal(0, 0.5, 12)[g])))).astype(float)
    f = S.logit_sandwich(X, y, g)
    t, p = P1.coefficient_table(X, y, g)
    assert np.allclose(f["w"], t.coef) and np.allclose(np.sqrt(np.diag(f["cov"])), t.se_cluster_event) and np.allclose(f["p"], p)
    m, sm, a = S.linear_combination(f, [1, 2, 3])
    assert np.isclose(m, f["w"][1:].mean()) and np.isclose(sm ** 2, a @ f["cov"] @ a)


def test_joint_difference_se_known_answers():
    rng = np.random.default_rng(7)
    def fit(n, groups):
        g = rng.choice(groups, n); X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
        y = (rng.random(n) < 1 / (1 + np.exp(-X.a))).astype(float)
        return S.logit_sandwich(X, y, g)
    fa = fit(300, np.arange(8)); a = np.array([0, 1.0, 0])
    assert np.isclose(S.joint_difference_se(fa, a, fa, a), 0)                               # a model against itself
    fb = fit(300, np.arange(8, 14))                                                         # disjoint clusters: independent
    va, vb = a @ fa["cov"] @ a, a @ fb["cov"] @ a
    G = 14
    expect = G / (G - 1) * (7 / 8 * va + 5 / 6 * vb)                                       # each sandwich carries its own G/(G-1)
    assert np.isclose(S.joint_difference_se(fa, a, fb, a) ** 2, expect)


def test_mixed_model_design():
    rng = np.random.default_rng(8)
    raters = np.repeat([11, 12, 13], 4)
    d = pd.DataFrame(dict(rater=raters, dec=np.tile([0, 1.0, 1, 0], 3)))
    Z = pd.DataFrame(rng.normal(size=(12, 6)), columns=S.ATTRS)
    St = pd.DataFrame(np.repeat(rng.normal(size=(3, 6)), 4, axis=0), columns=S.STATED)
    P = pd.DataFrame({p: Z[a] * St[f"stated_{a}"] for a, p in zip(S.ATTRS, S.PROD)})
    m = S.build_mixed(d, Z, St, P)
    assert m.fep_names == ["Intercept"] + S.ATTRS + S.STATED + S.PROD
    assert np.allclose(m.exog, np.column_stack([np.ones(12), Z, St, P]))
    assert m.vcp_names == ["rater_intercept"] + [f"slope_{a}" for a in S.ATTRS]
    assert list(np.bincount(m.ident)) == [3] * 7
    V = m.exog_vc.toarray()
    for j, name in enumerate(m.vcp_names):
        cols = V[:, m.ident == j]
        for c, r in enumerate((11, 12, 13)):                                                # columns in rater order
            x = np.ones(12) if name == "rater_intercept" else Z[name.replace("slope_", "")].to_numpy()
            assert np.allclose(cols[:, c], np.where(raters == r, x, 0.0))


def test_r2_and_mixed_simulation():
    assert np.isclose(S.r2(0.3, 0.4), 0.36) and np.isclose(S.r2(0.0, 0.2), 0.0)
    class Fake:                                                                             # no random effects: plain logistic draws
        exog = np.ones((20000, 1)); exog_vc = np.zeros((20000, 2)); ident = np.array([0, 1]); k_vc = 2
    vb = dict(fe=pd.Series([np.log(0.3 / 0.7)]), vcp=pd.Series([-50.0, -50.0]))
    y = S.simulate_mixed(Fake, vb, np.random.default_rng(9))
    assert abs(y.mean() - 0.3) < 0.015


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
