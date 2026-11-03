"""Known-answer tests for draft Proposition 11 (scripts/prop11_linear.py, scripts/sim_cv_unequal.py): the linear
representation reproduces Proposition 6 and the balanced cancellation, the two closed forms agree with its Monte Carlo,
and the simulation's fold statistics agree with the representation. Seeds 996900 to 996910 (scratch).
Run from the repository root: OMP_NUM_THREADS=1 .venv/bin/python tests/test_sim_cv_unequal.py"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import prop11_linear as P
import sim_cv_unequal as U

SIZES = np.array([25, 36, 42, 48, 81, 90, 90, 100, 100, 100, 140, 196, 225, 256, 304, 324, 342, 360, 400, 441, 484], float)
# per-row and same-cluster covariances of (0-1 loss, b) and the log-loss variance, from the scratch pilot (seed 997001)
ROW = np.array([[0.20151, -0.00167], [-0.00167, 0.02811]]); CC = np.array([[0.017425, -0.008204], [-0.008204, 0.009495]])
S2_LOG, C_LOG = 0.28217, 0.031112


def test_real_sizes():
    assert np.array_equal(U.real_sizes(), SIZES)


def test_balanced_cancellation():
    rng = np.random.default_rng(996900); G, m = 20, 30
    a, b = rng.normal(size=(5, G)) * 3, rng.normal(size=(5, G))
    st = P.linear_stats(np.full(G, m), a, b, rng)
    assert np.allclose(st["cv"] - st["err"], a.sum(1) / (G * m))        # omega_h = 1 when the folds have equal row counts


def test_proposition6_balanced():
    rng = np.random.default_rng(996901); G, m, D = 20, 30, 40000; n = G * m
    a1, b1 = rng.normal(size=G) * 5, rng.normal(size=G) * 2
    A, B = np.broadcast_to(a1, (D, G)).copy(), np.broadcast_to(b1, (D, G)).copy()
    st = P.linear_stats(np.full(G, m), A, B, rng, ks_r={5: 12})
    c = a1 - b1 / 4
    pred = st["fac_K5"][0] * G / (G - 1) * np.sum((c - c.mean()) ** 2) / n ** 2     # Proposition 6 on the totals c_h
    assert abs(st["vnb_own_K5"].mean() / pred - 1) < 0.005, st["vnb_own_K5"].mean() / pred
    st0 = P.linear_stats(np.full(G, m), A, 0 * B, rng, ks_r={5: 12})
    assert np.ptp(st0["vcr"]) == 0 and abs(st0["ratio_own_K5"].mean() - 1) < 0.005


def test_proposition6_unequal_residual_totals():
    """With residual totals divided by n/K and K | G, Proposition 6 holds exactly for any sizes (caveat 4 of 6e)."""
    rng = np.random.default_rng(996902); m = SIZES[1:]; G, D = 20, 40000
    A = np.broadcast_to(rng.normal(size=G) * m, (D, G)).copy()
    st = P.linear_stats(m, A, 0 * A, rng, ks_r={5: 12})
    r = st["vnb_res_K5"].mean() / (st["fac_K5"][0] * st["vcr"].mean())
    assert abs(r - 1) < 0.005, r


def test_gaussian_totals_covariance():
    rng = np.random.default_rng(996903); m = np.full((200000, 1), 50.0)
    x, y = P.gaussian_totals(m, ROW, CC, rng)
    target = 50 * ROW + 50 * 49 * CC
    assert np.allclose(np.cov(x[:, 0], y[:, 0]) / target, 1, atol=0.03)


def test_closed_form_balanced():
    rng = np.random.default_rng(996904)
    for G, m in ((20, 40), (20, 199)):
        V = m * ROW + m * (m - 1) * CC
        cr, nb = P.balanced_closed_form(V[0, 0], V[0, 1], V[1, 1], G, K_cv=5, K_nb=5)
        sm = P.summarize(P.surrogate(lambda D, r: np.full((D, G), m), ROW, CC, rng, D=40000, ks_r={5: 12}), ks=(5,))
        assert abs(sm["ratioofmeans_own_K5"] / nb - 1) < 0.01, (G, m, sm["ratioofmeans_own_K5"], nb)
        assert abs(1 / sm["var_ratio"] / cr - 1) < 0.025, (G, m, 1 / sm["var_ratio"], cr)


def test_closed_form_balanced_different_fold_counts():
    """V_CR from a 5-fold run, Nadeau-Bengio partitions into 10 folds: the V_CR term uses K_cv and Var(c) uses K_nb. Known
    answer: the review's independent Monte Carlo, 1.0480 +- 0.0057 (G = 20, m = 199; scratchpad/theory/review/rv_closed_forms.log)."""
    G, m = 20, 199
    V = m * ROW + m * (m - 1) * CC
    cr, nb = P.balanced_closed_form(V[0, 0], V[0, 1], V[1, 1], G, K_cv=5, K_nb=10)
    assert abs(nb - 1.0480) < 2 * 0.0057, nb
    _, wrong = P.balanced_closed_form(V[0, 0], V[0, 1], V[1, 1], G, K_cv=10, K_nb=10)    # one K for both terms (the error)
    assert abs(nb - 1.0490) < 0.001 and abs(wrong - 1.0528) < 0.001, (nb, wrong)
    rng = np.random.default_rng(996910)
    sm = P.summarize(P.surrogate(lambda D, r: np.full((D, G), m), ROW, CC, rng, D=40000, K_cv=5, ks_r={10: 6}), ks=(10,))
    assert abs(sm["ratioofmeans_own_K10"] / nb - 1) < 0.01, (sm["ratioofmeans_own_K10"], nb)


def test_closed_form_beta0_unequal():
    rng = np.random.default_rng(996905)
    v = SIZES * S2_LOG + SIZES * (SIZES - 1) * C_LOG
    row, cc = np.array([[S2_LOG, 0], [0, 1e-12]]), np.array([[C_LOG, 0], [0, 1e-13]])
    sm = P.summarize(P.surrogate(lambda D, r: np.broadcast_to(SIZES, (D, 21)).copy(), row, cc, rng, D=40000))
    for K in (5, 10):
        cr, nb, ratio = P.beta0_closed_form(SIZES, v, K, rng)
        assert abs(sm[f"ratioofmeans_own_K{K}"] / ratio - 1) < 0.01, (K, sm[f"ratioofmeans_own_K{K}"], ratio)
    assert abs(cr - 0.946) < 0.002, cr                                   # the CR1 leverage factor for the real sizes


def test_fold_stats_match_linear_representation():
    """With fixed per-row losses (no refitting) the simulation's fold statistics equal the linear representation with
    beta = 0 on the same partition."""
    rng = np.random.default_rng(996906); m = SIZES.astype(int); G = 21; g = np.repeat(np.arange(G), m)
    l = rng.random(len(g)) + rng.normal(size=G)[g]; mu = l.mean()
    a = np.bincount(g, weights=l - mu)
    for K in (5, 10):
        lab = P.deal(1, G, K, rng)
        own, res, shift = U.fold_stats(l, lab[0][g], K)
        cv, Lres, nF, AF, BT, NT, n = P.cv_terms(lab, K, m[None, :].astype(float), a[None, :], np.zeros((1, G)))
        assert np.allclose(np.array(own) - mu, (AF / nF)[0])
        assert np.allclose(res, (AF - nF * cv[:, None])[0] * K / n[0])
        assert np.allclose(np.array(shift) - np.array(res), l.mean())


def test_nb_ratio_balanced_fixed_losses():
    """Proposition 6 through the simulation's own functions: fixed losses, balanced clusters, K | G, many partitions."""
    rng = np.random.default_rng(996907); G, m, K = 20, 10, 5; g = np.repeat(np.arange(G), m)
    l = rng.random(G * m) + rng.normal(size=G)[g]
    import modeling as M
    vcr = M.cluster_robust_var_of_mean(l, g)
    r = []
    for _ in range(3000):
        vals = []
        for _ in range(12):
            f = P.deal(1, G, K, rng)[0][g]; vals += U.fold_stats(l, f, K)[0]
        r.append(U.nb_ratio(vals, K, vcr))
    assert abs(np.mean(r) - 1) < 0.02, np.mean(r)


def test_quadrature_and_population():
    rng = np.random.default_rng(996908); X = rng.normal(size=(4, 3))
    mc = np.array([np.mean(U.sigmoid(U.B0 + x @ U.BETA + rng.normal(size=400000))) for x in X])
    assert np.allclose(U.pi_of(X), mc, atol=0.002)
    pop = U.Population(1000, rng); w = np.array([-1.0, 0.8, 0.4, 0.0]); t = 0.3
    p = U.sigmoid(np.hstack([np.ones((1000, 1)), pop.X]) @ w)
    assert np.isclose(pop.A(w, t)[0], np.mean(np.where(p > t, 1 - pop.pi, pop.pi)))


def test_plugin_totals_sum_to_zero():
    rng = np.random.default_rng(996909); X, y, g = U.draw(SIZES.astype(int), rng)
    T = U.plugin_totals(X, y, g)
    assert abs(T["a01"].sum()) < 1e-8 and abs(T["all"].sum()) < 1e-8
    assert abs(T["beta"].sum()) < 1e-3 * np.sqrt(np.sum(T["beta"] ** 2))     # score equations: sum of influences is 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
