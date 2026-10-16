"""Known-answer tests for Part 9, the cross-validated AUC under clustering (scripts/sim_cv_auc.py).
Run from the repository root: python tests/test_part9.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import modeling as M
import sim_cv_auc as A
import auc_ratio_quadrature as Q


def brute(score, y, g, same):
    """Concordant match / non-match pairs (ties half) and the pair count, over pairs in the same cluster (same=True)
    or in different clusters."""
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    conc = pairs = 0.0
    for i in i1:
        for j in i0:
            if (g[i] == g[j]) == same:
                conc += 1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0
                pairs += 1
    return conc, pairs


def toy(rng, G=7, m=9):
    g = np.repeat(np.arange(G), m)
    y = (rng.random(G * m) < 0.35).astype(int); y[0], y[1] = 1, 0
    s = np.round(rng.normal(size=G * m) + y + rng.normal(size=G)[g], 1)                         # rounding makes ties
    return s, y, g


def test_within_cluster_counts_match_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(5):
        s, y, g = toy(rng)
        conc, pairs = A.within_cluster_counts(s, y, g)
        bc, bp = brute(s, y, g, True)
        assert np.isclose(conc.sum(), bc) and np.isclose(pairs.sum(), bp)


def test_between_cluster_auc_matches_brute_force():
    rng = np.random.default_rng(2)
    for _ in range(5):
        s, y, g = toy(rng)
        bc, bp = brute(s, y, g, False)
        assert np.isclose(A.between_cluster_auc(s, y, g), bc / bp)
    # one row per cluster: there are no same-cluster pairs, so it is the ordinary AUC
    s, y, _ = toy(rng); g = np.arange(len(s))
    assert np.isclose(A.between_cluster_auc(s, y, g), M.auc(s, y))


def test_fold_auc_splits_into_between_and_within():
    rng = np.random.default_rng(3)
    s, y, g = toy(rng, G=10, m=12)
    conc, pairs = A.within_cluster_counts(s, y, g)
    n1 = y.sum(); n0 = len(y) - n1; f = pairs.sum() / (n1 * n0)
    A_w = conc.sum() / pairs.sum()
    assert np.isclose(M.auc(s, y), (1 - f) * A.between_cluster_auc(s, y, g) + f * A_w)


def test_linearization_reproduces_obuchowski():
    rng = np.random.default_rng(4)
    for G, m in ((12, 30), (25, 17)):
        g = np.repeat(np.arange(G), m); y = (rng.random(G * m) < 0.3).astype(int)
        s = rng.normal(size=G * m) + y + 0.5 * rng.normal(size=G)[g]
        u = A.auc_linearization(s, y)
        assert abs(u.mean()) < 1e-12
        assert np.isclose(M.cluster_robust_var_of_mean(u, g), M.auc_clustered_var(s, y, g))


def test_cv_scores_are_held_out_and_folds_are_whole_clusters():
    rng = np.random.default_rng(5)
    X, y, g = A.draw(20, rng, "linear", 1.0, 1.0)
    fo = A.grouped(20, rng)
    s, f = A.cv_scores(X, y, g, fo)
    assert all(len(np.unique(f[g == h])) == 1 for h in range(20))                               # clusters stay whole
    assert np.bincount(fo).tolist() == [4] * 5
    te = f == 0; w, _ = M.fit_logit(X[~te], y[~te])                                             # fold 0 refitted by hand
    assert np.allclose(s[te], w[0] + X[te] @ w[1:])


def test_pair_density_estimators_known_answers():
    # matches' scores N(1, 1), non-matches' N(0, 1): D = s_i - s_j ~ N(1, 2), whose density at 0 is
    # exp(-1/4) / sqrt(4 pi) and whose slope there is half of that; weights F = 2 and W = 1.5 scale both
    rng = np.random.default_rng(6)
    n = 400000; y = (rng.random(n) < 0.3).astype(int); s = rng.normal(size=n) + y
    f0 = np.exp(-0.25) / np.sqrt(4 * np.pi)
    F = np.full((n, 1), 2.0)
    assert abs(A.pair_density_term(s, y, F, F) / (4 * f0) - 1) < 0.01
    assert abs(A.pair_density_slope(s, y, F, F, np.array([[1.5]])) / (6 * 0.5 * f0) - 1) < 0.05
    # a weight that depends on the match's score: E[delta(D) s_i] = E[s_i | D = 0] f_D(0) = (1/2) f_D(0)
    assert abs(A.pair_density_term(s, y, s[:, None], np.ones((n, 1))) / (0.5 * f0) - 1) < 0.02


def test_quadrature_gives_one_half_for_independent_rows():
    # Proposition 10: with independent rows and a correct logistic model, c2 = C_mix / 2 exactly
    for b0, beta in ((-1.5, (1.0, 0.5, 0.0)), (-0.5, (2.0, 0.0, 0.0))):
        r = Q.compute(b0=b0, beta=beta, su=0.0, akind="gauss", apar=0.0)
        assert abs(r["ratio"] - 0.5) < 1e-5, r["ratio"]


def test_quadrature_agrees_with_the_binned_constants():
    # an independent method: the binned estimator of sim_cv_auc.components for design 2 (sa 1, su 0)
    r = Q.compute(su=0.0, akind="gauss", apar=1.0)
    c = A.components(2)
    assert abs(c["c_mix"] / r["C_mix"] - 1) < 0.03, (c["c_mix"], r["C_mix"])
    assert abs(c["c_mix2"] / r["c2"] - 1) < 0.05, (c["c_mix2"], r["c2"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
