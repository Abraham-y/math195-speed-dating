"""Known-answer tests for the Part 5 shared pieces (the paired AUC variance; Proposition 6's identity).
Run from the repository root: python tests/test_part5_core.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import modeling as M


def test_auc_diff_variance_known_answers():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(12), 30); y = (rng.random(360) < 0.3).astype(int); s = rng.normal(size=360) + y
    assert np.isclose(M.auc_diff_clustered_var(s, s, y, g), 0)                                  # identical models
    assert np.isclose(M.auc_diff_clustered_var(s, np.zeros(360), y, g), M.auc_clustered_var(s, y, g))   # vs a constant
    t = s + rng.normal(size=360)
    va, vb, vd = M.auc_clustered_var(s, y, g), M.auc_clustered_var(t, y, g), M.auc_diff_clustered_var(s, t, y, g)
    assert 0 < vd < va + vb + 2 * np.sqrt(va * vb)


def nb_variance(fold_means, K):
    J = len(fold_means)
    return (1 / J + 1 / (K - 1)) * np.var(fold_means, ddof=1)


def test_proposition_6_identity():
    """Fixed losses: averaged over random partitions, the Nadeau-Bengio variance equals (J+K-1)/(J-1) times the
    cluster-robust variance (grouped folds) or the naive variance (row folds) (proposals/06, Proposition 6)."""
    rng = np.random.default_rng(2)
    G, m, K, R = 20, 8, 5, 4
    g = np.repeat(np.arange(G), m); loss = rng.normal(size=G)[g] + rng.normal(size=G * m)
    J = R * K; factor = (J + K - 1) / (J - 1)
    v_cr = M.cluster_robust_var_of_mean(loss, g); v_naive = loss.var(ddof=1) / len(loss)
    grouped, rows = [], []
    for _ in range(4000):
        fm_g, fm_r = [], []
        for _ in range(R):
            to = np.empty(G, int); to[rng.permutation(G)] = np.arange(G) % K; f = to[g]
            fm_g += [loss[f == k].mean() for k in range(K)]
            fr = rng.permutation(len(loss)) % K
            fm_r += [loss[fr == k].mean() for k in range(K)]
        grouped.append(nb_variance(fm_g, K)); rows.append(nb_variance(fm_r, K))
    assert abs(np.mean(grouped) / (factor * v_cr) - 1) < 0.02
    assert abs(np.mean(rows) / (factor * v_naive) - 1) < 0.02


def test_proposition_7_fixed_design_overstatement():
    """Fixed clusters with expected losses mu_g: E[V_CR | design] = Var(ybar | design) + S_mu^2 / G
    (proposals/06, Proposition 7)."""
    rng = np.random.default_rng(3)
    G, m, su, sv = 15, 6, 0.5, 1.0
    g = np.repeat(np.arange(G), m); mu = rng.normal(scale=0.8, size=G)
    vcr, ybar = [], []
    for _ in range(20000):
        loss = mu[g] + rng.normal(scale=su, size=G)[g] + rng.normal(scale=sv, size=G * m)
        vcr.append(M.cluster_robust_var_of_mean(loss, g)); ybar.append(loss.mean())
    var_fixed = su ** 2 / G + sv ** 2 / (G * m)
    assert abs(np.var(ybar) / var_fixed - 1) < 0.04
    assert abs(np.mean(vcr) / (var_fixed + mu.var(ddof=1) / G) - 1) < 0.03


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
