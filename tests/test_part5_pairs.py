"""Known-answer tests for the Part 5 pair models: chemistry (C3, scripts/chemistry.py) and the bivariate probit (J1,
scripts/bivariate_probit.py).
Run from the repository root: OMP_NUM_THREADS=1 python tests/test_part5_pairs.py"""
import sys, pathlib
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.special import ndtr
from scipy.stats import multivariate_normal
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import modeling as M
import yes_extensions as YE
import chemistry as C
import bivariate_probit as BP


# ---------------------------------------------------------------- bivariate normal and probit

def test_phi2_matches_scipy_on_a_grid():
    grid = [-3.0, -1.2, -0.4, 0.0, 0.5, 1.7, 3.0]
    for rho in (-0.95, -0.6, -0.2, 0.0, 0.3, 0.7, 0.95):
        mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], abseps=1e-12, releps=1e-12)
        h, k = np.meshgrid(grid, grid); h, k = h.ravel(), k.ravel()
        ours = BP.phi2(h, k, rho)
        ref = np.array([mvn.cdf([a, b]) for a, b in zip(h, k)])
        assert np.max(np.abs(ours - ref)) < 1e-8, (rho, np.max(np.abs(ours - ref)))


def test_phi2_closed_forms():
    rho = np.linspace(-0.99, 0.99, 23)
    assert np.allclose(BP.phi2(0.0, 0.0, rho), 0.25 + np.arcsin(rho) / (2 * np.pi), atol=1e-14)     # orthant probability
    h, k = np.random.default_rng(1).normal(size=(2, 200))
    assert np.allclose(BP.phi2(h, k, 0.0), ndtr(h) * ndtr(k), atol=1e-14)                            # independence
    assert np.allclose(BP.phi2(h, 40.0, 0.5), ndtr(h), atol=1e-14)                                    # one margin
    assert np.allclose(BP.phi2(h, h, 0.999999), ndtr(h), atol=2e-3)                                   # near-perfect correlation


def test_cell_probabilities_sum_to_one():
    rng = np.random.default_rng(2)
    h, k = rng.normal(0, 1.5, 500), rng.normal(0, 1.5, 500)
    h[:50], k[25:75] = 0.0, 0.0                                                                       # include the h = 0, k = 0 cases
    rho = np.tanh(rng.normal(0, 1, 500))
    P = np.array(BP.cell_probs(h, k, rho))
    assert np.all(P >= 0) and np.all(P <= 1)
    assert np.allclose(P.sum(0), 1, atol=1e-12)
    assert np.allclose(P[0] + P[1], ndtr(h), atol=1e-12) and np.allclose(P[0] + P[2], ndtr(k), atol=1e-12)   # margins


def simulate_probit(n, par, rng, events=40):
    zf, zm = rng.normal(0, 0.6, n), rng.normal(0, 0.6, n)
    rho = np.tanh(par[4])
    e1 = rng.standard_normal(n); e2 = rho * e1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    yf = (par[0] + par[1] * zf + e1 > 0).astype(float); ym = (par[2] + par[3] * zm + e2 > 0).astype(float)
    return zf, zm, yf, ym, np.arange(n) % events


def test_probit_scores_match_finite_differences():
    rng = np.random.default_rng(3)
    zf, zm, yf, ym, _ = simulate_probit(300, [-0.2, 0.8, 0.1, 1.2, 0.4], rng)
    par = np.array([-0.1, 0.9, 0.2, 1.1, 0.3])
    _, S = BP.loglik_scores(par, zf, zm, yf, ym)
    for j in range(5):
        e = np.zeros(5); e[j] = 1e-6
        num = (BP.loglik_scores(par + e, zf, zm, yf, ym)[0] - BP.loglik_scores(par - e, zf, zm, yf, ym)[0]) / 2e-6
        assert np.allclose(S[:, j], num, atol=1e-6)


def test_probit_recovers_known_parameters_and_se():
    rng = np.random.default_rng(4)
    truth = np.array([-0.3, 0.9, 0.2, 1.1, np.arctanh(0.35)])
    zf, zm, yf, ym, g = simulate_probit(20000, truth, rng, events=200)
    par, _, ok = BP.fit(zf, zm, yf, ym)
    Vc, Vm = BP.sandwich(par, zf, zm, yf, ym, g)
    se = np.sqrt(np.diag(Vm))
    assert ok and np.all(np.abs(par - truth) < 4 * se)
    assert np.all(np.abs(np.sqrt(np.diag(Vc)) / se - 1) < 0.25)            # no clustering: the sandwich agrees with the model


# ---------------------------------------------------------------- chemistry

def test_offset_irls_recovers_known_coefficients():
    rng = np.random.default_rng(5)
    n = 40000; X = rng.normal(size=(n, 3)); w = np.array([-0.4, 0.8, -0.5, 0.3]); off = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + w[0] + X @ w[1:])))).astype(float)
    est = C.fit_logit_offset(X, y, off, ridge=1e-8)
    assert np.max(np.abs(est - w)) < 0.05
    base, _ = M.fit_logit(X, y, ridge=1.0)                                  # a constant offset moves only the intercept
    shifted = C.fit_logit_offset(X, y, 0.7, ridge=1.0)
    assert np.isclose(shifted[0], base[0] - 0.7, atol=1e-8) and np.allclose(shifted[1:], base[1:], atol=1e-8)
    assert np.allclose(C.fit_logit_offset(sp.csr_matrix(X), y, off, ridge=1.0), C.fit_logit_offset(X, y, off, ridge=1.0), atol=1e-10)


def test_factor_step_solves_each_group():
    rng = np.random.default_rng(6)
    n, G, lam = 600, 30, 3.0
    g = rng.integers(0, G, n); X = rng.normal(size=(n, 2)); off = rng.normal(0, 0.5, n)
    y = (rng.random(n) < 0.4).astype(float)
    F = C.factor_step(g, X, off, y, rng.normal(size=(G, 2)), lam)
    p = 1 / (1 + np.exp(-(off + np.sum(F[g] * X, 1))))
    grad = np.column_stack([np.bincount(g, X[:, a] * (y - p), minlength=G) for a in range(2)]) - lam * F
    assert np.max(np.abs(grad)) < 1e-8                                    # each group's penalized score is zero


def test_d0_equals_the_w1_fit():
    clean = pd.read_csv(ROOT / "data" / "clean.csv", low_memory=False)
    d, X, y, ri, ci, folds = C.setup(clean)
    rng = np.random.default_rng(YE.WITHIN_SEED)                             # W1's folds, as yes_extensions.within_event deals them
    assert np.array_equal(folds, YE.within_rater_folds(d.rater.to_numpy(), 5, rng))
    rows = np.where(d.event.isin([1, 2, 3]).to_numpy())[0]
    tr, te = rows[folds[rows] != 0], rows[folds[rows] == 0]
    Xd = YE.indicator_design(d.rater.to_numpy(), d.ratee.to_numpy())
    w, _ = M.fit_logit(Xd[tr], y[tr], ridge=1.0, iters=200)                  # W1's fit
    f = C.fit_chemistry(X[tr], ri[tr], ci[tr], y[tr], 0, 0.0, None, ri.max() + 1, ci.max() + 1)
    assert np.allclose(f["w"], w, atol=1e-10)
    assert np.allclose(C.predict(f, X[te], ri[te], ci[te]), M.predict_logit(w, Xd[te]), atol=1e-12)


def chemistry_data(rng, events=3, side=20, scale=2.0):
    rows, pid = [], 0
    for e in range(events):
        women, men = range(pid, pid + side), range(pid + side, pid + 2 * side); pid += 2 * side
        rows += [dict(event=e, rater=a, ratee=b) for a in women for b in men] + [dict(event=e, rater=b, ratee=a) for a in women for b in men]
    d = pd.DataFrame(rows)
    people = pid
    a, b = rng.normal(0, 0.5, people), rng.normal(0, 0.5, people)
    u, v = rng.normal(0, 1, people), rng.normal(0, 1, people)
    uv = scale * u[d.rater] * v[d.ratee]
    y = (rng.random(len(d)) < 1 / (1 + np.exp(-(-0.3 + a[d.rater] + b[d.ratee] + uv)))).astype(float)
    X = sp.csr_matrix(YE.indicator_design(d.rater.to_numpy(), d.ratee.to_numpy()))
    return d, X, y, pd.factorize(d.rater)[0], pd.factorize(d.ratee)[0], uv


def test_alternating_steps_lower_the_penalized_loss():
    rng = np.random.default_rng(7)
    d, X, y, ri, ci, uv = chemistry_data(rng)
    nr, nc = ri.max() + 1, ci.max() + 1
    f0 = C.fit_chemistry(X, ri, ci, y, 0, 0.0, None, nr, nc)
    for dim in (1, 2):
        f = C.fit_chemistry(X, ri, ci, y, dim, 1.0, np.random.default_rng(8), nr, nc, starts=2, rounds=30, trace=True)
        tr = np.array(f["trace"])
        assert np.all(np.diff(tr) <= 1e-9 * np.abs(tr[1:]))                 # every block update lowers it (or leaves it)
        assert f["loss"] < f0["loss"] - 10
    f = C.fit_chemistry(X, ri, ci, y, 1, 1.0, np.random.default_rng(8), nr, nc, starts=2, rounds=30)
    fitted = np.sum(f["U"][ri] * f["V"][ci], 1)
    assert np.corrcoef(fitted, uv)[0, 1] > 0.7                               # the rank-1 term finds the built-in chemistry


def test_large_lambda_collapses_to_w1():
    rng = np.random.default_rng(9)
    d, X, y, ri, ci, _ = chemistry_data(rng, scale=0.0)
    nr, nc = ri.max() + 1, ci.max() + 1
    f0 = C.fit_chemistry(X, ri, ci, y, 0, 0.0, None, nr, nc)
    f = C.fit_chemistry(X, ri, ci, y, 1, 50.0, np.random.default_rng(10), nr, nc, starts=1, rounds=25)
    assert np.max(np.abs(f["U"] * f["V"].max())) < 1e-6 and np.allclose(C.predict(f, X, ri, ci), C.predict(f0, X, ri, ci), atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
