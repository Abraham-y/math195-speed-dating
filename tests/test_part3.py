"""Known-answer tests for the Part 3 scripts (mean_reversion, stable_matching, srm, sim_cv_designs).
Run from the repository root: python tests/test_part3.py"""
import sys, pathlib, itertools
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import modeling as M
import mean_reversion as MR
import stable_matching as SMT
import srm as SRM
import sim_cv_designs as SD


def test_shrink_known_answer():
    raw, var_w, n = np.array([1.0, 2, 3, 4, 5]), np.full(5, 2.0), np.full(5, 3)
    shrunk, keep = MR.shrink(raw, var_w, n)
    tau2 = 2.5 - 2 / 3                                    # var(raw) minus the mean sampling variance 2/3
    assert np.allclose(keep, tau2 / (tau2 + 2 / 3))
    assert np.allclose(shrunk, 3 + tau2 / (tau2 + 2 / 3) * (raw - 3))
    flat, keep0 = MR.shrink(np.array([2.0, 2.2, 1.8]), np.full(3, 4.0), np.full(3, 2))   # spread below the noise: all the way
    assert np.allclose(keep0, 0) and np.allclose(flat, 2.0)


def all_matchings(men, women):
    for perm in itertools.permutations(women, len(men)):
        yield dict(zip(men, perm))


def test_gale_shapley_is_stable_and_proposer_optimal():
    mp = {"A": ["X", "Y", "Z"], "B": ["Y", "Z", "X"], "C": ["Z", "X", "Y"]}
    wp = {"X": ["B", "C", "A"], "Y": ["C", "A", "B"], "Z": ["A", "B", "C"]}
    men_opt, wom_opt = SMT.gale_shapley(mp, wp), SMT.gale_shapley(wp, mp)
    assert men_opt == {"A": "X", "B": "Y", "C": "Z"} and wom_opt == {"X": "B", "Y": "C", "Z": "A"}
    stable = [m for m in all_matchings(list(mp), list(wp)) if not SMT.blocking_pairs(m, mp, wp)]
    assert men_opt in stable and {w: m for m, w in men_opt.items()} != wom_opt
    for m in mp:                                           # each man gets his best partner over all stable matchings
        assert min(mp[m].index(s[m]) for s in stable) == mp[m].index(men_opt[m])
    assert SMT.blocking_pairs({"A": "Y", "B": "X", "C": "Z"}, mp, wp)       # A and X prefer each other: blocked
    rng = np.random.default_rng(0)                         # random lists, unequal sides: always stable, everyone on the short side matched
    for _ in range(50):
        men, women = [f"m{i}" for i in range(3)], [f"w{i}" for i in range(4)]
        mp = {m: list(rng.permutation(women)) for m in men}; wp = {w: list(rng.permutation(men)) for w in women}
        res = SMT.gale_shapley(mp, wp)
        assert len(res) == 3 and not SMT.blocking_pairs(res, mp, wp)


def toy_event(noise=0.0, seed=1):
    """3 women (ids 1-3) and 4 men (ids 11-14); each person's partner effect equals their actor effect, and the pair
    residuals of the two directions are equal (double-centred), so both reciprocities are 1 when noise = 0."""
    rng = np.random.default_rng(seed)
    w, m = [1, 2, 3], [11, 12, 13, 14]
    eff = dict(zip(w + m, rng.normal(size=7)))
    E = rng.normal(size=(3, 4)); E = E - E.mean(0) - E.mean(1, keepdims=True) + E.mean()
    rows = []
    for i, a in enumerate(w):
        for j, b in enumerate(m):
            rows.append(dict(event=1, gender=0, rater=a, ratee=b, like=6 + eff[a] + eff[b] + E[i, j] + noise * rng.normal()))
            rows.append(dict(event=1, gender=1, rater=b, ratee=a, like=5 + eff[b] + eff[a] + E[i, j] + noise * rng.normal()))
    return pd.DataFrame(rows), eff, E


def test_additive_recovers_exact_effects_with_missing_cells():
    rng = np.random.default_rng(3)
    r, c = np.repeat(np.arange(5), 6), np.tile(np.arange(6), 5)
    a, b = rng.normal(size=5), rng.normal(size=6)
    keep = rng.random(30) > 0.2; keep[[0, 7, 14, 21, 28]] = True
    x = (2 + a[r] + b[c])[keep]
    mu, ah, bh, e = SRM.additive(x, r[keep], c[keep])
    assert np.allclose(e, 0, atol=1e-8)
    assert np.allclose(ah.sort_index().to_numpy(), a - a.mean(), atol=1e-8) and np.allclose(bh.sort_index().to_numpy(), b - b.mean(), atol=1e-8)


def test_reciprocities_known_answer():
    ev, eff, E = toy_event()
    s = SRM.estimates([SRM.event_pieces(ev)])
    assert np.isclose(s["dyadic_reciprocity"], 1) and np.isclose(s["generalized_reciprocity"], 1)
    assert np.isclose(s["generalized_reciprocity_corrected"], 1)
    assert s["dyads"] == 12 and s["people"] == 7
    assert np.isclose(s["actor_share"] + s["partner_share"] + s["residual_share"], 1)


def test_corrected_generalized_reciprocity_removes_the_leak():
    """Actor and partner effects independent (truth 0), pair effects strongly reciprocal (0.8): the registered estimator
    picks up the pair correlation, the corrected one does not (averaged over replicates). 12 events of 8 women and 8
    men; with events of 3 or 4 a side the corrected variances can approach zero and the ratio is unstable."""
    rows = [dict(event=s, gender=g, rater=(a if g == 0 else b), ratee=(b if g == 0 else a), like=6.0)
            for s in range(12) for a in range(100 * s, 100 * s + 8) for b in range(100 * s + 50, 100 * s + 58) for g in (0, 1)]
    clean = pd.DataFrame(rows)
    truth = dict(generalized=0.0, dyadic=0.8, actor_var=0.3, partner_var=0.3, residual_var=1.0)
    rng = np.random.default_rng(7); naive, corr = [], []
    for _ in range(100):
        sim = SRM.simulate(clean, truth, rng)
        s = SRM.estimates([SRM.event_pieces(e) for _, e in sim.groupby("event")])
        naive.append(s["generalized_reciprocity"]); corr.append(s["generalized_reciprocity_corrected"])
    assert np.mean(naive) > 0.15                            # the leak: about 0.23 on this design
    assert abs(np.mean(corr)) < 0.04                        # replicate SD about 0.11, so the mean's SE is about 0.011


def brute_dyadic(e, a, b):
    r, n = e - e.mean(), len(e)
    N = len(set(a) | set(b))
    tot = sum(r[i] * r[j] for i in range(n) for j in range(n) if {a[i], b[i]} & {a[j], b[j]})
    return N / (N - 1) * tot / n ** 2


def test_dyadic_and_two_way_variances_against_brute_force():
    rng = np.random.default_rng(5)
    w, m = np.arange(4), np.arange(10, 15)
    W, Mn = [v.ravel() for v in np.meshgrid(w, m, indexing="ij")]
    a, b = np.r_[W, Mn], np.r_[Mn, W]                       # every pair in both directions
    e = rng.random(len(a))
    assert np.isclose(SD.dyadic_var_of_mean(e, a, b), brute_dyadic(e, a, b))
    ab = pd.factorize(pd.Series(a).astype(str) + "|" + pd.Series(b).astype(str))[0]
    assert np.isclose(SD.two_way_var_of_mean(e, a, b),
                      M.cluster_robust_var_of_mean(e, a) + M.cluster_robust_var_of_mean(e, b) - M.cluster_robust_var_of_mean(e, ab))
    one = SD.dyadic_var_of_mean(e[:1].repeat(len(e)), a, b)
    assert np.isclose(one, 0)


def test_two_way_bias_matches_proposition_4():
    """proposals/06, Proposition 4(ii): in a balanced R x C array with rater, ratee and cell effects, E[two-way CGM]
    minus Var(mean) = -(C(R-1) s_a^2 + R(C-1) s_b^2) / (n(n-1))."""
    R, C, sa, sb, sg = 6, 8, 1.0, 0.7, 0.5
    n = R * C
    a, b = np.repeat(np.arange(R), C), np.tile(np.arange(C), R) + 100
    rng = np.random.default_rng(11); est, means = [], []
    for _ in range(20000):
        e = sa * rng.normal(size=R)[a] + sb * rng.normal(size=C)[b - 100] + sg * rng.normal(size=n)
        est.append(SD.two_way_var_of_mean(e, a, b)); means.append(e.mean())
    truth = sa ** 2 / R + sb ** 2 / C + sg ** 2 / n
    bias = -(C * (R - 1) * sa ** 2 + R * (C - 1) * sb ** 2) / (n * (n - 1))
    assert abs(np.var(means) / truth - 1) < 0.03
    assert abs(np.mean(est) - (truth + bias)) < 0.03 * truth


def test_resample_events_gives_fresh_people():
    rng = np.random.default_rng(2)
    e = np.repeat(np.arange(21), 4); f = np.repeat(np.arange(42), 2); m = np.arange(84) % 50
    X = rng.normal(size=(84, 2))
    Xs, fs, ms, es = SD.resample_events(X, f, m, e, 40, rng)
    assert es.max() + 1 == 40 and len(Xs) == 160
    assert all(len(set(fs[es == k]) & set(fs[es != k])) == 0 for k in range(40))      # no woman in two drawn events
    Xs, fs, ms, es = SD.resample_events(X, f, m, e, 10, rng)
    assert es.max() + 1 == 10 and len(Xs) == 40
    X21 = SD.resample_events(X, f, m, e, 21, rng)
    assert X21[0] is X


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
