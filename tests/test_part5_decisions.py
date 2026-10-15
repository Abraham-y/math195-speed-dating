"""Known-answer tests for the Part 5 decision analyses: the matchmaker (M1), multiple-testing control and power.
Run from the repository root: python tests/test_part5_decisions.py"""
import sys, pathlib, itertools
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import matchmaker as MM
import multiple_testing as MT
import power as PW


def brute_force_best(W):
    """The largest total weight of any one-to-one matching (not necessarily complete) on the present (non-NaN) pairs."""
    n, m = W.shape
    best = 0.0
    def rec(i, used, tot):
        nonlocal best
        if i == n:
            best = max(best, tot); return
        rec(i + 1, used, tot)                                    # row i stays unpaired
        for j in range(m):
            if j not in used and not np.isnan(W[i, j]):
                rec(i + 1, used | {j}, tot + W[i, j])
    rec(0, frozenset(), 0.0)
    return best


def test_assignment_never_picks_absent_pairs_and_is_optimal():
    rng = np.random.default_rng(7)
    for shape in [(3, 4), (4, 3), (4, 4), (5, 3), (2, 5)] * 8:
        W = rng.random(shape)
        W[rng.random(shape) < 0.35] = np.nan
        chosen = MM.assign(W)
        assert all(not np.isnan(W[i, j]) for i, j in chosen)
        assert len({i for i, _ in chosen}) == len(chosen) == len({j for _, j in chosen})
        assert np.isclose(sum(W[i, j] for i, j in chosen), brute_force_best(W))
    # only one pair can be formed: the solver's second pair would be absent, so it is dropped
    assert MM.assign(np.array([[0.9, np.nan], [0.8, np.nan]])) == [(0, 0)]
    # two pairs beat the single best pair
    assert sorted(MM.assign(np.array([[0.5, 0.6], [np.nan, 0.55]]))) == [(0, 0), (1, 1)]
    # complete grid: a full one-to-one assignment of min(rows, columns) pairs
    assert len(MM.assign(rng.random((4, 6)))) == 4


def test_expected_random_pairing():
    assert np.isclose(MM.expected_random(3, 5, 3, 15), 0.6)
    assert np.isclose(MM.expected_random(22, 22, 72, 484), 22 * 72 / 484)
    # the formula is the mean over every one-to-one assignment of min(women, men) pairs on a complete grid
    for match in (np.array([[1, 0, 1], [0, 1, 0]]), np.array([[1, 0], [0, 0], [1, 1]])):
        nw, nm = match.shape
        if nw <= nm:
            totals = [sum(match[i, p[i]] for i in range(nw)) for p in itertools.permutations(range(nm), nw)]
        else:
            totals = [sum(match[p[j], j] for j in range(nm)) for p in itertools.permutations(range(nw), nm)]
        assert np.isclose(np.mean(totals), MM.expected_random(nw, nm, match.sum(), match.size))


def test_precision_at_k():
    pred = pd.DataFrame({"event": 1, "f_iid": [1, 1, 1, 2, 2, 2], "m_iid": [10, 11, 12, 10, 11, 12],
                         "match": [1, 0, 1, 0, 0, 1], "p": [0.9, 0.5, 0.1, 0.3, 0.3, 0.2]})
    w1 = MM.precision_at_k(pred, "women", "p", 1).set_index("iid")
    assert w1.precision.tolist() == [1.0, 0.0]                  # woman 2: tie at 0.3 broken by partner iid -> man 10
    w2 = MM.precision_at_k(pred, "women", "p", 2).set_index("iid")
    assert w2.precision.tolist() == [0.5, 0.0] and np.allclose(w2.own_rate, [2 / 3, 1 / 3])
    m1 = MM.precision_at_k(pred, "men", "p", 1).set_index("iid")
    assert m1.precision.tolist() == [1.0, 0.0, 1.0]             # man 11: woman 1 at 0.5 > woman 2 at 0.3, no match


def test_bh_by_textbook():
    """Benjamini and Hochberg (1995), section 4: 15 p-values; at q = 0.05 BH rejects 4 (Bonferroni 3)."""
    p = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719,
                  0.6528, 0.7590, 1.0000])
    assert MT.bh_reject(p, 0.05).sum() == 4 and MT.bh_reject(p, 0.05)[:4].all()
    assert MT.by_reject(p, 0.05).sum() == 3                     # thresholds i x 0.05 / (15 x 3.318)
    assert MT.bh_reject(p, 0.10).sum() == 9                     # p_(9) = 0.0459 <= 9 x 0.10 / 15 = 0.06
    assert MT.by_reject(p, 0.10).sum() == 3                     # p_(4) = 0.0095 > 4 x 0.10 / (15 x 3.318) = 0.0080
    shuffled = np.random.default_rng(3).permutation(len(p))     # the order of the input does not matter
    assert np.allclose(MT.bh_adjusted(p[shuffled]), MT.bh_adjusted(p)[shuffled])
    from statsmodels.stats.multitest import multipletests
    assert np.allclose(MT.bh_adjusted(p), multipletests(p, 0.10, "fdr_bh")[1])
    assert np.allclose(MT.by_adjusted(p), multipletests(p, 0.10, "fdr_by")[1])
    assert np.isclose(MT.two_sided_p(1.959963984540054), 0.05) and np.isclose(MT.two_sided_p(-1.959963984540054), 0.05)


def test_storey_pi0():
    p = np.array([0.001] * 6 + [0.2, 0.6, 0.75, 0.9])          # 3 of 10 above 0.5: pi0 = 3 / (10 x 0.5)
    assert np.allclose(MT.storey_pi0(p), (0.6, 0.6))
    assert np.allclose(MT.storey_pi0(np.array([0.6, 0.7, 0.8, 0.9])), (1.0, 2.0))     # capped at 1
    assert np.allclose(MT.storey_pi0(np.array([0.5, 0.5, 0.01, 0.9])), (0.5, 0.5))    # p = lambda is not above it


def test_events_needed():
    G = 21
    assert np.isclose(PW.Z_SUM, 1.959964 + 0.841621, atol=1e-6)
    # a difference that sits exactly at z_0.975 + z_0.80 standard errors needs the events already there
    assert np.isclose(PW.events_needed(0.03 / PW.Z_SUM, 0.03, G), G)
    assert np.isclose(PW.power_now(0.03 / PW.Z_SUM, 0.03), 0.80, atol=1e-3)
    # (1.959964 + 0.841621)^2 x (sqrt(21) x 0.02)^2 / 0.04^2 = 7.84888 x 5.25
    assert np.isclose(PW.events_needed(0.02, 0.04, G), 41.2066, atol=1e-3)
    assert np.isclose(PW.events_needed(0.02, 0.02, G), 4 * PW.events_needed(0.02, 0.04, G))   # half the effect, 4x the events


def test_resampled_events_are_separate_clusters():
    ev = np.array([1, 1, 2, 3, 3, 3])
    idx, lab = PW.resample_rows(ev, np.array([1, 2, 3]), np.array([2, 2, 0]))
    assert idx.tolist() == [3, 4, 5, 3, 4, 5, 0, 1] and lab.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
