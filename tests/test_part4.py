"""Known-answer tests for the Part 4 code (race block, reciprocity of yes decisions, error-bar repair).
Run from the repository root: python tests/test_part4.py"""
import sys, pathlib, json
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import modeling as M
import race_models as RM
import srm as SRM
import sim_cv_repair as SR


def test_imprace_out_of_range_is_missing():
    v = RM.imprace_clean(pd.Series([0, 1, 5.0, 10, 11, np.nan]))
    assert v.isna().tolist() == [True, False, False, False, True, True]


def test_race_block_columns_and_feature_set():
    sets = json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "feature_sets.json").read_text())
    pairs = pd.read_csv(pathlib.Path(__file__).resolve().parents[1] / "data" / "pairs.csv")
    X, y, ev, ids = M.build_frame(pairs, sets, M.HEADLINE, "pre_race")
    base, *_ = M.build_frame(pairs, sets, M.HEADLINE, "pre_interactions")
    assert list(X.columns) == list(base.columns[:-len(sets["flags_pre"])]) + sets["race_block"] + sets["flags_pre"]
    assert np.allclose(X.samerace_x_f_imprace.fillna(-1), (pairs.samerace * pairs.f_imprace).fillna(-1))
    for fs in ("pre", "pre+date", "pre_individual", "pre_interactions"):           # race enters no other set
        Xo, *_ = M.build_frame(pairs, sets, M.HEADLINE, fs)
        assert not any("race" in c for c in Xo.columns)
    assert pairs.groupby("f_iid").f_imprace.nunique(dropna=False).max() == 1      # one stated importance per person


def test_binary_srm_simulation_hits_the_yes_rate():
    rows = [dict(event=s, gender=g, rater=(a if g == 0 else b), ratee=(b if g == 0 else a), like=6.0, dec=1.0)
            for s in range(3) for a in range(100 * s, 100 * s + 6) for b in range(100 * s + 50, 100 * s + 56) for g in (0, 1)]
    clean = pd.DataFrame(rows)
    sim = SRM.simulate_binary(clean, dict(actor_share=0.3, partner_share=0.2, residual_share=0.5), 0.2, 0.4, np.random.default_rng(1))
    assert set(np.unique(sim.dec)) <= {0.0, 1.0} and abs(sim.dec.mean() - 0.4) < 1 / len(sim)


def test_nadeau_bengio_known_answer():
    losses = np.array([0.30, 0.34, 0.32, 0.36, 0.28, 0.33, 0.31, 0.35, 0.29, 0.32])
    c, se, q = SR.nadeau_bengio(losses, K=5)
    assert np.isclose(c, losses.mean())
    assert np.isclose(se, np.sqrt((1 / 10 + 1 / 4) * losses.var(ddof=1)))
    assert np.isclose(q, 1.8331129326536335)                 # t quantile 0.95 with 9 degrees of freedom


def test_dersimonian_laird_known_answers():
    m, se, tau2 = SR.dersimonian_laird([0.3, 0.3, 0.3], [0.01, 0.01, 0.01])            # homogeneous: fixed-effect mean
    assert np.isclose(m, 0.3) and np.isclose(se, np.sqrt(0.01 / 3)) and tau2 == 0
    est, var = np.array([0.1, 0.5]), np.array([0.01, 0.01])                               # two studies, by hand
    Q = np.sum((est - 0.3) ** 2 / 0.01)                                                   # 8.0
    tau2_hand = (Q - 1) / (200 - 20000 / 200)                                             # (8 - 1) / 100 = 0.07
    m, se, tau2 = SR.dersimonian_laird(est, var)
    assert np.isclose(tau2, tau2_hand) and np.isclose(m, 0.3) and np.isclose(se, np.sqrt(1 / (2 / (0.01 + tau2_hand))))


def test_bootstrap_folds_keep_copies_together():
    rng = np.random.default_rng(3)
    orig = np.repeat(rng.integers(0, 21, 21), 7)
    f = SR.bootstrap_folds(orig, 5, rng)
    assert all(len(set(f[orig == v])) == 1 for v in np.unique(orig))
    assert set(np.bincount(pd.Series(f).groupby(orig).first())) <= {len(np.unique(orig)) // 5, len(np.unique(orig)) // 5 + 1}


def test_redraw_events_always_redraws():
    rng = np.random.default_rng(4)
    e = np.repeat(np.arange(21), 4); f = np.repeat(np.arange(42), 2); m = np.arange(84) % 50
    X = rng.normal(size=(84, 2))
    Xs, fs, ms, es = SR.redraw_events(X, f, m, e, 21, rng)
    assert es.max() + 1 == 21 and len(Xs) == 84 and not np.array_equal(Xs, X)          # with replacement, so not the identity
    assert all(len(set(fs[es == k]) & set(fs[es != k])) == 0 for k in range(21))
    Xs, fs, ms, es = SR.redraw_events(X, f, m, e, 10, rng)
    assert es.max() + 1 == 10 and len(Xs) == 40


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
