"""Known-answer tests for Part 10 (scripts/limits.py): the learning curve's draws and extrapolation, the dyad folds,
the leakage-free person ceiling, the person-effect estimates, the split-half reliability and the corrected R2.
All on simulated data. Run from the repository root: python tests/test_limits.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import modeling as M
import limits as L


def test_extrapolate_recovers_a_power_law():
    n = np.array([800, 1600, 2400, 3300.0])
    for c in (0.5, 1.0):
        a = 0.6 - 2.0 * n ** (-c)
        ainf, b = L.extrapolate(np.column_stack([n, a]), c)
        assert np.isclose(ainf, 0.6) and np.isclose(b, 2.0)


def test_subsets_keep_test_folds_and_draw_from_training_events():
    folds = {e: (e - 1) % 5 for e in range(1, 22)}
    draws = L.subsets(folds)
    assert len(draws) == 5 * (len(L.SIZES) * L.REPS + 1)
    for k, m, r, S in draws:
        train = {e for e, f in folds.items() if f != k}
        assert set(S) <= train and not any(folds[e] == k for e in S)
        assert len(set(S)) == (len(train) if m == "all" else m)
    assert draws == L.subsets(folds)                                         # seeded: the same draws every time


def make_dates(G=8, nf=8, nm=8, sa=1.0, sb=1.0, sd=0.0, seed=0):
    """Simulated events: every woman meets every man; logit P(yes) = -0.3 + a_rater + b_ratee + dyad effect."""
    rng = np.random.default_rng(seed); rows, pid, dyad = [], 0, 0
    a, b = {}, {}
    for e in range(G):
        women, men = list(range(pid, pid + nf)), list(range(pid + nf, pid + nf + nm)); pid += nf + nm
        for p in women + men:
            a[p], b[p] = rng.normal(0, sa), rng.normal(0, sb)
        for w in women:
            for m in men:
                u = rng.normal(0, sd)
                for rater, ratee, g in ((w, m, 0), (m, w, 1)):
                    eta = -0.3 + a[rater] + b[ratee] + u
                    rows.append(dict(rater=rater, ratee=ratee, event=e, dyad=dyad, gender=g, dec=int(rng.random() < 1 / (1 + np.exp(-eta))),
                                     **{r: rng.normal() for r in L.RATINGS}))
                dyad += 1
    d = pd.DataFrame(rows)
    d["match"] = d.groupby("dyad").dec.transform("min")
    return d, pd.Series(a), pd.Series(b)


def test_dyad_folds_hold_out_whole_dyads_evenly():
    d, _, _ = make_dates(G=3, nf=5, nm=6)
    f = L.dyad_folds(d.dyad.to_numpy(), d.event.to_numpy())
    assert set(f) == set(d.dyad)
    for _, g in d.drop_duplicates("dyad").groupby("event"):
        sizes = pd.Series([f[x] for x in g.dyad]).value_counts()
        assert sizes.max() - sizes.min() <= 1 and len(sizes) == 5


def test_person_ceiling_does_not_leak_the_pair():
    """Decisions driven only by a shared dyad effect: if a dyad's own decisions reached its training set the match
    AUC would be high. Without leakage, nothing about the pair is learnable, so it is near 0.5."""
    d, _, _ = make_dates(G=10, sa=0.0, sb=0.0, sd=3.0, seed=1)
    f = L.dyad_folds(d.dyad.to_numpy(), d.event.to_numpy())
    s = L.match_scores(d, L.person_scores(d, f, "person"))
    assert abs(M.auc(s.score, s.match) - 0.5) < 0.06


def test_person_ceiling_finds_person_effects_and_ignores_noise_ratings():
    """The fitted ceiling recovers most of what the true effects give (about 6 training dates per person here) and
    never more."""
    d, a, b = make_dates(G=10, sa=1.2, sb=1.2, seed=2)
    f = L.dyad_folds(d.dyad.to_numpy(), d.event.to_numpy())
    person = L.match_scores(d, L.person_scores(d, f, "person"))
    date = L.match_scores(d, L.person_scores(d, f, "date"))                  # ratings are pure noise here
    eta = -0.3 + a.loc[d.rater].to_numpy() + b.loc[d.ratee].to_numpy()
    oracle = L.match_scores(d, 1 / (1 + np.exp(-eta)))
    A, A_oracle = M.auc(person.score, person.match), M.auc(oracle.score, oracle.match)
    assert 0.5 + 0.6 * (A_oracle - 0.5) < A < A_oracle + 0.02, (A, A_oracle)
    assert abs(M.auc(date.score, date.match) - 0.5) < 0.06


def test_person_effects_track_the_truth():
    """12 decisions per person give a logit-scale noise variance of about 1 / (12 x 0.24), so with unit-variance
    effects the correlation should be about sqrt(1 / 1.35) = 0.86 before shrinkage; seeds 3 to 5 give 0.80 to 0.83."""
    d, a, b = make_dates(G=10, nf=12, nm=12, seed=3)
    ea, eb = L.person_effects(d)
    assert np.corrcoef(ea, a.loc[ea.index])[0, 1] > 0.7 and np.corrcoef(eb, b.loc[eb.index])[0, 1] > 0.7


def test_split_half_reliability_matches_the_truth():
    """With known effects, the reliability Var(true) / Var(estimate) is about corr(estimate, true)^2."""
    assert np.isclose(L.spearman_brown(0.5), 2 / 3)
    d, a, b = make_dates(G=12, nf=10, nm=10, seed=4)
    rel = L.reliability(d, halves=6)
    ea, eb = L.person_effects(d)
    for key, est, true in (("yes_rate", ea, a), ("appeal", eb, b)):
        target = np.corrcoef(est, true.loc[est.index])[0, 1] ** 2
        assert abs(rel[key][0] - target) < 0.1, (key, rel[key], target)


def test_corrected_r_squared():
    """A target measured with reliability 0.6 and a predictor explaining 30% of the true target: the held-out R2 on
    the noisy target is about 0.18, and dividing by the reliability gives back about 0.30."""
    rng = np.random.default_rng(5); n = 200_000
    x = rng.normal(size=n); true = np.sqrt(0.3) * x + np.sqrt(0.7) * rng.normal(size=n)
    noisy = true + np.sqrt(1 / 0.6 - 1) * rng.normal(size=n)
    r2 = M.r_squared_and_rmse(noisy, np.sqrt(0.3) * x, np.arange(n) % 50)["r_squared"]
    assert abs(r2 - 0.18) < 0.01 and abs(r2 / 0.6 - 0.30) < 0.015


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
