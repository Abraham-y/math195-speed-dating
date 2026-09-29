"""Known-answer tests for scripts/pairs.py (convention 7).
Run from the repository root: python tests/test_pairs.py   (pytest also collects it)
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import pairs as P


def person(**kw):
    """One person's pre-event answers in clean.csv names, with defaults for anything not given."""
    d = {"age": 25, "field_cd": 1, "career_c": 1, "goal": 1, "date": 5, "go_out": 2, "exphappy": 6,
         **{c: 5 for c in P.INTERESTS}, **{f"{a}3_1": 7 for a in P.SELF_ATTRS},
         **{f"{a}1_1_share": 100 / 6 for a in P.ATTRS}}
    d.update(kw)
    return d


def dyad(f_iid, m_iid, f, m, f_gives=None, m_gives=None, f_dec=1, m_dec=1, event=1):
    """The two rows clean.csv holds for one dyad: the woman's (gender 0) and the man's (gender 1)."""
    rows = []
    for iid, pid, g, own, gives, dec in ((f_iid, m_iid, 0, f, f_gives, f_dec), (m_iid, f_iid, 1, m, m_gives, m_dec)):
        r = {"rater": iid, "ratee": pid, "gender": g, "event": event, "dyad": f"{min(iid, pid)}_{max(iid, pid)}",
             "dec": dec, "match": f_dec * m_dec, "int_corr": 0.1, "condtn": 2, "pref_scale": "100pt",
             "wave_undergrad": 0, "wave_yes_budget": 0, "wave_diff_mc": 0, "wave_brought_item": 0,
             **own, **{a: 6 for a in P.ATTRS + ["like"]}, "prob": 5}
        r.update(gives or {})
        rows.append(r)
    return rows


def toy():
    """Woman 1 meets men 3 and 4. Values chosen so every pair feature has a hand-computed answer."""
    w = person(age=25, field_cd=2, goal=1,
               **dict(zip(P.INTERESTS, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7])),
               **{"attr1_1_share": 40, "sinc1_1_share": 10, "intel1_1_share": 10, "fun1_1_share": 20,
                  "amb1_1_share": 10, "shar1_1_share": 10},
               **{"attr3_1": 8, "sinc3_1": 6, "intel3_1": 7, "fun3_1": 9, "amb3_1": 5})
    m3 = person(age=30, field_cd=2, goal=3,
                **dict(zip(P.INTERESTS, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7])),
                **{"attr3_1": 9, "sinc3_1": 6, "intel3_1": 7, "fun3_1": 8, "amb3_1": 5})
    m4 = person(age=23, field_cd=np.nan, goal=1,
                **dict(zip(P.INTERESTS, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 3])))
    rows = (dyad(1, 3, w, m3, f_gives={"attr": 7, "prob": 4}, m_gives={"attr": 9, "like": 8}, f_dec=1, m_dec=1)
            + dyad(1, 4, w, m4, f_dec=0, m_dec=1))
    return pd.DataFrame(rows[::-1])   # reversed, so the men's rows come first


def built():
    return P.build_pairs(toy()).set_index("m_iid")


def test_row_corr_known():
    A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 3], [1, 1, 1, 1], [1, 2, np.nan, 4]])
    B = np.array([[2, 4, 6, 8], [4, 3, 2, 1], [1, 3, 2, 2], [1, 2, 3, 4], [1, 2, 3, 4]])
    r = P.row_corr(A, B)
    assert np.isclose(r[0], 1) and np.isclose(r[1], -1)
    assert np.isclose(r[2], np.corrcoef(A[2], B[2])[0, 1])
    assert np.isnan(r[3]) and np.isnan(r[4])   # no spread; missing value


def test_pref_fit_known():
    W = np.array([[40, 10, 10, 20, 10], [20, 20, 20, 20, 20], [40, 10, np.nan, 20, 10]])
    S = np.array([[9, 6, 7, 8, 5], [9, 6, 7, 8, 5], [9, 6, 7, 8, 5]])
    r = P.pref_fit(W, S)
    assert np.isclose(r[0], (40 * 9 + 10 * 6 + 10 * 7 + 20 * 8 + 10 * 5) / 90)   # 700 / 90
    assert np.isclose(r[1], 7.0)                                                  # equal weights: plain mean
    assert np.isnan(r[2])


def test_same_code_known():
    r = P.same_code([1, 2, np.nan, 4], [1, 3, 1, np.nan])
    assert r[0] == 1 and r[1] == 0 and np.isnan(r[2]) and np.isnan(r[3])


def test_build_pairs_assigns_f_and_m():
    b = built()
    assert len(b) == 2 and set(b.index) == {3, 4} and (b.f_iid == 1).all()
    assert b.loc[3, "f_age"] == 25 and b.loc[3, "m_age"] == 30 and b.loc[4, "m_age"] == 23
    assert b.loc[3, "f_rating_attr"] == 7 and b.loc[3, "m_rating_attr"] == 9      # rating each gave the other
    assert b.loc[3, "f_prob"] == 4 and b.loc[3, "m_rating_like"] == 8
    assert b.loc[3, "match"] == 1 and b.loc[4, "match"] == 0 and b.loc[4, "f_dec"] == 0 and b.loc[4, "m_dec"] == 1


def test_build_pairs_pair_features_known():
    b = built()
    assert b.loc[3, "age_diff"] == 5 and b.loc[3, "abs_age_diff"] == 5
    assert b.loc[4, "age_diff"] == -2 and b.loc[4, "abs_age_diff"] == 2
    assert b.loc[3, "same_field"] == 1 and np.isnan(b.loc[4, "same_field"])       # man 4's field is missing
    assert b.loc[3, "same_goal"] == 0 and b.loc[4, "same_goal"] == 1
    assert np.isclose(b.loc[3, "int_corr17"], 1.0)                                  # identical interests
    w_int = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7])
    m4_int = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 3])
    assert np.isclose(b.loc[4, "int_corr17"], np.corrcoef(w_int, m4_int)[0, 1])
    # her weights on (attr, sinc, intel, fun, amb) are 40, 10, 10, 20, 10; man 3 rates himself 9, 6, 7, 8, 5
    assert np.isclose(b.loc[3, "fit_to_f_prefs"], 700 / 90)
    # man 3 has equal shares on all six, so equal weights on five: the plain mean of her self-ratings 8, 6, 7, 9, 5
    assert np.isclose(b.loc[3, "fit_to_m_prefs"], 7.0)


def test_swapping_gender_swaps_directional_features_only():
    t = toy()
    b, s = built(), P.build_pairs(t.assign(gender=1 - t.gender)).set_index("f_iid")   # men now in the f_ slot
    for c in ["abs_age_diff", "int_corr17", "same_field", "same_goal"]:
        assert np.allclose(b[c], s.loc[b.index, c], equal_nan=True), c
    assert np.allclose(b.fit_to_f_prefs, s.loc[b.index, "fit_to_m_prefs"])
    assert np.allclose(b.age_diff, -s.loc[b.index, "age_diff"])


def test_feature_sets_consistent():
    sets = P.feature_sets()
    pre, full = sets["pre"], sets["pre+date"]
    assert len(set(full)) == len(full) and full[:len(pre)] == pre and full[len(pre):] == sets["date"]
    assert set(pre) == set(sets["pre_individual"]) | set(sets["pre_pair"])
    banned = {"match", "f_dec", "m_dec", "age_diff", "int_corr17"} | {f"{p}_{c}" for p in "fm" for c in ["race", "imprace", "samerace"]}
    assert not banned & set(full)
    assert not set(sets["date"]) & set(pre)                                      # no in-date column in `pre`
    assert set(sets["categorical"]) <= set(pre)
    assert set(full) <= set(built().reset_index().columns)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
