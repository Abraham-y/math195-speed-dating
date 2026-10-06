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
    for c in P.RECORDED:                      # recorded value defaults to the fixed one
        d.setdefault(f"{c}_recorded", d[c])
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


W_INT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7]
M4_INT = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 3]


def toy():
    """Woman 1 meets men 3 and 4. Values chosen so every pair feature has a hand-computed answer.
    Man 3's gaming was recorded as 14 and fixed to missing; man 4 left his field blank and skipped one rating."""
    w = person(age=25, field_cd=2, goal=1, **dict(zip(P.INTERESTS, W_INT)),
               **{"attr1_1_share": 40, "sinc1_1_share": 10, "intel1_1_share": 10, "fun1_1_share": 20,
                  "amb1_1_share": 10, "shar1_1_share": 10},
               **{"attr3_1": 8, "sinc3_1": 6, "intel3_1": 7, "fun3_1": 9, "amb3_1": 5})
    m3 = person(age=30, field_cd=2, goal=3, **dict(zip(P.INTERESTS, W_INT)),
                **{"attr3_1": 9, "sinc3_1": 6, "intel3_1": 7, "fun3_1": 8, "amb3_1": 5})
    m3["gaming"], m3["gaming_recorded"] = np.nan, 14
    m4 = person(age=23, field_cd=np.nan, goal=1, **dict(zip(P.INTERESTS, M4_INT)))
    rows = (dyad(1, 3, w, m3, f_gives={"attr": 7, "prob": 4}, m_gives={"attr": 9, "like": 8}, f_dec=1, m_dec=1)
            + dyad(1, 4, w, m4, m_gives={"sinc": np.nan}, f_dec=0, m_dec=1))
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


def test_row_corr_pairwise_known():
    A = np.array([[1, 2, np.nan, 4, 5], [1, 2, 3, 4, 5], [1, np.nan, np.nan, np.nan, 5], [np.nan] * 5])
    B = np.array([[2, 1, 9, 5, 4], [2, 1, 9, 5, 4], [3, 4, 5, 6, 7], [1, 2, 3, 4, 5]])
    r = P.row_corr_pairwise(A, B)
    keep = [0, 1, 3, 4]
    assert np.isclose(r[0], np.corrcoef(A[0, keep], B[0, keep])[0, 1])   # column 2 dropped for that row only
    assert np.isclose(r[1], np.corrcoef(A[1], B[1])[0, 1])               # complete row: ordinary correlation
    assert np.isnan(r[2]) and np.isnan(r[3])                              # 2 shared values (< 3); none shared
    full = np.array([[1, 2, 3, 4], [4, 1, 2, 2]], float)
    assert np.allclose(P.row_corr_pairwise(full, full[::-1]), P.row_corr(full, full[::-1]))


def test_pref_fit_and_compat_known():
    W = np.array([[40, 10, 10, 20, 10], [20, 20, 20, 20, 20], [40, 10, np.nan, 20, 10]])
    S = np.array([[9, 6, 7, 8, 5], [9, 6, 7, 8, 5], [9, 6, 7, 8, 5]])
    a, b = P.pref_fit(W, S), P.compat_fit(W, S)
    assert np.isclose(a[0], 700 / 90) and np.isclose(a[1], 7.0) and np.isnan(a[2])
    # her shares minus 0.2: (4/9 - 1/5, 1/9 - 1/5, 1/9 - 1/5, 2/9 - 1/5, 1/9 - 1/5); his deviations: 2, -1, 0, 1, -2
    assert np.isclose(b[0], (4/9 - .2) * 2 + (1/9 - .2) * -1 + 0 + (2/9 - .2) * 1 + (1/9 - .2) * -2)
    assert np.isclose(b[0], 700 / 90 - 7.0)          # option A = mean self-rating + compatibility term
    assert np.isclose(b[1], 0.0) and np.isnan(b[2])   # equal weights carry no compatibility information
    assert np.isclose(P.compat_fit(W[:1], np.full((1, 5), 6.0))[0], 0.0)   # flat self-ratings: none either


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
    assert np.isnan(b.loc[3, "m_gaming"]) and b.loc[3, "m_gaming_recorded"] == 14
    assert b.loc[3, "int_corr_released"] == 0.1                                     # carried as released


def test_build_pairs_pair_features_known():
    b = built()
    assert b.loc[3, "age_diff"] == 5 and b.loc[3, "abs_age_diff"] == 5
    assert b.loc[4, "age_diff"] == -2 and b.loc[4, "abs_age_diff"] == 2
    assert b.loc[3, "same_field"] == 1 and np.isnan(b.loc[4, "same_field"])
    assert b.loc[3, "same_goal"] == 0 and b.loc[4, "same_goal"] == 1
    # int_corr: 16 interests (no yoga), skipping man 3's missing gaming (index 7): identical otherwise, so 1
    assert np.isclose(b.loc[3, "int_corr"], 1.0)
    w16, m16 = np.array(W_INT[:16], float), np.array(M4_INT[:16], float)
    assert np.isclose(b.loc[4, "int_corr"], np.corrcoef(w16, m16)[0, 1])
    assert np.isclose(b.loc[3, "fit_to_f_prefs"], 700 / 90 - 7.0) and np.isclose(b.loc[3, "fit_raw_to_f_prefs"], 700 / 90)
    assert np.isclose(b.loc[3, "fit_to_m_prefs"], 0.0) and np.isclose(b.loc[3, "fit_raw_to_m_prefs"], 7.0)


def test_missing_flags_count_blanks_not_fixes():
    b = built()
    assert b.loc[3, "m_pre_missing"] == 0     # only gaming fixed to missing; recorded value present
    assert b.loc[4, "m_pre_missing"] == 1     # field left blank
    assert b.loc[4, "m_date_missing"] == 1 and b.loc[3, "m_date_missing"] == 0   # skipped a rating in dyad 1-4
    assert (b.f_pre_missing == 0).all() and (b.f_date_missing == 0).all()


def test_swapping_gender_swaps_directional_features_only():
    t = toy()
    b, s = built(), P.build_pairs(t.assign(gender=1 - t.gender)).set_index("f_iid")   # men now in the f_ slot
    for c in ["abs_age_diff", "int_corr", "int_corr17", "same_field", "same_goal"]:
        assert np.allclose(b[c], s.loc[b.index, c], equal_nan=True), c
    assert np.allclose(b.fit_to_f_prefs, s.loc[b.index, "fit_to_m_prefs"])
    assert np.allclose(b.fit_raw_to_f_prefs, s.loc[b.index, "fit_raw_to_m_prefs"])
    assert np.allclose(b.age_diff, -s.loc[b.index, "age_diff"])


def test_feature_sets_consistent():
    sets = P.feature_sets()
    pre, full = sets["pre"], sets["pre+date"]
    assert len(set(full)) == len(full) and full[:len(pre)] == pre and full[len(pre):] == sets["date"]
    assert set(pre) == set(sets["pre_individual"]) | set(sets["pre_pair"])
    alternates = {c for d in sets["alternatives"].values() for c in d.values()}
    banned = ({"match", "f_dec", "m_dec", "age_diff", "int_corr17"} | alternates | set(sets["flags_pre"] + sets["flags_date"])
              | {f"{p}_{c}" for p in "fm" for c in ["race", "imprace", "samerace"]})
    assert not banned & set(full)
    assert not set(sets["date"]) & set(pre)
    assert set(sets["categorical"]) <= set(pre) and set(sets["spline_candidates"]) <= set(pre)
    assert set(sets["format_standardized"]) <= set(pre) and set(sets["pref_reference"]) <= set(pre)
    cols = set(built().reset_index().columns)
    assert set(full) <= cols and alternates <= cols and set(sets["binary"]) <= cols


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
