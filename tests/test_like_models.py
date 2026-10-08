"""Known-answer tests for scripts/like_models.py and scripts/participants.py.
Run from the repository root: python tests/test_like_models.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import like_models as L
import participants as PT
import pairs as P
from test_pairs import toy


def toy_clean():
    t = toy().copy()
    likes = {(3, 1): 7.0, (1, 3): 9.0, (4, 1): 2.0, (1, 4): np.nan}       # (rater, ratee) -> like
    t["like"] = [likes[(r, e)] for r, e in zip(t.rater, t.ratee)]
    return t


def test_dates_table_maps_rater_and_partner():
    clean = toy_clean()
    D, rater, partner, pair = L.dates_table(P.build_pairs(clean), P.feature_sets(), clean)
    assert len(D) == 3                                                      # the date with no like is dropped
    w = D[(D.r_iid == 1) & (D.p_iid == 3)].iloc[0]; m = D[(D.r_iid == 3) & (D.p_iid == 1)].iloc[0]
    assert w.r_age == 25 and w.p_age == 30 and w.r_male == 0 and w.like == 9.0
    assert m.r_age == 30 and m.p_age == 25 and m.r_male == 1 and m.like == 7.0
    # (in the toy both men have fit 0 to their own preferences, so that standardized column is NaN; compare NaN-aware)
    assert np.isclose(w.fit_to_rater_prefs, m.fit_to_partner_prefs, equal_nan=True) and np.isclose(w.fit_to_partner_prefs, m.fit_to_rater_prefs, equal_nan=True)
    assert not np.isnan(w.fit_to_rater_prefs)
    assert np.isclose(w.abs_age_diff, m.abs_age_diff) and "fit_to_rater_prefs" in pair and "r_male" in rater


def test_column_types_same_goal_is_binary():
    sets = P.feature_sets()
    cols = ["r_goal", "p_goal", "r_field_cd", "p_career_c", "same_goal", "same_field", "r_age", "r_male", "r_pre_missing"]
    cat, binary = L.column_types(cols, sets)
    assert cat == ["r_goal", "p_goal", "r_field_cd", "p_career_c"]
    assert set(binary) == {"same_goal", "same_field", "r_male", "r_pre_missing"}


def test_complete_dates_drop_incomplete_rows_and_flags():
    D = pd.DataFrame({"r_age": [25, np.nan, 30], "r_pre_missing": [0, 1, 0], "p_age": [30, 25, 22], "p_pre_missing": [0, 0, 0],
                      "fit_to_rater_prefs": [0.1, 0.2, np.nan], "like": [5, 6, 7]})
    d, r, p, q = L.complete_dates(D, ["r_age", "r_pre_missing"], ["p_age", "p_pre_missing"], ["fit_to_rater_prefs"])
    assert len(d) == 1 and d.like.iloc[0] == 5 and r == ["r_age"] and p == ["p_age"] and q == ["fit_to_rater_prefs"]


def test_baseline_predicts_the_training_fold_mean():
    D = pd.DataFrame({"event": [1, 1, 2, 2, 3, 3], "like": [4.0, 6, 8, 8, 1, 3]})
    idx, pred = L.fit_fold(0, D, [], {1: 0, 2: 1, 3: 1}, sets=None)
    assert list(idx) == [0, 1] and np.allclose(pred, 5.0)                     # mean of events 2 and 3


def test_r_squared_known_answer():
    y = np.array([1.0, 2, 3, 4]); assert np.isclose(L.r_squared(y, y), 1) and np.isclose(L.r_squared(y, np.full(4, 2.5)), 0)


def test_participant_catalog_one_row_per_person():
    clean = toy_clean().assign(field="x", career="y", **{f"{a}1_1_share": 100 / 6 for a in ["attr", "sinc", "intel", "fun", "amb", "shar"]})
    people = PT.build(clean)
    assert sorted(people.iid) == [1, 3, 4] and people.loc[people.iid == 4, "any_pre_missing"].iloc[0] == 1   # man 4 left field blank
    assert people.loc[people.iid == 3, "any_pre_missing"].iloc[0] == 0                                      # only a recording-error fix


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
