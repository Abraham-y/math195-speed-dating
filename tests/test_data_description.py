"""Known-answer tests for scripts/data_description.py. Run from the repository root: python tests/test_data_description.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import data_description as DD


def toy():
    """Wave 1: women 1, 2 and men 3, 4 (four dyads). Wave 2: woman 5 and man 6 (one dyad)."""
    rows = []
    def add(w, f, m, fd, md):
        rows.append(dict(event=w, rater=f, ratee=m, gender=0, dec=fd, match=fd * md, goal=1, age=25))
        rows.append(dict(event=w, rater=m, ratee=f, gender=1, dec=md, match=fd * md, goal=2, age=30))
    add(1, 1, 3, 1, 1); add(1, 1, 4, 0, 1); add(1, 2, 3, 1, 0); add(1, 2, 4, 1, 1); add(2, 5, 6, 0, 0)
    clean = pd.DataFrame(rows)
    pairs = clean[clean.gender == 0].rename(columns={"rater": "f_iid", "ratee": "m_iid", "dec": "f_dec"})
    pairs["m_dec"] = clean[clean.gender == 1].dec.to_numpy()
    return clean, pairs


def test_participants_by_wave():
    clean, _ = toy()
    t = DD.participants_by_wave(clean)
    assert t.loc[1, "women"] == 2 and t.loc[1, "men"] == 2 and t.loc[2, "women"] == 1 and t.loc[2, "men"] == 1


def test_rates_by_wave():
    clean, pairs = toy()
    r = DD.rates_by_wave(clean, pairs)
    assert np.isclose(r.loc[1, "yes_rate"], 6 / 8) and np.isclose(r.loc[2, "yes_rate"], 0)      # 6 yeses of 8 decisions
    assert np.isclose(r.loc[1, "yeses_per_participant"], 1.5)                                     # 1, 2, 2, 1 yeses
    assert np.isclose(r.loc[1, "match_rate"], 2 / 4) and r.loc[1, "dyads"] == 4 and r.loc[1, "decisions"] == 8


def test_survey_snapshot_counts_people_once():
    clean, _ = toy()
    age, goal, missing = DD.survey_snapshot(clean)
    assert len(age) == 6 and goal["Seemed like a fun night out"] == 3 and goal["To meet new people"] == 3 and missing == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
