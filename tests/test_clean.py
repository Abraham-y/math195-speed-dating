"""Known-answer tests for scripts/clean.py. Run from the repository root: python tests/test_clean.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import clean as C


def test_recording_errors_set_to_missing_and_kept():
    df = pd.DataFrame({"rater": [1, 1, 2, 3, 4], "gaming": [14, 14, 10, 3, np.nan], "reading": [5, 5, 13, 10, 7]})
    out, people, values = C.apply_recording_errors(df.copy())
    assert np.isnan(out.gaming.iloc[0]) and np.isnan(out.gaming.iloc[1]) and out.gaming.iloc[2] == 10 and out.gaming.iloc[3] == 3
    assert np.isnan(out.reading.iloc[2]) and out.reading.iloc[3] == 10            # a genuine 10 is untouched
    assert out.gaming_recorded.tolist()[:4] == [14, 14, 10, 3] and out.reading_recorded.iloc[2] == 13
    assert people == {"gaming": 1, "reading": 1}                                   # people, not rows
    assert values == {"gaming": 2, "reading": 1}


def test_blank_allocations_filled_only_when_nothing_was_left():
    blk = lambda *v: dict(zip([f"{a}1_1" for a in C.ATTRS], v))
    df = pd.DataFrame([
        dict(rater=1, event=5, **blk(40, 20, 20, 20, np.nan, np.nan)),       # sums to 100: blanks become 0
        dict(rater=2, event=5, **blk(40, 20, 20, 19.5, np.nan, np.nan)),     # 99.5, within 1 point: filled
        dict(rater=3, event=5, **blk(40, 20, 20, 10, np.nan, np.nan)),       # 90: points left over, stays blank
        dict(rater=4, event=7, **blk(8, 7, 9, 7, np.nan, 5)),                 # a 1-10 wave: untouched
        dict(rater=5, event=5, **blk(*[np.nan] * 6)),                         # skipped entirely: untouched
        dict(rater=6, event=5, **blk(20, 20, 20, 20, 10, 10))])               # complete: untouched
    out, changed = C.fill_blank_allocations(df.copy())
    assert out.loc[0, "amb1_1"] == 0 and out.loc[0, "shar1_1"] == 0 and out.loc[1, "shar1_1"] == 0
    assert out.loc[2:, [f"{a}1_1" for a in C.ATTRS]].isna().sum().sum() == df.loc[2:, [f"{a}1_1" for a in C.ATTRS]].isna().sum().sum()
    assert out.pref_blanks_set_to_0.tolist() == [2, 2, 0, 0, 0, 0] and changed["1_1"] == ([1, 2], 2)
    # blank on the form, whether or not it was then set to 0
    assert out.amb1_1_was_blank.tolist() == [1, 1, 1, 1, 1, 0] and out.attr1_1_was_blank.tolist() == [0, 0, 0, 0, 1, 0]


def test_clean_frame_on_the_release():
    df, notes = C.clean_frame(pd.read_csv(C.RAW, encoding="latin-1"))
    assert len(df) == 8368 and int(df.ratings_complete.sum()) == 7031
    assert df.groupby("pref_scale").event.unique()["1-10"].tolist() == [6, 7, 8, 9]
    filled = df[df.pref_blanks_set_to_0 > 0]
    assert sorted(filled.rater.unique()) == [129, 130, 528] and len(filled) == 40
    shares = filled[[f"{a}1_1_share" for a in C.ATTRS]]
    assert shares.notna().all().all() and np.allclose(shares.sum(axis=1), 100)
    assert any("Dropped 10 rows with no partner id (waves [5]) and 0 further rows" in n for n in notes)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
