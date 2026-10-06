"""Known-answer tests for scripts/clean.py. Run from the repository root: python tests/test_clean.py"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import clean as C


def test_recording_errors_set_to_missing_and_kept():
    df = pd.DataFrame({"rater": [1, 1, 2, 3, 4], "gaming": [14, 14, 10, 3, np.nan], "reading": [5, 5, 13, 10, 7]})
    out, people = C.apply_recording_errors(df.copy())
    assert np.isnan(out.gaming.iloc[0]) and np.isnan(out.gaming.iloc[1]) and out.gaming.iloc[2] == 10 and out.gaming.iloc[3] == 3
    assert np.isnan(out.reading.iloc[2]) and out.reading.iloc[3] == 10            # a genuine 10 is untouched
    assert out.gaming_recorded.tolist()[:4] == [14, 14, 10, 3] and out.reading_recorded.iloc[2] == 13
    assert people == {"gaming": 1, "reading": 1}                                   # people, not rows


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
