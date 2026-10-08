"""A catalog of participants: one row per person with their sign-up (pre-date) answers.

Input : data/clean.csv
Output: data/participants.csv (one row per participant) and data/participants_summary.md
Columns: iid, wave, gender, the pre-event answers used as predictors (age, field and career codes, goal, dating and
going-out frequency, expected happiness, the 17 interests after the recording-error fix and as recorded, five
self-ratings, six stated-preference shares and the survey format), and any_pre_missing (an answer left blank on
the form: recorded values count, and so do preference items the blank-allocation rule set to 0). A convenience
table; the models use data/pairs.csv.
Run from the repository root after clean.py: python scripts/participants.py
"""
import pathlib
import pandas as pd

INTERESTS = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing",
             "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"]
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
ANSWERS = (["age", "field", "field_cd", "career", "career_c", "goal", "date", "go_out", "exphappy"] + INTERESTS
           + [f"{a}3_1" for a in ATTRS[:5]] + [f"{a}1_1_share" for a in ATTRS])


def blank_on_form(people):
    """True where an answer was left blank on the form: the recorded interest values count (not the recording-error
    fix), and so do preference shares that the blank-allocation rule set to 0."""
    b = people.assign(gaming=people.gaming_recorded, reading=people.reading_recorded)[[c for c in ANSWERS if c not in ("field", "career")]].isna()
    for a in ATTRS:
        b[f"{a}1_1_share"] |= people[f"{a}1_1_was_blank"].eq(1)
    return b


def build(clean):
    blank_cols = [f"{a}1_1_was_blank" for a in ATTRS]
    cols = ["rater", "event", "gender"] + ANSWERS + ["gaming_recorded", "reading_recorded", "pref_scale"] + blank_cols
    people = clean[cols].drop_duplicates("rater").rename(columns={"rater": "iid", "event": "wave"})
    numeric = [c for c in ANSWERS if c not in ("field", "career")]
    people["any_pre_missing"] = blank_on_form(people)[numeric].any(axis=1).astype(int)
    varying = clean.groupby("rater")[numeric].nunique(dropna=False).gt(1).any()
    assert not varying.any(), f"answers that vary within a person: {list(varying[varying].index)}"
    return people.sort_values("iid").reset_index(drop=True)


def main():
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    people = build(clean)
    assert len(people) == people.iid.nunique() == 551
    people.to_csv("data/participants.csv", index=False)
    blank = blank_on_form(people).sum()
    fixed = {c: int((people[c].isna() & people[f"{c}_recorded"].notna()).sum()) for c in ("gaming", "reading")}
    pathlib.Path("data/participants_summary.md").write_text(
        f"# Participant catalog\n\n- {len(people)} participants ({int((people.gender == 0).sum())} women, {int((people.gender == 1).sum())} men) in "
        f"{people.wave.nunique()} waves\n- With no pre-event answer left blank on the form (values set to missing as recording errors were "
        f"answered, so they do not count; preference items set to 0 were blank, so they do): {int((people.any_pre_missing == 0).sum())} "
        f"({(people.any_pre_missing == 0).mean():.1%})\n"
        f"- Blank answers by column, where any: " + ", ".join(f"{c} {v}" for c, v in blank[blank > 0].sort_values(ascending=False).items()) + "\n"
        f"- Set to missing as recording errors (not blank): " + ", ".join(f"{c} {v} people" for c, v in fixed.items()) + "\n")
    print(pathlib.Path("data/participants_summary.md").read_text()); print("wrote data/participants.csv, data/participants_summary.md")


if __name__ == "__main__":
    main()
