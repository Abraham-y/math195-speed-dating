"""Basic descriptive figures for the Data Description section of the written report.

Input : data/clean.csv, data/pairs.csv, data/speed_dating.csv (the raw file, for independent cross-checks)
Output: results/data_description_*.png, results/data_description_*.csv and results/data_description_captions.md
No random operations, so no seed.

1. Participants in each wave, women and men.
2. Yes rate and match rate in each wave. Rates rather than counts, because waves had 5 to 22 dates per person;
   the average number of yeses per participant is in the table.
3. The pre-event survey: participants' ages and their main reason for coming.
Consistency checks (the script stops if any fails). They confirm that cleaning lost no participant and that the
tables agree with each other; they are not independent recomputations of the caption numbers, which come straight
from clean.csv and the pair table: participant counts per wave and gender, ages and reasons for coming match the raw
file one row per person (so the dropped rows removed nobody); every participant's number of dates equals the number
of opposite-sex participants in their wave; the yes rate per wave agrees across clean.csv, the pair table's two
decisions and the raw file after the same drop; the match rate per wave agrees between clean.csv and the pair table
(the match = both decisions check itself is in clean.py and pairs.py). Counts are of the release, whose wave table
differs from the codebook's by one person in five waves. The captions file is generated here, so no number is typed
into the report by hand.
Run from the repository root after clean.py and pairs.py: python scripts/data_description.py
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from eda import INK, INK2, MUTED, GRID, AXIS, GENDER, pct_axis   # shared chart style (sets rcParams on import)
import matplotlib.pyplot as plt

RES = pathlib.Path("results")
ACCENT = "#4a3aa7"   # one hue for all-participant series (the match colour used in eda.py)
GOALS = {1: "Seemed like a fun night out", 2: "To meet new people", 3: "To get a date",
         4: "Looking for a serious relationship", 5: "To say I did it", 6: "Other"}   # codebook, goal


def participants_by_wave(clean):
    """Women and men per wave, one count per participant."""
    t = clean.drop_duplicates("rater").groupby(["event", "gender"]).size().unstack(fill_value=0)
    return t.rename(columns={0: "women", 1: "men"}).rename_axis("wave")


def rates_by_wave(clean, pairs):
    """Yes rate (share of dates the rater said yes to), average yeses per participant, and match rate per wave."""
    yes = clean.groupby("event").dec.mean()
    per_person = clean.groupby(["event", "rater"]).dec.sum().groupby("event").mean()
    dates = clean.groupby(["event", "rater"]).size().groupby("event").median()
    match = pairs.groupby("event").match.mean()
    return pd.DataFrame({"decisions": clean.groupby("event").size(), "yes_rate": yes, "yeses_per_participant": per_person,
                         "dates_per_participant_median": dates, "dyads": pairs.groupby("event").size(),
                         "match_rate": match}).rename_axis("wave")


def survey_snapshot(clean):
    """One row per participant: age, and counts of the main reason for coming."""
    people = clean.drop_duplicates("rater")
    goal = people.goal.map(GOALS).value_counts().reindex(list(GOALS.values()), fill_value=0)
    return people.age, goal, int(people.goal.isna().sum())


def cross_check(clean, pairs, raw, parts, rates, age, goal):
    """Independent recomputations; every one must agree."""
    rp = raw.drop_duplicates("iid").groupby(["wave", "gender"]).size().unstack(fill_value=0)
    assert (rp[0].to_numpy() == parts.women.to_numpy()).all() and (rp[1].to_numpy() == parts.men.to_numpy()).all(), "participant counts"
    assert parts.to_numpy().sum() == clean.rater.nunique() == 551
    # each participant's number of dates equals the number of opposite-sex participants in the wave
    n = clean.groupby(["event", "rater", "gender"]).size().reset_index(name="dates")
    other = n.apply(lambda r: parts.loc[r.event, "men" if r.gender == 0 else "women"], axis=1)
    assert (n.dates == other).all(), "dates per participant"     # wave 5's absent woman (iid 118) is in neither count
    # yes rate from the pair table (both decisions of every dyad) equals the rate from clean.csv
    yes_pairs = (pairs.groupby("event").f_dec.sum() + pairs.groupby("event").m_dec.sum()) / (2 * pairs.groupby("event").size())
    assert np.allclose(yes_pairs.to_numpy(), rates.yes_rate.to_numpy()), "yes rate"
    # match rate from clean.csv (each dyad appears twice) equals the rate from the pair table
    assert np.allclose(clean.groupby("event").match.mean().to_numpy(), rates.match_rate.to_numpy()), "match rate"
    # yes rate from the raw file, after dropping the rows clean.py drops
    r = raw.dropna(subset=["dec", "pid"]).groupby("wave").dec.mean()
    assert np.allclose(r.to_numpy(), rates.yes_rate.to_numpy()), "yes rate against the raw file"
    # ages and reasons for coming from the raw file, one row per participant
    people = raw.drop_duplicates("iid").set_index("iid")
    assert np.allclose(people.age.sort_values().dropna().to_numpy(), age.sort_values().dropna().to_numpy()), "ages"
    assert (people.goal.map(GOALS).value_counts().reindex(goal.index, fill_value=0) == goal).all(), "reasons for coming"


def bar(ax, x, h, color, width=0.62, **kw):
    ax.bar(x, h, width=width, color=color, zorder=3, **kw)


def fig_participants(parts, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.1), layout="constrained")
    x = parts.index.to_numpy()
    for g, off in ((0, -0.2), (1, 0.2)):
        name, color = GENDER[g]
        bar(ax, x + off, parts[name.lower()], color, width=0.38, label=name)
    ax.grid(axis="x", visible=False); ax.set_xticks(x); ax.set_xlabel("Wave"); ax.set_ylabel("Participants")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper right", ncol=2)
    fig.suptitle(f"Participants in each wave ({parts.women.sum()} women, {parts.men.sum()} men)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.savefig(path, dpi=200); plt.close(fig)


def fig_rates(rates, overall_yes, overall_match, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True, layout="constrained")
    x = rates.index.to_numpy()
    for ax, col, overall, title, top in ((a1, "yes_rate", overall_yes, "Yes rate: share of dates the participant said yes to", 0.7),
                                         (a2, "match_rate", overall_match, "Match rate: share of pairs where both said yes", 0.35)):
        bar(ax, x, rates[col], ACCENT)
        ax.axhline(overall, color=MUTED, lw=0.9, zorder=4)
        ax.text(21.6, overall, f"all waves\n{overall:.1%}", va="center", fontsize=7.5, color=INK2)
        ax.set_title(title); pct_axis(ax, top); ax.grid(axis="x", visible=False)
    w12 = rates.loc[12, "yes_rate"]
    a1.annotate("wave 12: each person could\nsay yes to only half their dates", (12, w12), (14.6, 0.63), fontsize=7.5, color=INK2,
                ha="center", va="top", arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    a2.set_xticks(x); a2.set_xlabel("Wave"); a2.set_xlim(0.3, 23.2)
    fig.suptitle("Yes and match rates in each wave", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(path, dpi=200); plt.close(fig)


def fig_survey(age, goal, goal_missing, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.0), width_ratios=[1, 1.25], layout="constrained")
    a = age.dropna()
    a1.hist(a, bins=np.arange(17.5, a.max() + 1.5, 2), color=ACCENT, edgecolor="white", linewidth=1, zorder=3)
    a1.axvline(a.median(), color=INK2, lw=1, zorder=4)
    a1.set_title(f"Age ({len(a)} participants; line = median {a.median():.0f})"); a1.set_xlabel("Years"); a1.set_ylabel("Participants"); a1.grid(axis="x", visible=False)
    g = goal.sort_values()
    a2.barh(range(len(g)), g.to_numpy(), color=ACCENT, height=0.6, zorder=3)
    a2.set_yticks(range(len(g)), g.index, fontsize=8); a2.grid(axis="y", visible=False)
    for i, v in enumerate(g.to_numpy()):
        a2.text(v + 3, i, str(v), va="center", fontsize=7.5, color=INK2)
    a2.set_xlim(0, g.max() * 1.15)
    a2.set_title(f"Main reason for coming ({goal_missing} did not answer)"); a2.set_xlabel("Participants")
    fig.suptitle("Who took part: answers from the sign-up survey", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(path, dpi=200); plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    pairs = pd.read_csv("data/pairs.csv")
    raw = pd.read_csv("data/speed_dating.csv", encoding="latin-1")
    parts, rates = participants_by_wave(clean), rates_by_wave(clean, pairs)
    age, goal, goal_missing = survey_snapshot(clean)
    cross_check(clean, pairs, raw, parts, rates, age, goal)
    overall_yes, overall_match = clean.dec.mean(), pairs.match.mean()
    dpp = clean.groupby("rater").size()                                     # dates per participant

    table = parts.join(rates)
    table.to_csv(RES / "data_description_by_wave.csv")
    pd.DataFrame({"participants": goal}).rename_axis("main_reason").to_csv(RES / "data_description_goals.csv")
    fig_participants(parts, RES / "data_description_participants.png")
    fig_rates(rates, overall_yes, overall_match, RES / "data_description_rates.png")
    fig_survey(age, goal, goal_missing, RES / "data_description_survey.png")

    lo, hi = table.yes_rate.idxmin(), table.yes_rate.idxmax()
    mlo, mhi = table.match_rate.idxmin(), table.match_rate.idxmax()
    top_goal = goal.idxmax()
    captions = f"""# Data Description: figure captions (generated by scripts/data_description.py)

**Figure 1. Participants in each wave.** The experiment ran 21 speed-dating events ("waves") between October 2002
and April 2004. The release holds {parts.sum().sum()} participants: {parts.women.sum()} women and {parts.men.sum()} men
(the codebook's own table differs by one person in five waves). Waves ranged from {parts.sum(axis=1).min()} to
{parts.sum(axis=1).max()} participants; in the release each person dated every participant of the opposite sex in
their wave, so the number of dates ranged from {int(dpp.min())} to {int(dpp.max())} (median {int(dpp.median())} per
participant).

**Figure 2. Yes and match rates in each wave.** Across all {len(clean):,} dating decisions, participants said yes
{overall_yes:.1%} of the time, from {table.yes_rate.min():.1%} (wave {lo}) to {table.yes_rate.max():.1%} (wave {hi}). A
match, where both people said yes, happened in {overall_match:.1%} of the {len(pairs):,} pairs, from
{table.match_rate.min():.1%} (wave {mlo}) to {table.match_rate.max():.1%} (wave {mhi}). In wave 12 each person could say
yes to at most half of their dates.

**Figure 3. Who took part.** Participants' median age was {age.median():.0f} (range {age.min():.0f} to {age.max():.0f};
{int(age.isna().sum())} did not give an age). The most common reason for coming was "{top_goal.lower()}"
({goal.max()} participants); {goal.loc['Looking for a serious relationship']} said they were looking for a serious
relationship.

Source: Columbia speed-dating experiment (Fisman, Iyengar, Kamenica and Simonson), Kaggle release. Counts leave out
10 scorecards with no partner id: in wave 5, ten men each rated a woman who has no record in the release (no
decision is missing anywhere). Tables: results/data_description_by_wave.csv and
results/data_description_goals.csv.
"""
    (RES / "data_description_captions.md").write_text(captions)
    print(captions)
    print("consistency checks passed: participants, ages and reasons for coming against the raw file; dates per participant "
          "against the wave's opposite-sex count; yes rate across clean.csv, the pair table and the raw file; match rate across "
          "clean.csv and the pair table")
    print("wrote " + ", ".join(sorted(p.name for p in RES.glob("data_description_*"))))


if __name__ == "__main__":
    main()
