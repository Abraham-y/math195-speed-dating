"""Exploratory analysis for the data section of the written report.

Input : data/clean.csv (one row per date from one side), data/pairs.csv and data/feature_sets.json (one row per dyad)
Output: results/eda_*.csv, the numbers behind every figure and printed line, and results/eda_*.png
No random operations, so no seed.

1. Yes rate by gender and match rate, overall and by event.
2. Distributions of the six in-date attribute ratings, by the rater's gender.
3. Dates per person, by gender.
4. Missingness of every feature in the `pre` and `pre+date` sets.
5. Interest similarity and age difference by match status, with a summary of all six pair features.
6. Values outside the codebook's 1 to 10 scales, and the evidence on whether they are typos.
7. Stated preferences by survey format (waves 6 to 9 rated 1 to 10; the other waves split 100 points).
8. The preference-fit feature split into the partner's average self-rating and the compatibility term.
9. Yes and match rates by semester, season and academic year, for all events and for events without a design
   change (event dates from the codebook's wave table).
Sections 6 to 8 are the evidence for the decisions in PREREG.md; section 8 uses no match outcomes.
Run from the repository root after clean.py and pairs.py: python scripts/eda.py
"""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("results")
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
ATTR_NAMES = dict(attr="Attractive", sinc="Sincere", intel="Intelligent", fun="Fun", amb="Ambitious", shar="Shared interests")
SELF_ATTRS = ATTRS[:5]   # no self-rating of shared interests
INTERESTS = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing",
             "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"]
FORMATS = {"1-10": "1-10 (waves 6-9)", "100pt": "100 points (other waves)"}
# event dates, transcribed from the codebook's wave table (data/speed_dating_key.doc)
EVENT_DATES = {1: "2002-10-16", 2: "2002-10-23", 3: "2002-11-12", 4: "2002-11-12", 5: "2002-11-20", 6: "2003-03-26",
               7: "2003-03-26", 8: "2003-04-02", 9: "2003-04-02", 10: "2003-09-24", 11: "2003-09-24", 12: "2003-10-07",
               13: "2003-10-08", 14: "2003-10-08", 15: "2004-02-24", 16: "2004-02-25", 17: "2004-02-25", 18: "2004-04-06",
               19: "2004-04-06", 20: "2004-04-07", 21: "2004-04-07"}
# events whose procedure differed (codebook "Variations"): undergraduates, yes budget, different host, book or magazine
DESIGN_FLAGS = ["wave_undergrad", "wave_yes_budget", "wave_diff_mc", "wave_brought_item"]

# Chart style. Gender is categorical slots 1 and 2 of the reference palette; match status is an emphasis pair
# (violet against the de-emphasis gray). Both pairs checked with the dataviz skill's palette validator on white.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GENDER = {0: ("Women", "#2a78d6"), 1: ("Men", "#eb6834")}
MATCH = {1: ("Match", "#4a3aa7"), 0: ("No match", MUTED)}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False, "legend.labelcolor": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


def line(ax, x, y, color, label=None):
    """A 2px line with ringed markers (the ring is the surface color)."""
    ax.plot(x, y, color=color, lw=1.5, marker="o", ms=5, mec="white", mew=1.2, label=label,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)


def pct_axis(ax, top=None):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, top)


def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)


def save(fig, name):
    fig.savefig(RES / f"eda_{name}.png"); plt.close(fig)


def rates(clean, pairs):
    """Section 1: yes rate by gender and match rate, overall and by event."""
    yes = clean.groupby(["event", "gender"]).dec.mean().unstack()
    people = clean.drop_duplicates("rater").groupby(["event", "gender"]).size().unstack()
    flags = clean.groupby("event")[DESIGN_FLAGS].first()
    by_event = pd.DataFrame({"women": people[0], "men": people[1], "dyads": pairs.groupby("event").size(),
                             "yes_rate_women": yes[0], "yes_rate_men": yes[1],
                             "match_rate": pairs.groupby("event").match.mean()}).join(flags)
    assert set(EVENT_DATES) == set(by_event.index), "an event has no date in EVENT_DATES"
    dates = pd.to_datetime(pd.Series(EVENT_DATES)).reindex(by_event.index)
    by_event["date"] = dates.dt.strftime("%Y-%m-%d")
    by_event["term"] = dates.map(lambda d: f"{'Autumn' if d.month >= 8 else 'Spring'} {d.year}")
    by_event["design_change"] = by_event[DESIGN_FLAGS].any(axis=1).astype(int)
    by_event.to_csv(RES / "eda_rates_by_event.csv")
    overall = {"people": clean.rater.nunique(), "women": int((clean.drop_duplicates("rater").gender == 0).sum()),
               "men": int((clean.drop_duplicates("rater").gender == 1).sum()), "events": clean.event.nunique(),
               "dates (rows, one per side)": len(clean), "dyads": len(pairs),
               "yes rate, all": clean.dec.mean(), "yes rate, women": clean.dec[clean.gender == 0].mean(),
               "yes rate, men": clean.dec[clean.gender == 1].mean(), "match rate (per dyad)": pairs.match.mean(),
               "yes rate, wave 12 (yes budget)": clean.dec[clean.event == 12].mean(),
               "yes rate, other waves": clean.dec[clean.event != 12].mean()}
    pd.Series(overall, name="value").rename_axis("metric").to_csv(RES / "eda_overall.csv")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True, height_ratios=[1.4, 1], layout="constrained")
    ev = by_event.index.to_numpy(); off = {0: -0.14, 1: 0.14}      # women left, men right, so equal rates stay visible
    for a in (a1, a2): a.grid(axis="x", visible=False)
    a1.plot(np.vstack([ev + off[0], ev + off[1]]), np.vstack([by_event.yes_rate_women, by_event.yes_rate_men]),
            color=AXIS, lw=1, zorder=2)                             # dumbbell stem
    for g, (name, color) in GENDER.items():
        a1.plot(ev + off[g], by_event[f"yes_rate_{name.lower()}"], "o", ms=6, color=color, mec="white", mew=1.2, label=name, zorder=3)
    a1.set_title("Share of dates the rater said yes to"); pct_axis(a1, 0.8)
    a1.text(12, by_event.loc[12, ["yes_rate_women", "yes_rate_men"]].min() - 0.12, "wave 12: yes limited\nto half of dates",
            ha="center", va="top", fontsize=7.5, color=INK2)
    a2.axhline(pairs.match.mean(), color=MUTED, lw=0.8, zorder=1)
    a2.text(21.6, pairs.match.mean(), f"all events {pairs.match.mean():.1%}", va="center", fontsize=7.5, color=INK2)
    a2.plot(ev, by_event.match_rate, "o", ms=6, color=MATCH[1][1], mec="white", mew=1.2, zorder=3)
    a2.set_title("Share of dyads that matched"); pct_axis(a2, 0.4)
    a2.set_xticks(ev); a2.set_xlabel("Event (wave)"); a2.set_xlim(0.3, 21.7)
    fig.legend(*a1.get_legend_handles_labels(), loc="outside upper right", ncol=2)
    title(fig, "Yes and match rates by event"); save(fig, "rates_by_event")
    return overall


def ratings(clean):
    """Section 2: the six in-date attribute ratings, as given by each gender. Half-point ratings are rounded."""
    rows, summ = [], []
    for a in ATTRS + ["like", "prob"]:
        for g, (name, _) in GENDER.items():
            v = clean.loc[clean.gender == g, a]
            summ.append(dict(rating=a, rater_gender=name, dates=len(v), missing=int(v.isna().sum()),
                             missing_rate=v.isna().mean(), mean=v.mean(), sd=v.std(), zeros=int((v == 0).sum()),
                             half_points=int((v.dropna() % 1 != 0).sum())))
            share = np.floor(v.dropna() + 0.5).value_counts(normalize=True).reindex(range(11), fill_value=0)
            rows += [dict(rating=a, rater_gender=name, value=k, share=s) for k, s in share.items()]
    dist = pd.DataFrame(rows); summary = pd.DataFrame(summ)
    dist.to_csv(RES / "eda_ratings_distribution.csv", index=False)
    summary.to_csv(RES / "eda_ratings_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.6), sharex=True, sharey=True, layout="constrained")
    for ax, a in zip(axes.flat, ATTRS):
        for g, (name, color) in GENDER.items():
            d = dist[(dist.rating == a) & (dist.rater_gender == name)]
            line(ax, d.value, d.share, color, name)
        ax.set_title(ATTR_NAMES[a]); ax.set_xticks([0, 2, 4, 6, 8, 10])
    pct_axis(axes[0, 0], 0.35)
    for ax in axes[1]: ax.set_xlabel("Rating given")
    for ax in axes[:, 0]: ax.set_ylabel("Share of ratings")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper right", ncol=2)
    title(fig, "In-date ratings of the partner, by the rater's gender"); save(fig, "ratings")
    return summary


def dates_per_person(clean):
    """Section 3: how many dates each person had."""
    n = clean.groupby(["rater", "gender"]).size().rename("dates").reset_index()
    tab = n.groupby(["gender", "dates"]).size().unstack(0, fill_value=0).rename(columns=lambda g: GENDER[g][0])
    tab = tab.reindex(range(tab.index.min(), tab.index.max() + 1), fill_value=0).rename_axis("dates")
    tab.to_csv(RES / "eda_dates_per_person.csv")
    fig, ax = plt.subplots(figsize=(7.5, 2.9), layout="constrained")
    for g, (name, color) in GENDER.items():   # paired columns with a small surface gap between them
        ax.bar(tab.index + (0.2 if g else -0.2), tab[name], width=0.38, color=color, label=name, zorder=3)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(tab.index); ax.set_xlabel("Dates in the evening"); ax.set_ylabel("People")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper right", ncol=2)
    title(fig, "Dates per person"); save(fig, "dates_per_person")
    return n.groupby("gender").dates.describe()


def missingness(pairs, sets):
    """Section 4: missing values of every model feature, per dyad, and per person for person-level features."""
    kind = {c: "in-date" for c in sets["date"]} | {c: "pair" for c in sets["pre_pair"]} | {c: "person" for c in sets["pre_individual"]}
    rows = []
    for c in sets["pre+date"]:
        r = dict(feature=c, set="pre" if c in sets["pre"] else "pre+date only", kind=kind[c],
                 missing_dyads=int(pairs[c].isna().sum()), missing_rate_dyads=pairs[c].isna().mean())
        if kind[c] == "person":
            p = c[:1]; per = pairs.drop_duplicates(f"{p}_iid")[c]
            r |= dict(missing_people=int(per.isna().sum()), people=len(per))
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values("missing_dyads", ascending=False)
    tab.to_csv(RES / "eda_missingness.csv", index=False)
    complete = {s: pairs[sets[s]].notna().all(axis=1) for s in ("pre", "pre+date")}
    return tab, complete


def pair_features_by_match(pairs, sets):
    """Section 5: pair features by match status; figure for interest similarity and age difference."""
    feats = sets["pre_pair"] + ["int_corr_released", "fit_raw_to_f_prefs", "fit_raw_to_m_prefs", "age_diff", "int_corr17"]
    summ = pairs.groupby("match")[feats].agg(["count", "mean", "std", "median"]).stack(0, future_stack=True)
    summ = summ.rename_axis(["match", "feature"]).reset_index()
    summ.to_csv(RES / "eda_pair_features_by_match.csv", index=False)

    int_bins = np.round(np.arange(-1, 1.0001, 0.1), 1)
    rows = []
    for m, (name, _) in MATCH.items():
        sub = pairs[pairs.match == m]
        ic = pd.cut(sub.int_corr_released.dropna(), int_bins, include_lowest=True).value_counts(normalize=True, sort=False)
        rows += [dict(feature="int_corr", match=name, bin_mid=iv.mid, share=s) for iv, s in ic.items()]
        ad = sub.age_diff.dropna().value_counts(normalize=True).sort_index()
        rows += [dict(feature="age_diff", match=name, bin_mid=k, share=s) for k, s in ad.items()]
    bins = pd.DataFrame(rows); bins.to_csv(RES / "eda_pair_features_bins.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
    lim = 15
    for m, (name, color) in MATCH.items():
        d = bins[(bins.feature == "int_corr") & (bins.match == name)]
        line(a1, d.bin_mid, d.share, color, name)
        d = bins[(bins.feature == "age_diff") & (bins.match == name) & (bins.bin_mid.abs() <= lim)]
        line(a2, d.bin_mid, d.share, color, name)
    means = pairs.groupby("match")[["int_corr_released", "age_diff"]].mean().rename(columns={"int_corr_released": "int_corr"})
    a1.set_title("Interest similarity (int_corr, as released)"); a1.set_xlabel("Correlation of the two people's interest ratings")
    a1.set_ylabel("Share of dyads"); pct_axis(a1)
    a1.text(0.02, 0.62, f"mean\nmatch {means.int_corr[1]:.2f}\nno match {means.int_corr[0]:.2f}",
            transform=a1.transAxes, va="top", fontsize=7.5, color=INK2)
    outside = int((pairs.age_diff.abs() > lim).sum())
    a2.set_title("Age difference"); a2.set_xlabel("Man's age minus woman's age (years)"); pct_axis(a2)
    a2.text(0.98, 0.97, f"mean\nmatch {means.age_diff[1]:.2f}\nno match {means.age_diff[0]:.2f}\n\n"
            f"{outside} dyads beyond ±{lim}\nnot shown", transform=a2.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    fig.legend(*a1.get_legend_handles_labels(), loc="outside upper right", ncol=2)
    title(fig, "Pair features by match status"); save(fig, "pair_features_by_match")
    return summ


def rates_by_period(by_event):
    """Section 9: yes and match rates by semester, season and academic year, weighted by dyads (each dyad holds one
    decision from each side), for all events and for events without a design change."""
    ev = by_event.copy()
    d = pd.to_datetime(ev.date)
    ev["season"] = np.where(d.dt.month >= 8, "autumn", "late winter/spring")
    ev["academic_year"] = [f"{y}-{(y + 1) % 100:02d}" if m >= 8 else f"{y - 1}-{y % 100:02d}" for y, m in zip(d.dt.year, d.dt.month)]
    w = lambda x, c: (x[c] * x.dyads).sum() / x.dyads.sum()
    rows = []
    for period in ("term", "season", "academic_year"):
        for events, sub in (("all", ev), ("no design change", ev[ev.design_change == 0])):
            for key, x in sub.groupby(period, sort=False):
                rows.append(dict(period=period, value=key, events=events, n_events=len(x), dyads=int(x.dyads.sum()),
                                 first_date=x.date.min(), last_date=x.date.max(), yes_rate_women=w(x, "yes_rate_women"),
                                 yes_rate_men=w(x, "yes_rate_men"), match_rate=w(x, "match_rate")))
    tab = pd.DataFrame(rows)
    tab["yes_gap_men_minus_women"] = tab.yes_rate_men - tab.yes_rate_women
    tab.to_csv(RES / "eda_rates_by_period.csv", index=False)
    return tab


def out_of_range(clean, pairs):
    """Section 6: in-date ratings of 0 and interest ratings outside 1 to 10. A '10' typed as '0' would sit on
    a card the rater liked, so the liking and yes rate on cards with a 0 are the test."""
    R = ATTRS + ["like", "prob"]
    any_zero = clean[R].eq(0).any(axis=1)
    givers = clean.rater.isin(clean.rater[any_zero].unique())
    zeros = pd.DataFrame([dict(rating=a, zero_cards=int((clean[a] == 0).sum()), raters=clean.rater[clean[a] == 0].nunique(),
                               like_on_zero_cards=clean.like[clean[a] == 0].mean(), like_all_cards=clean.like.mean(),
                               yes_rate_on_zero_cards=clean.dec[clean[a] == 0].mean(), yes_rate_all_cards=clean.dec.mean(),
                               zero_cards_with_like_8_plus=int(((clean[a] == 0) & (clean.like >= 8)).sum())) for a in R])
    zeros.to_csv(RES / "eda_zero_ratings.csv", index=False)
    people = clean.drop_duplicates("rater").set_index("rater")
    people = people.assign(**{c: people[f"{c}_recorded"] for c in ("gaming", "reading")})   # the values as recorded
    off = people[INTERESTS].lt(1) | people[INTERESTS].gt(10)
    inter = pd.DataFrame([dict(interest=c, value=v, people=int((people[c] == v).sum()), events=people.event[people[c] == v].nunique())
                          for c in INTERESTS for v in sorted(people.loc[off[c], c].unique())])
    inter.to_csv(RES / "eda_interest_out_of_range.csv", index=False)
    recorded_error = ((pairs.f_gaming_recorded == 14) | (pairs.m_gaming_recorded == 14)
                      | (pairs.f_reading_recorded == 13) | (pairs.m_reading_recorded == 13))
    summary = {"scorecards with a rating of 0": int(any_zero.sum()), "people giving a 0": int(clean.rater[any_zero].nunique()),
               "events with a 0": int(clean.event[any_zero].nunique()),
               "N/A rate in ratings, people giving a 0": clean.loc[givers, R].isna().mean().mean(),
               "N/A rate in ratings, everyone else": clean.loc[~givers, R].isna().mean().mean(),
               "people with an interest rating of 0": int(people[INTERESTS].eq(0).any(axis=1).sum()),
               "most interest zeros from one person": int(people[INTERESTS].eq(0).sum(axis=1).max()),
               "dyads involving a gaming-14 or reading-13 person": int(recorded_error.sum())}
    pd.Series(summary, name="value").rename_axis("metric").to_csv(RES / "eda_out_of_range_summary.csv")
    return zeros, inter, summary


def pref_format(clean, pairs):
    """Section 7: stated-preference shares (block 1_1) by survey format."""
    people = clean.drop_duplicates("rater")
    S = people[[f"{a}1_1_share" for a in ATTRS]].set_axis(ATTRS, axis=1)
    rows = []
    for code, name in FORMATS.items():
        g = S[people.pref_scale == code]
        rows.append(dict(format=name, people=len(g), dyads=int((pairs.pref_scale == code).sum()),
                         **{f"mean_share_{a}": g[a].mean() for a in ATTRS}, **{f"sd_share_{a}": g[a].std() for a in ATTRS},
                         mean_within_person_sd=g.std(axis=1).mean(), mean_largest_share=g.max(axis=1).mean(),
                         share_all_six_equal=g.dropna().nunique(axis=1).eq(1).mean()))
    tab = pd.DataFrame(rows); tab.to_csv(RES / "eda_pref_format.csv", index=False)
    return tab


def pref_fit_split(pairs):
    """Section 8: fit = partner's average self-rating + compatibility term, checked as an identity, then the
    share of fit's variation the average explains, overall and within each survey format. No match outcomes."""
    rows = []
    for me, other in (("f", "m"), ("m", "f")):
        W = pairs[[f"{me}_pref_{a}" for a in SELF_ATTRS]].to_numpy(float); W = W / W.sum(axis=1, keepdims=True)
        S = pairs[[f"{other}_self_{a}" for a in SELF_ATTRS]].to_numpy(float)
        level = S.mean(axis=1); compat = ((W - 0.2) * (S - level[:, None])).sum(axis=1)
        fit = pairs[f"fit_raw_to_{me}_prefs"].to_numpy(); ok = ~np.isnan(fit)
        assert np.allclose(fit[ok], level[ok] + compat[ok]), "fit is not level + compatibility"
        assert np.allclose(compat[ok], pairs[f"fit_to_{me}_prefs"].to_numpy()[ok]), "pairs.py compatibility term differs"
        for name, m in [("all", ok)] + [(FORMATS[c], ok & (pairs.pref_scale == c).to_numpy()) for c in FORMATS]:
            r = np.corrcoef(fit[m], level[m])[0, 1]
            rows.append(dict(preferences_of=me, format=name, dyads=int(m.sum()), corr_fit_with_partner_mean_self=r,
                             r_squared=r * r, sd_level=level[m].std(ddof=1), sd_compatibility=compat[m].std(ddof=1)))
    tab = pd.DataFrame(rows); tab.to_csv(RES / "eda_pref_fit.csv", index=False)
    people = pd.concat([pairs[[f"{p}_iid"] + [f"{p}_self_{a}" for a in SELF_ATTRS] + [f"{p}_pref_{a}" for a in SELF_ATTRS]]
                        .set_axis(["iid"] + [f"self_{a}" for a in SELF_ATTRS] + [f"pref_{a}" for a in SELF_ATTRS], axis=1)
                        for p in "fm"]).drop_duplicates("iid").dropna()
    flat = {"people with complete answers": len(people),
            "all five self-ratings equal": int(people[[f"self_{a}" for a in SELF_ATTRS]].nunique(axis=1).eq(1).sum()),
            "all five weights equal": int(people[[f"pref_{a}" for a in SELF_ATTRS]].nunique(axis=1).eq(1).sum())}
    pd.Series(flat, name="value").rename_axis("metric").to_csv(RES / "eda_pref_fit_flat_answers.csv")
    return tab, flat


def main():
    RES.mkdir(exist_ok=True)
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    pairs = pd.read_csv("data/pairs.csv")
    sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())

    ov = rates(clean, pairs)
    print(f"{ov['people']} people ({ov['women']} women, {ov['men']} men) in {ov['events']} events; "
          f"{ov['dates (rows, one per side)']} rows, {ov['dyads']} dyads")
    print(f"yes rate: women {ov['yes rate, women']:.3f}, men {ov['yes rate, men']:.3f}, all {ov['yes rate, all']:.3f}; "
          f"match rate {ov['match rate (per dyad)']:.3f}")
    print(f"wave 12 (yes budget) yes rate {ov['yes rate, wave 12 (yes budget)']:.3f} vs {ov['yes rate, other waves']:.3f} elsewhere")
    ev = pd.read_csv(RES / "eda_rates_by_event.csv")
    print(f"by event: match rate {ev.match_rate.min():.3f} to {ev.match_rate.max():.3f}; "
          f"men's yes rate above women's in {int((ev.yes_rate_men > ev.yes_rate_women).sum())} of {len(ev)} events")

    rs = ratings(clean)
    print("\nratings given (mean; women / men):")
    for a in ATTRS + ["like", "prob"]:
        r = rs[rs.rating == a].set_index("rater_gender")
        print(f"  {a:6s} {r.loc['Women', 'mean']:.2f} / {r.loc['Men', 'mean']:.2f}   missing {r.missing.sum() / r.dates.sum():.1%}"
              f"   zeros {r.zeros.sum()}   half-points {r.half_points.sum()}")

    d = dates_per_person(clean)
    print("\ndates per person: " + "; ".join(f"{GENDER[g][0].lower()} median {d.loc[g, '50%']:.0f} (range {d.loc[g, 'min']:.0f} to {d.loc[g, 'max']:.0f})" for g in d.index))

    tab, complete = missingness(pairs, sets)
    print("\nmissingness (per dyad), top 8:")
    print(tab.head(8)[["feature", "set", "missing_dyads", "missing_rate_dyads"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for s, cc in complete.items():
        print(f"complete dyads, `{s}`: {int(cc.sum())} of {len(pairs)}, match rate among them {pairs.match[cc].mean():.3f}")
    both = complete["pre+date"]
    print(f"match rate among dyads complete on `pre` but not `pre+date`: {pairs.match[complete['pre'] & ~both].mean():.3f}")
    recorded = pairs.assign(**{f"{p}_{c}": pairs[f"{p}_{c}_recorded"] for p in "fm" for c in ("gaming", "reading")})
    rows = []
    for basis, frame in (("as recorded", recorded), ("after the recording-error fix", pairs)):
        pre_ok, both_ok = frame[sets["pre"]].notna().all(axis=1), frame[sets["pre+date"]].notna().all(axis=1)
        groups = {"all dyads": pd.Series(True, index=pairs.index), "complete on pre": pre_ok, "complete on pre+date": both_ok,
                  "complete on pre, missing an in-date feature": pre_ok & ~both_ok, "missing a pre feature": ~pre_ok}
        rows += [dict(interest_values=basis, sample=k, dyads=int(m.sum()), matches=int(pairs.match[m].sum()),
                      match_rate=pairs.match[m].mean()) for k, m in groups.items()]
    cc_tab = pd.DataFrame(rows); cc_tab.to_csv(RES / "eda_complete_cases.csv", index=False)
    print("complete cases (as recorded / after the fix): " + "; ".join(
        f"{s} {a} / {b}" for s, a, b in zip(cc_tab["sample"][:5], cc_tab.dyads[:5], cc_tab.dyads[5:])))

    summ = pair_features_by_match(pairs, sets)
    print("\npair features by match status (mean; match / no match):")
    for f in ["abs_age_diff", "int_corr_released", "int_corr", "same_field", "same_goal", "fit_raw_to_f_prefs", "fit_raw_to_m_prefs", "age_diff"]:
        s = summ[summ.feature == f].set_index("match")["mean"]
        print(f"  {f:15s} {s[1]:.3f} / {s[0]:.3f}")

    zeros, inter, oor = out_of_range(clean, pairs)
    z = zeros[zeros.rating != "like"]
    print(f"\nout-of-range: {oor['scorecards with a rating of 0']} scorecards with a 0 from {oor['people giving a 0']} people in "
          f"{oor['events with a 0']} events; N/A rate {oor['N/A rate in ratings, people giving a 0']:.3f} (people giving a 0) "
          f"vs {oor['N/A rate in ratings, everyone else']:.3f}; liking on cards with a 0 averages "
          f"{z.like_on_zero_cards.min():.1f} to {z.like_on_zero_cards.max():.1f} by attribute, against {clean.like.mean():.1f} on all cards")
    print("interest values outside 1 to 10: " + "; ".join(f"{r.interest} = {r.value:g}: {r.people} people in {r.events} events" for r in inter.itertuples()))
    print(f"people with an interest of 0: {oor['people with an interest rating of 0']} (one gave {oor['most interest zeros from one person']}); "
          f"dyads involving a gaming-14 or reading-13 person: {oor['dyads involving a gaming-14 or reading-13 person']}")

    fmt = pref_format(clean, pairs).set_index("format")
    print("\nstated preferences by survey format (" + " / ".join(fmt.index) + "):")
    for col, label in [("people", "people"), ("dyads", "dyads"), ("mean_share_attr", "mean share, attractive"),
                       ("mean_share_amb", "mean share, ambitious"), ("mean_within_person_sd", "SD of a person's six shares"),
                       ("mean_largest_share", "largest share, average")]:
        print(f"  {label:28s} " + " / ".join(f"{v:.1f}" if isinstance(v, float) else f"{v}" for v in fmt[col]))

    fit, flat = pref_fit_split(pairs)
    print("\npreference fit = partner's average self-rating + compatibility term (identity checked):")
    for r in fit.itertuples():
        print(f"  {r.preferences_of}'s preferences, {r.format:24s}: average explains {r.r_squared:.0%}; SD average {r.sd_level:.2f}, compatibility {r.sd_compatibility:.3f}")
    print(f"  of {flat['people with complete answers']} people: all five self-ratings equal {flat['all five self-ratings equal']}, "
          f"all five weights equal {flat['all five weights equal']}")
    per = rates_by_period(pd.read_csv(RES / "eda_rates_by_event.csv", index_col="event"))
    print("\nby period (weighted by dyads): yes women / yes men / gap / match")
    for r in per.itertuples():
        print(f"  {r.period:13s} {r.value:19s} {r.events:16s} {r.n_events:2d} events: {r.yes_rate_women:.3f} / {r.yes_rate_men:.3f} / "
              f"{r.yes_gap_men_minus_women:+.3f} / {r.match_rate:.3f}")
    print("\nwrote " + ", ".join(sorted(p.name for p in RES.glob("eda_*"))))


if __name__ == "__main__":
    main()
