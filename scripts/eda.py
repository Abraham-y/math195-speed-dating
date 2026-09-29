"""Exploratory analysis for the data section of the written report.

Input : data/clean.csv (one row per date from one side), data/pairs.csv and data/feature_sets.json (one row per dyad)
Output: results/eda_*.csv, the numbers behind every figure and printed line, and results/eda_*.png
No random operations, so no seed.

1. Yes rate by gender and match rate, overall and by event.
2. Distributions of the six in-date attribute ratings, by the rater's gender.
3. Dates per person, by gender.
4. Missingness of every feature in the `pre` and `pre+date` sets.
5. Interest similarity and age difference by match status, with a summary of all six pair features.
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
    flags = clean.groupby("event")[["wave_undergrad", "wave_yes_budget", "wave_diff_mc", "wave_brought_item"]].first()
    by_event = pd.DataFrame({"women": people[0], "men": people[1], "dyads": pairs.groupby("event").size(),
                             "yes_rate_women": yes[0], "yes_rate_men": yes[1],
                             "match_rate": pairs.groupby("event").match.mean()}).join(flags)
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
    feats = sets["pre_pair"] + ["age_diff", "int_corr17"]
    summ = pairs.groupby("match")[feats].agg(["count", "mean", "std", "median"]).stack(0, future_stack=True)
    summ = summ.rename_axis(["match", "feature"]).reset_index()
    summ.to_csv(RES / "eda_pair_features_by_match.csv", index=False)

    int_bins = np.round(np.arange(-1, 1.0001, 0.1), 1)
    rows = []
    for m, (name, _) in MATCH.items():
        sub = pairs[pairs.match == m]
        ic = pd.cut(sub.int_corr.dropna(), int_bins, include_lowest=True).value_counts(normalize=True, sort=False)
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
    means = pairs.groupby("match")[["int_corr", "age_diff"]].mean()
    a1.set_title("Interest similarity (int_corr)"); a1.set_xlabel("Correlation of the two people's interest ratings")
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
    groups = {"all dyads": pd.Series(True, index=pairs.index), "complete on pre": complete["pre"],
              "complete on pre+date": both, "complete on pre, missing an in-date feature": complete["pre"] & ~both,
              "missing a pre feature": ~complete["pre"]}
    pd.DataFrame([dict(sample=k, dyads=int(m.sum()), matches=int(pairs.match[m].sum()), match_rate=pairs.match[m].mean())
                  for k, m in groups.items()]).to_csv(RES / "eda_complete_cases.csv", index=False)

    summ = pair_features_by_match(pairs, sets)
    print("\npair features by match status (mean; match / no match):")
    for f in sets["pre_pair"] + ["age_diff"]:
        s = summ[summ.feature == f].set_index("match")["mean"]
        print(f"  {f:15s} {s[1]:.3f} / {s[0]:.3f}")
    print("\nwrote " + ", ".join(sorted(p.name for p in RES.glob("eda_*"))))


if __name__ == "__main__":
    main()
