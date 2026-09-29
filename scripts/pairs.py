"""Build the one-row-per-dyad table for match prediction.

Input : data/clean.csv (from scripts/clean.py)
Output: data/pairs.csv, data/feature_sets.json, data/pairs_summary.md

Every dyad (one woman and one man in the same event) appears twice in clean.csv, once from each side. The row
from the woman's side supplies her own pre-event answers and her in-date view of the man; the row from the man's
side supplies the same for him. Person columns are prefixed f_ and m_ by gender, so each row is well defined.

Columns
- ids and response: dyad, event, f_iid, m_iid, match. f_dec and m_dec (each side's yes) are kept for checks and
  are never features.
- individual pre-event features, for p in {f, m}:
    p_age; p_field_cd, p_career_c, p_goal (codes, categorical); p_date, p_go_out (frequency, 1 = most often);
    p_exphappy; p_<interest> for the 17 interests; p_self_<a> self-ratings for five attributes (attr3_1 ...;
    there is no self-rating of shared interests); p_pref_<a> stated preference shares for the six attributes
    (attr1_1_share ..., summing to 100).
- pair features (pre-event):
    abs_age_diff    |m_age - f_age|. The signed age_diff = m_age - f_age is kept for description but is not a
                    feature: it is a linear combination of f_age and m_age.
    int_corr        correlation of the two people's interest ratings, as released. It equals the Pearson
                    correlation over 16 interests (yoga excluded) rounded to 2 decimals; main() checks this.
    int_corr17      the same correlation over all 17 interests, unrounded. In the table, not in the feature sets.
    same_field, same_goal   1 if the two codes are equal, 0 if not, NaN if either is missing.
    fit_to_f_prefs  the man's self-ratings weighted by the woman's stated preference shares, over the five
                    attributes that have self-ratings, weights renormalized to sum to 1 (a weighted mean on the
                    self-rating scale): how well he fits what she says she wants, by his own account.
    fit_to_m_prefs  the same with the roles swapped.
- in-date features, for p in {f, m}: p_rating_<a> for the six attributes and like (the rating p gave the
  partner), p_prob (p's guess that the partner would say yes).
- event-level columns carried for sensitivity checks, not features: condtn, pref_scale, the wave flags.

Missing values are left as NaN; the model script decides how to handle them.
Run from the repository root: python scripts/pairs.py
"""
import json, pathlib
import numpy as np, pandas as pd

CLEAN = pathlib.Path("data/clean.csv")
OUT = pathlib.Path("data/pairs.csv")
SETS = pathlib.Path("data/feature_sets.json")
SUMMARY = pathlib.Path("data/pairs_summary.md")
EXPECTED_ROWS, EXPECTED_MATCH_RATE = 4184, 0.165

ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
SELF_ATTRS = ATTRS[:5]
INTERESTS = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing",
             "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"]
INTERESTS_INT_CORR = [c for c in INTERESTS if c != "yoga"]   # the 16 the released int_corr uses
CATEGORICAL = ["field_cd", "career_c", "goal"]
# person-level pre-event columns: name in clean.csv -> name after the f_/m_ prefix
PERSON = {**{c: c for c in ["age", "field_cd", "career_c", "goal", "date", "go_out", "exphappy"] + INTERESTS},
          **{f"{a}3_1": f"self_{a}" for a in SELF_ATTRS},
          **{f"{a}1_1_share": f"pref_{a}" for a in ATTRS}}
# in-date columns, the rater's view of the partner
IN_DATE = {**{a: f"rating_{a}" for a in ATTRS + ["like"]}, "prob": "prob"}
EVENT_LEVEL = ["condtn", "pref_scale", "wave_undergrad", "wave_yes_budget", "wave_diff_mc", "wave_brought_item"]
PAIR_FEATURES = ["abs_age_diff", "int_corr", "same_field", "same_goal", "fit_to_f_prefs", "fit_to_m_prefs"]


def row_corr(A, B):
    """Pearson correlation between matching rows of A and B. NaN if either row has a missing value or no spread."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ac, bc = A - A.mean(axis=1, keepdims=True), B - B.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ac * bc).sum(axis=1) / np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))


def pref_fit(weights, self_ratings):
    """Mean of one person's self-ratings weighted by the other's stated shares, renormalized over the columns given.
    NaN if any weight or rating in the row is missing."""
    W, S = np.asarray(weights, float), np.asarray(self_ratings, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * S).sum(axis=1) / W.sum(axis=1)


def same_code(a, b):
    """1.0 where two code columns agree, 0.0 where they differ, NaN where either is missing."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = (a == b).astype(float)
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out


def side(clean, gender, p):
    """The rows rated by one gender, renamed to that gender's prefix."""
    cols = {"rater": f"{p}_iid", "dec": f"{p}_dec", **{k: f"{p}_{v}" for k, v in PERSON.items()},
            **{k: f"{p}_{v}" for k, v in IN_DATE.items()}}
    return clean.loc[clean.gender == gender, ["dyad"] + list(cols)].rename(columns=cols)


def build_pairs(clean):
    """One row per dyad from the two-sided rows of clean.csv. Dyad-level columns are taken from the woman's row;
    check_two_sided() confirms they agree with the man's."""
    base = clean.loc[clean.gender == 0, ["dyad", "event", "match", "int_corr"] + EVENT_LEVEL]
    pairs = (base.merge(side(clean, 0, "f"), on="dyad", validate="one_to_one")
                 .merge(side(clean, 1, "m"), on="dyad", validate="one_to_one"))
    assert len(pairs) == clean.dyad.nunique(), "some dyad is not seen from exactly one woman and one man"
    cols = lambda p, names: pairs[[f"{p}_{n}" for n in names]]
    age_diff = pairs.m_age - pairs.f_age
    new = pd.DataFrame({
        "age_diff": age_diff,
        "abs_age_diff": age_diff.abs(),
        "int_corr17": row_corr(cols("f", INTERESTS), cols("m", INTERESTS)),
        "same_field": same_code(pairs.f_field_cd, pairs.m_field_cd),
        "same_goal": same_code(pairs.f_goal, pairs.m_goal),
        "fit_to_f_prefs": pref_fit(cols("f", [f"pref_{a}" for a in SELF_ATTRS]), cols("m", [f"self_{a}" for a in SELF_ATTRS])),
        "fit_to_m_prefs": pref_fit(cols("m", [f"pref_{a}" for a in SELF_ATTRS]), cols("f", [f"self_{a}" for a in SELF_ATTRS])),
    }, index=pairs.index)
    pairs = pd.concat([pairs, new], axis=1)
    sets = feature_sets()
    order = (["dyad", "event", "f_iid", "m_iid", "match", "f_dec", "m_dec"] + PAIR_FEATURES + ["age_diff", "int_corr17"]
             + sets["pre_individual"] + sets["date"] + EVENT_LEVEL)
    return pairs[order].sort_values(["event", "f_iid", "m_iid"]).reset_index(drop=True)


def feature_sets():
    """The named feature sets (convention 2), plus their parts, so part1_models.py can also test P3."""
    individual = [f"{p}_{v}" for p in ("f", "m") for v in PERSON.values()]
    in_date = [f"{p}_{v}" for p in ("f", "m") for v in IN_DATE.values()]
    pre = individual + PAIR_FEATURES
    return {"pre": pre, "pre+date": pre + in_date, "pre_individual": individual, "pre_pair": PAIR_FEATURES,
            "date": in_date, "categorical": [f"{p}_{c}" for p in ("f", "m") for c in CATEGORICAL],
            "response": "match", "ids": ["dyad", "event", "f_iid", "m_iid"],
            "notes": {"f_/m_": "the woman's / the man's own answer; for rating_* and prob, her / his view of the partner",
                      "categorical": "integer codes; one-hot encode before fitting",
                      "excluded": "race variables (PREREG decision), f_dec/m_dec (define the response), age_diff "
                                  "(collinear with f_age and m_age), int_corr17 (near-duplicate of int_corr)"}}


def check_two_sided(clean):
    """Counts of rows that break the two-sided structure; all should be 0."""
    g = clean.groupby("dyad")
    both = clean.pivot(index="dyad", columns="gender", values=["dec", "age_diff"])
    return {
        "dyads not seen from exactly one woman and one man": int(g.gender.agg(lambda s: sorted(s) != [0, 1]).sum()),
        **{f"dyads whose two rows disagree on {c}": int((g[c].nunique(dropna=False) > 1).sum())
           for c in ["event", "match", "int_corr", "same_field", "condtn"]},
        "dyads where age_diff is not antisymmetric": int(((both.age_diff[0] + both.age_diff[1]).abs() > 1e-9).sum()),
        "dyads where match != her yes * his yes": int((g.match.first() != both.dec[0] * both.dec[1]).sum()),
    }


def check_pairs(pairs, clean):
    """Real-data checks with known answers; all counts should be 0."""
    f_rows = clean[clean.gender == 0].set_index("dyad").loc[pairs.dyad]
    ours16 = np.round(row_corr(pairs[[f"f_{c}" for c in INTERESTS_INT_CORR]], pairs[[f"m_{c}" for c in INTERESTS_INT_CORR]]), 2)
    rel = pairs.int_corr.to_numpy()
    same_field_clean = f_rows.same_field.to_numpy()
    # rebuilding with the gender labels swapped must swap f_ and m_ and leave the symmetric pair features unchanged
    flipped = build_pairs(clean.assign(gender=1 - clean.gender)).set_index("dyad").loc[pairs.dyad]
    sym = ["abs_age_diff", "int_corr", "int_corr17", "same_field", "same_goal"]
    neq = lambda a, b: int((~np.isclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True)).sum())
    return {
        "dyads where our 16-interest correlation, rounded, differs from int_corr": neq(ours16, rel),
        "dyads where int_corr is present but ours is not": int((np.isnan(ours16) & ~np.isnan(rel)).sum()),
        "dyads where same_field differs from clean.csv": neq(pairs.same_field, same_field_clean),
        **{f"dyads where {c} changes when f and m are swapped": neq(flipped[c], pairs[c]) for c in sym},
        "dyads where fit_to_f_prefs does not become fit_to_m_prefs when swapped": neq(flipped.fit_to_f_prefs, pairs.fit_to_m_prefs),
        "dyads where age_diff does not change sign when swapped": neq(flipped.age_diff, -pairs.age_diff),
    }


def main():
    clean = pd.read_csv(CLEAN, low_memory=False)
    notes = [f"Input: {CLEAN}, {len(clean)} rows, {clean.dyad.nunique()} dyads"]
    two = check_two_sided(clean)
    pairs = build_pairs(clean)
    checks = check_pairs(pairs, clean)
    rate = pairs.match.mean()
    notes += [f"Dyad rows: {len(pairs)} (expected {EXPECTED_ROWS}); matches {int(pairs.match.sum())}, rate {rate:.4f} (expected {EXPECTED_MATCH_RATE})"]
    notes += [f"{k} (should be 0): {v}" for k, v in {**two, **checks}.items()]

    sets = feature_sets()
    missing_cols = [c for s in ("pre", "pre+date", "categorical") for c in sets[s] if c not in pairs.columns]
    assert not missing_cols, f"feature-set columns not in the table: {missing_cols}"
    for name in ("pre", "pre+date"):
        cc = int(pairs[sets[name]].notna().all(axis=1).sum())
        notes.append(f"Feature set `{name}`: {len(sets[name])} columns ({len(sets['categorical'])} categorical codes); "
                     f"complete dyads {cc} of {len(pairs)}, matches among them {int(pairs.match[pairs[sets[name]].notna().all(axis=1)].sum())}")
    miss = pairs[sets["pre+date"]].isna().sum()
    notes.append("Missing values by feature, where any: " + ", ".join(f"{c} {v}" for c, v in miss[miss > 0].sort_values(ascending=False).items()))
    person = pd.concat([pairs[[f"{p}_iid"] + [f"{p}_{c}" for c in INTERESTS]].set_axis(["iid"] + INTERESTS, axis=1) for p in "fm"]).drop_duplicates("iid")
    off = person[INTERESTS].lt(1) | person[INTERESTS].gt(10)
    notes.append(f"People with an interest rating outside the codebook's 1 to 10 scale (kept as recorded): {int(off.any(axis=1).sum())} "
                 f"(values {sorted(set(person[INTERESTS].where(off).stack().dropna().astype(int)))})")
    notes.append(f"int_corr17 vs int_corr: correlation {pairs[['int_corr17', 'int_corr']].corr().iloc[0, 1]:.3f}")

    assert len(pairs) == EXPECTED_ROWS and round(rate, 3) == EXPECTED_MATCH_RATE
    assert all(v == 0 for v in {**two, **checks}.values()), "a consistency check failed; see the summary"

    pairs.to_csv(OUT, index=False)
    SETS.write_text(json.dumps(sets, indent=2) + "\n")
    SUMMARY.write_text("# Pair table summary\n\n" + "\n".join(f"- {n}" for n in notes)
                       + f"\n\nOutput: {OUT} with {len(pairs)} rows and {pairs.shape[1]} columns; feature sets in {SETS}.\n")
    print("\n".join(notes)); print(f"wrote {OUT}, {SETS}, {SUMMARY}")


if __name__ == "__main__":
    main()
