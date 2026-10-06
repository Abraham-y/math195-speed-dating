"""Build the one-row-per-dyad table for match prediction.

Input : data/clean.csv (from scripts/clean.py)
Output: data/pairs.csv, data/feature_sets.json, data/pairs_summary.md

Every dyad (one woman and one man in the same event) appears twice in clean.csv, once from each side. The row
from the woman's side supplies her own pre-event answers and her in-date view of the man; the row from the man's
side supplies the same for him. Person columns are prefixed f_ and m_ by gender, so each row is well defined.
The definitions follow PREREG.md (decisions committed before any model ran).

Columns
- ids and response: dyad, event, f_iid, m_iid, match. f_dec and m_dec (each side's yes) are kept for checks and
  are never features.
- individual pre-event features, for p in {f, m}:
    p_age; p_field_cd, p_career_c, p_goal (codes, categorical); p_date, p_go_out (frequency, 1 = most often);
    p_exphappy; p_<interest> for the 17 interests, with the recording errors (gaming 14, reading 13) already
    missing; p_self_<a> self-ratings for five attributes (no self-rating of shared interests); p_pref_<a> stated
    preference shares for the six attributes (summing to 100). p_gaming_recorded and p_reading_recorded keep the
    values as recorded, for the robustness grid.
- pair features (pre-event):
    abs_age_diff    |m_age - f_age|. The signed age_diff = m_age - f_age is kept for description only: it is a
                    linear combination of f_age and m_age.
    int_corr        Pearson correlation of the two people's ratings of the 16 interests other than yoga (the
                    released definition), over the interests both answered, unrounded, after the recording-error
                    fix. int_corr_released is the released value; main() checks that it equals our formula on the
                    recorded values, rounded to 2 decimals.
    same_field, same_goal   1 if the two codes are equal, 0 if not, NaN if either is missing.
    fit_to_f_prefs  compatibility of the man with the woman's stated preferences: sum over the five self-rated
                    attributes of (her weight share - 0.2) x (his self-rating - his mean self-rating), her weights
                    rescaled to sum to 1 over the five. fit_to_m_prefs swaps the roles. fit_raw_to_f_prefs and
                    fit_raw_to_m_prefs are the plain weighted means (option A in PREREG.md), kept for the grid;
                    each equals the partner's mean self-rating plus the compatibility term. Standardizing within
                    survey format happens in part1_models.py.
- missing-data flags: f_pre_missing, m_pre_missing (any pre-event answer left blank, counting the recorded
  values, not the recording-error fix) and f_date_missing, m_date_missing (any in-date answer left blank).
- in-date features, for p in {f, m}: p_rating_<a> for the six attributes and like (the rating p gave the
  partner), p_prob (p's guess that the partner would say yes).
- event-level columns carried for sensitivity checks, not features: condtn, pref_scale, the wave flags.

Other missing values are left as NaN; part1_models.py fills them inside each training fold.
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
RECORDED = ["gaming", "reading"]                               # interests with a recording-error fix in clean.py
CATEGORICAL = ["field_cd", "career_c", "goal"]
# person-level pre-event columns: name in clean.csv -> name after the f_/m_ prefix
PERSON = {**{c: c for c in ["age", "field_cd", "career_c", "goal", "date", "go_out", "exphappy"] + INTERESTS},
          **{f"{a}3_1": f"self_{a}" for a in SELF_ATTRS},
          **{f"{a}1_1_share": f"pref_{a}" for a in ATTRS}}
PERSON_RECORDED = {f"{c}_recorded": f"{c}_recorded" for c in RECORDED}
# in-date columns, the rater's view of the partner
IN_DATE = {**{a: f"rating_{a}" for a in ATTRS + ["like"]}, "prob": "prob"}
EVENT_LEVEL = ["condtn", "pref_scale", "wave_undergrad", "wave_yes_budget", "wave_diff_mc", "wave_brought_item"]
PAIR_FEATURES = ["abs_age_diff", "int_corr", "same_field", "same_goal", "fit_to_f_prefs", "fit_to_m_prefs"]
FLAGS_PRE, FLAGS_DATE = ["f_pre_missing", "m_pre_missing"], ["f_date_missing", "m_date_missing"]


def row_corr(A, B):
    """Pearson correlation between matching rows of A and B. NaN if either row has a missing value or no spread."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ac, bc = A - A.mean(axis=1, keepdims=True), B - B.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ac * bc).sum(axis=1) / np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))


def row_corr_pairwise(A, B, min_n=3):
    """Pearson correlation between matching rows of A and B over the columns where both are present.
    NaN when fewer than min_n columns are shared or either side has no spread there."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ok = ~(np.isnan(A) | np.isnan(B))
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(ok, A, 0.0).sum(axis=1) / n
        mb = np.where(ok, B, 0.0).sum(axis=1) / n
        ac, bc = np.where(ok, A - ma[:, None], 0.0), np.where(ok, B - mb[:, None], 0.0)
        r = (ac * bc).sum(axis=1) / np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    r[n < min_n] = np.nan
    return r


def pref_fit(weights, self_ratings):
    """Option A: one person's self-ratings weighted by the other's stated shares, renormalized over the columns given.
    NaN if any weight or rating in the row is missing."""
    W, S = np.asarray(weights, float), np.asarray(self_ratings, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * S).sum(axis=1) / W.sum(axis=1)


def compat_fit(weights, self_ratings):
    """Option B (PREREG.md): sum over the k attributes of (weight share - 1/k) x (self-rating - own mean self-rating),
    weights rescaled to sum to 1. Equals pref_fit minus the mean self-rating. NaN if any value in the row is missing."""
    W, S = np.asarray(weights, float), np.asarray(self_ratings, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = W / W.sum(axis=1, keepdims=True)
    return ((P - 1 / W.shape[1]) * (S - S.mean(axis=1, keepdims=True))).sum(axis=1)


def same_code(a, b):
    """1.0 where two code columns agree, 0.0 where they differ, NaN where either is missing."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = (a == b).astype(float)
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out


def side(clean, gender, p):
    """The rows rated by one gender, renamed to that gender's prefix."""
    cols = {"rater": f"{p}_iid", "dec": f"{p}_dec", **{k: f"{p}_{v}" for k, v in {**PERSON, **PERSON_RECORDED}.items()},
            **{k: f"{p}_{v}" for k, v in IN_DATE.items()}}
    return clean.loc[clean.gender == gender, ["dyad"] + list(cols)].rename(columns=cols)


def recorded_person_cols(p):
    """The person's pre-event columns with the recorded interest values in place of the fixed ones."""
    return [f"{p}_{v}_recorded" if v in RECORDED else f"{p}_{v}" for v in PERSON.values()]


def build_pairs(clean):
    """One row per dyad from the two-sided rows of clean.csv. Dyad-level columns are taken from the woman's row;
    check_two_sided() confirms they agree with the man's."""
    base = clean.loc[clean.gender == 0, ["dyad", "event", "match", "int_corr"] + EVENT_LEVEL].rename(columns={"int_corr": "int_corr_released"})
    pairs = (base.merge(side(clean, 0, "f"), on="dyad", validate="one_to_one")
                 .merge(side(clean, 1, "m"), on="dyad", validate="one_to_one"))
    assert len(pairs) == clean.dyad.nunique(), "some dyad is not seen from exactly one woman and one man"
    cols = lambda p, names: pairs[[f"{p}_{n}" for n in names]]
    prefs, selfs = (lambda p: cols(p, [f"pref_{a}" for a in SELF_ATTRS])), (lambda p: cols(p, [f"self_{a}" for a in SELF_ATTRS]))
    age_diff = pairs.m_age - pairs.f_age
    new = pd.DataFrame({
        "age_diff": age_diff,
        "abs_age_diff": age_diff.abs(),
        "int_corr": row_corr_pairwise(cols("f", INTERESTS_INT_CORR), cols("m", INTERESTS_INT_CORR)),
        "int_corr17": row_corr_pairwise(cols("f", INTERESTS), cols("m", INTERESTS)),
        "same_field": same_code(pairs.f_field_cd, pairs.m_field_cd),
        "same_goal": same_code(pairs.f_goal, pairs.m_goal),
        "fit_to_f_prefs": compat_fit(prefs("f"), selfs("m")),
        "fit_to_m_prefs": compat_fit(prefs("m"), selfs("f")),
        "fit_raw_to_f_prefs": pref_fit(prefs("f"), selfs("m")),
        "fit_raw_to_m_prefs": pref_fit(prefs("m"), selfs("f")),
        **{f"{p}_pre_missing": pairs[recorded_person_cols(p)].isna().any(axis=1).astype(int) for p in "fm"},
        **{f"{p}_date_missing": cols(p, IN_DATE.values()).isna().any(axis=1).astype(int) for p in "fm"},
    }, index=pairs.index)
    pairs = pd.concat([pairs, new], axis=1)
    sets = feature_sets()
    order = (["dyad", "event", "f_iid", "m_iid", "match", "f_dec", "m_dec"] + PAIR_FEATURES
             + ["int_corr_released", "fit_raw_to_f_prefs", "fit_raw_to_m_prefs", "age_diff", "int_corr17"]
             + FLAGS_PRE + FLAGS_DATE + sets["pre_individual"] + [f"{p}_{c}_recorded" for p in "fm" for c in RECORDED]
             + sets["date"] + EVENT_LEVEL)
    return pairs[order].sort_values(["event", "f_iid", "m_iid"]).reset_index(drop=True)


def feature_sets():
    """The named feature sets (convention 2), their parts, and what part1_models.py needs to apply PREREG.md."""
    individual = [f"{p}_{v}" for p in ("f", "m") for v in PERSON.values()]
    in_date = [f"{p}_{v}" for p in ("f", "m") for v in IN_DATE.values()]
    pre = individual + PAIR_FEATURES
    return {"pre": pre, "pre+date": pre + in_date, "pre_individual": individual, "pre_pair": PAIR_FEATURES,
            "date": in_date, "categorical": [f"{p}_{c}" for p in ("f", "m") for c in CATEGORICAL],
            "flags_pre": FLAGS_PRE, "flags_date": FLAGS_DATE,
            "binary": ["same_field", "same_goal"] + FLAGS_PRE + FLAGS_DATE,
            "pref_reference": ["f_pref_shar", "m_pref_shar"],
            "format_standardized": [f"{p}_pref_{a}" for p in ("f", "m") for a in ATTRS] + ["fit_to_f_prefs", "fit_to_m_prefs"],
            "spline_candidates": ["f_age", "m_age", "abs_age_diff", "int_corr", "fit_to_f_prefs", "fit_to_m_prefs"],
            "alternatives": {"out_of_range_as_recorded": {**{f"{p}_{c}": f"{p}_{c}_recorded" for p in "fm" for c in RECORDED},
                                                          "int_corr": "int_corr_released"},
                             "preference_fit_A": {"fit_to_f_prefs": "fit_raw_to_f_prefs", "fit_to_m_prefs": "fit_raw_to_m_prefs"}},
            "response": "match", "ids": ["dyad", "event", "f_iid", "m_iid"],
            "notes": {"f_/m_": "the woman's / the man's own answer; for rating_* and prob, her / his view of the partner",
                      "categorical": "integer codes; rare codes merged and one-hot encoded inside each training fold",
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
    rec16 = lambda p: pairs[[f"{p}_{c}_recorded" if c in RECORDED else f"{p}_{c}" for c in INTERESTS_INT_CORR]]
    released_formula = np.round(row_corr(rec16("f"), rec16("m")), 2)
    rel = pairs.int_corr_released.to_numpy()
    fixed = (pairs[[f"{p}_{c}" for p in "fm" for c in RECORDED]].isna()
             & pairs[[f"{p}_{c}_recorded" for p in "fm" for c in RECORDED]].notna().to_numpy()).any(axis=1).to_numpy()
    clean_dyad = ~fixed & ~np.isnan(rel)
    level = pairs[[f"m_self_{a}" for a in SELF_ATTRS]].mean(axis=1)
    flipped = build_pairs(clean.assign(gender=1 - clean.gender)).set_index("dyad").loc[pairs.dyad]
    sym = ["abs_age_diff", "int_corr", "int_corr_released", "int_corr17", "same_field", "same_goal"]
    neq = lambda a, b: int((~np.isclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True)).sum())
    return {
        "dyads where the released formula on recorded values differs from int_corr_released": neq(released_formula, rel),
        "dyads with no recording-error fix where our int_corr, rounded, differs from the released": neq(np.round(pairs.int_corr[clean_dyad], 2), rel[clean_dyad]),
        "dyads where same_field differs from clean.csv": neq(pairs.same_field, f_rows.same_field.to_numpy()),
        "dyads where fit_raw_to_f_prefs != his mean self-rating + fit_to_f_prefs": neq(pairs.fit_raw_to_f_prefs, level + pairs.fit_to_f_prefs),
        **{f"dyads where {c} changes when f and m are swapped": neq(flipped[c], pairs[c]) for c in sym},
        **{f"dyads where {a}_f_prefs does not become {a}_m_prefs when swapped": neq(flipped[f"{a}_f_prefs"], pairs[f"{a}_m_prefs"])
           for a in ("fit_to", "fit_raw_to")},
        "dyads where age_diff does not change sign when swapped": neq(flipped.age_diff, -pairs.age_diff),
    }, int(fixed.sum())


def main():
    clean = pd.read_csv(CLEAN, low_memory=False)
    notes = [f"Input: {CLEAN}, {len(clean)} rows, {clean.dyad.nunique()} dyads"]
    two = check_two_sided(clean)
    pairs = build_pairs(clean)
    checks, n_fixed = check_pairs(pairs, clean)
    rate = pairs.match.mean()
    notes += [f"Dyad rows: {len(pairs)} (expected {EXPECTED_ROWS}); matches {int(pairs.match.sum())}, rate {rate:.4f} (expected {EXPECTED_MATCH_RATE})"]
    notes += [f"{k} (should be 0): {v}" for k, v in {**two, **checks}.items()]
    notes.append(f"Dyads involving a person with a recording-error fix (gaming 14, reading 13): {n_fixed}")

    sets = feature_sets()
    alt = [c for d in sets["alternatives"].values() for c in d.values()]
    missing_cols = [c for s in ("pre", "pre+date", "categorical", "flags_pre", "flags_date", "format_standardized", "spline_candidates")
                    for c in sets[s] if c not in pairs.columns] + [c for c in alt if c not in pairs.columns]
    assert not missing_cols, f"feature-set columns not in the table: {missing_cols}"
    for name in ("pre", "pre+date"):
        cc = pairs[sets[name]].notna().all(axis=1)
        notes.append(f"Feature set `{name}`: {len(sets[name])} columns ({len(sets['categorical'])} categorical codes); "
                     f"complete dyads {int(cc.sum())} of {len(pairs)}, matches among them {int(pairs.match[cc].sum())}")
    notes.append("Missing-data flags set (dyads): " + ", ".join(f"{c} {int(pairs[c].sum())}" for c in FLAGS_PRE + FLAGS_DATE))
    miss = pairs[sets["pre+date"]].isna().sum()
    notes.append("Missing values by feature, where any: " + ", ".join(f"{c} {v}" for c, v in miss[miss > 0].sort_values(ascending=False).items()))
    notes.append(f"int_corr (ours) vs int_corr_released: correlation {pairs[['int_corr', 'int_corr_released']].corr().iloc[0, 1]:.4f}; "
                 f"int_corr17 vs int_corr: {pairs[['int_corr17', 'int_corr']].corr().iloc[0, 1]:.3f}")

    assert len(pairs) == EXPECTED_ROWS and round(rate, 3) == EXPECTED_MATCH_RATE
    assert all(v == 0 for v in {**two, **checks}.values()), "a consistency check failed; see the summary"

    pairs.to_csv(OUT, index=False)
    SETS.write_text(json.dumps(sets, indent=2) + "\n")
    SUMMARY.write_text("# Pair table summary\n\n" + "\n".join(f"- {n}" for n in notes)
                       + f"\n\nOutput: {OUT} with {len(pairs)} rows and {pairs.shape[1]} columns; feature sets in {SETS}.\n")
    print("\n".join(notes)); print(f"wrote {OUT}, {SETS}, {SUMMARY}")


if __name__ == "__main__":
    main()
