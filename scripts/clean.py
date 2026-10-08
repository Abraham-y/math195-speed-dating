"""Clean the speed-dating data.

Input : data/speed_dating.csv (Kaggle release of the Columbia speed-dating experiment)
Output: data/clean.csv, data/clean_summary.md

What it does
1. Drops rows with no decision recorded or no partner id (in this release: 10 wave-5 rows whose partner, the
   woman in position 7, did not attend; no row lacks a decision).
2. Adds identifiers: rater (iid), ratee (pid), event (wave), dyad (unordered pair), date_id (ordered pair).
3. Harmonizes the stated-preference blocks. The codebook says waves 6 to 9 rated each attribute on a
   1 to 10 scale and the other waves split 100 points ("Preference Scale"). In the release, blocks 1_1 and
   2_1 for waves 6 to 9 are already rescaled to shares of 100, while block 4_1 still holds the 1 to 10
   answers, and 100-point answers do not always sum to exactly 100. Every block is converted to shares
   summing to 100 (a no-op where it already does); the survey format is kept in `pref_scale`. A check
   confirms which blocks arrived already converted. Shares from the two formats are on the same units but
   not equally spread; see PREREG.md on standardizing within format. In the 100-point waves, a block with some
   items blank whose answered items already sum to 100 (within 1 point) gets 0 for the blank items, since no
   points were left to give (PREREG.md, fixes of 2026-10-08); the number of items set to 0 is kept in
   `pref_blanks_set_to_0`.
4. Flags missing attribute ratings and marks complete-case rows.
5. Flags the event-level variations listed in the codebook (undergraduate wave, yes-budget wave,
   different MC, brought a book or magazine) and keeps `condtn` (limited vs extensive choice).
6. Sets the interest values PREREG.md treats as recording errors (gaming 14, reading 13) to missing, keeping the
   recorded values in `gaming_recorded` and `reading_recorded` for the robustness grid.
7. Adds a few derived predictors (age difference, same field of study).
8. Runs consistency checks and writes a summary of everything dropped or flagged.
Run from the repository root: python scripts/clean.py
"""
import pandas as pd, numpy as np, pathlib

RAW = pathlib.Path("data/speed_dating.csv")
OUT = pathlib.Path("data/clean.csv")
SUMMARY = pathlib.Path("data/clean_summary.md")

ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
# stated-preference blocks that the codebook says used a wave-dependent scale
PREF_BLOCKS = {"1_1": "own preferences, pre-event", "2_1": "what the opposite sex wants, pre-event",
               "4_1": "what same-sex peers want, pre-event"}
SCALE_1_TO_10_WAVES = {6, 7, 8, 9}
# whether the release already holds waves 6 to 9 as shares of 100, per block (checked below)
PRECONVERTED_IN_RELEASE = {"1_1": True, "2_1": True, "4_1": False}
# interest values that recur in the same question across several events (PREREG.md, out-of-range values)
RECORDING_ERRORS = {"gaming": 14, "reading": 13}
WAVE_FLAGS = {"wave_undergrad": {5}, "wave_yes_budget": {12}, "wave_diff_mc": {13, 14},
              "wave_brought_item": {18, 19, 20, 21}}

def apply_recording_errors(df):
    """Set each known recording error to missing, keeping the recorded column as <name>_recorded.
    Returns the frame, the number of people affected and the number of values changed per column."""
    people, values = {}, {}
    for col, bad in RECORDING_ERRORS.items():
        df[f"{col}_recorded"] = df[col]
        hit = df[col] == bad
        people[col], values[col] = int(df.loc[hit, "rater"].nunique()), int(hit.sum())
        df.loc[hit, col] = np.nan
    return df, people, values


def fill_blank_allocations(df):
    """In the 100-point waves, set the blank items of a stated-preference block to 0 when the answered items
    already sum to 100 within 1 point. Adds `pref_blanks_set_to_0` (items set to 0 in the row, over all blocks).
    Returns the frame and, per block, (people, rows) changed."""
    changed = {}
    df["pref_blanks_set_to_0"] = 0
    for blk in PREF_BLOCKS:
        cols = [f"{a}{blk}" for a in ATTRS]
        if not all(c in df.columns for c in cols):
            continue
        block = df[cols].astype(float)
        blank = block.isna()
        hit = (~df.event.isin(SCALE_1_TO_10_WAVES) & blank.any(axis=1) & ~blank.all(axis=1)
               & block.sum(axis=1).sub(100).abs().le(1))
        df[cols] = block.mask(blank & hit.to_numpy()[:, None], 0.0)
        df["pref_blanks_set_to_0"] += (blank & hit.to_numpy()[:, None]).sum(axis=1)
        changed[blk] = (sorted(df.loc[hit, "rater"].unique().tolist()), int(hit.sum()))
    return df, changed


def clean_frame(raw):
    """Every cleaning step on the raw release; returns (clean frame, summary notes)."""
    notes = [f"Raw rows: {len(raw)}; participants: {raw.iid.nunique()}; waves: {raw.wave.nunique()}"]
    df = raw.copy()

    # 1. drop unusable rows
    no_pid, no_dec = df.pid.isna(), df.dec.isna()
    df = df[~(no_pid | no_dec)]
    waves = sorted(raw.loc[no_pid, "wave"].unique().tolist())
    notes.append(f"Dropped {int(no_pid.sum())} rows with no partner id (waves {waves}) and {int((no_dec & ~no_pid).sum())} "
                 "further rows with no decision")
    dup = int(df.duplicated(subset=["iid", "pid"]).sum())
    df = df.drop_duplicates(subset=["iid", "pid"])
    notes.append(f"Dropped {dup} duplicate (rater, ratee) rows")

    # 2. identifiers
    df["rater"] = df.iid.astype(int)
    df["ratee"] = df.pid.astype(int)
    df["event"] = df.wave.astype(int)
    lo, hi = np.minimum(df.rater, df.ratee), np.maximum(df.rater, df.ratee)
    df["dyad"] = lo.astype(str) + "_" + hi.astype(str)
    df["date_id"] = df.rater.astype(str) + "->" + df.ratee.astype(str)

    # 3. harmonize stated-preference blocks
    df["pref_scale"] = np.where(df.event.isin(SCALE_1_TO_10_WAVES), "1-10", "100pt")
    df, filled = fill_blank_allocations(df)
    notes.append("Blank items set to 0 where the answered items of a 100-point block already sum to 100 (PREREG.md, 2026-10-08): "
                 + "; ".join(f"block {b}: iids {ids} ({n} rows)" for b, (ids, n) in filled.items() if n))
    for blk, desc in PREF_BLOCKS.items():
        cols = [f"{a}{blk}" for a in ATTRS]
        if not all(c in df.columns for c in cols):
            notes.append(f"Block {blk} ({desc}): columns not all present, skipped"); continue
        block = df[cols].astype(float)
        total = block.sum(axis=1, min_count=1)
        shares = block.div(total, axis=0) * 100
        for a, c in zip(ATTRS, cols):
            df[f"{a}{blk}_share"] = shares[c]
        w = (df.pref_scale == "1-10") & block.notna().all(axis=1)
        preconverted = bool((total[w].sub(100).abs() <= 0.1).all())
        assert preconverted == PRECONVERTED_IN_RELEASE[blk], f"block {blk}: waves 6-9 conversion state differs from the release this script was written for"
        off = (df.pref_scale == "100pt") & total.notna() & (total.sub(100).abs() > 2)
        state = ("already shares of 100 in the release (every complete row within 0.1 of 100)" if preconverted
                 else f"raw 1-10 answers in the release (row sums {total[w].min():.0f} to {total[w].max():.0f})")
        notes.append(f"Block {blk} ({desc}): converted to shares of 100. Waves 6-9: {state}. 100-point waves: "
                     f"{df.loc[off, 'rater'].nunique()} people ({int(off.sum())} rows) summed more than 2 away from 100 before rescaling")

    # 4. missing ratings
    for a in ATTRS:
        df[f"{a}_missing"] = df[a].isna().astype(int)
    df["ratings_complete"] = df[ATTRS].notna().all(axis=1).astype(int)
    notes.append("Missing rate per rating: " + ", ".join(f"{a} {df[a].isna().mean():.1%}" for a in ATTRS))
    notes.append(f"Complete-case rows (all six ratings present): {int(df.ratings_complete.sum())} of {len(df)}")

    # 5. event-level variations from the codebook
    for flag, waves in WAVE_FLAGS.items():
        df[flag] = df.event.isin(waves).astype(int)
    notes.append("Event flags added: " + ", ".join(f"{k} (waves {sorted(v)})" for k, v in WAVE_FLAGS.items()) + "; condtn kept (1 = limited choice, 2 = extensive)")

    # 6. recording errors in the interest ratings
    df, people, values = apply_recording_errors(df)
    notes.append("Set to missing as recording errors (PREREG.md): " + ", ".join(f"{c} = {RECORDING_ERRORS[c]} for {n} people ({values[c]} rows)" for c, n in people.items())
                 + "; recorded values kept in " + ", ".join(f"{c}_recorded" for c in RECORDING_ERRORS))

    # 7. derived predictors
    df["age_diff"] = df.age - df.age_o
    partner_field = df.drop_duplicates("rater").set_index("rater")["field_cd"]
    pf = df.ratee.map(partner_field)
    df["same_field"] = (df.field_cd == pf).astype(float)
    df.loc[df.field_cd.isna() | pf.isna(), "same_field"] = np.nan

    # 8. consistency checks
    assert df.duplicated(subset=["rater", "ratee"]).sum() == 0
    assert set(df.dec.unique()) <= {0, 1}
    partner_wave = df.drop_duplicates("rater").set_index("rater")["event"]
    pw = df.ratee.map(partner_wave)
    notes.append(f"Rows whose partner is in a different wave (should be 0): {int((pw.notna() & (pw != df.event)).sum())}")
    both = df.dropna(subset=["dec_o"])
    mism = int((both.match != (both.dec * both.dec_o)).sum())
    notes.append(f"Rows where match != dec * dec_o (should be 0): {mism}")
    pairs = df.groupby("dyad").size()
    notes.append(f"Dyads: {len(pairs)}; seen from both sides: {int((pairs == 2).sum())}; one side only: {int((pairs == 1).sum())}")
    dpr = df.groupby("rater").size()
    notes.append(f"Dates per rater: min {dpr.min()}, median {int(dpr.median())}, max {dpr.max()}")
    return df, notes


def main():
    df, notes = clean_frame(pd.read_csv(RAW, encoding="latin-1"))
    df.to_csv(OUT, index=False)
    SUMMARY.write_text("# Cleaning summary\n\n" + "\n".join(f"- {n}" for n in notes) + f"\n\nOutput: {OUT} with {len(df)} rows and {df.shape[1]} columns.\n")
    print("\n".join(notes)); print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
