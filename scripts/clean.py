"""Clean the speed-dating data.

Input : data/speed_dating.csv (Kaggle release of the Columbia speed-dating experiment)
Output: data/clean.csv, data/clean_summary.md

What it does
1. Drops rows with no decision recorded or no partner id.
2. Adds identifiers: rater (iid), ratee (pid), event (wave), dyad (unordered pair), date_id (ordered pair).
3. Harmonizes the stated-preference blocks. The codebook says waves 6 to 9 rated each attribute on a
   1 to 10 scale and the other waves split 100 points ("Preference Scale"). In the release, blocks 1_1 and
   2_1 for waves 6 to 9 are already rescaled to shares of 100, while block 4_1 still holds the 1 to 10
   answers, and 100-point answers do not always sum to exactly 100. Every block is converted to shares
   summing to 100 (a no-op where it already does); the survey format is kept in `pref_scale`. A check
   confirms which blocks arrived already converted. Shares from the two formats are on the same units but
   not equally spread; see PREREG.md on standardizing within format.
4. Flags missing attribute ratings and marks complete-case rows.
5. Flags the event-level variations listed in the codebook (undergraduate wave, yes-budget wave,
   different MC, brought a book or magazine) and keeps `condtn` (limited vs extensive choice).
6. Adds a few derived predictors (age difference, same field of study).
7. Runs consistency checks and writes a summary of everything dropped or flagged.
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
WAVE_FLAGS = {"wave_undergrad": {5}, "wave_yes_budget": {12}, "wave_diff_mc": {13, 14},
              "wave_brought_item": {18, 19, 20, 21}}

def main():
    raw = pd.read_csv(RAW, encoding="latin-1")
    notes = [f"Raw rows: {len(raw)}; participants: {raw.iid.nunique()}; waves: {raw.wave.nunique()}"]
    df = raw.copy()

    # 1. drop unusable rows
    n0 = len(df)
    df = df.dropna(subset=["dec", "pid"])
    notes.append(f"Dropped {n0 - len(df)} rows with no decision or no partner id")
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

    # 6. derived predictors
    df["age_diff"] = df.age - df.age_o
    partner_field = df.drop_duplicates("rater").set_index("rater")["field_cd"]
    pf = df.ratee.map(partner_field)
    df["same_field"] = (df.field_cd == pf).astype(float)
    df.loc[df.field_cd.isna() | pf.isna(), "same_field"] = np.nan

    # 7. consistency checks
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

    df.to_csv(OUT, index=False)
    SUMMARY.write_text("# Cleaning summary\n\n" + "\n".join(f"- {n}" for n in notes) + f"\n\nOutput: {OUT} with {len(df)} rows and {df.shape[1]} columns.\n")
    print("\n".join(notes)); print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
