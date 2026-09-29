# Math 195 capstone: speed-dating data, with an honest-error-bar study

A statistical-learning course project on the Columbia speed-dating experiment. Part 1 predicts whether a
participant says yes to a partner, and whether a date becomes a match, from attribute ratings, and interprets
what drives choices. Part 2 asks whether the usual cross-validation error bars are trustworthy when the same
people appear in many rows, and one further question. See `proposals/05-speed-dating-dataset-first.md` for the
full plan, `PREREG.md` for the hypotheses, and `CLAUDE.md` for working conventions.

## Setup

    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python scripts/clean.py                                 # writes data/clean.csv and data/clean_summary.md

## Layout

    data/        raw CSV and codebook (from Kaggle: "Speed Dating Experiment"), cleaned CSV is generated
    scripts/     one script per analysis step; each writes to results/
    proposals/   project proposals and the theory note
    results/     generated tables and figures (not committed)
    PREREG.md    pre-registered hypotheses, committed before analyses run
    CLAUDE.md    conventions for anyone (or any agent) working in this repo

## Scripts

    clean.py               cleaning and consistency checks (run first)
    layerA_real_data.py    first error-bar comparison on the real data
    sim_cv_cluster.py      simulation of cross-validation intervals under clustering

## Data provenance

Fisman, Iyengar, Kamenica and Simonson, speed-dating experiment at Columbia University, 2002 to 2004.
Public release on Kaggle ("Speed Dating Experiment", uploaded by Anna Montoya). The copy in `data/` was taken
from a public mirror and matches the Kaggle release's dimensions (8,378 rows, 195 columns); replace it with the
official download and cite Kaggle in the report.
