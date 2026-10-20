# Abraham's part of the written report and slides

Everything here is drawn from the repository's results. Every number traces to a file in `results/` through the
script that made it. Nothing was typed in by hand.

## What goes where in the Overleaf report

| File | Where it goes | What it is |
| --- | --- | --- |
| `abraham_background.tex` | End of Section 2 (Background), after "Random Forests" | **Methods beyond the course textbook**: grouped CV, nested CV, cluster-robust SEs, the AUC and its clustered SE, the corrected resampled t, regularized discriminant analysis, the lasso path for logistic regression, pre-registration, simulation studies. Each entry says what the method is, gives 2-3 sentences on why we used it, and cites a primary source and a textbook. |
| `abraham_technical_implementation.tex` | Replaces the draft's Section 4 subsections from "Cross-Validation" through "Model Evaluation" | The same content with the fact-check corrections (below) and citations added. Keep the draft's "Prediction Problem", "Constructed Pair-Level Predictors", "Pre-Date Feature Sets" and "Survey-Format Standardization". |
| `abraham_errorbars_section.tex` | A new section after Section 5 (Results) | **Error bars when rows share people and events**: Lemma 1 (proof), cluster-robust variance, Proposition 6 (proof), Proposition 8, simulation design, results with four figures, the cross-validated AUC with Proposition 10 (proof sketch), recommendations. About 6 pages with figures. To shorten it, cut the AUC subsection to its first paragraph and Figure `paper_auc_bias`. |
| `references_abraham.bib` | Append to `references.bib` | BibTeX for every key used above (verified; see the notes column in `citations.md`). |
| `figures/paper_*.png` | Upload to Overleaf in a `figures/` folder | Untitled figures for LaTeX (the caption carries the message). |

The three `.tex` files compile together in the template's preamble: 10 pages, no errors, no overfull lines. They
need these labels from the draft, which it already has: `sec:lda_theory` and `sec:random_forest_theory`.

### A figure for Section 5 (Results)

`figures/paper_auc_by_model.png` shows every headline model's held-out AUC on `pre` and `pre+date`, with 90%
intervals. Suggested caption:

> Before the date every model is near chance; the four-minute date adds most of the signal. Held-out AUC (event
> 5-fold cross-validation, headline specification, 4,184 dyads) with AUC $\pm t_{20}$ times the clustered SE
> (results/part2\_models\_metrics.csv).

## Fact-check of the existing draft (full table in `factcheck_draft.md`)

**No number in the draft is wrong.** Every AUC, SE, count and gap matches the stored results and was reproduced
with the current code. The problems are wording that leaves out the specification, a few implementation details
stated more broadly than the code does them, and robustness results the draft leaves out.

Corrections already made in `abraham_technical_implementation.tex`:

- **Splines.** "Logistic" and "spline-augmented logistic" are one model. No spline was chosen in any fold, so the
  reported full logistic model is linear.
- **Format standardization.** It is computed once, before CV, from all participants' answers. No outcomes are
  used, so no match information leaks, but it is not done within folds.
- **Skewness, clipping and standardization.** Skewness for person-level variables is computed over one value per
  participant. Clipping applies only to the log-transformed predictors. Dummies stay 0/1.
- **Tuning landed on the edge of the grids.** QDA chose r = 0.9 in every fold, and the `pre` forest chose leaf
  size 50 in every fold.
- **The AUC SEs run slightly narrow.** In simulation they are 0.84 to 0.95 of the true spread, so a nominal 90%
  interval behaves like roughly an 83% one.

Suggested fixes for the other sections (exact sentences in `factcheck_draft.md`, section 2):

- **Missing data, complete cases.** 2,685 dyads (477 matches) is right. Say that it is the set complete on every
  `pre+date` predictor, used for both feature sets. Requiring only the `pre` predictors would keep 3,847 (632).
- **Missing data, indicators.** There is one indicator per participant (`f_pre_missing`, `m_pre_missing`), not one
  "either participant" flag. `pre+date` adds two more for in-date ratings.
- **Pre-date variables.** The shared-interests share is left out as a main effect but enters `pre_interactions`
  through the two interaction terms.
- **Preference fit.** Define w_a as her share rescaled to sum to 1 over the five self-rated attributes. Say that
  `fit_raw` is used only in a robustness specification.
- **Results labels.** Label results "headline specification, event 5-fold CV". Note that 0.8459 is the best among
  the pre-registered models (Part 2's stacked ensemble reached 0.8464). Explain the negative liking R-squared: it
  is worse than predicting the training mean (-0.0028), on 8,128 dates.

Caveats the draft omits, which change how a reader takes the results (details in `factcheck_draft.md`, section 3):

1. **P1 passes narrowly.** It clears 0.58 by 0.0025. Forest seeds move the AUC from 0.573 to 0.587, and fold
   seeds move the logistic AUC with SD 0.013. P1 fails on complete cases and in all six wave-exclusion versions.
2. **P4 depends on the specification.** It is not supported in all 16 imputed specifications and supported in 7 of
   the 8 complete-case ones (gap 0.0215).
3. **The attractiveness margin is thin for women.** Its lead over shared interests is 2.9 points.

## Slides (Speed-dating deck, version 14)

Changes to Abraham's slides:

- **"One pipeline" (8).** `pre+date` corrected from 92 to 90 variables: 92 listed minus the two reference shares,
  as for `pre`'s 74. Citations added to the footer: grouped CV, nested tuning, pre-registration.
- **"How each model was fitted" (9).** Added what the tuning chose (no spline, r = 0.9, leaf 50) and a footer with
  citations: James et al. 2023; Friedman 1989; Breiman 2001. The speaker notes explain why: the pre-date signal is
  too weak for flexible fits.
- **New "We tested every error bar on simulated events" (10).** The coverage figure, a three-line takeaway, and
  citations: Liang & Zeger; Cameron & Miller; Nadeau & Bengio; Morris et al.
- **New "The usual cross-validated AUC runs low when events differ" (11).** The AUC-bias figure, a takeaway, and
  citations: Parker et al.; Airola et al.; DeLong et al.; Obuchowski.
- **Later slides.** Page numbers shifted by two: 12 to 17.

### For the team to decide (not edited)

**Jenna's slide "The usual error bars are 2 to 3 times too narrow" (15) is out of date.**

- Its fourth bar presents "our cluster-aware nested CV" as the method. With events redrawn it covered only 72% to
  86% (N1, N5).
- Its "79%" figure is from the early simulation that reused the same 21 events. With events redrawn, the
  rows-independent interval covers 46% to 66%.

Suggested replacement for the fourth bar and the side text, all from `results/errorbars_real.csv` and
`errorbars_final.csv` (full logistic model, `pre`):

- Bars (SE of the error rate): rows independent 0.008; clustered by event 0.017 (2.3×); clustered by woman and by
  man 0.019 (2.5×); Nadeau–Bengio on event folds 0.019 (2.5×).
- Side text: "In simulations built like our data, the usual 90% interval caught the truth only about half the time;
  event-aware intervals caught it 85% to 98% of the time."

## Regenerating

```bash
.venv/bin/python scripts/report_figures.py && .venv/bin/python scripts/report_figures.py --paper && .venv/bin/python scripts/report_figures.py --slides
```
