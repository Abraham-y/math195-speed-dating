# Abraham's part of the written report and slides

Everything here comes from the repository's results. Every number traces to a file in `results/` through the
script that made it.

## For the midterm (now)

| File | Where it goes in Overleaf | What it is |
| --- | --- | --- |
| `abraham_background.tex` | End of Section 2 (Background), after "Random Forests" | **Methods beyond the course textbook**, written at ISL level: grouped CV, nested CV, cluster-robust standard errors, the AUC's standard error, regularized QDA, the lasso for logistic regression, pre-registration. Each entry says what the method is and why we used it (2-3 sentences), with a precise citation. |
| `abraham_technical_implementation.tex` | Replaces the draft's Section 4 subsections from "Cross-Validation" through "Model Evaluation" | The same content with the fact-check corrections (below) and citations added. Keep the draft's "Prediction Problem", "Constructed Pair-Level Predictors", "Pre-Date Feature Sets" and "Survey-Format Standardization". |
| `references_abraham.bib` | Append to `references.bib` | 47 entries, with DOIs checked against Crossref. `James:2023` and `Fisman:2006` are omitted because `references.bib` already has them. Pointers and confidence levels are in `citations.md`. |
| `figures/paper_auc_by_model.png` | Optional, for Section 5 (Results) | Every model's held-out AUC on `pre` and `pre+date` with 90% intervals; suggested caption below. |

The two `.tex` files compile in the template's preamble: 5 pages, every citation resolved, no warnings. They use
two labels the draft already has: `sec:lda_theory` and `sec:random_forest_theory`.

Suggested caption for the Section 5 figure:

> Before the date every model is near chance; the four-minute date adds most of the signal. Held-out AUC (event
> 5-fold cross-validation, headline specification, 4,184 dyads) with AUC $\pm t_{20}$ times the clustered SE
> (results/part2\_models\_metrics.csv).

## For after the midterm: `after-midterm/`

The error-bar work is held back until after the midterm presentation.

| File | Where it goes | What it is |
| --- | --- | --- |
| `after-midterm/abraham_errorbars_section.tex` | A new section after Results | "How much should we trust a cross-validated number?", at ISL level plus three small additions: Lemma 1 and Result 1 (Proposition 6) with proofs, Results 2 and 3 in words, four figures, recommendations. |
| `after-midterm/abraham_appendix_theory.tex` | After `\appendix` | Formal Propositions 8, 9 and 10 with proof sketches. |
| `after-midterm/abraham_background_errorbars.tex` | End of `abraham_background.tex` | Two background entries only that section needs: repeated CV (Nadeau-Bengio) and simulation studies. |

These use `figures/paper_leakage.png`, `paper_coverage.png`, `paper_nb_theory.png`, `paper_real_intervals.png` and
`paper_auc_bias.png`. Together with the midterm files they compile to 12 pages.

In the deck, the two error-bar slides ("We tested every error bar on simulated events" and "The usual
cross-validated AUC runs low when events differ") are **hidden**, not deleted. Unhide them after the midterm.

## Fact-check of the existing draft (full table in `factcheck_draft.md`)

**No number in the draft is wrong.** Every AUC, SE, count and gap matches the stored results and was reproduced
with the current code. The problems are wording that leaves out the specification, a few implementation details
stated more broadly than the code does them, and robustness results the draft leaves out.

Already corrected in `abraham_technical_implementation.tex`:

- **Splines.** No spline was chosen in any fold, so the reported full logistic model is linear.
- **Format standardization.** It is computed once, before cross-validation, from all participants' answers. It
  uses no outcomes, so nothing leaks, but it is not done within folds.
- **Skewness, clipping and standardization.** Skewness for person-level variables is computed over one value per
  participant. Clipping applies only to log-transformed predictors. Dummies stay 0/1.
- **Tuning landed on the edge of the grids.** QDA chose r = 0.9 in every fold, and the `pre` forest chose leaf
  size 50 in every fold.
- **The AUC SEs run slightly narrow.** In simulation they are 0.84 to 0.95 of the true spread.

For the other sections (exact sentences in `factcheck_draft.md`, section 2):

- **Missing data.** The 2,685 complete-case dyads (477 matches) are those complete on every `pre+date` predictor.
  The missing-data indicators are one per participant.
- **Pre-date variables.** The shared-interests share enters `pre_interactions` through the interaction terms.
- **Preference fit.** Define w_a. Say that `fit_raw` is used only in a robustness specification.
- **Results.** Label results "headline specification, event 5-fold CV". Note that 0.8459 is the best among the
  pre-registered models. Explain the negative liking R-squared.

Caveats the draft omits:

- **P1 passes narrowly.** It clears 0.58 by 0.0025 and fails on complete cases and in the wave-exclusion versions.
- **P4 depends on the specification.**
- **The attractiveness gap.** For women, its lead over shared interests is only 2.9 points.

To check: the draft cites ISL "Section 4.4.1, pp. 153-155" for ROC and AUC. The section number was not
verified, and `abraham_background.tex` uses only the pages.

## Slides (Speed-dating deck, version 16)

- **"One pipeline" (8).** `pre+date` corrected from 92 to 90 variables. Citations added.
- **"How each model was fitted" (9).** Now says what the tuning chose (no spline, r = 0.9, leaf 50). Citations
  added.
- **Error-bar slides.** Hidden until after the midterm.
- **Next steps (15).** Presents the error-bar work as upcoming, with no results given away.
- **Jenna's slide "The usual error bars are 2 to 3 times too narrow" (13).** Also about error bars, and out of
  date: it presents our nested CV as the method, though it under-covers. If error bars are out of the midterm, it
  should probably be hidden too. That is the team's call; it has not been edited.

## Regenerating the figures

```bash
.venv/bin/python scripts/report_figures.py && .venv/bin/python scripts/report_figures.py --paper && .venv/bin/python scripts/report_figures.py --slides
```
