# Abraham's part of the written report and slides

Everything here comes from the repository's results. Every number traces to a file in `results/` through the
script that made it.

## For the midterm (now)

Everything is written at ISL level. Files marked **yours** are new material. Files marked **drop-in** add to a
teammate's section without rewording it. Each `.tex` file says in a comment where it goes.

| # | File | Where it goes | Whose section | What it is |
| --- | --- | --- | --- | --- |
| 1 | `references_midterm.bib` | Append to `references.bib` | all | The 17 entries the midterm files cite. `James:2023` and `Fisman:2006` are already in `references.bib`. |
| 2 | `abraham_background.tex` | End of Section 2, after "Random Forests" | **yours** | Methods beyond the course textbook: grouped CV, nested CV, cluster-robust SEs, the AUC's SE, regularized QDA, the lasso for logistic regression, pre-registration. |
| 3 | `section4_with_citations.tex` | Section 4, from "Survey-Format Standardization" to the end | teammate's | Their text word for word, with citations (`% CITE`) and five one-sentence fixes (`% FIX`, with sources) for the author to accept or drop. |
| 4 | `background_additions.tex` | Section 2, "Data Description" | drop-in | (1) A related-work paragraph: Joel et al. 2017, Eastwick & Finkel 2008, Fisman et al. 2008, each checked against its abstract. (2) A data figure (event sizes and match rates) with one sentence. |
| 5 | `results_additions.tex` | Section 5 (Results) | drop-in | The AUC figure, three tables (models, compatibility ladder, stated vs revealed) and a new subsection, "How robust are these results?". The insertion points are marked. |
| 6 | `drafts_abstract_intro_conclusion.tex` | Title, Abstract, Introduction, Conclusion | **drafts** | Replaces the template placeholders: an abstract, an introduction with a notation paragraph and a roadmap, and a conclusion. Error bars appear only as upcoming work. |

Upload to Overleaf **beside `Written_Report.tex`** (no subfolders; the drop-ins use bare file names, the same way
the template uses `Bessel_functions.png`): `figures/paper_auc_part1.png`, `figures/paper_events.png`,
`tables/models.tex`, `tables/ladder.tex`, `tables/stated_revealed.tex`.

### Shortcut: `Written_Report_midterm.tex`

The Overleaf file as of 2026-10-20 with all of the above applied. The template leftovers are removed and the drafts
and drop-ins are inserted; search for `DRAFT` and `ADDED`. Teammates' text is unchanged, and Section 4 still has no
citations (item 3 is for its author). Paste it over `Written_Report.tex` only when nobody else is editing, or copy
the marked blocks across one at a time. It compiles locally (plain bibtex standing in for biber) to 24 pages, with
every citation and reference resolved; the only warnings are two lines in teammates' text that run 3 to 4 pt into
the margin.

### Template leftovers to delete before submitting (all removed in the shortcut file)

- [ ] The instruction block at the top, the resource lists, and the red "Delete everything above…" note
- [ ] The opening paragraph of Background ("The background and literature review section discusses…")
- [ ] The two template paragraphs at the start of Data Cleaning ("This section should provide…", "Make sure to include…")
- [ ] The example figure (`Bessel_functions.png`) and example table at the end of Results, with the paragraph about `\ref`
- [ ] The Acknowledgments placeholder text (write one or two real sentences)
- [ ] The red paragraph about citing references
- [ ] The appendix examples: "Data Tables… Below are the data collected for…" and the `add_numbers` code listing

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

Suggested as one-sentence `% FIX` additions in `section4_with_citations.tex`:

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
