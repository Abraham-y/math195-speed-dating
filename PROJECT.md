# Project reference: "Predicting the Mutual Interest of Two Strangers Before Meeting"

Math 195 (Statistical Learning) capstone, Stanford, autumn 2026. Ten-week quarter. Textbook: James, Witten,
Hastie, Tibshirani, Taylor, An Introduction to Statistical Learning with Applications in Python (ISLP).

## Team

| Person | Role | Owns |
| --- | --- | --- |
| Sophia Qin | Project Manager | Project summary and goals in the CPP; coordination |
| Jenna Yang | Mathematical Lead | Weekly PRPR; the theoretical parts (variance derivations, simulation checks) |
| Abraham Yeung | Technical Specialist | Timeline in the CPP; repository, pipeline, scripts, this file |

Class meets Tuesday and Thursday. Optional synchronous check-in Sunday 5:30 to 6:00 pm weekly.
Week 1 decision: do the course's original project description first (predictive model plus exploratory
analysis), and add the more complex components only after that is achieved. The team brings different
background levels; the schedule should not assume everyone reads at the same pace.

## Research question

To what extent can a mutual match between two strangers be predicted using only information available
before they meet, and which characteristics of a pair are most useful? Secondary: how much predictive
information does the four-minute interaction itself add?

## Data

Columbia speed-dating experiment, 2002 to 2004 (Fisman, Iyengar, Kamenica, Simonson). Public release on
Kaggle, "Speed Dating Experiment" (annavictoria). Each participant had a four-minute date with every
opposite-sex participant in their event (wave), then rated the partner on six attributes and said yes or no.
A match is both saying yes. Pre-event and post-event surveys cover demographics, dating goals and
frequency, interests (17 activities rated 1 to 10), stated partner preferences (six attribute weights),
self-ratings, and expectations. 8,378 rows (one per date from one side), 551 participants, 21 events,
4,184 dyads all seen from both sides, match rate 16.5%, yes rate 42%.

## Two feature sets, two model sets

Pre-meeting set (the main question). Everything known before the date: each person's age, field, career,
goals, dating and going-out frequency, 17 interest ratings, stated preference weights (harmonized shares),
self-ratings, expected happiness; and pair-level compatibility features constructed from them: age
difference, interest similarity (`int_corr` is already in the data; also compute our own), same field, same
goal, and preference alignment (sum over attributes of one person's stated weight times the other person's
self-rating, computed in both directions).

Pre-meeting plus in-date set (the comparison). Adds what was recorded during the date: each side's six
attribute ratings of the other, overall liking, and the guess that the partner will say yes.

Comparing the two model sets answers the secondary question. The gap in AUC between them is the value of
the four-minute meeting.

Unit of analysis for a match: one row per dyad. Because every date is opposite-sex, order the two people's
features by gender (prefix `f_` and `m_`) so the row is well defined. Unit of analysis for a yes: one row per
date from one side (rater and ratee), as in the raw data.

## Models and evaluation

Models: logistic regression, LDA, QDA, lasso-logistic, classification tree, bagging, random forest,
boosting (ISL chapters 4, 6, 8). Evaluation: cross-validation holding out whole events, so no person is in
both training and test data; random-row hold-out is reported only to show how much it overstates accuracy.
Metrics: classification error, balanced accuracy, sensitivity, specificity, AUC. Matches are 16.5% of dyads,
so balanced accuracy and AUC carry the comparison, and thresholds are chosen on training folds only.
Error bars on cross-validated metrics use the cluster-robust estimator at minimum (see proposals/06).

## The methods question (second half)

Rows are correlated through shared people and shared events, which makes the conventional cross-validation
standard error too narrow. The project measures by how much on this data (first run: about 40% too narrow)
and, in simulation, which error-bar constructions have honest coverage. Details and the theory behind it:
proposals/06-theory-track-cv-under-clustering.md.

## The further question (choose by week 4)

Option 1: rank participants by desirability with honest uncertainty (Rasch model, bootstrap, rank ranges).
Option 2: score participants' own per-date guesses of whether the partner would say yes (calibration,
Brier score) and their pre-event predictions of how many matches they would get.

## Stretch goal (only if everything above is done, and only with permission)

Apply the pre-meeting model to a Stanford dataset (Date Drop, Marriage Pact, Senior Scramble) or a dating
app. Any such data involves real students; it requires written permission from the data owner and a check
with the instructor about IRB requirements before a single row is loaded.

## Deliverables and milestones

Weekly: Written Report (WR) section and Project Reflection and Planning Report (PRPR).
Week 5: midterm WR. Week 6: midterm presentation and performance review. Week 8: poster first draft.
Week 9: final WR and final poster. Week 10: final presentation. April: ASURPS application (deadline end of
March, to be confirmed).

## Timeline (from the CPP; three tasks per week)

Week 1: choose dataset and read codebook; write the plan and the hypotheses; read ISL 2.1.1, 2.1.3,
2.2.1 to 2.2.3, 3.1 to 3.3, chapter 3 lab.
Week 2: repository, environment, pre-registration committed, cleaning script; exploratory analysis and the
data section of the WR; read ISL 4.1 to 4.5.1 and 5.1 to 5.2, chapter 4 and 5 labs, Fisman et al. 2006
sections I to III.
Week 3: logistic regression for a yes and for a match, coefficients by gender, stated-versus-revealed table;
CV harness with both hold-out schemes and correlation-aware error bars; random forest benchmark; read
4.3.4, 3.3.2, 8.2.2.
Week 4: LDA and QDA, model comparison table; lasso; choose the further question and pre-register it; read
6.1.3, 6.2.1 to 6.2.3.
Week 5: error-bar study on real data; matched simulation; read Bates, Hastie and Tibshirani 2024; midterm WR.
Week 6: midterm presentation; begin the further question; its one paper; act on the performance review.
Week 7: trees, bagging, boosting; leakage comparison with tree models; finish the further question; read
8.1, 8.2.1 to 8.2.3, 8.2.5.
Week 8: freeze results, robustness checks; poster first draft; optional ISL 13.1 to 13.3; theory appendix.
Week 9: final WR; final poster and ASURPS abstract; code review.
Week 10: final presentation; tag release; reflection.

## Key references

- Fisman, Iyengar, Kamenica, Simonson. "Gender Differences in Mate Selection: Evidence from a Speed Dating
  Experiment." QJE 2006. The source of the data; the stated-versus-revealed result to reproduce.
- Joel, Eastwick, Finkel. "Is Romantic Desire Predictable? Machine Learning Applied to Initial Romantic
  Attraction." Psychological Science 2017. Closest prior work to our main question: pre-meeting self-reports
  predicted people's general tendency to like and be liked, but not who liked whom specifically. Our
  pair-level compatibility features test exactly that.
- Bates, Hastie, Tibshirani. "Cross-Validation: What Does It Estimate and How Well Does It Do It?" JASA 2024.
  The error-bar question.
- ISLP chapters 2 to 6 and 8; ESL 7.10 to 7.12 for the theory.

## Status as of 2026-09-29 (end of week 1)

Done: dataset chosen and downloaded; codebook read; cleaning script written and checked; hypotheses drafted;
first error-bar comparison and simulation run (proposals/06 sections 7 and 7b). Next: pair-level table,
exploratory analysis, and the week 2 reading.

## Phase 2: additional discovery (the earlier plan, in priority order)

The team's week 1 decision: finish the plain predictive analysis (Part 1) first, then add complexity. This
section is the menu for that second phase. Each item says what it is, what it costs, what already exists,
and where the details live. Nothing here starts before the Part 1 model table exists.

### Tier 1: planned and already in the timeline

Error-bar study (weeks 5 and 8). Because the same people and events appear in many rows, the conventional
cross-validation standard error is too narrow. Measure by how much on the real data and, in simulation,
which constructions have honest coverage. Status: first real-data run done (naive bar about 40% too narrow;
`scripts/layerA_real_data.py`), simulation harness done (`scripts/sim_cv_cluster.py`), and the theory
written up with four proved propositions (proposals/06, sections 2 to 5). Remaining: rerun both with the
final Part 1 models, add the Bates et al. bias adjustment, report. Cost: about two weeks, mostly reuse.

Further question (weeks 6 and 7), one of:
- Desirability ranking with honest uncertainty. Rasch model (each participant a desirability, each rater a
  pickiness; logistic regression with one indicator per person), bootstrap, simultaneous rank ranges per
  event, and a dates-needed table. Note: the comparison graph is complete within an event and empty across
  events, so rankings are per event by necessity. Reference: Mogstad et al. 2024. Cost: two weeks.
- Accuracy of participants' own forecasts. Score each participant's per-date guess that the partner will say
  yes (`prob`, present on 96% of rows) against the actual decision: reliability diagram, Brier score and its
  calibration and resolution parts, calibration slope by gender; plus pre-event predicted match counts
  (`match_es`, 86% of participants) against realized. Reference: Murphy 1973. Cost: one to two weeks.

### Tier 2: small add-ons, each a section rather than a layer

- Regression to the mean at the top. Split each person's dates into odd and even; the top-rated person in an
  event on odd dates drops on even dates; test whether James-Stein shrinkage of per-person mean ratings
  predicts the held-out half better than raw means. One week. proposals/05, Layer D.
- Strategic yes-saying. Do people say yes more when they expect a yes back, holding fixed how much they
  liked the partner? One extra term in the Part 1 yes model. Half a day.
- Stable matching. Compute the Gale-Shapley stable matching within each event from the estimated
  preferences and measure how far the realized matches are from it. One week, plus a page of proof if wanted.
- Psychometrics of the ratings. Variance components (person rated vs rater vs noise), inter-rater
  reliability, halo effects (a high attractiveness rating pulling the other five up). Regressions the team
  already knows. A few days.
- Nonlinearity check. GAM on the six ratings (ISL 7.7.2) to see whether each rating's effect is linear.
  One day.

### Tier 3: the theory track (Jenna, optional, runs alongside)

Goal: a variance formula for cross-validation error under two-way (rater and ratee) clustering, with a
cluster-aware nested CV estimator whose intervals cover in simulation. Proved so far (proposals/06):
grouped and random-split CV estimate different quantities; the grouped-CV variance equals the independent-
rows formula plus one cluster term; the cluster-robust SE misses exactly the between-fold covariance; the
cluster-nested MSE estimate is unbiased. Next steps in order (proposals/06 section 8): add the published
bias adjustment and rerun; estimate the between-fold covariance directly instead of using a floor; extend
to unbalanced clusters; prove Proposition 4 for the cross-classified case; simulate the dyadic case at the
dating data's dimensions. Items 1 to 3 are about a week; 4 and 5 are the part that would be new.

### Tier 4: alternative directions, kept for reference (proposals/01 to 04)

Written before the team chose the dating dataset; the same statistical questions on other data. Useful if
the team wants a second dataset for the poster, a pivot, or a follow-on project.
- 01. Cross-validation intervals when labels come from a noisy LLM judge (Chatbot Arena data).
- 02. The winner's curse on LLM leaderboards: how much the top score is inflated, and shrinkage to fix it.
- 03. Rank uncertainty in Bradley-Terry arenas: are published leaderboard ranks honest, and how many votes
  separate two models.
- 04. Markets versus models: Polymarket's monthly "best AI model" contracts against a statistical forecaster
  built from arena votes.
Each has a "Connections and extensions" section (economics, prediction markets, pure and applied
mathematics) written for a general reader.

### Stretch: a Stanford dataset

Apply the pre-meeting model to Date Drop, Marriage Pact, Senior Scramble or a dating app. Requires the data
owner's written permission and the instructor's confirmation on IRB requirements before any row is loaded.
