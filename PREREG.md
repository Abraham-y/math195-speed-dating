# Pre-registered hypotheses

Committed before the analyses that test them are run. Results are reported whichever way they fall.
A hypothesis is never edited after its test has run; if the team changes its mind, a new numbered
hypothesis is added with the date and reason in the change log.

Data: `data/clean.csv` and the pair-level table from `scripts/pairs.py`. Evaluation: 5-fold cross-validation
holding out whole events, cluster-robust standard errors, thresholds chosen on training folds. Feature sets:
`pre` (known before the date) and `pre+date` (adds the in-date ratings). Metrics: AUC and balanced accuracy
unless stated.

## Main question: predicting a match before the meeting (committed week 2)

- P1. With the `pre` set only, the best model's event-held-out AUC for predicting a match is at least 0.58
  (pre-meeting information is predictive) and below 0.70 (but weakly so).
- P2. Adding the in-date ratings (`pre+date`) raises the best model's event-held-out AUC by at least 0.10.
  The four-minute meeting carries more predictive information than everything known beforehand.
- P3. Within the `pre` set, pair-level compatibility features (age difference, interest similarity, same
  field, same goal, preference alignment) add less than 0.02 AUC over the two individuals' features alone.
  This is the Joel, Eastwick and Finkel (2017) finding restated: pre-meeting data predicts who tends to like
  and be liked, not who likes whom.
- P4. The random forest improves event-held-out AUC over logistic regression by less than 0.03 on the `pre`
  set. The signal is mostly additive.
- P5. In the logistic regression predicting a yes from the six in-date ratings, attractiveness has the largest
  standardized coefficient for both genders, and the gap between stated importance (share of the 100-point
  allocation) and revealed importance (share of the positive standardized coefficients) is largest for
  attractiveness (the Fisman et al. 2006 result).

## Second half: the error-bar question (committed week 2)

- A1. Random-row cross-validation reports a higher AUC than event-held-out cross-validation for every model,
  and the gap is larger for the tree models than for the logistic model.
- A2. In a simulation matched to the data's structure (event sizes, dates per person, a fitted logistic
  truth), the conventional 90% cross-validation interval contains the true error in under 80% of replicates,
  while the event-held-out interval with a cluster-robust standard error contains it in at least 85%.

## Second half: the further question (to be committed in week 4, once chosen)

Option 1, desirability ranking: hypotheses B1 and B2 from proposals/05, copied here verbatim when chosen.
Option 2, forecast accuracy: hypotheses C1 and C2 from proposals/05, copied here verbatim when chosen.

## Decisions to record before analysis

- Wave 12 (participants limited to saying yes to half their dates): kept in the main analysis with a
  sensitivity check excluding it, unless the team decides otherwise here.
- Race and race-preference variables: excluded from the predictor sets unless the team decides otherwise here.
- Missing values: headline results use all 4,184 dyads for both feature sets, so `pre` and `pre+date` are
  compared on the same rows. Missing numeric features are filled with their training-fold median; a missing
  field, career or goal code becomes its own level. Four indicators mark whether each person has any missing
  pre-event feature (`f_pre_missing`, `m_pre_missing`, in both sets) or any missing in-date rating
  (`f_date_missing`, `m_date_missing`, `pre+date` only, since whether a rating was recorded is part of the date
  record). Medians and category levels are learned from the training folds only and applied unchanged to the
  held-out events; the indicators are computed row by row and count only answers left blank on the forms, not
  values set to missing under the out-of-range rule below.
  Reason: dropping incomplete dyads would leave 3,943 for `pre` and 2,766 for `pre+date`, and the dyads lost
  are not a random subset (the 1,177 complete on `pre` but missing an in-date rating match at 0.133, against
  0.177 for the 2,766 complete on both; `results/eda_complete_cases.csv`), so a complete-case P2 would compare
  different samples as well as different feature sets. The complete-case version (P1 and P2 recomputed on the
  2,766 dyads complete on both sets, with no imputation and no indicators) is reported side by side with the
  headline in every results table.
- Preference alignment (the pair feature in convention 3 and P3, `fit_to_f_prefs` and `fit_to_m_prefs`): the
  compatibility term only. Let p_a be one person's stated weight on attribute a, rescaled to sum to 1 over the
  five attributes that have self-ratings (attractive, sincere, intelligent, fun, ambitious); s_a the other
  person's self-rating on it; and s-bar the mean of those five self-ratings. Alignment = sum over a of
  (p_a - 0.2)(s_a - s-bar), computed in both directions. It is 0 when either person gave all-equal answers.
  Reason: the plain weighted mean of the partner's self-ratings equals s-bar plus this term, and s-bar
  explains 94% of its variation in one direction and 85% in the other (measured from the survey answers
  alone, without match outcomes; `results/eda_pref_fit.csv`). s-bar is already in the individual features through the self-ratings, so it
  is not pair information. Logistic regression, LDA and QDA give the same predictions under either definition;
  for the coefficient table, the lasso and the tree models, this one keeps individual information out of the
  pair features that P3 tests. Because waves 6 to 9 stated their preferences in a different format (see the
  survey-format entry below), the term's standard deviation there is about a third of that elsewhere (0.09
  against 0.28 for her preferences, 0.11 against 0.38 for his), so the term is standardized to mean 0 and SD 1
  within each format, as described in that entry.
- Out-of-range values (the codebook's scales all run 1 to 10): ratings of 0 are kept as recorded. These are 107
  scorecards from 50 people, plus hobby ratings of 0 from 6 people. They read as deliberate answers: the people
  who give them write N/A at the usual rate, and on cards with a 0 overall liking averages 0.0 to 4.2 depending
  on the attribute, against 6.1 on all cards. Gaming = 14 (5 people) and reading = 13 (3 people) are set to
  missing and then handled under the missing-values rule above: the same value recurs in the same question
  across different events, so they are a recording error whose true value cannot be recovered
  (`results/eda_zero_ratings.csv`, `results/eda_interest_out_of_range.csv`,
  `results/eda_out_of_range_summary.csv`). The released `int_corr` includes those
  values (127 dyads), so `int_corr` is recomputed for every dyad with the released definition (Pearson
  correlation over the 16 interests other than yoga), over the interests both people answered, unrounded. The
  rule is applied in `scripts/clean.py`, which reports how many values it changed; the raw file is not edited.
  Sensitivity check: the headline results for P1 to P5 recomputed with every value as recorded.
- Stated-preference survey format: waves 6 to 9 rated each attribute from 1 to 10, while the other waves split
  100 points (codebook, "Preference Scale"); the release has already converted waves 6 to 9 to shares of 100.
  The converted shares are flatter (a person's six shares have SD 3.4 against 9.2; attractiveness averages 16.2
  against 24.2), so the twelve stated-preference features (`f_pref_*`, `m_pref_*`) and the preference-
  alignment term are standardized to mean 0 and SD 1 within each format. This covers 102 people and 781
  dyads (`results/eda_pref_format.csv`). The means and SDs come from all participants' survey answers within each format: this uses no
  outcomes, and it still works when a fold holds out every waves 6 to 9 event at once. For P5, stated
  importance is the mean share in the 100-point waves only, as P5's wording says; waves 6 to 9 are reported
  separately. Sensitivity check: the headline results recomputed with unstandardized shares.
- Robustness grid: besides the sensitivity checks named above, every combination of each decision's headline
  choice and its named alternatives is run for every model and reported in one table, with the headline row
  marked: missing values (headline, no in-date indicators, complete cases) x preference alignment
  (compatibility term, plain weighted mean) x out-of-range values (this rule, as recorded) x preference format
  (standardized, unstandardized), 24 combinations. The headline choices are fixed by this file and are not
  changed in light of the grid. Reason: picking the best-scoring combination would overstate the results, and
  with 21 events chance differences between sensible choices are likely as large as the 0.02 and 0.03 AUC
  thresholds in P3 and P4.

- Model building (P1 to P4, one row per dyad). Models: logistic regression on the full feature set; stepwise
  logistic regression in three directions (forward from the intercept, backward from the full model, and both
  ways from the intercept), each stopping when no single addition or removal lowers BIC on the training folds;
  LDA; and QDA with its class covariances shrunk toward the identity (scikit-learn `reg_param`), the amount
  chosen from 0.1, 0.3, 0.5, 0.7 and 0.9 by event-grouped 4-fold cross-validation inside each training set.
  Field, career and goal codes enter and leave stepwise selection as blocks. Events are assigned to the 5 folds
  at random with seed 20261006. Every step below (filling, transformations, standardizing, splines, selection,
  tuning) is refitted inside each training fold, so the event-held-out score covers the whole procedure.
  Preparation: the shared-interests preference share is dropped as the reference, since the six shares sum to
  100; field and career codes held by fewer than 10 participants are merged into "other"; a continuous
  predictor whose skewness across participants exceeds 1 in absolute value is log-transformed (log1p after
  shifting its minimum to 0), judged from the predictor alone, never the outcome; numeric predictors are
  standardized. Shape: for the six continuous predictors where curvature is plausible (each person's age, the
  absolute age gap, `int_corr`, and the two alignment terms), a natural cubic spline with 3 degrees of freedom
  replaces the linear term in the logistic models when it lowers BIC on the training folds; rating-scale
  predictors stay linear, and LDA and QDA use linear terms. Thresholds: a dyad is predicted to match when its
  predicted probability exceeds the training folds' match rate. Error bars: error rate, sensitivity,
  specificity and balanced accuracy use the unbalanced cluster-robust estimator with events as clusters,
  applied to each dyad's contribution; AUC uses the clustered placement-value estimator (Obuchowski 1997) with
  events as clusters, checked against a simulation with known clustering before it is reported (convention 7).
  Diagnostics, reported but not used to change the model once outcomes are seen: variance inflation factors,
  calibration of held-out predictions by decile, binned residuals against each continuous predictor, and the
  most influential dyads. A problem they reveal is fixed only through a new dated entry in this file, with the
  original analysis still reported. Reporting: the coefficient table, with cluster-robust standard errors by
  event, comes from the full logistic model fitted to all dyads; stepwise models are reported by held-out
  performance and by how often each variable is selected across the five folds, without p-values, which
  selection makes too small. Stepwise selection runs on the headline specification and its complete-case
  version; the other models run on the whole robustness grid. Definitions for the hypotheses: the "best model"
  in P1 and P2 is the model with the highest headline event-held-out AUC among those listed here (and the random
  forest, once it exists); P3 compares the full logistic regression with and without the six pair features; in
  P4, "logistic regression" means the full logistic model.
- Clarification of the preparation rule above (2026-10-06, before any model has used match outcomes). The
  skewness rule applies only to numeric predictors with more than two distinct values; binary indicators
  (`same_field`, `same_goal`, the missing-data flags) are never transformed. A predictor with skewness above 1
  becomes log1p(x - min); one with skewness below -1 becomes -log1p(max - x), so the transformation reduces the
  skew in both directions and higher values still mean more; min and max come from the training folds, and
  held-out values beyond them are clipped to that range. Inside each training fold the steps run in this order:
  survey-format standardization (from all participants), the out-of-range rule, the skewness rule, median
  filling and flags, merging of rare codes, splines, standardization of numeric predictors. Reason: as first
  written, the rule would have increased the skew of the three left-skewed predictors (`f_self_sinc`,
  `f_movies`, `m_movies`) and transformed a binary indicator (`same_field`).

## Change log

- (date) (hypothesis) (what changed and why)
- 2026-10-06 (no hypothesis changed) Added the decisions on missing values, preference alignment, out-of-range
  values, survey format, the robustness grid and model building, and defined "best model" for P1 to P4, before
  any model has used match outcomes.
- 2026-10-06 (no hypothesis changed) Clarified the skewness rule (direction, binary indicators, order of
  preparation steps), found by checking predictor distributions only, before any model has used match outcomes.
