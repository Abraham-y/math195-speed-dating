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

## Additions after the Part 1 run (committed 2026-10-07, before any of them ran)

Proposed by a teammate after P1 to P5 had been tested (results/part1_models_hypotheses.csv as of commit
cea590b). Each is written before any model using it has run, and is reported whichever way it falls.

- P3b. Within the `pre` set, adding five more compatibility features to the full logistic regression that
  already has the six pair features raises event-held-out AUC by less than 0.02. The five: same career code
  (`same_career`), the absolute differences in dating frequency (`abs_date_diff`) and in going-out frequency
  (`abs_go_out_diff`), the mean absolute difference across the 17 interest ratings (`int_mad`), and the cosine
  similarity of the two 17-interest vectors (`int_cosine`), the last two over the interests both people answered.
  The signed age gap (man's age minus woman's) is not added: it is the difference of the two ages already in
  the model.
- P3c. Adding three declared nonlinear and interaction terms to the model of P3b raises event-held-out AUC by
  less than 0.02: the square of the signed age gap (so the best gap need not be zero), and `int_corr` times each
  person's stated weight on shared interests (shared hobbies should matter more to people who say they value
  them; the weight is the format-standardized share, used here although it is the reference share left out as
  a main effect).
- The ladder for P3, P3b and P3c is reported for the full logistic regression under the headline specification
  and its complete-case version: individual features; plus the six pair features; plus the five extended
  features; plus the three terms. The new columns follow the preparation rules of the model-building entry.
- Waves 6 to 9 check: P1 to P4 recomputed under the headline specification on the 17 waves that split 100
  points (3,403 dyads), every model except stepwise, with events assigned to 5 folds by seed 20261006; reported
  next to the headline as a check on the survey-format decision.
- Like model. Linear regression (ISL chapter 3) of each date's overall liking rating (`like`, 0 to 10 as
  recorded) on pre-date information, one row per date from one side, dates with `like` recorded. Predictors:
  the rater's and the partner's own pre-event answers (the `pre_individual` columns, labelled rater and partner
  instead of woman and man), the rater's gender, and the pair features of P3b. Same preparation, missing-data
  handling and event folds (seed 20261006) as the match models. Reported: held-out R-squared (1 minus the
  pooled squared error over the variance of `like`) and RMSE, with cluster-robust standard errors by event, for
  a ladder of rater only; plus partner; plus pair features.
- L1. The full like model's event-held-out R-squared is below 0.10.
- L2. The pair features add less than 0.01 to the like model's event-held-out R-squared over the rater's and
  partner's own answers.

## Second half: the error-bar question (committed week 2)

- A1. Random-row cross-validation reports a higher AUC than event-held-out cross-validation for every model,
  and the gap is larger for the tree models than for the logistic model.
- A2. In a simulation matched to the data's structure (event sizes, dates per person, a fitted logistic
  truth), the conventional 90% cross-validation interval contains the true error in under 80% of replicates,
  while the event-held-out interval with a cluster-robust standard error contains it in at least 85%.

## Second half: the further question (to be committed in week 4, once chosen)

Option 1, desirability ranking: hypotheses B1 and B2 from proposals/05, copied here verbatim when chosen.
Option 2, forecast accuracy: hypotheses C1 and C2 from proposals/05, copied here verbatim when chosen.

## Part 2 (committed 2026-10-08, after the Part 1 model table and before any of this ran)

The team chose both further questions (desirability ranking and forecast accuracy) and the extensions below.
As the placeholder above says, B1, B2, C1 and C2 are copied verbatim from proposals/05:

- B1. In a wave of 10, the median rank range spans at least four positions.
- B2. Pickiness varies across raters by more than desirability varies across ratees, in logit units.
- C1. Forecasts are informative (AUC above 0.6) but the calibration slope is below 0.8.
- C2. Predicted match counts exceed realized counts on average.

New hypotheses:

- T1. No tree model (a single pruned classification tree, bagging, boosting) beats the random forest's
  event-held-out AUC by 0.02 or more, on `pre` or on `pre+date`, under the headline specification.
- T2. On `pre`, the single classification tree has the lowest event-held-out AUC of the four tree models.
- D1. Predicting each side's yes separately and multiplying the two predicted probabilities raises the
  event-held-out AUC for a match on `pre` by less than 0.02 over the best pre-registered single-stage `pre`
  model on the same folds: the pre-date ceiling is in the information, not the model.
- D2. Pre-date information predicts each side's yes (hers and his) with event-held-out AUC of at least 0.60.
- E1. Neither ensemble (the mean of the base models' predicted probabilities; a stacked logistic regression)
  beats the best single pre-registered model's event-held-out AUC by 0.01 or more on `pre`.
- E2. The same on `pre+date`.
- W1. Knowing who rates and who is rated, from their other dates at the same event, predicts a held-out yes
  with AUC at least 0.10 higher than pre-date information predicts a yes for people at new events.
- S1. Holding the six attribute ratings and overall liking fixed, a rater's guess that the partner will say
  yes has a positive coefficient in the yes model, with a 90% interval above zero, for both genders.

Exploratory, no hypothesis: a nonlinearity check of the yes model (natural splines on the six ratings), variance
components and halo in the ratings, a dates-needed table for the rankings.

Settings, fixed now:

- Tree models (`scripts/part2_models.py`), on the Part 1 preparation without splines, each tuned by the
  inner event folds by pooled AUC: a classification tree (scikit-learn, Gini) with cost-complexity pruning,
  ccp_alpha in {0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02}, random_state 20261606 + fold;
  bagging as a random forest that considers every column at each split (500 trees), minimum leaf size in
  {1, 5, 20, 50}, random_state 20261706 + fold; boosting (HistGradientBoostingClassifier, learning rate 0.05,
  no early stopping), maximum depth in {1, 2, 3} by number of trees in {100, 300, 600}, random_state 20261806
  + fold. Run under the headline and complete-case specifications, `pre` and `pre+date`, event folds.
- A1 (the leakage comparison): every model (full and stepwise logistic, LDA, QDA, random forest, lasso, the
  three tree models above) is also run with random-row 5-fold CV under the headline specification: each dyad
  is its own unit, dealt into folds by seed 20261006, inner tuning folds also random rows (seed 20261006 + 1 +
  fold). A1 is supported if the random-row AUC exceeds the event AUC for every model in both feature sets, and
  each of the four tree models' gap exceeds the full logistic model's gap in both feature sets.
- A2 (`scripts/sim_cv_matched.py`, seed 20261906, 500 replicates): the real dyads, people and events; truth =
  a logistic model for the match with the four pre-date predictors chosen most often in Part 1 (age gap, the
  man's dating frequency, his self-rated fun, same field), plus a woman effect, a man effect and an event
  effect whose SDs and coefficients come from a crossed random-effects logistic regression fitted to the real
  matches (statsmodels BinomialBayesMixedGLM, variational Bayes). The procedure: logistic regression on the
  four predictors; error = misclassification at the training match rate. Conventional interval: random-row
  CV error plus or minus 1.645 naive SEs; event interval: event CV error plus or minus 1.645 cluster-robust
  SEs. Target: the full-data model's error on 300 new events (real events' covariates resampled with fresh
  effects). Also reported: coverage of the average error, and of the nested-CV intervals of sim_cv_cluster.py.
- D1 and D2: for each side, a full logistic regression and a random forest (Part 1 settings) on the same
  `pre` columns and preparation, fitted in each training fold to that side's yes; P(match) = P(her yes) x
  P(his yes). D1 uses the better of the two products (logistic x logistic, forest x forest) against the best
  pre-registered single-stage `pre` model on the same folds; D2 uses the better learner for each side.
  Headline specification; the complete-case version is reported.
- E1 and E2: base models = full logistic (with BIC splines), LDA, QDA, random forest, lasso and boosting.
  The mean ensemble averages their predicted probabilities. The stacked ensemble is a logistic regression of
  the match on the base models' logit probabilities (clipped to [0.0001, 0.9999]), fitted to inner event-CV
  predictions within the training fold (4 inner folds, seed 20261006 + 1 + fold, base settings as tuned for
  that outer fold) and applied to the outer test predictions. The comparison is the best of the P1 pool.
- W1 (`scripts/yes_extensions.py`): one row per date from one side; dates are dealt into 5 folds at random
  within each rater (seed 20262006), so every rater and ratee has training dates; a logistic regression with a
  rater indicator and a ratee indicator (ridge penalty 1 on the indicators) predicts the held-out yes. The
  comparison is the AUC of D2's pre-date yes predictions on new events, both sides pooled. W1 is supported if
  the difference is at least 0.10.
- S1: by gender, dates with all six attribute ratings, liking and the guess; ratings standardized within
  gender; logistic regression of the yes on the eight; the guess's coefficient with a 90% cluster-sandwich
  interval (events, G/(G-1)), checked by a parametric bootstrap on the real design (seed 20262406).
- Rankings (`scripts/ranking.py`): within each wave and rater gender, a Rasch model P(yes) = logistic(desirability
  of the ratee - pickiness of the rater), fitted by penalized likelihood (ridge 0.5 on all effects, so estimates
  stay finite when someone said yes to everyone or no one; desirability centred at 0); 1,000 bootstrap
  resamples of the block's dates (seed 20262106); simultaneous 90% rank ranges by the max-t method of Mogstad,
  Romano, Shaikh and Wilhelm (2024) over pairwise differences in desirability. B1 is tested on the nine
  wave-and-gender blocks with exactly 10 ratees: the median span (upper minus lower rank, plus 1) over their 90
  ratees is at least 4. The ranges' coverage is checked by simulating from each block's fitted model (seed
  20262206). B2: a crossed random-effects logistic regression (BinomialBayesMixedGLM, variational Bayes) for
  each rater gender, with wave fixed effects and rater and ratee random effects; B2 is supported if the
  rater SD exceeds the ratee SD for both rater genders.
- Forecasts (`scripts/forecasts.py`): dates with the guess (prob, on 96% of rows) and the partner's decision;
  forecast p = prob / 10 clipped to [0.025, 0.975]; AUC with the clustered SE; calibration slope = the
  coefficient of logit(p) in a logistic regression of the partner's decision, with a cluster-sandwich SE
  checked by a parametric bootstrap (seed 20262306); the Brier score with its reliability, resolution and
  uncertainty parts over the distinct forecast values; reported by gender. C1 uses the pooled AUC and slope.
  C2: for each participant with a predicted count (match_es, 479 people), predicted minus realized matches;
  supported if the mean is above 0, reported with a cluster-robust SE by event.
- Every metric is computed on event-held-out (or, where stated, within-event) predictions pooled over folds;
  every error bar is cluster-robust by event. Results go to `results/` under each script's name, earlier
  results stay in place, and each new table names its fold scheme and seeds.

- Error-bar study, the real-data part and the bias adjustment (2026-10-08, added before scripts/sim_cv_matched.py
  and scripts/errorbars_real.py ran; no new hypothesis):
  - Real data, descriptive (`scripts/errorbars_real.py`, seed 20262506): for the full logistic regression and LDA
    (the Part 1 models cheap enough for nested CV) under the headline specification, `pre` and `pre+date`: the
    event-CV error rate and AUC with naive, event cluster-robust and two-way (woman and man) cluster-robust SEs,
    the latter by Cameron, Gelbach and Miller (2011): the woman and man cluster variances minus the dyad one; the
    cluster-nested CV estimate of the error's root MSE (10 repetitions); and random-row CV with its naive SE and
    the nested interval of Bates, Hastie and Tibshirani (2024).
  - Bias adjustment (Bates, Hastie and Tibshirani 2024): the nested interval is centred at the nested estimate
    minus (1 + (K - 2)/K)(nested estimate - standard CV estimate), with half-width 1.645 sqrt((K - 1)/K) times the
    root of the nested MSE estimate, K = 5. It is reported for random-row nested CV and for event cluster-nested CV,
    in the A2 simulation and on the real data, next to the unadjusted intervals.

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
- Clarification of the complete-case version (2026-10-06, before any model has used match outcomes):
  "complete" means complete on both feature sets as built under each specification. Under the headline
  out-of-range rule that is 2,662 dyads (476 matches); with interest values as recorded it is the 2,766 quoted
  above, which was counted before the recording-error fix (`results/eda_complete_cases.csv` gives both).
- Correction to the count above (2026-10-08, after the models ran; no result changes): the complete-case runs
  use 2,679 dyads (476 matches), complete on every column the models use. The 2,662 quoted above also required
  the shared-interests preference share, which the models leave out as the reference share; the 17 extra dyads
  have no matches.
- Random forest (2026-10-06, before it has been run): scikit-learn `RandomForestClassifier` with 500 trees,
  the square root of the number of predictors tried at each split (ISL 8.2.2), and the minimum leaf size chosen
  from 1, 5, 20 and 50 by event-grouped 4-fold cross-validation (pooled AUC) inside each training fold, on the
  same prepared design as LDA and QDA; `random_state` 20261106 plus the fold number. It runs on the whole
  robustness grid, is a candidate for the "best model" in P1 and P2, and is the random forest of P4. Reason:
  the model-building entry named the random forest for P1, P2 and P4 without fixing its settings. The other
  models had already been run when this was written; the settings are the textbook defaults plus a small
  tuning grid, chosen without looking at any random-forest result.
- Lasso (2026-10-06, before it has been run): L1-penalized logistic regression (ISL 6.2.2) on the same prepared
  design as LDA and QDA (standardized; each dummy of a categorical code penalized on its own), fitted with
  scikit-learn's `liblinear` solver with `intercept_scaling` 100 so that the intercept is effectively
  unpenalized; `random_state` 20261206 plus the fold number. The penalty is chosen from 30 values spaced evenly
  on a log scale, from the smallest that sets every coefficient to zero down to one thousandth of it, by
  event-grouped 4-fold cross-validation (pooled AUC) inside each training fold, ties going to the stronger
  penalty. It runs on the whole robustness grid and is reported by its held-out metrics and by how often each
  predictor keeps a nonzero coefficient across the five folds. P1 to P4 had already been tested when this was
  written, so the lasso is not added to the "best model" of P1 and P2; any comparison that includes it is
  exploratory. The solver was chosen by speed and convergence on one training fold (`saga` took 20 times
  longer), without looking at any lasso result.

- Fixes after an independent check of the code (2026-10-08, after the models ran; each changed result is reported
  next to the earlier one, and no hypothesis changes):
  - Error bars for sensitivity, specificity and balanced accuracy treated the number of matches (or non-matches)
    as fixed, so each event's term absorbed that event's match rate and the SEs were overstated (up to 1.7 times).
    They now use the linearized ratio form: for sensitivity each matched dyad contributes (correct minus
    sensitivity) divided by the number of matches, likewise for specificity, and balanced accuracy averages the
    two; the cluster-robust formula of convention 6 is unchanged. Point estimates, the error rate and the AUC are
    unaffected.
  - Identical columns in a training design (a man missing both his field and his career code) are reduced to one
    before fitting. The coefficient table now comes from the converged IRLS fit with a cluster-robust sandwich
    (events, G/(G-1)); a dummy whose category has no matches is reported as not estimable rather than as a
    number. The variance inflation table reports an exactly collinear column as infinite.
  - Like model: `same_goal` is a binary feature, not a categorical code; the rare-code rule counts each
    participant once even when they appear as both rater and partner. Added: its complete-case version (dates
    complete on every predictor, no filling, no flags). It uses linear terms only; the spline step belongs to the
    logistic models.
  - Complete-case ladder (P3, P3b, P3c): all four steps use the dyads complete on every column any step uses,
    including the shared-interests shares in P3c's terms (2,662 dyads), so this version involves no filling.
  - Blank allocations: in the 100-point waves, a stated-preference block with some items left blank whose answered
    items already sum to 100 (within 1 point) gets 0 for the blank items, since no points were left to give: iids
    129, 130 and 528 in block 1_1, 130 in block 2_1 and 528 in block 4_1 (40 dyads). The evidence numbers quoted
    in the entries above were computed before this fix.
  - Clustered AUC error bars: the check now runs five seeds and a second design whose event effect (SD 0.4 on the
    logit) matches the real spread of match rates across events (SD 0.058, against 0.079 in the first design).
    The result is reported as a range and stored with every results table.
  - Wording: in the out-of-range entry, liking on cards with a 0 averages 0.0 to 3.7 across the six attributes
    (the 4.2 was the yes-guess, not an attribute); the complete-case count with interest values as recorded is
    2,784 dyads (490 matches) on the columns the models use.
  - Two pre-registered results that had not been produced are now run: the wave 12 sensitivity check (P1 to P4
    without wave 12, every model except stepwise) and P5's stated shares for waves 6 to 9, reported separately.

- Second set of fixes, after a second independent check of the code (2026-10-08, after the rerun at ecfc18e and
  before any rerun under these rules; each changed result is reported next to the earlier one, and no hypothesis
  changes):
  - Identical columns: the reduction to one column in the entry above is withdrawn. When columns are identical in
    a training fold, the data cannot divide their combined effect; keeping the first column put all of it on
    whichever column came first, and that decided the predictions for test dyads where the columns differ. All
    columns are kept, and the ridge of 1e-6 that IRLS has used since the first run to keep separated fits finite
    divides the effect equally among them, as in the original run. The coefficient table reports identical
    columns as one combined term; the variance inflation table reports them as infinite.
  - Separation spread over several columns: the coefficient table marks as not estimable every coefficient that
    moves by more than 0.01 when the ridge goes from 1e-4 to 1e-8 (one man with 19 dates and no matches separates
    the man's missing-goal and missing-field dummies together, which the one-column check missed), and reports
    the sum of those coefficients where the sum does not move.
  - Missing-data flags and the participant catalog count the preference items set to 0 by the blank-allocation
    rule as blank, since they were left blank on the form (the missing-values entry: flags count answers left
    blank on the forms).
  - Error bars checked by simulation before they are reported: the like model's R-squared SE now uses the
    linearized ratio form (the squared error and the variance of `like` both vary), checked with the RMSE's SE in
    a simulation of the like-model design; the coefficient table and the yes model use the cluster sandwich with
    G/(G-1) only (the yes model also carried statsmodels' (N-1)/(N-K) factor), checked by a parametric bootstrap
    on the real designs with an event effect of SD 0.4 on the logit. The AUC simulation's two designs get separate
    seeds, and its stored range adds sensitivity and specificity. Every results table carries the range from the
    check that applies to its error bars (`se_check`).
  - P5: the test is the signed gap (revealed minus stated, largest value), as coded since P5 first ran (44f8508);
    the reading by absolute size and the margins are exploratory. The waves 6 to 9 version is reported without a
    verdict, as the survey-format entry says "reported separately". Added: P5 without wave 12, since the wave 12
    entry promised a sensitivity check of the results without restricting it to P1 to P4.
  - Wave 12 and waves 6 to 9 checks: the primary version assigns the remaining events to 5 folds by seed 20261006,
    as the waves 6 to 9 entry says; each is also reported with the headline folds minus the excluded waves, and in
    its complete-case version, which the missing-values entry requires in every results table. The coefficient
    table also gets its complete-case version.
  - Reporting: hypothesis statistics print AUCs to 4 decimals; every results table names its fold scheme and
    seeds, and the folds table records the forest, lasso and inner-tuning seeds. The like model adds an
    intercept-only baseline row. Exploratory, not a test: the headline full logistic AUC under 10 further fold
    seeds (20261007 to 20261016) and the headline random forest under 5 further forest seeds (random_state offsets
    20261107 to 20261111), to show how the P1 and P4 margins compare with fold and seed noise
    (scripts/fold_seed_noise.py).
  - Earlier results: before this rerun every result table is copied to results/earlier_ecfc18e/; the first
    round's diagnostics that were overwritten (calibration, binned residuals, influential dyads, selection, folds,
    the one-seed simulation summary) are regenerated from commit 4948d9f under the headline specification; and
    scripts/compare_runs.py compares every table and writes a committed summary with the checksums of the files
    compared.
  - Changes in the previous round that its entry did not announce: random forest predictions run on one thread
    (summing trees across threads changed the last digits and could break a tuning tie); the coefficient table's
    ridge of 1e-6; the like model uses linear terms only, a deviation from "same preparation as the match models",
    whose spline step it never ran.
  - Corrections to the entry above: the overstated SEs were up to 2.1 times on the headline rows (3.1 across the
    grid), not 1.7; the first simulation design's spread of event match rates is 0.080, not 0.079; the
    complete-case ladder had run on 2,679 dyads with P3c's shared-interests terms filled, and 2,662 is what the
    corrected rule would have given before the blank-allocation fix, never run; 2,784 dyads (490 matches) as
    recorded became 2,790 (491) after that fix.

## Change log

- (date) (hypothesis) (what changed and why)
- 2026-10-06 (no hypothesis changed) Added the decisions on missing values, preference alignment, out-of-range
  values, survey format, the robustness grid and model building, and defined "best model" for P1 to P4, before
  any model has used match outcomes.
- 2026-10-06 (no hypothesis changed) Clarified the skewness rule (direction, binary indicators, order of
  preparation steps), found by checking predictor distributions only, before any model has used match outcomes.
- 2026-10-06 (no hypothesis changed) Stated the complete-case count under the out-of-range rule (2,662 dyads),
  which the recording-error fix lowered from 2,766.
- 2026-10-06 (no hypothesis changed) Fixed the random forest's settings before running it; logistic, stepwise,
  LDA and QDA had already run (results/part1_models_metrics.csv as of commit 44f8508).
- 2026-10-06 (no hypothesis changed) Fixed the lasso's settings before running it, and kept it out of the "best
  model" of P1 and P2, which had already been tested (commit 3e44b7b).
- 2026-10-07 (new hypotheses P3b, P3c, L1, L2) Added a teammate's extended compatibility features, three declared
  interaction terms, the waves 6 to 9 check and the like model, after P1 to P5 had been tested and before any of
  these ran.
- 2026-10-08 (no hypothesis changed) Corrected the complete-case count to the columns the models use (2,679).
- 2026-10-08 (no hypothesis changed) Fixes after an independent check of the code (error bars for sensitivity,
  specificity and balanced accuracy; the coefficient table; two like-model bugs; the complete-case ladder; blank
  100-point allocations; the AUC error-bar simulation; two wording errors); ran the wave 12 check and P5 for waves 6
  to 9.
- 2026-10-08 (no hypothesis changed) After the rerun: the blank-allocation fix raised the complete-case count on the
  columns the models use to 2,685 dyads (477 matches), the same on every step of the complete-case ladder (2,679 and
  2,662 before the fix), and 2,790 (491) with interest values as recorded (2,784 before). Earlier and new results
  are side by side in results/compare_runs_*.csv (scripts/compare_runs.py); no verdict changed.
- 2026-10-08 (no hypothesis changed) Second set of fixes after a second independent check (identical columns kept
  and split equally; separation across columns; flags count filled blanks; simulation checks for the like-model,
  coefficient and yes-model error bars; P5's signed gap stated, its waves 6 to 9 version without a verdict, and P5
  without wave 12 added; fold and complete-case versions of the wave checks; reporting; earlier results kept;
  unannounced changes logged; corrections), before the rerun.
- 2026-10-08 (new hypotheses B1, B2, C1, C2 copied from proposals/05; T1, T2, D1, D2, E1, E2, W1, S1 added) Part 2:
  both further questions, tree models, the leakage comparison and matched simulation for A1 and A2, the
  two-decision model, ensembles, within-event prediction and strategic yes-saying, committed before any ran.

- 2026-10-08 (no hypothesis changed) Error-bar study: the real-data comparison with the final models and the Bates,
  Hastie and Tibshirani bias adjustment, added before either ran.
