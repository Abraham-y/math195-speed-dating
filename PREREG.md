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

## Change log

- (date) (hypothesis) (what changed and why)
