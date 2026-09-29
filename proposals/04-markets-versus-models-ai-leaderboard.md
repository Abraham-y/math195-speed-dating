# Markets versus Models: Pricing the Top of the AI Leaderboard

Math 195 project proposal, draft of 2026-09-24.

## In plain terms

The setup. On Polymarket, each month there is a set of contracts: "Will Google have the best AI model at the end of the month?", "Will OpenAI?", and so on, resolved by checking which company holds the top spot on the LMArena leaderboard at a fixed time. A contract trading at 65 cents on the dollar means the crowd puts roughly a 65% chance on that outcome. Prices move every day as news arrives.

The other forecaster. The leaderboard is public and computed from public votes, so we can build our own probability from the data: fit each model's strength, resample the votes to see how uncertain the strengths are, and count in what fraction of resamples each company's best model comes out on top. Add one more ingredient, the chance that a company ships a new top model before month's end, estimated from how often each company has released recently.

The question. Which forecaster is better, the market or the statistical model, and when? A month out, the market may be guessing about releases; a day out, it has seen everything. And does combining them beat both? Combining forecasts is a classic problem in economics, and the standard method is a logistic regression on the two forecasts' log-odds, which learns both how to weight them and how far to push the combined forecast away from 50% when the two agree.

What we do. Collect every monthly contract since the markets began (more than a year of months, five to ten contracts each), with price histories and outcomes. Reconstruct our forecaster's probability as of each date, using only votes available by then, which is the part that must be audited carefully. Score both forecasters with the Brier score at several horizons, draw calibration plots, fit the combination, and evaluate it by leaving one month out at a time so the combination is never tested on a month it was trained on. A decision tree then shows where market and model disagree and which was right.

What comes out. A horizon table (30, 14, 7, 1 days before the check) with market Brier score, model Brier score and combination Brier score. Calibration plots for each. A test of the favorite-longshot bias, the well-documented tendency for unlikely outcomes to be overpriced in betting markets. And, if the model beats the market at long horizons, evidence of a mispricing that a trader could in principle exploit, which is the finance reading of the result.

The main risk. About 100 to 150 resolved contracts is a small dataset, so the evaluation has to be careful and the report has to show intervals rather than bare point comparisons. Adding similar questions from Manifold and Metaculus helps.

Glossary:

- Implied probability: a contract's price read as a probability; 30 cents means about 30%.
- Brier score: mean squared error between forecast probabilities and outcomes (0 or 1); lower is better.
- Calibration: whether 70% forecasts come true about 70% of the time. Resolution: whether the forecasts actually separate the outcomes that happen from those that do not.
- Logit: log-odds, the scale on which probabilities are combined; logistic regression works on this scale.
- Extremizing: pushing a combined forecast further from 50% than its inputs, because the inputs share information.
- Leave-one-month-out cross-validation: train the combination on all months but one, test on the held-out month, rotate.
- Favorite-longshot bias: low-probability contracts trade above their true frequency; high-probability ones below.
- Look-ahead bias: using information that was not available at forecast time. The classic way backtests cheat, and the thing our timestamp audit guards against.

## Question and motivation

Since mid-2025 Polymarket has run a monthly market cluster, "Which company has the best AI model at the end of [month]?", resolved by the top position on the LMArena text leaderboard with style control off at a fixed check time. Some clusters have traded over a million dollars, and each has one contract per company. Meanwhile LMArena publishes the vote data from which the leaderboard is computed, so a statistical forecaster can be built entirely from public inputs: current ratings, their bootstrap uncertainty, the gap to the leader, and each company's recent release cadence.

Three questions. Is the market calibrated, and is the statistical model calibrated? Which is better at each horizon (30, 14, 7 and 1 days before the check)? Does a learned combination, logistic regression on the two logits, beat both under leave-one-month-out cross-validation? The prediction-market literature says markets are hard to beat but show a favorite-longshot bias; whether either holds in a market whose resolution is a computable statistic is unknown.

This is forecast combination, a core topic in econometrics and quantitative finance (Bates and Granger 1969; Satopää et al. 2014), applied to a market that resolves on a quantity we can model directly. The methods are logistic regression, LDA/QDA, regularization, trees and cross-validation, so the project fits the course exactly.

## What is known and what is unsettled

- Prediction markets aggregate information well and are hard to beat at short horizons (Wolfers and Zitzewitz 2004; Arrow et al. 2008), but show a favorite-longshot bias: low-probability contracts trade above their resolution frequency (Thaler and Ziemba 1988).
- Forecast combination: a logit-linear combination with learned weights, which extremizes when inputs share information, is the standard method (Satopää et al. 2014); simple averages are surprisingly strong (Bates and Granger 1969).
- Arena ratings carry quantifiable uncertainty (bootstrap intervals in Chiang et al. 2024), so the probability that a given company holds the top rank is computable from the votes.

Unsettled: whether a purely statistical forecaster built from votes and release cadence matches the market; at which horizon each wins; whether the combination beats both; and whether the favorite-longshot bias appears when the resolution source is a public statistic rather than an event.

## Data

| Item | Detail |
| --- | --- |
| Markets | Polymarket monthly "best AI model" clusters from mid-2025 to the present: one contract per company (5 to 10 per month), price history and final resolution via the public Polymarket API. Confirm the number of clusters and the granularity of price history at download |
| Leaderboard | LMArena vote releases and dated leaderboard snapshots (LMArena data releases, community trackers, Internet Archive), to reconstruct ratings as of each forecast date without look-ahead |
| Releases | Model release dates per company from public changelogs, for a release-cadence predictor |
| Second market source | Manifold and Metaculus questions on leaderboard position and benchmark scores, to enlarge the set of resolved questions |
| Response | Binary: the company's model holds the top position at the check time |
| Predictors | Market price at horizon h; statistical probability from the bootstrap Bradley-Terry fit as of the same date; rating gap to the leader; days to resolution; releases by the company in the prior 180 days; market volume and liquidity |
| Sample size | Roughly 15 monthly clusters times 6 to 10 contracts, so about 100 to 150 resolved contracts, each observed at several horizons. Small, so evaluation is leave-one-month-out with standard errors clustered by month |
| Preprocessing | Align market timestamps with leaderboard snapshots; strict cutoff so no vote after the forecast date enters the statistical probability; drop zero-volume contracts; encode each month's stated resolution rule |

Note on resolution rules: early clusters resolved on the highest arena score, later ones on the top leaderboard rank with score as tiebreaker. Encode the rule per month rather than assuming one.

## Methods

- Statistical forecaster (chapters 4 and 5): fit Bradley-Terry by logistic regression to votes up to date t and bootstrap the ratings. The probability that a company holds the top position at month's end is the share of bootstrap replicates in which it does, adjusted by a simple release hazard (the chance of a new top model from that company in the remaining days, estimated from historical release cadence).
- Calibration analysis: reliability diagrams and the Brier decomposition (Murphy 1973) for the market and for the model at each horizon.
- Learned combination (chapter 4): logistic regression of the outcome on logit(market price) and logit(model probability). Coefficients summing above 1 indicate extremizing. LDA and QDA as baselines for the same task.
- Regularization (chapter 6): ridge and lasso over the full predictor set, to test whether anything beyond the two forecasts carries information.
- Trees and forests (chapter 8): find the regimes where market and model disagree and which is right (large rating gaps, thin markets, months with major releases).
- Cross-validation (chapter 5): leave-one-month-out; report Brier score and log loss with month-clustered standard errors.
- Favorite-longshot test: bin contracts by price and compare resolution frequency within bins.

## Experiments

Stage A, simulation. Generate synthetic months from a fitted Bradley-Terry model with random releases. Check that the statistical forecaster is calibrated by construction and that the combination code recovers known weights.

Stage B, real data. Build the horizon-by-horizon table for market, model and combination, with calibration plots and the disagreement tree.

Claims fixed before any run:

- H1: at one day before the check, the market's Brier score beats the statistical model's.
- H2: at 30 days, the model's calibration is at least as good as the market's, and the market overprices challengers early in the month (favorite-longshot bias).
- H3: the logistic combination beats both components at the 7-day and 14-day horizons under leave-one-month-out CV.
- H4: the tree's first split is on rating gap to the leader; above a 20-point gap, the market's disagreements with the model are systematically wrong.

## Timeline and deliverables

| Weeks | ISL chapters | Milestone |
| --- | --- | --- |
| 1 to 2 | 1 to 3 | Market price histories and resolutions downloaded; leaderboard snapshots aligned; look-ahead audit; EDA on price paths |
| 3 to 4 | 3 to 4 | Bradley-Terry forecaster with bootstrap; calibration plots; logistic combination, LDA, QDA |
| 5 to 6 | 5 to 6 | Leave-one-month-out CV; ridge and lasso over predictors; Brier decomposition; favorite-longshot test |
| 7 to 8 | 8 | Trees and forests on disagreement regimes; Manifold and Metaculus extension; write-up |

Deliverables: the course report; a short note with the horizon table and calibration plots; a repository that rebuilds the forecaster from public data each month.

## Risks and scope control

| Risk | Mitigation |
| --- | --- |
| Few resolved contracts | Add Manifold and Metaculus questions; evaluate at several horizons; report intervals rather than point comparisons |
| Look-ahead bias in reconstructed ratings | Strict timestamp cutoffs; audit by recomputing a past month's leaderboard and comparing with the archived snapshot |
| Gaps in market price history | Polymarket API history first; daily closes from archive services as fallback |
| Release timing dominates outcomes | Model it explicitly with the release hazard; report results with and without it |
| Resolution rule changed across months | Encode each month's rule; keep a column for it and test whether it matters |

Division of labour: coding agents build the download, alignment and bootstrap pipeline and the figures; the team owns the forecaster's design, the pre-registration and the interpretation. The look-ahead audit is done by hand before any evaluation number is trusted.

## Connections and extensions

Economics. Market efficiency (Fama 1970) and prediction-market design (Wolfers and Zitzewitz 2004) frame the question; the favorite-longshot bias (Thaler and Ziemba 1988) is the specific anomaly to test. Forecast combination (Bates and Granger 1969) is the method. If the model beats the market at long horizons, that is a mispricing an informed trader could exploit, which is the finance reading of the result.

Pure and applied mathematics. Proper scoring rules and the Brier decomposition (Murphy 1973) supply the evaluation theory; logit pooling (Satopää et al. 2014) supplies the combination theory; a hazard model for releases is elementary survival analysis; the Bradley-Terry bootstrap connects to the companion proposal on rank uncertainty, whose simultaneous rank sets are the honest version of the top-rank probability.

## References (from memory; confirm when assembling the reading list)

- Wolfers, Zitzewitz. "Prediction Markets." Journal of Economic Perspectives, 2004.
- Arrow et al. "The Promise of Prediction Markets." Science, 2008.
- Thaler, Ziemba. "Anomalies: Parimutuel Betting Markets." Journal of Economic Perspectives, 1988.
- Bates, Granger. "The Combination of Forecasts." Operational Research Quarterly, 1969.
- Satopää et al. "Combining Multiple Probability Predictions Using a Simple Logit Model." International Journal of Forecasting, 2014.
- Murphy. "A New Vector Partition of the Probability Score." Journal of Applied Meteorology, 1973.
- Chiang et al. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." 2024.
- James et al. An Introduction to Statistical Learning with Applications in Python. Chapters 4, 5, 6, 8.
