# Speed Dating: A Dataset-First Project with Optional Research Layers

Math 195 project proposal, draft of 2026-09-24.

## In plain terms

The dataset. Between 2002 and 2004, Columbia researchers ran speed-dating events: each participant had a four-minute date with every person of the opposite sex in their wave (10 to 20 dates), then rated each partner on attractiveness, sincerity, intelligence, fun, ambition and shared interests, guessed how likely the partner was to say yes, and said yes or no. A match happened when both said yes. Before the event, participants also stated how much they cared about each attribute and predicted how many matches they would get. The public release has about 8,400 date records from roughly 550 participants in 21 waves, with demographics, interests and follow-up surveys.

The standard project. Predict whether a participant says yes, and whether a date becomes a match, from the ratings and attributes. Fit logistic regression, LDA and QDA, then trees and forests, compare them with cross-validation, and interpret the coefficients: which attributes matter, how that differs by gender, and whether what people say they want matches what they choose. That last comparison is the famous result from the original paper, so the team can check whether it replicates.

The optional layers. Once the standard models exist, four extra questions can be asked of them without new data. Are the cross-validation error bars honest, given that the same people appear in many rows? Can we rank participants by desirability with honest uncertainty, and how many dates does it take to pin a rank down? Are people calibrated when they predict whether a partner will say yes? And does the "most desirable person in the wave" regress to the mean on a second look? Each is a chapter-5-and-later topic and each takes about two weeks.

What comes out. A normal course report with a model comparison and an interpretation section, plus one or two short sections that answer a methods question the literature has not settled for this kind of data.

Glossary:

- Dyadic data: every observation involves two people, and every date appears twice, once from each side. Rows are not independent.
- Clustered cross-validation: holding out whole people (or whole waves) rather than random rows, so the test fold contains no one the model has seen.
- Rasch model: each rater has a pickiness and each ratee has a desirability; the chance of a yes is a logistic function of desirability minus pickiness. It is Bradley-Terry with rater effects, fitted by logistic regression with indicators.
- Coverage: how often a "90% interval" actually contains the truth.
- Calibration: whether events forecast at 70% happen about 70% of the time.
- Stated versus revealed preference: what people say they weight versus what their choices show they weight.
- Regression to the mean: an extreme first measurement is usually followed by a less extreme second one, because part of the extremity was luck.

## Part 1: the dataset analysis (weeks 1 to 4)

Questions. What predicts a yes, and what predicts a match? Do stated attribute weights match the weights implied by decisions? How do the answers differ by gender? How much predictive accuracy do trees add over a logistic model?

| Item | Detail |
| --- | --- |
| Source | Speed Dating Experiment (Fisman, Iyengar, Kamenica, Simonson), public release on Kaggle; confirm row and participant counts at download |
| Unit | One row per date from one participant's side |
| Responses | dec (this participant says yes, binary); match (both say yes, binary) |
| Predictors | Partner ratings on six attributes (1 to 10), overall like, own guess that the partner will say yes, order of the date within the evening, age difference, same field of study, shared interests score, stated attribute weights, self-ratings, dating frequency and goals |
| Preprocessing | Harmonize the stated-preference scales, which differ across waves per the codebook; handle missing ratings; add rater and ratee identifiers for the clustered folds; keep demographic variables out of the predictive models unless the team decides otherwise (see Risks) |
| Checked at download (2026-09-24) | 8,378 rows, 551 participants, 21 waves, 5 to 22 dates per person (median 16); 4,184 of 4,194 dates present from both sides; yes-rate 42%, match rate 16.5%. The shared-interests rating is missing on 13% of rows, so a complete-case model on the six ratings keeps about 7,000 rows. The per-date guess that the partner will say yes (`prob`) is present on 96% of rows; the pre-event match-count prediction (`match_es`) on 86% of participants; the pre-event count of expected interested partners (`expnum`) was asked in only a few waves and is missing for 78%, so Layer C should use `prob` and `match_es` |
| Methods | Logistic regression, LDA, QDA (chapters 3 and 4); lasso for variable selection (chapter 6); classification tree, random forest, boosting (chapter 8); K-fold and grouped cross-validation (chapter 5) |

Pre-registered claims for Part 1:

- P1: attractiveness has the largest coefficient for both genders, and the gap between stated weight and revealed weight is largest for attractiveness.
- P2: a shared-interests rating adds predictive value beyond the six attribute ratings.
- P3: random forests improve held-out AUC over logistic regression by less than 0.03, meaning the signal is mostly additive.

## Part 2: research layers (weeks 5 to 8, choose one or two)

Layer A, honest cross-validation error bars under clustering. The dataset is clustered by rater, by ratee and by wave, and every date appears twice. Random-split CV puts the same person in training and test folds, which leaks and also makes fold errors dependent, the reason CV standard errors are too narrow (Bates, Hastie and Tibshirani 2024). Compare three procedures on the Part 1 models: random-split CV, grouped CV holding out whole people, and nested CV. Then build a simulation matched to the data's structure (same wave sizes, same number of dates per person, a fitted logistic truth) and measure the coverage of each procedure's 90% interval. Claims: A1, random-split CV reports a lower error than grouped CV on every model; A2, the random-split interval covers the true error in under 80% of simulated replicates while grouped-plus-nested covers in at least 87%.

Theory track for Layer A (optional, for whoever likes proofs). The companion note `06-theory-track-cv-under-clustering.md` sets this up formally and proves the first results: grouped and random-split CV estimate different quantities (new-person error versus seen-person error); under grouped folds the variance of the CV estimate is the independent-rows formula plus one cluster term, so the naive standard error is too small by at least the design effect $1+(m-1)\rho$; a cluster-robust standard error recovers that term exactly and is short only by the between-fold covariance from shared training data; and a cluster-nested CV, with clusters held out at both levels and a cluster-robust correction, has an unbiased MSE estimate. A first simulation matched to those assumptions supports all of it (naive coverage 0.67, cluster-robust 0.84, floored cluster-nested 0.91 at nominal 0.90). The open part, and the part that would be new, is the cross-classified case where every row has a rater and a ratee: the estimators are stated and the proofs are left for the team. The script `sim_cv_cluster.py` is the harness to extend.

Layer B, desirability ranking with honest uncertainty. Fit the Rasch model (yes ~ desirability of ratee minus pickiness of rater) by logistic regression with indicators. Note that the comparison graph is complete within a wave and disconnected across waves, so desirability is only identified within a wave; report per-wave rankings. Bootstrap over dates for rating intervals, then compute simultaneous rank ranges (Mogstad et al. 2024) and a dates-needed table: how many dates would pin a participant's rank to within one position. Claims: B1, in a wave of 10, the median rank range spans at least four positions; B2, pickiness varies across raters by more than desirability varies across ratees, in logit units.

Layer C, calibration of self-forecasts. Each participant guessed, for every date, how likely the partner was to say yes, and before the event predicted how many matches they would get. Treat the per-date guess as a probability forecast and score it against the partner's actual decision: reliability diagram, Brier score with its calibration and resolution parts, and a logistic regression of the outcome on the forecast logit, where a slope below 1 means overconfidence. Compare by gender. Claims: C1, forecasts are informative (AUC above 0.6) but the calibration slope is below 0.8; C2, predicted match counts exceed realized counts on average.

Layer D, regression to the mean at the top. Split each participant's dates into odd and even positions. Identify the top-rated participant in each wave on odd dates and record their rating on even dates; the drop is the winner's curse. Test whether James-Stein shrinkage of per-participant mean ratings predicts even-date ratings better than raw means (chapter 6 ideas). Claims: D1, the wave leader on odd dates drops by at least 0.5 rating points on even dates; D2, shrunk means beat raw means in RMSE on the held-out half.

Recommended pairing: A plus B, or A plus C. Layer A reuses Part 1 entirely; B and C each add one new fitted object.

## Timeline

| Weeks | ISL chapters | Milestone |
| --- | --- | --- |
| 1 to 2 | 1 to 3 | Download, codebook read, scale harmonization, EDA on ratings and decisions by gender, first logistic fits |
| 3 to 4 | 3 to 4 | Full Part 1 model set fitted and interpreted; stated-versus-revealed table; lasso selection |
| 5 to 6 | 5 to 6 | Grouped and nested CV on Part 1 models; Layer A simulation and coverage table; Layer B or C fitted |
| 7 to 8 | 8 | Trees and forests added to the comparison; chosen layers finished; write-up |

Deliverables: course report; a repository with the cleaning script and the clustered-CV harness; optionally a short note on the chosen layer.

## Risks and scope control

| Risk | Mitigation |
| --- | --- |
| Scale differences across waves in the stated-preference questions | Follow the codebook's wave notes; harmonize or restrict to consistent waves and say which |
| Demographic variables (race, income proxy) are present and were the subject of the original authors' second paper | Decide at the start which to analyze; report descriptively with the original papers' framing, and do not build models whose purpose is to score demographic preference |
| Small comparisons per participant (10 to 20 dates) make rankings weak | That is Layer B's finding; report ranges honestly rather than point ranks |
| Leakage between the two sides of a date | Grouped folds by date pair as well as by person; document the folding scheme |
| Layer creep | One or two layers, chosen by week 5, each with claims written before it is run |

Division of labour: coding agents build the cleaning pipeline, the CV harness and the figures; the team owns the folding scheme, the pre-registered claims and the interpretation. Any agent-written cleaning step is checked against the codebook by a team member before the models are fitted.

## Connections and extensions

Which of the earlier extensions survive. The econometrics of misclassified labels and the budget-allocation question do not apply here, because the labels are exact. Prediction markets are replaced by Layer C, which is the same calibration analysis with people forecasting their own outcomes. The U-statistic view of CV error carries over directly and is the theory behind Layer A. Winner's curse and shrinkage survive as Layer D. Rank inference survives as Layer B, with the twist that the comparison graph is disconnected across waves.

Economics. Matching markets are the natural frame: Becker's marriage market, Gale-Shapley stable matching, and Hitsch, Hortacsu and Ariely (2010), who estimated mate preferences from online-dating behavior with the same logit structure as Part 1. Extension: compute the stable matching implied by the estimated preferences within each wave and compare it with the realized matches; measure how far the observed outcome is from stable. Stated versus revealed preference is a core econometrics topic and Part 1 tests it directly. Behavioral economics supplies Layer C's overconfidence framing and a strategic-behavior question: do people say yes more readily when they think the partner will say yes, holding how much they liked the partner fixed?

Psychometrics and measurement. Each participant is rated by 10 to 20 different raters, so the six attribute ratings are noisy measurements of a latent quality. Variance components (how much of the variation in attractiveness ratings is the ratee, the rater, or noise) and inter-rater reliability are classical tools, and halo effects (a high attractiveness rating pulling the other five ratings up) are testable with the same regression machinery. This is the human-rater version of the noisy-judge problem in the companion proposal on CV intervals.

Pure and applied mathematics. Identifiability of Bradley-Terry and Rasch models requires a connected comparison graph (Ford 1957); here the graph is complete within a wave and empty across waves, which makes the within-wave analysis exact and the cross-wave analysis impossible without extra assumptions. Dyadic dependence (the social relations model of Kenny) gives the correct variance structure for Layer A's simulation. Stable matching is a short, elegant algorithm the team can implement in an afternoon, and its properties (existence, proposer-optimality) are provable in a page.

## References (from memory; confirm when assembling the reading list)

- Fisman, Iyengar, Kamenica, Simonson. "Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment." Quarterly Journal of Economics, 2006.
- Hitsch, Hortacsu, Ariely. "Matching and Sorting in Online Dating." American Economic Review, 2010.
- Gale, Shapley. "College Admissions and the Stability of Marriage." American Mathematical Monthly, 1962.
- Bates, Hastie, Tibshirani. "Cross-Validation: What Does It Estimate and How Well Does It Do It?" JASA, 2024.
- Mogstad, Romano, Shaikh, Wilhelm. "Inference for Ranks." Review of Economic Studies, 2024.
- Efron, Morris. "Data Analysis Using Stein's Estimator and Its Generalizations." JASA, 1975.
- Ford. "Solution of a Ranking Problem from Binary Comparisons." American Mathematical Monthly, 1957.
- James et al. An Introduction to Statistical Learning with Applications in Python. Chapters 3 to 6 and 8.
