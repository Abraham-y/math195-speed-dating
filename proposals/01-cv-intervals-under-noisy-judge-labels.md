# Cross-Validation Intervals Under Noisy Evaluation

Math 195 project proposal, draft of 2026-09-24.

## In plain terms

The setup. Suppose you build a classifier and want to know how often it will be wrong on new data. You do not have new data, so you use cross-validation: split what you have into five parts, train on four, test on the fifth, rotate through all five, and average the five test errors. That average is your estimate of the true error rate, for example "about 18% of predictions will be wrong."

Why an error bar matters. An estimate without an error bar is not much use: is it 18% plus or minus 1, or plus or minus 8? The natural error bar treats the five fold errors like five independent measurements and computes a standard error from them. The problem is that the five folds share most of their training data, so they are not independent. A 2024 paper by Stanford statisticians showed that the resulting error bar is too narrow: a "90% interval" built this way can contain the truth only 70% of the time. Their fix, nested cross-validation, runs a second cross-validation inside each fold to measure how much the estimate actually wobbles, instead of assuming it.

What this project adds. Most LLM evaluations today do not have human labels for every item; they use another language model as a judge. Judges make mistakes, and they make different mistakes on different runs: ask the same judge twice at nonzero temperature and you can get different answers. So the error you compute in each fold is not your model's true error; it is the true error plus judge noise, and possibly plus a systematic judge bias (for instance, favoring longer answers). Nobody has checked whether the nested fix still works in that case. It was not designed for it.

What we do. Two stages. First, a simulation where we control everything: we invent data, invent a judge that flips labels with a known probability, run three kinds of error bar (naive, nested, and our adjusted version that adds a judge-noise term), and count how often each contains the truth. Second, real data: Chatbot Arena pairs of model responses with human votes, plus labels we generate ourselves from three LLM judges, each asked several times. The human votes are the truth; the judge labels are what a practitioner would actually have.

What comes out. A table: for each error-bar method and each level of judge noise, the fraction of intervals that contained the truth. If the "nested CV, noisy judge" row reads 78% when it should read 90%, that is the finding, and the adjusted row shows whether the fix works. The course's required comparisons come for free along the way: logistic regression, LDA, QDA and trees are the classifiers whose error we are estimating, and their coefficients tell us which judges favor longer answers or the first position.

Why anyone outside the course cares. Teams that train language models choose checkpoints, reward models and judges from LLM-judged evaluations every week. If the error bars on those evaluations are too narrow, they are making confident choices on noise. A clean answer here is directly usable.

Glossary:

- Cross-validation (CV): estimate test error by splitting data into K folds, training on K minus 1 and testing on the held-out one, K times.
- Confidence interval: the error bar. A 90% interval is a range that should contain the true value in 90% of repeated experiments.
- Coverage: the fraction of the time it actually does. Undercoverage means the bar is too narrow, so you are more confident than the data justifies.
- Nested CV: a cross-validation inside each fold of the outer cross-validation, so the estimate's own variability can be measured.
- LLM judge: a language model asked to label an item, here "which of two answers is better." Cheap and fast, but noisy and biased.
- Test-retest variance: how much a judge's answer changes if you ask again.
- Judge bias: the systematic gap between judge labels and human labels. Variance averages away with more draws; bias does not.
- Variance components: separating the total wobble into named sources (sampling noise, judge noise) so each can be estimated on its own.
- Pre-registration: writing down the claims and the analysis plan before running anything, so the result cannot be steered after the fact.

## Question and motivation

Do cross-validation confidence intervals still cover the true test error when the labels used for training and evaluation come from a noisy LLM judge? Bates, Hastie and Tibshirani (2024) showed that standard K-fold intervals undercover even with exact labels, and proposed nested CV as the fix. Their analysis assumes each point's loss is observed exactly, so a second noise source (judge test-retest variance and judge bias) is outside it.

This matters beyond the course. Post-training teams choose checkpoints, reward models and judges from LLM-judged evals every week, and the statistics of that choice is a live problem (Miller 2024; Madaan et al. 2024; Heineman et al. 2025). A clean answer for the two-noise-source case is a small, real contribution, and it fits the course's required toolkit: logistic regression, LDA/QDA, trees, resampling, regularization.

## What is known and what is unsettled

Three results frame the project, and one gap sits between them.

- Bengio and Grandvalet (JMLR 2004): no unbiased estimator of the variance of the K-fold CV error exists, for any fixed K.
- Bates, Hastie and Tibshirani (JASA 2024): CV estimates the average error over training sets, not the error of the model you fitted. Naive intervals undercover; nested CV reaches near-nominal coverage in their simulations.
- Chen and Yang (Stats 2021): the one-standard-error rule from ISL chapter 6 has no theoretical support and frequently selects the wrong model.

The gap: all three assume the loss on each point is observed exactly. When labels come from an LLM judge, each loss carries test-retest variance (temperature, prompt phrasing) and judge bias relative to human labels. Whether nested CV still covers, and whether a variance-components correction restores coverage, has not been studied. Eval sizes in practice are hundreds to low thousands, which is the regime where it matters most.

## Data

Primary dataset: the public Chatbot Arena human-preference release. Each row holds a prompt, two model responses, the two generator IDs, and a human vote (A, B, or tie). The public releases hold tens of thousands of pairs; confirm the exact count at download and drop ties. The human vote is the ground truth. The full labelled set, minus the judged subset, serves as the population on which a fitted model's true error is computed (roughly 20,000+ pairs).

The team generates the judge labels itself on a subset of about 5,000 pairs: three LLM judges from different labs, one greedy draw plus five draws at temperature 0.7 each, with A/B order swapped on half the draws. That is roughly 90,000 short calls, a small budget. It yields within-judge test-retest variance, between-judge disagreement, and judge bias against the human vote.

| Item | Detail |
| --- | --- |
| Response | Human preference (binary, A or B) |
| Observed label during CV | LLM judge preference (binary) |
| Predictors | Length of each response and their difference, formatting counts (lists, code blocks, headers), position, generator model IDs (one-hot), judge score from a small cheap model |
| Training sizes n | 200, 500, 1,000, 2,000, drawn from the judged subset |
| Preprocessing | Deduplicate prompts, drop ties, standardize numeric predictors, hold out the population pool before anything else |

Fallback if Arena access is awkward: JudgeBench or RewardBench, which ship ground-truth correctness labels the judges can be scored against.

## Methods

The classification task is fixed and simple on purpose: predict which response the judge prefers from the predictors above. The models are the course's set, so the comparison across them is a deliverable in its own right.

- Logistic regression, LDA, QDA (ISL chapters 3 and 4), with coefficient interpretation of length, position and self-preference effects.
- Ridge and lasso logistic regression (chapter 6), with the one-SE rule tested as a secondary question.
- Classification tree, random forest, gradient boosting (chapter 8).

Error estimation compares three interval types for the CV estimate of test error at K = 5 and K = 10:

1. Naive: mean of fold errors, plus or minus 1.645 times the standard error across folds.
2. Nested CV (Bates, Hastie and Tibshirani), which corrects the standard error for the fold-to-fold dependence.
3. Nested CV plus a judge-noise term. Each observed loss is the true loss plus judge noise with variance sigma_J^2, estimated per item from the repeated judge draws. The first proposal to test is

```
SE_adj^2 = SE_nest^2 + sigma_J_hat^2 / n
```

The simulation, where every quantity is known, decides whether this form is right or needs a training-label term as well. Judge bias (the systematic gap between judge and human labels) is reported separately, since no interval can absorb it; a small human-labelled calibration set is the honest fix and its cost is quantified.

Metrics: coverage of the 90% interval against the true human-label error and against the true judge-label error (the difference isolates bias from variance), interval width, and model-selection accuracy (how often the CV-chosen model is within 1 point of the best model on the population).

## Experiments

Two stages, simulation first, so the real-data result can be read against a case where the truth is known.

Stage A, simulation. Draw predictors from a fixed distribution, labels from a known logistic model, and observe each label through a judge that flips it with probability p (or returns a score with Gaussian noise). Grid: n in {200, 500, 1,000, 2,000}, p in {0, 0.05, 0.10, 0.20}, K in {5, 10}, 1,000 replications per cell. Record coverage, width and selection accuracy for the three interval types. Add a biased judge (flip probability that depends on length) in a second grid.

Stage B, real data. Repeat the pipeline on Arena pairs with the three judges' labels; the population pool gives the true error. Estimate sigma_J^2 from the repeated draws and judge bias from the judge-versus-human gap. Report the same tables as Stage A.

Claims fixed before any run, reported whichever way they fall:

- H1: naive intervals undercover, and coverage falls as judge noise rises.
- H2: nested CV reaches nominal coverage at p = 0 but not at p = 0.20.
- H3: the adjusted interval is within 3 points of nominal across the Stage A grid.
- H4 (secondary): the one-SE rule's selection accuracy drops below plain minimum-CV selection once p is at least 0.10.

A result of "nested CV is already fine" would still be a finding worth writing up; the pre-registration is what makes it credible.

## Timeline and deliverables

| Weeks | ISL chapters | Milestone |
| --- | --- | --- |
| 1 to 2 | 1 to 3 | Arena data downloaded, population pool held out, judge labels generated on the 5,000-pair subset, EDA on judge agreement and biases; simulation harness built and unit-tested against known truth |
| 3 to 4 | 3 to 4 | Logistic, LDA, QDA fitted and interpreted: length, position and self-preference coefficients; Stage A grid launched |
| 5 to 6 | 5 to 6 | K-fold, nested CV and adjusted intervals implemented; Stage A coverage tables complete; ridge/lasso and one-SE rule added |
| 7 to 8 | 8 | Trees, random forest, boosting added to the model set; Stage B on real data; final tables and figures; write-up |

Deliverables: the course report; a four-to-six-page arXiv-style note with the coverage results; a public repository containing the harness, the judge-label dataset, and the pre-registration file committed before week 3.

## Risks, scope control, and division of labour

The main risk is a subtly wrong variance decomposition that every downstream table inherits. The mitigation is ordering: the team derives the adjusted-interval formula by hand, the coding agent implements it, and the implementation is validated on the simulation (where coverage is computable exactly) before a single real-data number is produced. Two team members review the derivation independently.

| Risk | Mitigation |
| --- | --- |
| Nested CV is compute-heavy (outer folds times inner refits times replications) | Models are cheap (logistic, LDA); cap n at 2,000; run the grid in parallel; boosting only in Stage B |
| Judge API cost or rate limits | About 90,000 short calls on a 5,000-pair subset; drop to two judges or three draws if needed |
| Arena data licensing or format changes | Fallback to JudgeBench or RewardBench, which have ground-truth labels |
| Result is null (nested CV already covers) | Still reported; the pre-registration makes it a finding rather than a failure |
| Scope creep into judge-bias modelling | One dataset, one question, three interval types; the one-SE rule and bias analysis run only if Stage A is done by week 5 |

Division of labour: coding agents build the harness, run the grids and produce figures; the team owns the derivation, the pre-registration, interpretation and writing. Every agent-produced intermediate result gets a check before it is used, ideally a unit test against simulation truth.

## Connections and extensions

The project sits on three older literatures. Each gives a team member a lane and the write-up a second audience.

Econometrics. A judge that flips labels is a misclassified binary dependent variable, a problem econometrics solved in the 1990s. Hausman, Abrevaya and Scott-Morton (1998) give a modified logit whose two extra parameters are the flip rates (judge says B when the truth is A, and the reverse), identifiable from a small human-labelled calibration set. Extension: replace "bias is reported separately" with a bias-corrected fit, and measure how many human labels the correction needs before it beats an uncorrected model. A second angle is budget allocation: given the cost ratio of a human label to a judge label, how should a fixed budget be split between human labels, judged items and repeated judge draws to minimize interval width? This is Cochran's optimal allocation and it produces one chart, interval width against budget split.

Pure mathematics. The CV error is an incomplete U-statistic, and its Hoeffding decomposition is the reason no unbiased variance estimator exists. A theory section can derive the two-noise-source variance from that decomposition, which is also the cleanest way to check the adjusted-interval formula before trusting the simulation.

Applied mathematics. Natarajan et al. (2013) give an unbiased loss correction for class-conditional label noise: with known flip rates, a "backward" corrected loss is unbiased for the clean loss. Training with the corrected loss is a one-line change; whether it alters the coverage results is a natural extra experiment.

Prediction markets. The nearest instance is comparing two forecasters on n resolved questions: scoring is exact, but questions are correlated and n is small, so the same interval machinery applies to "is forecaster A better than B." ForecastBench or Metaculus tournament data can serve as a fallback Stage B if judge-label generation proves difficult.

## References (from memory; confirm venues and links when assembling the reading list)

- Bates, Hastie, Tibshirani. "Cross-Validation: What Does It Estimate and How Well Does It Do It?" JASA, 2024.
- Bengio, Grandvalet. "No Unbiased Estimator of the Variance of K-Fold Cross-Validation." JMLR, 2004.
- Chen, Yang. "The One Standard Error Rule for Model Selection: Does It Work?" Stats, 2021.
- Miller. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." Anthropic, 2024.
- Madaan et al. "Quantifying Variance in Evaluation Benchmarks." 2024.
- Heineman et al. "Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation." Ai2, 2025.
- James, Witten, Hastie, Tibshirani, Taylor. An Introduction to Statistical Learning with Applications in Python. Chapters 4, 5, 6, 8.
