# The Winner's Curse on LLM Leaderboards

Math 195 project proposal, draft of 2026-09-24.

## In plain terms

The setup. A benchmark is a test with, say, 500 questions. Two hundred models take it and each gets a score. The leaderboard sorts the scores. You read off the top model, see 87.4%, and assume that is how good it is.

The catch. Every score is the model's true ability plus luck: on a different set of 500 questions, each model would score a little higher or lower. When you pick the top of the list, you are picking not only the most able model but also, in part, the one that happened to get lucky on this particular set of questions. So the top score overstates the top model's true ability. This is the winner's curse, named in auction theory: the winning bidder is the one who most overestimated the item's value. The effect is larger when the test is short (more luck in each score) and when many models are bunched near the top (more chances for a lucky one to edge ahead).

How big is it? Nobody has measured it on current LLM leaderboards. It could be 0.2 points (ignore it) or 3 points (the top-five order is mostly noise). That number is the main thing this project produces.

How we measure it. The leaderboard we use publishes results question by question. We split the questions randomly into two halves. Rank the models on the first half. Then look at the top model's score on the second half, which it was not selected on. The drop from first-half score to second-half score is the curse. Repeat with a thousand random splits and average. This is the same trick that finance researchers used to ask whether top mutual funds are skilled or lucky.

The fix. Shrinkage: pull every model's score part of the way toward the average score, with the amount depending on how noisy individual scores are relative to how spread out the models are. This idea dates to the 1950s and 60s (Stein, James, Efron and Morris) and it is the same mathematics as ridge regression on the course syllabus, where a penalty pulls coefficients toward zero. We test whether shrunk scores predict the second-half scores better than raw scores do, and whether shrinking changes who sits at the top.

What comes out. For each benchmark: the size of the curse in points; the regression-to-the-mean slope (if it is 0.9, a model that scored 10 points above average on one half scores about 9 above average on the other); and a corrected leaderboard. Plus a simulation that shows how the curse grows with the number of models and shrinks with the number of questions, so the result carries over to leaderboards we did not test. And a practical rule, from the logistic-regression part: given the gap between two models and the number of questions, how likely is it that the higher-ranked one is actually better?

Glossary:

- Benchmark and item: a test and its questions.
- Winner's curse: the top of a ranking of noisy scores is biased upward, because selecting the maximum favors lucky draws.
- Regression to the mean: extreme scores on one measurement are less extreme on a second one, because part of the extremity was luck.
- Shrinkage (empirical Bayes, James-Stein): pulling many estimates toward a common mean, with the amount estimated from the data.
- Ridge regression: shrinkage applied to regression coefficients; the penalty does the pulling.
- Split-half: divide the items at random into two halves, select on one, check on the other.
- Selective inference: the exact distribution of a score given that you only looked at it because it was the maximum. An alternative to shrinkage that we compare against.
- RMSE: root mean squared error, the standard measure of how close a set of predictions is to the truth.

## Question and motivation

By how many points does the reported score of the top-ranked model on an LLM leaderboard overstate its true accuracy, and does shrinkage fix it? A leaderboard rank is a maximum taken over many noisy estimates. The maximum of noisy estimates is biased upward, and the bias grows with the number of models compared and shrinks with the number of test items. This is the winner's curse (Smith and Winkler 2006), and the classical remedy is empirical-Bayes shrinkage of every score toward the grand mean (Efron and Morris 1975), which is exactly the idea behind ridge regression in ISL chapter 6.

The question is quantitative, not conceptual. Leaderboards today compare hundreds of models on benchmarks with a few hundred to a few thousand items, and the top few are often separated by less than a point. Nobody has published how large the curse is on these leaderboards, or whether a standard shrinkage estimator gives a better forecast of a model's score on fresh items than the raw score does. Both are answerable with public item-level data, and the answer is a number people will quote.

The same mathematics governs best-of-n sampling with a noisy verifier in post-training: choosing the highest-scored sample and trusting its score. A clean leaderboard result transfers directly to that setting, and the write-up should note the connection.

## What is known and what is unsettled

- Winner's curse and optimizer's curse: the selected maximum of noisy estimates is biased upward; Smith and Winkler (2006) give the decision-analysis form.
- Empirical-Bayes and James-Stein shrinkage (Efron and Morris 1975): shrinking many parallel estimates toward their mean dominates the raw estimates under squared error whenever three or more are compared.
- Selection bias after cross-validation model choice (Varma and Simon 2006; Tibshirani and Tibshirani 2009): the CV error of the chosen model is optimistically biased, and simple corrections exist.
- Error bars on evals (Miller 2024): per-benchmark standard errors for a single model, treating items as a sample, with clustered variants.

The gap: none of this has been applied to LLM leaderboards with item-level data. Open questions the project can settle: the size of the bias on current leaderboards; whether split-half validation confirms it; whether empirical-Bayes shrinkage, ridge regression of held-out score on observed score, or a selective-inference correction gives the best forecast of fresh-item accuracy; and whether shrinkage changes the top-3 order on any benchmark.

## Data

Primary: item-level results from a public leaderboard that publishes per-sample outputs. The Hugging Face Open LLM Leaderboard publishes per-sample results for each evaluated model on MMLU-Pro, GPQA, MATH, IFEval, BBH and MuSR, for hundreds of models. HELM publishes per-instance predictions as a second source. Confirm current availability and formats at download; leaderboard hosting changes often.

Design: for each benchmark, split items at random into a selection half and a validation half (stratified by subtask). Rank models on the selection half. The bias of the top-ranked model is its selection-half score minus its validation-half score, averaged over many random splits. The validation half stands in for fresh items.

| Item | Detail |
| --- | --- |
| Unit of analysis | A (model, benchmark) pair; several hundred models times six benchmarks |
| Response (regression) | Validation-half score of the model |
| Predictors | Selection-half score, number of items, selection-half variance, model family, parameter count, release date, number of models compared |
| Response (classification) | Whether model A truly beats model B on the validation half |
| Predictors (classification) | Observed gap, item count, both models' variances, family match |
| Preprocessing | Drop models with incomplete runs, drop contaminated or deprecated benchmarks flagged by the leaderboard, stratify splits by subtask |

Secondary: Chatbot Arena vote releases, where the same design applies to Bradley-Terry ratings by splitting votes. This links directly to the rank-uncertainty project and can share code.

## Methods

Course methods map onto the problem one to one.

- Linear regression (chapter 3): regress validation-half score on selection-half score. A slope below 1 is regression to the mean, and the fitted line is itself a shrinkage estimator. Interpret the slope per benchmark.
- Ridge regression and empirical Bayes (chapter 6): James-Stein shrinkage of all models' scores toward the benchmark mean, with the shrinkage factor estimated from the between-model and within-model variances. Compare to ridge with the penalty chosen by cross-validation.
- Resampling (chapter 5): the split-half procedure repeated 1,000 times per benchmark; bootstrap intervals on the bias estimate.
- Logistic regression, LDA, QDA (chapter 4): predict whether A truly beats B from the observed gap and item counts. This gives a practical rule for when a leaderboard difference is real.
- Trees and random forests (chapter 8): find where the curse is largest (few items, many close models, particular families) and present the tree as the interpretable summary.

Metrics: bias of the top model's raw score and of each corrected score; RMSE of raw versus shrunk scores as forecasts of validation-half accuracy across all models; rank agreement (Kendall tau) between selection-half ranks and validation-half ranks before and after shrinkage; the number of top-3 orderings that change.

## Experiments

Stage A, simulation. M models with true accuracies drawn from a Beta distribution matched to a real benchmark's spread, each observed on n items with binomial noise. Grid: M in {10, 50, 200, 500}, n in {200, 500, 2,000, 10,000}, 1,000 replications. Record the top model's bias, the RMSE of raw and shrunk estimates, and rank agreement. This gives the expected curse as a function of M and n and validates the shrinkage code where the truth is known.

Stage B, real data. Run the split-half design on each benchmark and on Arena ratings. Fit the regression, ridge, empirical-Bayes and classification models above. Report the bias table, the slope table and the corrected leaderboard.

Claims fixed before any run:

- H1: on every benchmark with fewer than 1,000 items, the top-ranked model's selection-half score exceeds its validation-half score by more than one standard error of a single model's score.
- H2: the regression slope of validation score on selection score is below 0.95 on every benchmark, and empirical-Bayes shrinkage reduces the top model's bias to within 0.5 points.
- H3: shrinkage changes the top-3 ordering on at least one benchmark.
- H4: the simulation's bias curve, evaluated at each benchmark's M and n, predicts the observed bias to within a factor of two.

## Timeline and deliverables

| Weeks | ISL chapters | Milestone |
| --- | --- | --- |
| 1 to 2 | 1 to 3 | Item-level data downloaded for at least four benchmarks; split-half pipeline built; Stage A simulation built and checked; first bias numbers |
| 3 to 4 | 3 to 4 | Regression slopes per benchmark; logistic, LDA, QDA for "is A truly better than B"; interpretation |
| 5 to 6 | 5 to 6 | Bootstrap intervals on bias; ridge and empirical-Bayes shrinkage; corrected leaderboards; Stage A complete |
| 7 to 8 | 8 | Trees and forests on where the curse is largest; Arena extension if time allows; write-up |

Deliverables: the course report; a short arXiv-style note with the headline bias table and corrected leaderboards; a repository with the split-half pipeline that anyone can point at a new leaderboard dump.

## Risks, scope control, and division of labour

| Risk | Mitigation |
| --- | --- |
| Item-level data unavailable or format changed | HELM as a second source; fallback is to evaluate 30 open models on one benchmark with a local runner, which is cheap |
| Items are not exchangeable (subtasks differ in difficulty) | Stratify the split by subtask; report per-subtask results |
| Benchmarks with very few items (GPQA is a few hundred) | Report them as the high-curse case rather than dropping them |
| Contaminated models inflate the top of the board | Flag known contamination; run the analysis with and without flagged models |
| Result is small (curse under 0.2 points) | Still a finding; report it with the M and n at which it would become material |

Division of labour: agents build the download and split pipeline, the simulation and the figures; the team owns the shrinkage derivation, the pre-registration and the interpretation. Verify the empirical-Bayes code on the simulation before any leaderboard number is produced.

## Connections and extensions

Economics and finance. The winner's curse was named in auction theory: Capen, Clapp and Campbell (1971) noticed that the winning bidder in oil-lease auctions was the one who had overestimated most, and Milgrom and Weber (1982) formalized it for common-value auctions. A leaderboard is the same object with scores in place of bids. Two further literatures supply ready-made tools: publication bias (Andrews and Kasy 2019), where selecting significant results inflates published effects, and empirical-Bayes ranking of schools, teachers and hospitals (Gu and Koenker 2023, "Invidious Comparisons"), which provides shrinkage methods designed for ranks rather than scores. In finance, Fama and French (2010) bootstrap mutual-fund alphas to ask whether top funds are skilled or lucky; that is the split-half design applied to fund returns. A second dataset of mutual-fund or forecaster leaderboards (Metaculus, Good Judgment Project) tests whether the curse is as large there as on model leaderboards.

Prediction markets. Metaculus and Manifold carry questions on future benchmark scores (state of the art on a benchmark by a date). Comparing the market-implied expected score with raw and shrunk leaderboard extrapolations tests whether forecasters already shrink. Forecaster leaderboards suffer the curse themselves, so shrinkage of superforecaster rankings is a natural application with a ready audience.

Pure mathematics. Extreme-value theory gives a closed-form prediction for the curse: the expected maximum of M independent Gaussian errors with standard deviation sigma grows like sigma times sqrt(2 log M). The simulation can check this against the finite-M bias. Stein's paradox (the James-Stein dominance theorem) is the theory behind shrinkage, and a short derivation belongs in the report.

Applied mathematics and statistics. Selective inference (Lee, Sun, Sun and Taylor 2016) gives the exact distribution of the top score conditional on its having been selected as the maximum, via the polyhedral lemma, and hence honest post-selection intervals for the leader. Taylor is a co-author of the course textbook and is at Stanford Statistics. Comparing selective-inference intervals with empirical-Bayes shrinkage on the same leaderboard is a clean methods comparison.

## References (from memory; confirm when assembling the reading list)

- Smith, Winkler. "The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis." Management Science, 2006.
- Efron, Morris. "Data Analysis Using Stein's Estimator and Its Generalizations." JASA, 1975.
- Varma, Simon. "Bias in Error Estimation When Using Cross-Validation for Model Selection." BMC Bioinformatics, 2006.
- Tibshirani, Tibshirani. "A Bias Correction for the Minimum Error Rate in Cross-Validation." Annals of Applied Statistics, 2009.
- Miller. "Adding Error Bars to Evals." Anthropic, 2024.
- James et al. An Introduction to Statistical Learning with Applications in Python. Chapters 3, 4, 5, 6, 8.
