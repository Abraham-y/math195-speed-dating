# Rank Uncertainty in Bradley-Terry Arenas

Math 195 project proposal, draft of 2026-09-24.

## In plain terms

The setup. Chatbot Arena shows a user two anonymous model answers to the same prompt and asks which is better. Millions of these votes become a leaderboard by fitting a model in which every AI has a hidden strength, and the probability that A beats B depends on the difference in their strengths. This is the Bradley-Terry model. Fitting it is logistic regression with one indicator per model, so it sits directly on the course syllabus; chess Elo ratings are a cousin.

The uncertainty. Each strength estimate has an error bar, computed by resampling the votes many times (the bootstrap). Arenas then define a model's rank as one plus the number of models whose error bar sits entirely above it. That sounds careful, but it mixes two different questions. An error bar on one model answers "how sure are we about this model's strength?" A rank asks "how sure are we about this model relative to all the others at the same time?" When ten models sit within a few points of one another, the second question needs wider intervals than the first, and the arena's procedure does not widen them. So the published ranks may claim more certainty than the votes support.

What we do. Fit the model ourselves on the public votes and reproduce the arena's ranks. Then apply a method from economics, developed for ranking countries by test scores and neighborhoods by economic mobility, that produces rank ranges that are honest for all models at once. Check both procedures in a simulation where we know the true ranks, and count how often each one's range contains the truth. Finally, answer the practical question: if two models are 10 rating points apart, how many head-to-head votes are needed to be 90% sure of their order?

What comes out. A coverage table: how often the arena's procedure and the honest one contain the true rank. Rank ranges for the top ten, in the form "model X is somewhere between 2nd and 6th." A votes-needed table by rating gap. And a bonus the course's logistic-regression material covers directly: the style coefficients, meaning how many rating points a longer or more heavily formatted answer buys, and whether controlling for style changes who is on top.

Why it matters. Leaderboard positions drive press coverage, hiring narratives and, as the companion proposal shows, real-money betting markets. If the ranks are less certain than they look, that is worth knowing, and the votes-needed table tells an arena how much more data would settle a given dispute.

Glossary:

- Bradley-Terry model: each competitor has a strength; the chance A beats B is a logistic function of the strength difference.
- Difference coding: in the regression, model A gets +1, model B gets -1, everyone else 0; the fitted coefficients are the strengths.
- Bootstrap: resample the votes with replacement many times, refit, and read off how much the estimates move.
- Marginal versus simultaneous intervals: one model at a time versus all models at once. Simultaneous intervals must be wider to be honest, because many comparisons are made together.
- Coverage: the fraction of the time an interval or range contains the true value.
- Power: the probability of detecting a real difference of a given size; it drives the votes-needed calculation.
- Style control: adding length and formatting covariates to the regression so that strength is not confounded with verbosity.
- Comparison graph: which pairs of models actually get compared and how often; sparse or uneven comparisons make some strengths harder to pin down.

## Question and motivation

How many pairwise votes does an arena need before a model's rank is settled, and do the rank intervals arenas publish actually cover the true rank? Chatbot Arena and its successors rank models by fitting a Bradley-Terry model to human votes, which is logistic regression with one indicator per model (ISL chapter 4). They report bootstrap confidence intervals on each rating (chapter 5) and define a model's rank as one plus the number of models whose interval sits entirely above it. That procedure uses marginal intervals to answer a simultaneous question, and marginal intervals are not built for that.

The statistics literature has methods for confidence sets on ranks (Xie, Singh and Zhang 2009; Klein, Wright and Wieczorek 2020; Mogstad, Romano, Shaikh and Wilhelm 2024) developed for ranking hospitals, schools and countries. They have not been applied to LLM arenas. Applying them, checking coverage in simulation, and computing the vote counts needed to separate models at realistic rating gaps is a well-posed project with an audience that already exists: everyone who reads a leaderboard.

## What is known and what is unsettled

- Bradley-Terry via logistic regression: the vote outcome is Bernoulli with logit equal to the rating difference. Sparse pairings need a small ridge penalty (chapter 6) to keep ratings finite.
- Arena methodology (Chiang et al. 2024): bootstrap over votes for rating intervals; rank defined from pairwise interval separation; later additions control for style (length, formatting) by adding covariates to the same logistic regression.
- Confidence sets for ranks: Mogstad et al. (2024) build simultaneous sets from simultaneous confidence intervals on all pairwise differences, with bootstrap calibration of the multiplicity adjustment. Coverage is guaranteed uniformly; the price is width.

Unsettled, and answerable here: whether the arena's marginal-interval rank procedure undercovers when many models sit within a few rating points of each other, which is the situation at the top of every current arena; how wide honest simultaneous rank sets are for the top ten; how many votes on a given pair are needed to separate two models at gaps of 5, 10, 20 and 50 rating points; and whether style-control covariates widen or narrow the rank sets.

## Data

Primary: the public Chatbot Arena vote releases (conversation, model A, model B, winner, timestamp, language, and in newer releases the prompt category). The releases contain on the order of a hundred thousand votes over more than a hundred models; confirm the count and the latest release at download. Drop ties or treat them as half-wins, following the arena's convention, and state which.

| Item | Detail |
| --- | --- |
| Response | Vote outcome: A wins (binary) |
| Predictors, base model | Model indicators in difference coding (+1 for A, -1 for B, 0 otherwise) |
| Predictors, style control | Difference in response length, difference in counts of lists, headers and code blocks, prompt language, prompt category |
| Sample | All votes; sub-samples of 10k, 30k, 100k for the sample-size study |
| Preprocessing | Deduplicate conversations, drop models with under 300 votes, difference-code the model indicators, standardize style covariates |

Secondary: any other pairwise-preference release with the same structure (text-to-image arenas, coding arenas), which lets the sample-size result be checked on a second population.

## Methods

- Logistic regression (chapter 4): fit Bradley-Terry ratings with and without style covariates; interpret the style coefficients (how many rating points a longer answer buys).
- Ridge (chapter 6): a small L2 penalty for pairs with few votes; penalty chosen by cross-validation on held-out vote prediction accuracy.
- LDA and QDA (chapter 4): baselines for the same vote-prediction task, to satisfy the course comparison and to check whether the logistic model's linearity is costing accuracy.
- Bootstrap (chapter 5): rating intervals as the arena computes them; the arena's rank procedure reproduced exactly.
- Simultaneous rank sets: Mogstad et al. construction on top of the bootstrap distribution of all pairwise rating differences. Report the set for each of the top ten models.
- Trees and random forests (chapter 8): predict which votes are least informative (which prompt categories and languages produce near-coin-flip outcomes), which is what an arena would use to allocate future votes.

Metrics: coverage of the true rank in simulation (nominal 95%); width of rank sets for the top ten; held-out vote prediction accuracy across models; the vote count needed for 90% power to separate two models at a given gap.

## Experiments

Stage A, simulation. Fit Bradley-Terry to the real data, then treat the fitted ratings as truth. Simulate votes from those ratings using the real pairing frequencies, at sample sizes of 10k, 30k, 100k and 300k. For each replicate compute the arena's ranks and the simultaneous rank sets, and record whether each model's true rank is covered. 500 replicates per sample size. Repeat with a compressed rating spread (top ten within 15 points) to represent the current state of the leaderboard top.

Stage B, real data. Fit the models above on the full release; report ratings, arena-style ranks, simultaneous rank sets and style-control effects. Run the power calculation for adjacent pairs at the top of the board and state, per pair, how many additional votes would settle the order.

Claims fixed before any run:

- H1: the arena's marginal-interval rank procedure covers the true rank of a top-ten model in under 90% of replicates when the top ten span fewer than 15 rating points.
- H2: simultaneous rank sets reach nominal coverage; for the top ten on the real data, the median set width is at least three ranks.
- H3: separating two models at a 10-point gap with 90% power requires on the order of ten thousand votes on that pair; the exact number is the deliverable.
- H4: adding style-control covariates changes at least one top-ten rank by two or more positions.

## Timeline and deliverables

| Weeks | ISL chapters | Milestone |
| --- | --- | --- |
| 1 to 2 | 1 to 3 | Vote data downloaded and cleaned; Bradley-Terry fitted and matched to the published ratings; simulation harness built from the fitted ratings |
| 3 to 4 | 3 to 4 | Style-control logistic regression, LDA, QDA; interpretation of style coefficients; held-out vote prediction comparison |
| 5 to 6 | 5 to 6 | Bootstrap intervals and arena-style ranks reproduced; ridge penalty chosen by CV; simultaneous rank sets implemented and coverage measured in Stage A |
| 7 to 8 | 8 | Trees on vote informativeness; Stage B power calculations for the top pairs; write-up |

Deliverables: the course report; a short arXiv-style note with the coverage table, the rank sets for the top ten and the votes-to-separate table; a repository with a function that takes any vote dump and returns honest rank sets.

## Risks, scope control, and division of labour

| Risk | Mitigation |
| --- | --- |
| Simultaneous rank sets are computationally heavy for a hundred models | Restrict the sets to the top 20; the bootstrap of a logistic regression with a hundred parameters is cheap |
| Fitted ratings do not match the published ones | Expected to differ slightly by release and tie handling; document the difference rather than chase it |
| Rank sets come out very wide (uninformative) | That is a finding about the leaderboard, and the power table says how many votes would fix it |
| Non-uniform pairing makes the sample-size answer pair-specific | Report per-pair numbers for the top of the board and a general curve as a function of gap and votes-per-pair |
| Scope creep into active vote allocation | One sentence in the discussion; not implemented |

Division of labour: agents build the data pipeline, the simulation and the bootstrap loops; the team owns the rank-set construction (it is the one piece that must be derived, not just coded), the pre-registration and the interpretation. Verify the rank-set code on the simulation, where the true rank is known, before any real-data table.

## Connections and extensions

Economics. Bradley-Terry is a discrete-choice model: each vote is a Luce (1959) choice and the logistic form is McFadden's (1974) logit. Style control is a hedonic adjustment, the same idea as adjusting house prices for square footage. Two extensions follow. Social choice: Bradley-Terry ranks by latent strength, but Copeland and Borda rankings computed from the same votes can disagree, and intransitive cycles among models are measurable; report where they occur. Mechanism design: the arena is a vote-elicitation mechanism with no incentive for truthful voting and known bot-vote problems; the rank sets quantify how many adversarial votes it takes to move a rank.

Prediction markets. Polymarket runs monthly markets on which company will own the top-ranked model on the LMArena text leaderboard at month's end, resolved from the leaderboard with style control off, with volumes above a million dollars on some clusters. The bootstrap distribution of ratings gives the statistical probability that each company holds the top rank; comparing it with the market's implied probability is one figure here and a full project in the companion proposal, "Markets versus Models."

Pure mathematics. Ranking from pairwise comparisons has a theory literature with real open questions. Chen and Suh (2015) and Shah et al. (2016) give minimax rates for top-k identification under the Bradley-Terry model; the error depends on the spectral gap of the comparison graph's Laplacian, which the arena's non-uniform pairing makes small. Sharp constants under non-uniform comparison graphs remain open.

Applied mathematics. Spectral ranking (Negahban, Oh and Shah, Rank Centrality) is an alternative estimator to logistic regression; comparing the two on held-out vote prediction is a cheap extra experiment. Which pairs to sample next is an optimal-design problem (D-optimality on the comparison graph), and the active-ranking literature (Heckel et al. 2019) gives algorithms; a simulation of active versus random pairing shows how many votes an arena could save.

## References (from memory; confirm when assembling the reading list)

- Chiang et al. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." 2024.
- Mogstad, Romano, Shaikh, Wilhelm. "Inference for Ranks with Applications to Mobility across Neighbourhoods and Academic Achievement across Countries." Review of Economic Studies, 2024.
- Klein, Wright, Wieczorek. "A Joint Confidence Region for an Overall Ranking of Populations." JRSS-C, 2020.
- Xie, Singh, Zhang. "Confidence Intervals for Population Ranks in the Presence of Ties and Near Ties." JASA, 2009.
- Bradley, Terry. "Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons." Biometrika, 1952.
- James et al. An Introduction to Statistical Learning with Applications in Python. Chapters 4, 5, 6, 8.
