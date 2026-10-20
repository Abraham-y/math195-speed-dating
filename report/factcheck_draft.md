# Fact-check of draft_excerpt.tex against the repository (2026-10-19)

Sources are relative to the repository root. CSV "line" = line in the file (header is line 1).
Reproduced with the current code (.venv/bin/python, nothing edited): complete-case counts from
`modeling.build_frame`; headline logistic AUCs `pre` 0.542471 / `pre_individual` 0.531949; random forest
`pre` 0.582521 (SE 0.019653) and `pre+date` 0.845877; like-model R-squared -0.034117 (SE 0.03044) and
-0.042536. All match the stored results exactly. Commits after the Part 1 rerun (fbaf2e7, e335742) only add
the race block and a new variance function, so they do not change any Part 1 number.

Overall: no number in the draft is wrong. The problems are wording that leaves out the specification, a few
implementation details stated more broadly than the code does them, and robustness results the draft leaves
out.

## 1. Claims

| # | Claim | Draft says | Repository says | Verdict |
|---|---|---|---|---|
| 1 | Dates of experiment | Columbia, 2002 to 2004 | Codebook wave table: 16 Oct 2002 (wave 1) to 7 Apr 2004 (wave 21) (data/speed_dating_key.doc) | correct |
| 2 | Recruitment | mainly graduate and professional schools | Not in repo. Codebook and clean.py:45 flag wave 5 as undergraduates | not found (literature claim) |
| 3 | Number of waves | 21 | data/clean_summary.md:3 | correct |
| 4 | Fisman et al. findings | men attractiveness; women intelligence, race; female selectivity rises with group size | Not in repo | not checked (literature; consistent with the paper's abstract) |
| 5 | Rows removed | 10 with missing partner id | clean_summary.md:4 (all wave 5; 0 rows without a decision; 0 duplicates, line 5); 8,368 rows remain (line 19) | correct |
| 6 | Identifiers | rater, ratee, event, pair | clean.py:98-103 (also `date_id`) | correct |
| 7 | Consistency checks | no cross-wave pair, no duplicate dates, match = both yes | clean_summary.md:5, 14, 15 (all 0); clean.py:153-160 | correct |
| 8 | Dyads, matches | 4,184; 690 (16.5%) | pairs_summary.md:4 (rate 0.1649) | correct |
| 9 | Participants | 551 | clean_summary.md:3; recomputed from data/pairs.csv: 551 (274 women, 277 men) | correct |
| 10 | f_dec, m_dec | checks only, never predictors | pairs.py:12-13, 226 | correct |
| 11 | Survey formats | waves 1-5 and 10-21: 100 points; 6-9: 1-10 | Codebook, attr1_1 item; clean.py:10-14, 40 | correct (imprecise: the release already stores waves 6-9 as shares of 100, clean_summary.md:7) |
| 12 | Shares and `pref_scale` | converted to shares of 100; format kept in `pref_scale` | clean.py:106, 114-118 | correct |
| 13 | Blank allocations | 3 participants, 40 date-level observations; answered items about 100 | clean_summary.md:6: block 1_1, iids 129, 130, 528, 40 rows; "about" = within 1 point (clean.py:74-75). The rule also ran on blocks 2_1 and 4_1 (same people), which are not model inputs | correct |
| 14 | Recording errors | 5 gaming = 14, 3 reading = 13 | clean_summary.md:13 (78 and 51 rows); raw file rechecked; 127 dyads involved (pairs_summary.md:32) | correct |
| 15 | Their treatment | missing in primary spec; recorded values kept for robustness | clean.py:48-57; modeling.py:41, 92-93 (`out_of_range=recorded`) | correct |
| 16 | Zeros | not modified | PREREG.md:585-588 (107 scorecards from 50 people; interest 0 from 6 people) | correct |
| 17 | Pre-date variable list, 17 activities | as listed | pairs.py:58-66 (35 per person; `pre` = 76 columns, pairs_summary.md:33) | correct |
| 18 | date, go_out scales | seven-point, lower = more often | Codebook: 1 = several times a week ... 7 = almost never | correct |
| 19 | Interests, self-ratings | 1-10 | Codebook item 12 and attr3_1 | correct |
| 20 | Shared-interests share | omitted from all feature sets | Dropped as a main effect (modeling.py:113), but it enters `pre_interactions` through int_corr x shar share (modeling.py:104-105; PREREG.md:41-45) | imprecise |
| 21 | Numeric missing values | training-fold median | modeling.py:167, 211-215 (median on the skew-transformed scale) | correct |
| 22 | Binary missing values | training-fold median | modeling.py:168, 227 | correct |
| 23 | Missing-data indicators | "whether either participant" left a pre-date answer blank | One flag per person (`f_pre_missing`, `m_pre_missing`; pairs.py:186-187). `pre+date` also gets `f_date_missing`, `m_date_missing` (modeling.py:118-120). Flags count blanks on the form, including the zero-filled ones, not the recording-error values | imprecise |
| 24 | Categorical missing, rare levels | "missing" level; field and career codes held by fewer than 10 participants become "other" | modeling.py:22, 170-182, 207-209. Each participant is counted once, women and men pooled. Goal is not merged | correct |
| 25 | Complete-case count | 2,685 dyads, 477 matches | pairs_summary.md:34; part1_models_metrics.csv lines 19-35 (`dyads`, `matches`); recomputed: 2,685/477 for `pre`, `pre+date`, `pre_individual` and every ladder step (part1_additions_metrics.csv lines 6-9). This is the set complete on every `pre+date` column (in-date ratings included) even for the `pre` models (modeling.py:114-116). Complete on `pre` alone would be 3,847 (632) (pairs_summary.md:33). Other figures: 2,662 (476) = the 2026-10-06 rule that also required the shared-interests share, before the blank-allocation fix, never run (PREREG.md:656-659, 761-765). 2,679 (476) = what the runs used before that fix (PREREG.md:660-663). 2,790 (491) = complete cases with interest values as recorded (PREREG.md:765, 789-791; metrics rows with `out_of_range=recorded`, `missing=complete`). 2,766 and 3,943 are counts from before the recording-error fix (PREREG.md:565-569) | correct number; imprecise description |
| 26 | int_corr | Pearson, 16 interests without yoga, at least 3 shared answers | pairs.py:60, 87-99, 173 (unrounded, recomputed after the recording-error fix) | correct |
| 27 | int_mad, int_cosine | over all 17 | pairs.py:102-120, 180-181 (interests both answered, at least 3) | correct |
| 28 | same_field, same_goal, same_career | missing if either missing | pairs.py:140-145 | correct |
| 29 | abs_date_diff, abs_go_out_diff, abs_age_diff, age_diff_sq | as listed | pairs.py:172, 178-179; modeling.py:103 | correct |
| 30 | fit_to_f_prefs | sum_a (w_a - 1/5)(s_a - sbar) over five | pairs.py:131-137, 182. w_a is her share rescaled to sum to 1 over the five self-rated attributes; s_a and sbar are his self-ratings and their mean (PREREG.md:571-575) | imprecise (w_a undefined) |
| 31 | fit_raw | listed among constructed predictors | pairs.py:184-185. Used only in robustness option A (modeling.py:94-95), in no headline feature set | imprecise |
| 32 | Interaction terms | int_x_f_pref_shar, int_x_m_pref_shar | modeling.py:104-105: int_corr x the format-standardized shared-interests share | correct |
| 33 | Feature sets | pre_individual; pre (+6); pre_extended (+5); pre_interactions (+3) | pairs.py:206-211; modeling.py:109-112 | correct |
| 34 | Survey-format standardization | shares and two fit variables standardized within format | modeling.py:66-80, 98-102; PREREG.md:597-605. Means and SDs come from all participants (one row per person, women and men pooled) and all dyads, computed once before cross-validation and using no outcomes | correct (not within fold; say so) |
| 35 | Outer folds | 21 events randomly ordered with a fixed seed, dealt into 5 folds | modeling.py:19, 28-32 (seed 20261006; 5/4/4/4/4 events) | correct |
| 36 | Pooling | held-out predictions pooled for performance | part1_models.py:170-178 (each dyad is thresholded at its own fold's rate) | correct |
| 37 | Nested CV | 4 inner event-grouped folds; highest pooled inner AUC | part1_models.py:74-86 (inner seed 20261006 + 1 + fold; preparation refitted on each inner part; ties go to the earlier grid value) | correct |
| 38 | Skewness rule and clipping | >1: log(1 + x - min); <-1: -log(1 + max - x); held-out values clipped | modeling.py:154-165, 199-204; PREREG.md:646-655. Skewness of person-level columns is computed over one value per participant. Clipping applies only to the log-transformed predictors | imprecise |
| 39 | Order: median fill, then dummies | as stated | modeling.py:166-183. Reference level = the most common training level | correct |
| 40 | Standardization | training mean and SD | modeling.py:194-195, 243-244: numeric and spline columns only. Dummies and binary indicators stay 0/1 (results/methods_doc.tex:445-446) | imprecise |
| 41 | Logistic | ML with ridge 1e-6, unpenalized intercept | modeling.py:263-280 (IRLS; pen[0] = 0) | correct |
| 42 | Splines | candidates f_age, m_age, abs_age_diff, int_corr, 2 fits; spline if training BIC lower | part1_models.py:59-71; pairs.py:218; modeling.py:184-189. Knots at the min, 1/3 and 2/3 quantiles and max (3 df). Each candidate is tried alone against the all-linear model. "Logistic" and "spline-augmented logistic" are one model in the code, and no spline was chosen in any headline fold (part1_models_folds.csv, `splines` empty), so the reported full logistic is linear | correct; imprecise as two models |
| 43 | Stepwise | BIC, blocks, forward/backward/both | modeling.py:247-258, 291-315 (forward and both start from the intercept, backward from all blocks); run only for the headline and complete-case specs (part1_models.py:436) | correct |
| 44 | LDA | priors from training proportions | part1_models.py:144 (sklearn defaults) | correct |
| 45 | QDA | (1 - r) Sigma_k + r I, r in {0.1, ..., 0.9}, highest pooled inner AUC | part1_models.py:41, 97-98, 145-147; sklearn 1.9.1 `_solve_svd`: scaling = (1 - r) S^2/n + r. r = 0.9 was chosen in all 10 headline folds (part1_models_folds.csv) | correct |
| 46 | Random forest | 500 trees, sqrt(p), Gini, min leaf {1, 5, 20, 50} | part1_models.py:42-43, 113-115; PREREG.md:664-671 (Gini is the sklearn default; random_state 20261106 + fold). Min leaf 50 was chosen in all 5 `pre` folds | correct |
| 47 | L1 grid | 30 log-spaced C from C_min = 1/max_j abs(sum_i D_ij (Y_i - Ybar)) to 1000 C_min | part1_models.py:44, 101-110, 152-158 (D = prepared outer-training design; liblinear, intercept_scaling 100; ties go to the stronger penalty) | correct |
| 48 | Lasso added afterwards | added after the pre-registered comparisons | PREREG.md:672-683; part1_models.py:15-16 | correct |
| 49 | Threshold | training-fold match rate, about 0.165 | part1_models.py:123; part1_models_folds.csv `threshold` 0.158 to 0.171 | correct |
| 50 | Uncertainty | cluster-robust at event level | modeling.py:320-326, 401-422; AUC SE = Obuchowski (1997) clustered placement values (348-363) | correct |
| 51 | Best `pre` model | random forest 0.5825 (SE 0.0197) | part1_models_metrics.csv:8 (0.582521, 0.019653); part1_models_hypotheses.csv:2 | correct (headline spec only; label it) |
| 52 | Best `pre+date` model and gain | 0.8459; +0.2634 | metrics.csv:16 (rf 0.845877); hypotheses.csv:3. Part 2's stacked ensemble reached 0.8464 (part2_models_metrics.csv, `ensemble_stacked`, `pre+date`) | correct within the pre-registered pool |
| 53 | Liking R-squared | -0.0341 (SE 0.0304) | like_models_metrics.csv:5; like_models_hypotheses.csv:2. Computed on 8,128 dates. It is below the training-mean baseline (-0.0028, line 2) | correct; imprecise (negative R-squared not explained) |
| 54 | Pair features in liking model | +0.0084 | like_models_hypotheses.csv:3 (-0.0341 minus -0.0425); the eleven pair features | correct |
| 55 | Ladder | 0.5319; 0.5425 (+0.0105); 0.5454 (+0.0030); 0.5465 (+0.0010) | part1_additions_metrics.csv:2-5; part1_additions_hypotheses.csv:2-3; part1_models_hypotheses.csv:4 (differences 0.010522, 0.002961, 0.001027) | correct (headline) |
| 56 | P4 | RF - logistic 0.0401 > 0.03, not supported | part1_models_hypotheses.csv:5 (0.040050); PREREG.md:22. Complete-case: 0.0215, supported (hypotheses.csv:9) | correct for headline |
| 57 | Largest std. coefficient | attractiveness, both genders | part1_models_stated_vs_revealed.csv:2-13: women attr 0.817 (next shar 0.609); men attr 1.264 (next shar 0.566) | correct |
| 58 | Revealed-minus-stated gaps | 18.4 (women), 25.3 (men), largest | Same file, `gap_revealed_minus_stated` 18.36, 25.25; hypotheses.csv:10-11. Stated = mean share in the 100-point waves only (219 women, 224 men); yes model on 3,457 and 3,574 dates with all six ratings | correct; imprecise on definitions |

## 2. Suggested replacement sentences

- (#25) "As a sensitivity analysis, every model was also fitted to complete cases, with no imputation and no
  missing-data indicators. The complete-case sample is the set of dyads with every \texttt{pre+date} predictor
  recorded, so both feature sets and every step of the pair-feature ladder use the same 2,685 dyads (477
  matches); requiring only the pre-date predictors would have kept 3,847 dyads (632 matches)."
- (#20) "We left the shared-interests preference share out of every feature set as a main effect; it enters only
  through the two interest-by-preference interaction terms in the largest feature set."
- (#23) "We also included one indicator per participant (\texttt{f\_pre\_missing}, \texttt{m\_pre\_missing})
  for whether that person left any pre-date survey item blank; models with the in-date ratings add two more
  indicators for a missing in-date rating."
- (#30) "$\texttt{fit\_to\_f\_prefs} = \sum_a (w_a - 1/5)(s_a - \bar s)$, where $w_a$ is the woman's stated
  share for attribute $a$ rescaled to sum to 1 over the five self-rated attributes, $s_a$ is the man's
  self-rating and $\bar s$ his mean self-rating; \texttt{fit\_to\_m\_prefs} swaps the roles."
- (#31) "The plain weighted mean (\texttt{fit\_raw}) is used only in a robustness specification."
- (#34) "The twelve stated-preference shares and the two preference-fit variables were standardized to mean 0
  and SD 1 within questionnaire format. The means and SDs came from all participants' survey answers (no
  outcomes) and were computed once, before cross-validation."
- (#38, #40) "Within each training fold, a numeric predictor with more than two distinct values was
  transformed to $\log(1 + x - \min x)$ if its skewness exceeded 1 and to $-\log(1 + \max x - x)$ if it was
  below $-1$. Skewness for person-level variables was computed over one value per participant. The minimum and
  maximum came from the training fold, and held-out values of transformed predictors were clipped to that
  range. Missing values were then filled with training-fold medians, rare field and career codes were merged,
  and categorical variables were one-hot coded with the most common training level as reference. Finally,
  numeric columns (including spline columns) were standardized with the training mean and SD; dummies and
  binary indicators were left as 0/1."
- (#42) "In the full logistic model, a natural cubic spline (3 degrees of freedom, knots at the minimum, the
  1/3 and 2/3 quantiles and the maximum) replaced the linear term of \texttt{f\_age}, \texttt{m\_age},
  \texttt{abs\_age\_diff}, \texttt{int\_corr} or either fit measure if, tried one at a time, it lowered the
  full model's training-fold BIC. No spline was selected in any fold, so the reported full logistic model is
  linear."
- (#45, #46) Add: "The selected values were $r = 0.9$ in every fold and, for the pre-date forest, a minimum
  leaf size of 50 in every fold, the largest value in each grid."
- (#51) "Under the headline specification (event 5-fold cross-validation, 4,184 dyads), the random forest had
  the highest held-out AUC using the pre-date variables: 0.5825 (SE 0.0197)."
- (#52) "Among the pre-registered models, the best model using both the pre-date variables and the in-date
  ratings was also the random forest: AUC 0.8459, 0.2634 above the best pre-date model."
- (#53, #54) "The linear model of each date's liking rating (8,128 dates) had a held-out $R^2$ of $-0.0341$
  (SE 0.0304), worse than predicting the training-fold mean ($-0.0028$). Adding the eleven pair-level features
  to the rater's and partner's own answers changed held-out $R^2$ by only $+0.0084$ (from $-0.0425$)."
- (#56) "Under the headline specification, the random forest's AUC exceeded the full logistic model's by 0.0401
  (0.5825 against 0.5425). This is above the pre-registered 0.03, so P4 was not supported. In the
  complete-case version the gap was 0.0215, and P4 was supported."
- (#57, #58) "In the logistic model of each participant's yes decision on the six in-date ratings (dates with all
  six ratings), attractiveness had the largest standardized coefficient for both genders. It also had the
  largest gap between revealed importance (its share of the positive standardized coefficients) and stated
  importance (its mean share of the 100-point allocation, waves 1--5 and 10--21): 18.4 percentage points for
  women and 25.3 for men."
- (#11, optional) "...in Waves 6--9 participants instead rated each characteristic on a 1--10 scale; the public
  release already stores these answers as shares of 100."

## 3. Omitted caveats that change the reading

1. **P1 passes narrowly and is fragile.** The best `pre` AUC of 0.5825 clears the 0.58 bar by 0.0025.
   - Forest seeds alone move the forest's AUC from 0.5732 to 0.5870 (mean 0.5816; one of the 5 seeds is below
     0.58). Ten other fold seeds move the logistic AUC with SD 0.0127 (0.5226 to 0.5700)
     (results/fold_seed_noise_summary.csv:2, 5).
   - P1 fails in the complete-case version (0.5796, part1_models_hypotheses.csv:6) and in all six
     wave-exclusion versions: three without waves 6-9 and three without wave 12 (part1_additions_hypotheses.csv:6-29).
   - Across the 24-specification robustness grid, P1 holds in 14 and fails in all 8 complete-case
     specifications (from part1_models_metrics.csv).
   - results/report_midterm.md:39 already says this.
2. **P4 depends on the specification.** It is not supported in all 16 imputed specifications but is supported
   in 7 of the 8 complete-case ones (complete-case headline 0.0215). Wave checks are mixed: without waves 6-9
   the gap is 0.0032, 0.0374 and -0.0021 across the three versions; without wave 12 it is 0.0526 to 0.0607. The
   0.0101 margin over 0.03 is smaller than the fold-seed SD of the logistic AUC (0.0127), and the forest was not
   rerun on other fold seeds.
3. **AUC SEs run narrow.** In simulation, the clustered AUC SE is 0.84 to 0.95 of the true SD, and nominal
   90% intervals cover 80% to 87% (`se_check` column of every metrics file; methods_doc.tex:1179, 1194). Read
   "SE 0.0197" as somewhat optimistic; a 90% interval behaves like about an 83% one. The coefficient-table SEs
   are about 20% too narrow (methods_doc.tex:1159), which matters if the report quotes z-values.
4. **Tuning hit the edge of the grids.** QDA chose r = 0.9 in all 10 headline folds. The `pre` forest chose
   min leaf 50 in all 5 folds (part1_models_folds.csv). The grids may cut off better values. This is also why
   the spline step had no effect.
5. **The liking model is worse than a constant.** Every step of the ladder is below the intercept-only baseline
   (-0.0028); the complete-case version gives -0.0240 and +0.0029 (like_models_hypotheses.csv:4-5).
6. **P5 margins.** For women, attractiveness's gap leads shared interests by only 2.89 points. By absolute size,
   attractiveness leads sincerity (-18.2) by just 0.15 (part1_models_hypotheses.csv:10). With waves 6-9 stated
   shares the gaps are 22.4 and 37.1 (no verdict, lines 12-13). Without wave 12 they are 17.8 and 25.0 (still
   attractiveness; part1_additions_hypotheses.csv:30-31).
7. **Best `pre+date` model.** It is "best" only within the pre-registered pool; Part 2's stacked ensemble is
   0.8464. This is minor.
8. Not a problem: pooling held-out predictions instead of averaging per-fold AUCs changes little on the real
   data (pooled minus bias-corrected: -0.0013, SD 0.0086, for logistic `pre`; results/auc_estimators_real_run.log).
