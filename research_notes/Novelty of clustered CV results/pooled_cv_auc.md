# Pooled cross-validated AUC under clustering: prior work and the novelty of Proposition 9

Scope: the pooling bias of cross-validated (CV) AUC, especially when rows are clustered and the folds hold out whole clusters. The aim is to judge whether Proposition 9 (proposals/06 section 6f; proposals/08 Part 9) is known, partly known or plausibly new. Searches ran on 2026-11-05.

About the quotations: the copyright policy this research runs under allows at most one short verbatim quote. Every other prior statement is paraphrased, with its exact section and page so that the team can quote it from the source. The verification status of every citation is in the table under Q5.

## Q1. What do Bradley (1997), Forman and Scholz (2010), Parker et al. (2007), Airola et al. (2010, 2011) and the leave-pair-out papers show about pooling scores across folds versus averaging fold AUCs?

### Takeaway
All of these papers assume independent rows. Three things have been established since Bradley (1997) and Parker et al. (2007):
- the phenomenon: pooling held-out scores from different fold models lowers the CV AUC, sometimes far below 0.5;
- its cause: the fold models' scores are not on a common scale, mostly because each training set's class balance, and with it the intercept or prior, is different;
- the remedies: averaging fold AUCs, leave-pair-out (LPO) CV, and rebalancing the training sets.

None of these papers gives a closed-form expression for the bias or its dependence on K; the evidence is simulation and examples.

### Cited Findings
- **Bradley (1997), Pattern Recognition 30(7):1145-1159.**
  - He used 10-fold CV, computed the AUC on each of the 10 test partitions and averaged them. The SE came from the SD of the 10 areas, which he describes as in effect a jackknife SE (Section 6, "The performance measures", p. 1150). — [Bradley 1997, full text](https://math.ucdavis.edu/~saito/data/roc/bradley-auc.pdf); [DOI 10.1016/S0031-3203(96)00142-2](https://doi.org/10.1016/S0031-3203(96)00142-2)
  - In a Remark in the same section (p. 1150) he separates two ways of combining partitions:
    - "pooling": the true-positive and false-positive frequencies at each threshold are averaged into one group ROC curve;
    - "averaging": one AUC per partition, then the mean.
  - He says pooling assumes that the classifiers from all training partitions come from one population. That is defensible for overall accuracy but less so for individual ROC points. He states that "pooling the data in this way depresses the combined index of accuracy, AUC" (p. 1150), citing Swets and Pickett (his ref. 30). He uses pooled responses only to draw ROC curves. — [Bradley 1997](https://math.ucdavis.edu/~saito/data/roc/bradley-auc.pdf)
- **Forman and Scholz (2010), ACM SIGKDD Explorations 12(1):49-57.**
  - Section 2.4 (p. 52) defines AUC_merge (all folds' scores sorted into one ROC curve) and AUC_avg (the mean of the fold AUCs). It argues that AUC_merge implicitly requires probability estimates that are calibrated across folds, so it penalizes classifiers that rank well but are calibrated differently in different folds. It also reports that WEKA 3.6.1 uses AUC_merge in its Explorer GUI and Evaluation class but AUC_avg in its Experimenter. — [Forman & Scholz 2010, PDF](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf); [DOI 10.1145/1882471.1882479](https://doi.org/10.1145/1882471.1882479)
  - Section 3.2 (p. 53) gives a single illustration: 4-fold stratified CV of a linear SVM on a text task with 38 positives (2.5%). The fold AUCs are 96%, 91%, 94% and 87%, so AUC_avg is 92%, but AUC_merge is only 80%. Figure 3 shows the folds' FPR-against-threshold and TPR-against-threshold curves shifted horizontally from one another. — [Forman & Scholz 2010](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf)
  - The bias-and-variance study (Section 4) covers the F-measure only, not the AUC. The paper has no theory of the AUC pooling bias and does not cite Bradley or Parker (its reference list has 7 items). The conclusion (p. 57) urges AUC_avg. — [Forman & Scholz 2010](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf)
- **Parker, Günter and Bedo (2007), BMC Bioinformatics 8:326 (open access, CC BY).**
  - "Review of ROC and AUC calculation":
    - pooling and averaging are both in use, and the literature does not settle which to prefer;
    - pooling assumes that each fold's results are a sample from the same population (citing Swets and Pickett 1982).

    — [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/); [DOI 10.1186/1471-2105-8-326](https://doi.org/10.1186/1471-2105-8-326)
  - "Theoretical analysis" and "Analysis of bias in error rate and AUC estimation" give the mechanism:
    1. Classifiers carry the class prior estimated from the training set, explicitly as in LDA or implicitly.
    2. Under CV a training set's class proportions are negatively correlated with those of its test set (correlation -1 for one train/test pair).
    3. If posterior probabilities or decision values are used for ranking, a fold whose training set over-represents class 1 lifts the scores of all its test samples, and that test set under-represents class 1. So negatives are ranked too high when folds are pooled.
    4. The bias is pessimistic, toward an AUC of 0 for a predictor with no signal, and it shrinks as the signal grows.
    5. Averaging is immune, because ranking within one test set does not depend on the class proportions.

    — [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/)
  - "Application to practical induction algorithms" names a second, separate pooling bias: non-systematic noise across folds, from the slightly different classifiers learnt on each training set, which perturbs the scores and pulls the pooled AUC toward 0.5. — [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/)
  - Results:
    - With no signal (DLDA, n = 30, unstratified 10-fold CV), the pooled AUC lies well below 0.5 at every class proportion; the abstract reports values under 0.3.
    - The pooled bias tracks the covariance between training-set and test-set class proportions (Figs 4a and 5).
    - Stratified CV removes most of the bias but not all of it.
    - The proposed balanced stratified CV (BSCV) subsamples each training set so that its class counts are equal across folds, and this removes the stratification bias.
    - With BSCV the pooled AUC is still slightly low, which the paper attributes to inter-fold noise (text with Fig. 8).
    - Conclusion: average per-fold AUCs; unbalanced leave-one-out (LOO) CV should not be used to estimate the AUC.

    — [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/)
- **Airola, Pahikkala, Waegeman, De Baets and Salakoski (2010), JMLR W&CP 8:3-13 (MLSB 2009).**
  - Section 2 (p. 6) presents pooling and averaging, citing Bradley and Parker. It notes that the same-population assumption is doubtful for the AUC because some positive-negative pairs are built from instances in different folds.
  - It defines leave-pair-out CV (LPOCV): each positive-negative pair is held out in turn. Citing Cortes et al. (2007), LPOCV is unbiased for the expected AUC of training sets of size m-2.
  - Results: the pooled estimators (LOO, balanced LOO, pooled 10-fold) have a substantial pessimistic bias with RLS on low-dimensional data, and much less with RankRLS. The averaged 5-fold and 10-fold estimates have a higher variance.

  — [Airola et al. 2010, PMLR](https://proceedings.mlr.press/v8/airola10a.html)
- **Airola et al. (2011), Computational Statistics & Data Analysis 55(4):1828-1844.**
  - Section 3 repeats the argument and adds that averaging never compares predictions for instances from different folds.
  - Section 4 notes that the RankRLS optimum always sets the intercept coefficient to zero.
  - The pooled estimates were much less biased for RankRLS than for RLS on low-dimensional data (end of Section 5.2). In the no-signal runs, LOO's bias disappeared as the dimension grew (Fig. 4).

  — [Airola et al. 2011, self-archived](https://www.utupub.fi/items/639ce106-68e0-4b27-be21-2fe77dba9dfb/full); [DOI 10.1016/j.csda.2010.11.018](https://doi.org/10.1016/j.csda.2010.11.018)
- **Montoya Perez, Airola, Boström, Jambor and Pahikkala (2019), Statistical Methods in Medical Research 28(10-11):2975-2991 (arXiv v2, pp. 6-7).**
  - Pooling violates the assumption that a single classifier made all the predictions. The predictions for units i and j may come from different prediction functions, which biases the AUC when the learner is unstable.
  - Writing the prediction as f(i) = g(i) + c, the constant (intercept) term alone can bias the pooled AUC, most severely for LOO. A constant predictor built from the training-set class frequencies gives a LOO AUC of 1.
  - A similar negative bias arises when the fitted functions correlate with the class proportions (their ref. 8 is Perlich and Świrszcz 2011).
  - Averaging removes the pooling bias at the cost of a high variance. The paper proposes tournament LPO (TLPO).

  — [Montoya Perez et al., arXiv 1801.09386](https://arxiv.org/abs/1801.09386); [DOI 10.1177/0962280218795190](https://doi.org/10.1177/0962280218795190)
- The real-data experiment in Montoya Perez et al. uses 85,876 MRI voxels from 20 prostate cancer patients. Each experiment draws 30 voxels without replacement and ignores the patient clustering. — [Montoya Perez et al., arXiv v2](https://arxiv.org/abs/1801.09386)
- **Smith et al. (2014), American Journal of Epidemiology 180(3):318-324.** Airola et al. (2019, Section 2.2) cite it as showing that LPO removes the pooling bias. The paper itself was not opened. — [Airola et al. 2019](https://www.utupub.fi/items/6ca8f721-0d1b-4b82-9b51-e1cf531f2365/full); [Smith et al. DOI 10.1093/aje/kwu140](https://doi.org/10.1093/aje/kwu140)
- **Numminen, Montoya Perez, Jambor, Pahikkala and Airola (2023), Computational Statistics 38(3):1579-1595.** "Quicksort leave-pair-out cross-validation for ROC curve analysis." The metadata is verified; the content was seen only in search snippets: an O(n log n) LPO ranking. — [DOI 10.1007/s00180-022-01288-3](https://doi.org/10.1007/s00180-022-01288-3)
- **Austin, Pe'er and Korem (2025), Science Advances 11(48):eadx6976 (arXiv 2406.01652).** They call the mechanism "distributional bias".
  - Under LOO and leave-P-out CV, the mean label of each training set is negatively correlated with the label of its held-out instance.
  - A dummy predictor that returns minus the training mean achieves a perfect auROC.
  - Logistic regression on random data is pushed below an auROC of 0.5.
  - The effect weakens as P grows, and stratification cannot always remove it.
  - Their fix, RLOOCV, removes a random opposite-label instance from each training set.
  - They say the phenomenon was briefly observed before. Their setting is independent rows only.

  — [Austin et al. 2025, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12662204/); [DOI 10.1126/sciadv.adx6976](https://doi.org/10.1126/sciadv.adx6976)

### Inferences
- **Parker et al. (2007) already describe, in words, both parts of Proposition 9 (iii)'s mixing bias:**
  - a correlated part, in which a fold's training composition shifts that fold model's prior or intercept opposite to the test fold's composition (the analogue of C_mix);
  - a non-systematic part, in which inter-fold model noise pulls the pooled AUC toward 0.5 (the analogue of the c2 term, which comes from Var(xi)).

  Neither part is quantified. The project should cite Parker for this two-part structure and claim only its quantification.
- **Bradley pools differently.** His "pooling" combines TP/FP counts at common thresholds. With thresholds shared across folds this is the same as merging the scores, but the write-up should not describe Bradley as sorting raw scores together.
- **What to cite Forman and Scholz for.** Their contributions are the calibration explanation and the software practice (WEKA). They did not measure the AUC bias. proposals/07 and 08 currently cite them alongside Parker for the "pooling bias"; that is acceptable if worded as "illustrated", not "measured".

### Gaps
- Swets and Pickett (1982) is the source that Bradley and Parker both cite for the same-population assumption and the depressed pooled AUC. It is a book and was not opened.
- Smith et al. (2014), Perlich and Świrszcz (2011) and Numminen et al. (2023) were verified through Crossref metadata only; their content is reported second-hand above.

## Q2. Is the mechanism known: (a) for independent data, where pooling mixes scores from fold models with different calibrations; (b) for clustered or grouped data, where a cross-fold pair is linked through the other side's training cluster?

### Takeaway
**(a) Independent data: yes, qualitatively.** The following all explain the mechanism:
- Parker et al. (training-set class-prior shift, plus inter-fold noise);
- Forman and Scholz (cross-fold calibration);
- Montoya Perez et al. (the intercept term);
- Perlich and Świrszcz;
- Austin et al. (2025).

No closed form was found. Proposition 10's expression for the pooled bias, -((2K-1)/(2(K-1)))(1/n)E[delta(D) x_i'H^-1 x_j], and the identity c2 = C_mix/2 appear in none of these sources.

**(b) Clustered data: only an empirical hint.** Airola et al. (2019) found that the pooling bias of spatial LOO grows as the excluded neighbourhood grows. No source frames a cross-fold pair as linked through the other side's training cluster. None derives the order 1/G, the K-free leading term or computable constants.

### Cited Findings
- **Parker et al.: the training prior.** Training-set class proportions are anti-correlated with test-set proportions. This shifts each fold model's prior term, so pooled scores from different folds are not comparable ("Theoretical analysis"). — [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/)
- **Forman and Scholz: calibration.** The soft scores of different fold classifiers are not calibrated with one another (Section 3.2, Fig. 3). — [Forman & Scholz 2010](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf)
- **Montoya Perez et al.: the intercept.** A constant or intercept component of the prediction function biases the pooled AUC, and so do fitted functions that track the class proportions. — [Montoya Perez et al.](https://arxiv.org/abs/1801.09386)
- **Airola et al. (2011): no intercept, less bias.** RankRLS, which has no intercept by construction, shows a much smaller pooled bias than RLS with an intercept on low-dimensional data. — [Airola et al. 2011](https://www.utupub.fi/items/639ce106-68e0-4b27-be21-2fe77dba9dfb/full)
- **Austin et al. (2025).** Under LOO the training mean label and the held-out label are perfectly negatively correlated; the fix is to rebalance the training set (RLOOCV). — [Austin et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12662204/)
- **Airola, Pohjankukka, Torppa, Middleton, Nykänen, Heikkonen and Pahikkala (2019), Data Mining and Knowledge Discovery 33(3):730-747.**
  - The abstract names two biases in spatial CV AUC: pooling, which pushes it down, and spatial autocorrelation, which pushes it up.
  - Section 2.4 defines LPO-SCV. It leaves out the test pair and every instance within radius r of either member, then trains on the rest, so that only predictions from the same CV round are compared.
  - Section 4.1 reports that the pooling bias grows as the dead-zone radius grows. At 30 km, LPO-SCV gives 0.84 and LOO-SCV gives 0.70 for the model trained on features. For the model trained on coordinates alone, LOO-SCV falls well below 0.5.

  — [Airola et al. 2019, self-archived](https://www.utupub.fi/items/6ca8f721-0d1b-4b82-9b51-e1cf531f2365/full); [DOI 10.1007/s10618-018-00607-x](https://doi.org/10.1007/s10618-018-00607-x)

### Inferences
- **Parker's mechanism operates per cluster under grouped CV.** This framing is ours, not a source's.
  - Holding out a cluster whose residual outcome rate is high lowers the training prevalence and the fold model's intercept.
  - The cluster is large, so the shift is of order 1/G rather than 1/n.
  - Row-level stratification cannot remove it, because the folds must keep clusters whole.

  This matches the sign of Proposition 9 (C_mix < 0 under outcome clustering) and its scaling with the number of clusters rather than rows.
- **Closest prior evidence.** Airola et al.'s dead-zone result is the nearest evidence that holdouts which respect dependence enlarge the pooling bias. It gives no rate and no dependence on K. Proposition 9 (iii)'s K-free leading term C_mix/(G-1) and its c2/((K-1)(G-1)) correction have no counterpart there.
- **Rebalancing would fix only part of it.** A rebalancing fix adapted to grouped folds, in the style of Parker's BSCV or Austin's RLOOCV (subsampling clusters or rows so every training set has the same prevalence), would act only on the intercept channel of C_mix. BCWF avoids cross-fold pairs entirely. This is untested and could be an extra comparison arm.

### Gaps
- No theoretical treatment was found of the pooled LOO or K-fold AUC bias for a logistic or other M-estimator in closed form, even for independent rows. Older signal-detection sources (Swets and Pickett 1982) were not checked.
- Google Scholar "cited by" lists for Forman and Scholz (2010) and Airola et al. (2011) could not be read, because Scholar blocks automated access. A grouped-data follow-up citing them could have been missed.

## Q3. Does any paper study the cross-validated AUC under clustering or grouped CV: its bias, a pairs-within-fold estimator, or the validity of a clustered (Obuchowski) variance for CV AUCs?

### Takeaway
The closest work is LeDell, Petersen and van der Laan (2015) on grouped (id-level) folds. Its estimator is the fold-averaged within-fold AUC; its variance comes from influence curves summed within id; its target is the data-dependent "true CV AUC". It contains no bias analysis against the new-cluster AUC, no cross-fold pairs, and a coverage simulation for independent rows only.

Clustered ROC variance (Obuchowski 1997; Emir et al. 1998) and the within-cluster versus between-cluster concordance (Van Oirbeek and Lesaffre 2012; van Klaveren et al. 2014) are established, but not for CV estimates. Spatial LPO (Airola et al. 2019) is the only CV estimator for dependent data found that restricts which pairs are compared.

Nothing was found that analyses the bias of pooled CV AUC under grouped CV, proposes a between-cluster within-fold (BCWF) estimator, or checks the Obuchowski variance for CV AUCs.

### Cited Findings
- **LeDell, Petersen and van der Laan (2015), Electronic Journal of Statistics 9(1):1583-1607.**
  - Section 5 covers "pooled repeated measures" data. The independent units are ids, and all of an id's observations fall in the same validation fold.
  - The estimator (5.2-5.3) is the average over folds of the AUC within each validation fold, using that fold's training fit.
  - The target (5.6) is the average over folds of the true probability, under P0, that a random positive single-time-point observation outranks a random negative one under that fold's fitted function. The paper notes that this target is random.
  - Theorem 5.1 gives the influence curve for this case.

  — [LeDell et al. 2015, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4533123/); [DOI 10.1214/15-EJS1035](https://doi.org/10.1214/15-EJS1035)
- LeDell et al.'s coverage simulation (Section 7.1) generates i.i.d. training sets with n from 500 to 20,000 and k from 10 to 200 covariates, and reports that coverage can fall below nominal for small n. No simulation of the pooled repeated-measures interval was found in the paper. — [LeDell et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4533123/)
- **The cvAUC R package (version 1.1.4, 2022-01-17).**
  - `cvAUC()` returns the mean of the fold AUCs.
  - `ci.pooled.cvAUC()` takes cluster or entity ids and treats the clusters as the independent units.
  - The package's example builds folds that keep each patient's observations together, stratified by whether the id has any positive outcome.

  — [cvAUC reference manual](https://cran.r-project.org/web/packages/cvAUC/cvAUC.pdf)
- **Clustered ROC variance (metadata verified; content not re-opened here).**
  - Obuchowski (1997), "Nonparametric analysis of clustered ROC curve data", Biometrics 53(2):567-578. — [DOI 10.2307/2533958](https://doi.org/10.2307/2533958)
  - Emir, Wieand, Su and Cha (1998), "Analysis of repeated markers used to predict progression of cancer", Statistics in Medicine 17(22):2563-2578. — [Crossref record](https://api.crossref.org/works/10.1002/(sici)1097-0258(19981130)17:22%3C2563::aid-sim952%3E3.0.co;2-o)
- **van Klaveren, Steyerberg, Perel and Vergouwe (2014), BMC Medical Research Methodology 14:5.**
  - For clustered data they separate two concordance probabilities: within-cluster, for pairs from the same cluster, and between-cluster, for pairs from different clusters. They credit the distinction to Van Oirbeek and Lesaffre (2010 Stat Med; 2012 CSDA).
  - They argue that the within-cluster concordance is the relevant one when a model supports decisions within clusters, and estimate it by meta-analysing cluster-specific c-indexes.
  - Their example is the CRASH trial, with 2,081 patients in 35 centres. Fixed-effect summaries range from 0.75 to 0.84 depending on the weights, and the random-effects mean is 0.77.

  — [van Klaveren et al. 2014, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3897966/); [DOI 10.1186/1471-2288-14-5](https://doi.org/10.1186/1471-2288-14-5)
- **Van Oirbeek and Lesaffre (2012)**, "Assessing the predictive ability of a multilevel binary regression model", Computational Statistics & Data Analysis 56(6):1966-1980. The metadata is verified; the content is known only through van Klaveren et al.'s description. — [DOI 10.1016/j.csda.2011.11.023](https://doi.org/10.1016/j.csda.2011.11.023)
- **Wynants, Vergouwe, Van Huffel, Timmerman and Van Calster (2018), Statistical Methods in Medical Research 27(6):1723-1736.** They set out a framework for validating at the centre level and at the population level, with marginal or conditional predictions. With heavy clustering (ICC 20%), centre-specific predictions perform best at both levels. — [DOI 10.1177/0962280216668555](https://doi.org/10.1177/0962280216668555)
- **Benkeser, Petersen and van der Laan (2020), JASA 115(532):1917-1932.** Standard CV estimators of nonlinear metrics such as the AUC can perform poorly in small samples, and they propose CV-TMLE estimators. Only the abstract was checked. — [DOI 10.1080/01621459.2019.1668794](https://doi.org/10.1080/01621459.2019.1668794)
- **Escanciano and Terschuur, "Debiased Machine Learning U-statistics" (arXiv 2206.05235, v5).** Cross-fitted debiased U-statistic estimators with machine-learning first steps, with the AUC as one application. Only the abstract was checked; it has no clustered or grouped-CV content. — [arXiv 2206.05235](https://arxiv.org/abs/2206.05235)

### Inferences
- **Proposition 9 (i) is partly known.** It is in effect LeDell et al.'s Theorem 5.1 with clusters as units, extended in two ways: to the pooled and BCWF estimators, which contain cross-fold pairs or restrict pairs, and to the full model's new-cluster AUC as the target. That the Obuchowski variance of the pooled scores agrees with the id-summed influence-curve variance to first order is the expected consequence.
- **LeDell et al. do not see the same-cluster bias.** Their within-fold estimator includes pairs from the same id, while the target in (5.6) compares independently drawn single observations, so the estimator and target differ by the same-cluster-pair term of Proposition 9 (iii)1. Under their asymptotics, with V fixed and the number of ids growing, the share of same-id pairs vanishes, so the term is below first order and the paper does not discuss it. This is our reading of their definitions.
- **BCWF-pooled is a known idea in a new combination.** It is the grouped K-fold analogue of LPO and LPO-SCV (compare only scores from the same CV round, and exclude dependent neighbours), applied to van Klaveren's between-cluster concordance. That combination was not found as a named estimator, nor was any coverage study of it with an Obuchowski SE.

### Gaps
- The clustered rank-sum and AUC work of Rosner and Glynn (Biometrics, Statistics in Medicine) was not opened; status UNVERIFIED.
- The spatial-CV AUC literature beyond Airola et al. (2019) was not covered systematically. Pohjankukka et al. (arXiv 2005.14263, spatial k-fold CV) appeared in search results but was not opened.
- Two 2025-2026 arXiv items surfaced in search but were not opened, so their content is UNVERIFIED. Neither looked like theory of the pooling bias.
  - arXiv 2607.02394 (contrast-enhanced mammography): the snippet mentions patient-grouped CV with a patient-cluster bootstrap.
  - arXiv 2507.22299: "Comparing Cluster-Based Cross-Validation Strategies".

## Q4. Does any software (scikit-learn, mlr3, tidymodels, cvAUC, pROC) or guidance (TRIPOD+AI, medical ML checklists) say to pool or to average under grouped CV?

### Takeaway
The software defaults average per fold or resample:
- scikit-learn's `cross_val_score` and `cross_validate`;
- mlr3's default `average = "macro"`;
- tidymodels' `collect_metrics(summarize = TRUE)`;
- cvAUC, also with grouped ids.

Pooling is an option you choose: mlr3's "micro" setting, or scikit-learn's `cross_val_predict` (with a warning in its documentation). WEKA's Explorer pooled by default, as Forman and Scholz reported. No software documentation found addresses AUC pooling under grouped CV specifically.

Guidance: TRIPOD+AI says nothing on pooling or folds. TRIPOD-Cluster asks for performance overall and for each cluster, with heterogeneity and random-effects meta-analysis. That is a within-cluster target, not the pooled between-cluster AUC.

### Cited Findings
- **scikit-learn (1.9.1 documentation).** The `cross_val_predict` page warns that feeding its predictions to a metric may not validly measure generalization performance. Results can differ from `cross_validate` and `cross_val_score` unless the test sets are of equal size and the metric decomposes over samples, which the AUC does not. — [scikit-learn cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
- **mlr3.** A Measure's `average` field defaults to "macro": each resampling iteration is scored and the scores are combined by the aggregator, the mean by default. "micro" combines all predictions into one Prediction object and scores it once; "macro_weighted" also exists. The documentation suggests micro averaging as one way around iterations with missing classes. — [mlr3 Measure reference](https://mlr3.mlr-org.com/reference/Measure.html)
- **tidymodels.** `collect_metrics()` has `summarize = TRUE` by default, which summarizes the per-resample metrics over resamples; `FALSE` returns each resample's value. — [tune collect_metrics](https://tune.tidymodels.org/reference/collect_predictions.html)
- **cvAUC.** It averages fold AUCs, and `ci.pooled.cvAUC` keeps each id within one fold (see Q3). — [cvAUC manual](https://cran.r-project.org/web/packages/cvAUC/cvAUC.pdf)
- **WEKA 3.6.1.** Its Explorer GUI and Evaluation class compute AUC_merge, while the Experimenter computes AUC_avg (Forman and Scholz, Section 2.4 and conclusion). — [Forman & Scholz 2010](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf)
- **TRIPOD+AI (Collins et al. 2024, BMJ 385:e078378).** The main text (PMC11019967) never uses the words "cross-validation", "fold", "pooled" or "pooling". It refers clustered-data studies to TRIPOD-Cluster. — [DOI 10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378); [PMC11019967](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/)
- **TRIPOD-Cluster explanation and elaboration (Debray et al. 2023, BMJ 380:e071058).**
  - Item 13a: report performance measures, with uncertainty intervals, overall and for each cluster. Item 13b: report any heterogeneity in performance across clusters.
  - The text suggests validating the model separately in each cluster, showing the estimates in a forest plot, and summarizing them by random-effects meta-analysis with a prediction interval.
  - Its examples include internal-external CV (IECV) over regions, summarized by random-effects meta-analysis.

  — [TRIPOD-Cluster E&E, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9903176/); [DOI 10.1136/bmj-2022-071058](https://doi.org/10.1136/bmj-2022-071058)

### Inferences
- The project's recommendation (do not pool across fold models when clusters differ in outcome rate; use BCWF-pooled) agrees with the defaults and goes further than any documentation found. The defaults' fold averaging under grouped CV still contains the same-cluster-pair term K·kappa·(A_w - A_b)/G of Proposition 9 (iii)1, and no documentation mentions it.
- TRIPOD-Cluster's per-cluster meta-analysis targets A_w, the "matchmaker" target in proposals/06, not A_b. It is worth citing when the write-up explains which target fits which use.

### Gaps
- The pROC documentation was not checked. pROC is believed to have no CV facility (UNVERIFIED).
- The scikit-learn `GroupKFold` and `StratifiedGroupKFold` pages were not checked for AUC-specific advice.

## Q5. Overall: is Proposition 9 known, partly known or plausibly new?

### Takeaway
- **Known:** the phenomenon, the mechanism for independent rows (described qualitatively, with both of its parts), and the remedy family.
- **Partly known:** the first-order result (i), as an extension of LeDell et al. (2015).
- **Plausibly new:**
  - the clustered second-order theory (iii): order 1/G, a K-free leading term C_mix/(G-1), the next term c2/((K-1)(G-1)), constants computable from the population, and its test;
  - the same-cluster-pair bias of the fold-averaged and pooled estimates;
  - the Nadeau-Bengio ratio for AUC (ii);
  - the closed form of Proposition 10 with c2 = C_mix/2.

  None of these was found in any source. BCWF-pooled is a natural combination of known ideas (LPO, LPO-SCV, between-cluster concordance), but no paper was found that names it or tests its coverage.

### Cited Findings

#### Component verdicts

| Component | Verdict | Closest prior work |
| --- | --- | --- |
| Pooled CV AUC is biased downward (independent rows) | Known | [Bradley 1997, p. 1150](https://math.ucdavis.edu/~saito/data/roc/bradley-auc.pdf); [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/); [Forman & Scholz 2010, Section 3.2](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf); [Airola et al. 2010](https://proceedings.mlr.press/v8/airola10a.html), [2011](https://doi.org/10.1016/j.csda.2010.11.018); [Montoya Perez et al.](https://arxiv.org/abs/1801.09386); [Austin et al. 2025](https://doi.org/10.1126/sciadv.adx6976) |
| Mechanism: training composition shifts the prior or intercept, plus inter-fold noise | Known qualitatively, without a formula | [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/) (both parts); [Montoya Perez et al.](https://arxiv.org/abs/1801.09386) (intercept) |
| Remedies: averaging, LPO, rebalancing the training sets | Known | [Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/) (BSCV); [Airola et al. 2011](https://doi.org/10.1016/j.csda.2010.11.018) (LPO); [Austin et al. 2025](https://doi.org/10.1126/sciadv.adx6976) (RLOOCV) |
| (i) First-order Hajek projection; clustered or influence-curve variance valid to first order | Partly known | [LeDell et al. 2015, Section 5, Theorem 5.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4533123/) (fold-averaged, id-level influence curve, data-dependent target); [Obuchowski 1997](https://doi.org/10.2307/2533958) |
| (ii) Nadeau-Bengio ratio for fold AUCs tends to Var(a - beta/(K-1))/Var(a), with beta near 0 | No prior found | None found for the AUC |
| (iii) Mixing bias C_mix/(G-1) + c2/((K-1)(G-1)), K-free leading term, order 1/G | Plausibly new | Only an empirical hint: [Airola et al. 2019, Section 4.1](https://www.utupub.fi/items/6ca8f721-0d1b-4b82-9b51-e1cf531f2365/full) (pooling bias grows with the dead zone) |
| (iii) Same-cluster-pair bias, kappa(A_w - A_b)/G pooled and K times that fold-averaged | Plausibly new as a CV bias result | The concepts A_w and A_b: [van Klaveren et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3897966/); [Van Oirbeek & Lesaffre 2012](https://doi.org/10.1016/j.csda.2011.11.023) |
| BCWF-pooled estimator, and its coverage of 0.848 to 0.938 at nominal 0.90 | Construction partly known; named estimator and its evaluation not found | [Airola et al. 2010](https://proceedings.mlr.press/v8/airola10a.html) (LPO); [Airola et al. 2019, Section 2.4](https://www.utupub.fi/items/6ca8f721-0d1b-4b82-9b51-e1cf531f2365/full) (LPO-SCV) |
| c2/C_mix = 1/2 (Proposition 10) and the closed-form pooled bias for independent rows | Plausibly new | Nothing similar found; Parker et al. give only the qualitative covariance link |

#### Citation verification log

| Citation | Status | Source opened |
| --- | --- | --- |
| Bradley AP (1997). Pattern Recognition 30(7):1145-1159. doi:10.1016/S0031-3203(96)00142-2 | VERIFIED (full text and Crossref) | [PDF](https://math.ucdavis.edu/~saito/data/roc/bradley-auc.pdf) |
| Forman G, Scholz M (2010). ACM SIGKDD Explorations 12(1):49-57. doi:10.1145/1882471.1882479 | VERIFIED (full text and Crossref) | [PDF](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf) |
| Parker BJ, Günter S, Bedo J (2007). BMC Bioinformatics 8:326. doi:10.1186/1471-2105-8-326 | VERIFIED (full text) | [PMC2211509](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/) |
| Airola A, Pahikkala T, Waegeman W, De Baets B, Salakoski T (2010). JMLR W&CP 8:3-13. No DOI | VERIFIED (full text) | [PMLR](https://proceedings.mlr.press/v8/airola10a.html) |
| Airola A et al. (2011). Computational Statistics & Data Analysis 55(4):1828-1844. doi:10.1016/j.csda.2010.11.018 | VERIFIED (self-archived full text and Crossref) | [UTUPub](https://www.utupub.fi/items/639ce106-68e0-4b27-be21-2fe77dba9dfb/full) |
| Airola A, Pohjankukka J, Torppa J, Middleton M, Nykänen V, Heikkonen J, Pahikkala T (2019). Data Mining and Knowledge Discovery 33(3):730-747. doi:10.1007/s10618-018-00607-x | VERIFIED (self-archived full text and Crossref) | [UTUPub](https://www.utupub.fi/items/6ca8f721-0d1b-4b82-9b51-e1cf531f2365/full) |
| Montoya Perez I, Airola A, Boström PJ, Jambor I, Pahikkala T (2019). Statistical Methods in Medical Research 28(10-11):2975-2991. doi:10.1177/0962280218795190 | VERIFIED (arXiv v2 full text and Crossref) | [arXiv](https://arxiv.org/abs/1801.09386) |
| Smith GCS, Seaman SR, Wood AM, Royston P, White IR (2014). American Journal of Epidemiology 180(3):318-324. doi:10.1093/aje/kwu140 | Metadata VERIFIED (Crossref); content not opened | [DOI](https://doi.org/10.1093/aje/kwu140) |
| Perlich C, Świrszcz G (2011). ACM SIGKDD Explorations 12(2):11-15. doi:10.1145/1964897.1964901 | Metadata VERIFIED (Crossref); content not opened | [DOI](https://doi.org/10.1145/1964897.1964901) |
| Numminen R, Montoya Perez I, Jambor I, Pahikkala T, Airola A (2023). Computational Statistics 38(3):1579-1595. doi:10.1007/s00180-022-01288-3 | Metadata VERIFIED (Crossref); content not opened | [DOI](https://doi.org/10.1007/s00180-022-01288-3) |
| Austin GI, Pe'er I, Korem T (2025). Science Advances 11(48):eadx6976. doi:10.1126/sciadv.adx6976 | VERIFIED (PMC full text) | [PMC12662204](https://pmc.ncbi.nlm.nih.gov/articles/PMC12662204/) |
| LeDell E, Petersen M, van der Laan M (2015). Electronic Journal of Statistics 9(1):1583-1607. doi:10.1214/15-EJS1035 | VERIFIED (PMC full text) | [PMC4533123](https://pmc.ncbi.nlm.nih.gov/articles/PMC4533123/) |
| cvAUC R package 1.1.4 (2022-01-17) | VERIFIED (reference manual) | [CRAN](https://cran.r-project.org/web/packages/cvAUC/cvAUC.pdf) |
| Obuchowski NA (1997). Biometrics 53(2):567-578. doi:10.2307/2533958 | Metadata VERIFIED (Crossref, first page 567); content not re-opened | [DOI](https://doi.org/10.2307/2533958) |
| Emir B, Wieand S, Su JQ, Cha S (1998). Statistics in Medicine 17(22):2563-2578 | Metadata VERIFIED (Crossref); content not opened | [Crossref](https://api.crossref.org/works/10.1002/(sici)1097-0258(19981130)17:22%3C2563::aid-sim952%3E3.0.co;2-o) |
| van Klaveren D, Steyerberg EW, Perel P, Vergouwe Y (2014). BMC Medical Research Methodology 14:5. doi:10.1186/1471-2288-14-5 | VERIFIED (PMC full text) | [PMC3897966](https://pmc.ncbi.nlm.nih.gov/articles/PMC3897966/) |
| Van Oirbeek R, Lesaffre E (2012). Computational Statistics & Data Analysis 56(6):1966-1980. doi:10.1016/j.csda.2011.11.023 | Metadata VERIFIED (Crossref) | [DOI](https://doi.org/10.1016/j.csda.2011.11.023) |
| Wynants L et al. (2018). Statistical Methods in Medical Research 27(6):1723-1736. doi:10.1177/0962280216668555 | Metadata and abstract VERIFIED (Europe PMC) | [DOI](https://doi.org/10.1177/0962280216668555) |
| Benkeser D, Petersen M, van der Laan MJ (2020). JASA 115(532):1917-1932. doi:10.1080/01621459.2019.1668794 | Metadata and abstract VERIFIED (Europe PMC, PMC7954141) | [DOI](https://doi.org/10.1080/01621459.2019.1668794) |
| Escanciano JC, Terschuur JR. Debiased Machine Learning U-statistics. arXiv 2206.05235 | Abstract VERIFIED (arXiv API) | [arXiv](https://arxiv.org/abs/2206.05235) |
| Collins GS et al. (2024). TRIPOD+AI. BMJ 385:e078378. doi:10.1136/bmj-2023-078378 | VERIFIED (PMC full text searched) | [PMC11019967](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/) |
| Debray TPA et al. (2023). TRIPOD-Cluster explanation and elaboration. BMJ 380:e071058. doi:10.1136/bmj-2022-071058 | VERIFIED (PMC full text) | [PMC9903176](https://pmc.ncbi.nlm.nih.gov/articles/PMC9903176/) |
| Swets JA, Pickett RM (1982). Evaluation of Diagnostic Systems. Academic Press | UNVERIFIED (cited by Bradley and by Parker; not opened) | — |
| Rosner and Glynn, clustered rank-sum and AUC papers | UNVERIFIED (not opened) | — |
| Tsamardinos I, Greasidou E, Borboudakis G (2018). Machine Learning 107(12):1895-1922. doi:10.1007/s10994-018-5714-4 (BBC-CV pools out-of-sample predictions) | Metadata VERIFIED (Crossref); content not opened, so whether it discusses AUC pooling is UNVERIFIED | [DOI](https://doi.org/10.1007/s10994-018-5714-4) |

### Inferences
- **Suggested framing for proposals/08, "What is new and what is known":**
  - Origin of the pooling bias: Bradley (1997), with Swets and Pickett.
  - Mechanism for independent rows: Parker et al. (2007), whose two parts become C_mix and c2 here; Montoya Perez et al. for the intercept; Austin et al. (2025) as a recent restatement.
  - Remedies: Airola et al. (2010, 2011) and Montoya Perez et al. for LPO; Parker's BSCV.
  - Evidence that dependence enlarges the bias: Airola et al. (2019).
  - First order and the variance with clusters as units: LeDell et al. (2015) and Obuchowski (1997).
  - Within-cluster against between-cluster concordance: Van Oirbeek and Lesaffre (2012) and van Klaveren et al. (2014).
  - What to claim as new: the quantitative clustered expansion (order 1/G, the K-free leading term, the constants and their test), the same-cluster-pair bias of fold-averaged CV AUC, BCWF-pooled as a grouped estimator with its coverage, the Nadeau-Bengio ratio for AUC, and Proposition 10's closed form with c2 = C_mix/2.
- **One sentence in proposals/06 Remarks should be narrowed.** It currently says the mixing bias "is the pooling bias of Forman and Scholz (2010) and of Parker et al. (2007)". The mechanism comes from Parker, and Forman and Scholz illustrate the calibration view.
- **A possible extra comparison arm.** A grouped version of Parker's BSCV or Austin's RLOOCV would separate the intercept channel of C_mix from the slope channel.

### Gaps
- The search was not exhaustive. It used several web searches, Crossref, Europe PMC and the arXiv API, but not Google Scholar "cited by" lists or library databases (Scopus, Web of Science). A manual cited-by check of Airola et al. (2011), Airola et al. (2019) and LeDell et al. (2015) is the most valuable remaining step before any novelty claim.
- (ii)'s claim that d = grad A(theta*) = 0 under correct specification rests on the AUC-optimality of the true risk score (Neyman-Pearson). No source was opened for it here; a citation is still needed (UNVERIFIED).
- Exact page numbers are given where the PDF has them: Bradley, Forman and Scholz, and Airola et al. (2010). BMC, PMC and self-archived versions are cited by section only.
- Copyright policy limited verbatim quotation to one short quote (Bradley, p. 1150). The text extracted for quoting is in the session scratchpad, which may be cleaned up later:
  - `/private/tmp/claude-501/-Users-abrahamyeung-projects-Math-195-Project/6f0764a0-a8c9-434c-98f8-6ee470a2ce37/scratchpad/pdfs/forman.txt`
  - `/private/tmp/claude-501/-Users-abrahamyeung-projects-Math-195-Project/6f0764a0-a8c9-434c-98f8-6ee470a2ce37/scratchpad/pdfs/airola2010.txt`
  - in `/private/tmp/claude-501/-Users-abrahamyeung-projects-Math-195-Project/6f0764a0-a8c9-434c-98f8-6ee470a2ce37/scratchpad/pooled_auc_dl/`: `bradley.txt`, `parker.txt`, `airola2011.txt`, `airola2019.txt`, `montoya.txt`, `ledell.txt`, `cvauc.txt`, `pmc_3897966.txt` (van Klaveren et al.) and `tripodc.txt` (TRIPOD-Cluster).
