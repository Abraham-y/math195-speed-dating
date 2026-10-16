# 07. Literature check: cross-validation error bars under clustering

Date: 2026-10-15. Why: before claiming that the error-bar track (proposals/06) adds anything new. A background
agent searched the literature and read papers and package source code. This note records what it found and what
we can claim. Venues marked "unverified" were not confirmed from the paper itself.

## Verdicts

| Question | Verdict |
| --- | --- |
| (a) Nested CV with clusters held out at both levels, a cluster-robust outer variance and the Bates, Hastie and Tibshirani (BHT) bias adjustment | Partly done: every ingredient is published, but we found no paper that combines all three and tests coverage |
| (b) CV error intervals for two-person (dyadic, cross-classified) data | Apparently open for prediction error; ingredients exist for structural parameters |
| (c) Random-row CV and its intervals under-cover under clustering | Mostly done; our A1 and A2 confirm it |

### (a) Cluster-aware nested CV

- **Bates, Hastie and Tibshirani (2024, JASA 119(546); arXiv 2104.00673).** Assumes i.i.d. data. The discussion
  lists adapting nested CV to dependent data as future work. Their R package `nestedcv` assigns folds by row with
  no grouping option. Theorem 3 (the nested-CV MSE estimate is unbiased for the MSE of (K-1)-fold CV) carries over
  directly when clusters are treated as the i.i.d. units. Our "cluster-nested MSE is unbiased" proposition is
  therefore close to a corollary. What is new is unequal cluster sizes and ratio-type metrics.
- **Cai, Luo, Guo et al. ("Bootstrapping the cross-validation estimate", arXiv 2307.00260v2).** i.i.d. Remark 4
  suggests resampling whole subjects for longitudinal data but does not evaluate it.
- **LeDell, Petersen and van der Laan (2015, EJS 9:1583-1607) and the `cvAUC` package.** `ci.pooled.cvAUC` does
  grouped CV with influence curves summed within id and a variance over ids. This is essentially our
  cluster-robust AUC estimator. It ignores the covariance between folds.
- **Schulz-Kumpel et al. (2025, DMLR; arXiv 2409.18836) and the `mlr3inferr` package.** A benchmark of 13 CI
  methods, i.i.d. only. Its nested-CV resampling inherits mlr3's group handling, so grouped folds at both levels
  are possible, but its variance is computed row by row (read from the code, not run).
- **Bayle, Bayle, Janson and Mackey (2020, NeurIPS).** A CV central limit theorem for independent, not identically
  distributed points. With clusters as the points, it gives a grouped-CV Wald interval.
- **Rabinowicz and Rosset (2022, JASA 117(538)).** Separate prediction for seen clusters from prediction for new
  clusters, and correct the bias of CV under correlation. No variances or intervals.
- **Internal-external CV (for example de Jong et al. 2021, Stat Med; `metamisc`).** Leave one cluster out and
  pool the per-cluster results with a random-effects meta-analysis. Overlap between training sets is not modelled.

### (b) The two-person (dyadic) case

- **Cross-fitting.** Chiang, Kato, Ma and Sasaki (JBES 2022; arXiv 1909.03489) use K-squared-fold multiway
  cross-fitting with a two-way cluster-robust SE. This is exactly a held-out-rater-and-ratee scheme, but for a
  structural parameter, not prediction loss. Chiang, Ma, Rodrigue and Sasaki (arXiv 2110.04365) do dyadic
  cross-fitting, again for a structural target.
- **Variance theory:**
  - Fafchamps and Gubert (2007, JDE) and Aronow, Samii and Assenova (2015, Political Analysis): dyadic-robust
    variance.
  - Tabord-Meehan: asymptotics of the dyadic-robust t (venue unverified).
  - Cameron, Gelbach and Miller (2011, JBES): multiway clustering.
  - MacKinnon, Nielsen and Webb (J. Econometrics): two-way cluster-robust variance often performs badly in finite
    samples and can fail to be positive definite.
  - Davezies, D'Haultfoeuille and Guyonvarch (2021, AoS 49(2)): central limit theorems and bootstrap for
    exchangeable arrays.
  - Menzel: non-Gaussian limits (venue unverified).
  - Owen (2007, AoAS): pigeonhole bootstrap; Owen and Eckles (2012).
- **Hold-out designs without intervals:**
  - Park and Marcotte (2012, Nat Methods): C1, C2 and C3 test pairs.
  - Pahikkala et al. (2015, Brief Bioinform) and Stock et al. (2020, Brief Bioinform).
  - Network CV for model selection: Chen and Lei; Li, Levina and Zhu.
- **This dataset.** Aronow, Samii and Assenova reanalysed this same speed-dating data and found dyadic-robust SEs
  barely larger than SEs clustered on the subject. Everyone attends one wave, so all shared-person dependence sits
  inside events, and holding out events with event clustering already covers it. The two-way problem appears only
  for within-event targets (a new rater and a new ratee at an event already seen), which is our exploratory dyadic
  design.

### (c) Random-row CV is optimistic and naive error bars are too small

- **Random-row CV is optimistic under grouping:** Roberts et al. (2017, Ecography; content not opened), Saeb et al.
  (2017, GigaScience), Rabinowicz and Rosset (2022), Park and Marcotte (2012), Pahikkala et al. (2015).
- **Naive CV standard errors are too small:** Varoquaux (2018, NeuroImage), Bengio and Grandvalet (2004, JMLR), and
  Nadeau and Bengio (2003, Machine Learning; read through secondary sources).
- **Not found:** a paper that runs BHT nested CV on clustered data and reports coverage. Our A2 demonstration is
  new but small.
- **Interpretation.** Most of the shortfall to 68-74% coverage of the new-event error is a centring failure: random
  rows estimate the error for people already seen. It is not mainly a variance failure, so we should present it as
  expected.

## What we can claim

**Superseded on 2026-10-15 by Parts 4 to 8.** The adapted nested-CV interval below under-covers once the events are
redrawn in every data set (N1; 83.6% in N5), while existing intervals applied at the event level reach the target:
the cluster-robust SE with t quantiles (N4, N11), the wild cluster bootstrap-t (N12) and the Nadeau-Bengio correction
on event folds (conservative, N6). What we now claim is in proposals/06, "Where the error-bar track stands": the
evidence for those intervals at the event level, and Proposition 8 (ii), which predicts how conservative the
Nadeau-Bengio correction is at any fold count (checked out of sample, N14). The original wording is kept for the record:

"We adapt the Bates, Hastie and Tibshirani nested-CV interval to one-way clustered data: clusters are held out at
both levels, the outer-fold term is cluster-robust, and the bias adjustment is kept. In a simulation calibrated to
the data, with about 21 clusters, it gives near-nominal coverage of the new-cluster error, where the published
version and grouped CV with a cluster-robust SE do not."

How to frame the parts:

- **Propositions in proposals/06.** Present them as clusters-as-units extensions of BHT Theorem 3, Bengio and
  Grandvalet, and Bayle et al., not as new theory.
- **Result (c).** Present it as a confirming demonstration.
- **The dyadic case.** Claim nothing unless we implement a two-way scheme, for example the K-squared blocks of
  Chiang et al. within events. Even then, state that in this dataset the problem is nested inside events.

## Still to do

1. **More comparisons in the simulation.** (Done in Parts 4, 5 and 7: `sim_cv_repair.py`, `sim_cv_confirm.py`, `sim_cv_sizes.py`.) Add these intervals:
   - the Nadeau-Bengio corrected t on grouped folds;
   - Bayle et al.'s Wald interval with clusters as units;
   - a cluster bootstrap of grouped CV (Cai et al., Remark 4);
   - internal-external CV with a random-effects meta-analysis;
   - grouped nested CV with a row-level variance (as `mlr3inferr` would compute it).
2. **Over-coverage.** (Explained: the events were fixed across replicates; Proposition 7 and N4.) Explain why event CV with a cluster-robust SE covers 99% (A2). Possible causes are the
   G/(G-1) factor, the choice of normal against t quantiles at G = 21, or a positive covariance between the CV
   estimate and the target.
3. **Citations in the write-up:** Bates et al.; Rabinowicz and Rosset; Bengio and Grandvalet; Nadeau and Bengio;
   Bayle et al.; Austern and Zhou; LeDell et al.; Cai et al.; Schulz-Kumpel et al.; Roberts et al.; Saeb et al.;
   Aronow et al.; Cameron, Gelbach and Miller; MacKinnon, Nielsen and Webb; Chiang et al. (2022); Owen (2007).

## Part 9: the cross-validated AUC (added 2026-10-15)

A quick search (one extended and one standard web search) found the following.

- **Known: the pooling bias.** Pooled K-fold and leave-one-out AUC estimates are negatively biased because they
  compare scores from different fold models. See Parker, Gunter and Bedo (2007, BMC Bioinformatics 8:326,
  "stratification bias"), Forman and Scholz (2010, SIGKDD Explorations 12(1)), and Airola et al. (2010, JMLR W&CP 8;
  2011, Computational Statistics and Data Analysis).
- **Known: the remedies.** Averaging over folds removes the pooling bias at a cost in variance. Leave-pair-out CV is
  nearly unbiased (Airola et al.; Montoya Perez et al. 2019, Statistical Methods in Medical Research, the tournament
  version).
- **Closest work on dependent data.** Spatial leave-pair-out CV (Airola, Pohjankukka, Torppa, Middleton, Nykanen,
  Heikkonen and Pahikkala, 2019, Data Mining and Knowledge Discovery) corrects both the pooling bias and a spatial
  bias for spatial classifiers. BCWF is the grouped analogue: within a fold, only pairs from different clusters.
- **Not found: anything like Proposition 9 (iii).** The search turned up no closed-form expansion of the pooled
  bias under clustering. That covers three points:
  - its order is 1/G rather than 1/n, larger by about the residuals' design effect;
  - its leading term C_mix / G does not depend on K, with the next term c2 / ((K - 1) G);
  - the constants are computable from the population.
  It also turned up nothing on the separate same-cluster-pair bias of fold-averaged AUC (K kappa (A_w - A_b) / G).
  The identity c2 = C_mix / 2 is also not in these sources. It is now proved for independent rows (Proposition 10)
  and shown false in general under clustering (proposals/09).

Framing for the write-up: the phenomenon and the remedy family are known. What may be new is the clustered-data
theory that predicts the bias's size and its dependence on K, and the test of those predictions (Part 9). Before
claiming novelty, search further for clustered or multi-site AUC pooling (for example "leave-site-out AUC pooling"
and "grouped cross-validation ROC"). The Airola et al. spatial paper should be read in full.
