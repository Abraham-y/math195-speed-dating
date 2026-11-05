# Grouped-fold Nadeau–Bengio theory should anchor the preprint

The team's searches found nothing that anticipates any of the three results in full. Each one, though, is built from published pieces, so for all three the honest claim is a new quantification or a new combination of known parts, not a discovery. **Result A (Proposition 8(ii) with Proposition 6) is the strongest.** It is the first-order value of Bengio and Grandvalet's (2004) between-block eigenvalue, combined with the influence-function CV variance of Austern and Zhou and of Li. No paper found writes the expected Nadeau–Bengio variance this way, gives the conservativeness factor Var(a−β/(K−1))/Var(a) or its sign condition, or treats grouped folds. The result also explains the "factor of about 2" that Markatou et al. observed in 2005; a seeded simulation run for this report confirms that factor to within one Monte Carlo standard error at n = 1000. **Result B (Proposition 9) builds on a phenomenon first reported in 1997.** Pooled CV AUC is biased low, Parker et al. (2007) described both halves of its mechanism in words, and LeDell et al. (2015) already gave first-order theory with clusters as units for the fold-averaged estimator. What the searches did not find is the order-1/G clustered expansion with its K-free leading term, the same-cluster-pair bias, BCWF-pooled as a named estimator with a coverage study, or c₂ = C_mix/2. **Result C (Proposition 11) is mostly classical.** Its leverage factor is exactly Bell and McCaffrey's (2002) eq. (3), which reproduces the project's 0.945 to three decimals. Its plausibly new parts are the own-row normalization term in the Nadeau–Bengio variance and the fix of dividing residual totals by n/K, and both await one unread survey paper. The applied audience is real: practitioners who already hold out whole subjects, patients, sites or events, but then report a fold SD, a corrected-t interval or a pooled AUC, because neither TRIPOD+AI nor the main libraries give them a method. Recommendation: make A (with Propositions 6 and 7) the core of a preprint, add C's normalization result as an extension, and publish B as a separate companion. Claim nothing as new until the open checks below are done and the mathematical lead, Jenna Yang, has reviewed the theory.

## Proposition 8(ii) is Bengio and Grandvalet's eigenvalue, evaluated to first order

**What the result says.** Proposition 6 concerns grouped folds with the losses held fixed. Averaged over random partitions, the Nadeau–Bengio (NB) variance equals the cluster-robust (random-group) variance times **(J+K−1)/(J−1)**, where the J = RK fold values come from R partitions. On row folds it equals the naive variance times the same factor. Proposition 8(ii) adds refitting. To first order, G·E[V_NB] tends to that factor times **Var(a − β/(K−1))**, where a is a cluster's loss residual and β = dᵀψ is its influence on the fold model's expected loss. Three consequences follow. Relative to the cluster-robust variance, the correction is inflated by Var(a−β/(K−1))/Var(a). It is conservative whenever Var(β) > 2(K−1)Cov(a,β). It is exact when the evaluation loss is the training criterion, because then d = 0. In the team's simulations the factor was **about 1.2 for 0–1 error at K = 5 and about 1 for log-loss and for the AUC**. Predictions built from K = 5 runs matched the observed ratios at K = 3 and K = 10 to within 0.053 (N14, N15).

**Prior work supplies every ingredient but never puts them together.**

- **Nadeau and Bengio** derived their surrogate correlation n₂/(n₁+n₂) for J independent random splits. They assumed each loss depends only on the test point and the training-set size, and warned that the corrected estimator can over- or underestimate the variance, depending on how the surrogate compares with the true correlation ([Nadeau & Bengio 1999, Section 3, p. 310](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf)).
- **Bouckaert and Frank** carried the factor over to repeated K-fold CV by analogy. Their only argument is that K-fold CV is random subsampling whose test sets do not overlap ([Bouckaert & Frank 2004, Section 3.3](https://researchcommons.waikato.ac.nz/handle/10289/1451)).
- **Bengio and Grandvalet** gave the exact eigenstructure of the K-fold error covariance. The between-block eigenvalue λ₂ is precisely what one partition's fold-mean variance estimates, and λ₃/n is Var(CV). They never evaluate the covariance terms through an expansion of the fitted model, and never compute NB's expectation ([Bengio & Grandvalet 2004, Lemma 7, p. 1099](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)).
- **Austern and Zhou** ([arXiv 2001.11111, Theorem 3 and Proposition 3](https://arxiv.org/pdf/2001.11111)) and **Li** ([JAIR 2023, Theorem 3.1](https://jair.org/index.php/jair/article/view/13974)) derived the first-order CV variance, Var(a)+Var(β)+2Cov(a,β) in the team's notation. Austern and Zhou's Remark 6 shows that the covariance term vanishes when the model is evaluated on the loss it was trained on, which is exactly Proposition 8(iv)'s mechanism. Neither paper mentions Nadeau and Bengio.
- **Bates, Hastie and Tibshirani** note that the NB and Markatou estimates do not address the covariance term at all ([BHT arXiv v4, Section 1](https://arxiv.org/pdf/2104.00673)).
- **Bayle et al.** compare NB only empirically, on random 90/10 splits ([Bayle et al. 2020, Section 5.1](https://arxiv.org/pdf/2007.12671)).

**Two lines of algebra connect these papers, and that should be the preprint's first sentence about Result A.** Substitute the first-order loss eᵢ = aᵢ + β̄_T into Bengio and Grandvalet's within-block covariance ω and between-block covariance γ, and use n − m = (K−1)m. This gives **λ₂ = Var(a − β/(K−1)) and λ₃ = Var(a + β)**. The second identity is the Austern–Zhou and Li variance. The first is the quantity behind Proposition 8(ii), and no source computes it. I re-derived both identities from the eigenvalue formulas while writing this report, and they hold. With one partition the expected NB variance is (2K−1)/(K−1)·λ₂/n, which is Proposition 8(ii) at R = 1. Described honestly, Proposition 8(ii) is the first-order value of a known exact quantity, extended to R partitions through Proposition 6 and to clusters as units, and compared with the cluster-robust variance rather than with Var(CV).

**The same algebra explains Markatou's factor of about 2.** Markatou et al. found that the NB correction on a single K-fold run was about twice their moment-approximation variance, for a sample mean under squared loss ([Markatou et al. 2005, Table 6, pp. 1147–1149](https://www.jmlr.org/papers/volume6/markatou05a/markatou05a.pdf)). In that setting β = 0, so the prediction is (2K−1)/(K−1). Their ratios are close to it at K = 10 and for LOOCV, but fall 4–15% short at K = 4 and 5. Their denominator is a moment approximation, and the table does not say how many replications it averages, so I reran the setting with a Monte Carlo variance. The n = 1000 ratios fall within one SE of the prediction at every K. The n = 100 ratios fall 1–3% below it, which is a second-order effect. So Markatou's shortfall at K = 4 comes from their approximate denominator or from Monte Carlo noise, not from the theory. Their table can be cited as an independent, if rough, published confirmation, with the team's own simulation giving the exact check.

| K | Predicted (2K−1)/(K−1) | Markatou NB/MA, n = 100 | Simulation, n = 100 (SE) | Markatou NB/MA, n = 1000 | Simulation, n = 1000 (SE) |
| --- | --- | --- | --- | --- | --- |
| 4 | 2.333 | 1.99 | 2.267 (0.005) | 2.12 | 2.321 (0.014) |
| 5 | 2.250 | 2.16 | 2.192 (0.007) | 2.06 | 2.240 (0.013) |
| 10 | 2.111 | 2.12 | 2.085 (0.007) | 2.03 | 2.107 (0.016) |
| LOOCV | 2.010 / 2.001 | 1.99 | 1.982 (0.011) | 1.99 | 1.989 (0.024) |

The simulation (N(0,1) data, sample mean, squared loss, one random K-fold partition per data set, 20 batches, 10,000 to 200,000 data sets per cell, seeds 20261106 plus the design index) is in `/private/tmp/claude-501/-Users-abrahamyeung-projects-Math-195-Project/6f0764a0-a8c9-434c-98f8-6ee470a2ce37/scratchpad/markatou_check/nb_mean_se.py`. It is a scratch check: move it into `scripts/` with a test before citing it.

**Proposition 6 is only partly new.** Its J = K case is the random-group method of survey sampling. Wolter's Chapter 2 is the standard source, but only its table of contents was seen, so the "whole clusters" rule that proposals/06 and /08 attribute to it is UNVERIFIED ([Wolter 2007, catalogue record](https://find.library.upatras.gr/Record/978-0-387-35099-8/Details)). The searches did not find:

- the exact factor for R repeated partitions;
- the statement that NB equals the cluster-robust variance on grouped folds but the naive variance on row folds;
- the consequence that the corrected t handles clustering only when the folds are grouped.

The survey literature on averaging over several random-group divisions, and Kott's delete-a-group jackknife, were not searched. They are the likeliest places for an R > 1 analogue.

**The question is still open in applied work.** A June 2026 neuroimaging preprint runs NB on family-aware, 20-times-repeated 10-fold CV. It asserts, without derivation, that the correction is conservative because its 1/(k−1) term dominates ([Marraffini et al. 2026, Appendix B](https://arxiv.org/pdf/2606.04010)). Proposition 6 shows the opposite: that term is exactly what converts the spread of fold means into the variance of the overall mean. The excess comes from the β terms and from the finite-J factor. A 2025 benchmark likewise calls the NB factor a heuristic ([Schulz-Kümpel et al., Section 4.5](https://arxiv.org/pdf/2409.18836)). The literature on grouped, blocked and spatial CV offers no variance estimator at all:

- a full-text search of Roberts et al. found none for block CV ([Roberts et al. 2017](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf));
- Rabinowicz and Rosset's abstract describes a bias correction only; just the abstract was read ([Rabinowicz & Rosset 2022](https://ideas.repec.org/a/taf/jnlasa/v117y2022i538p718-731.html)).

One weakness must travel with any claim. Proposition 8(i)'s variance-ratio tests (N8, N8b) failed at the registered precision, so the preprint has to report them next to the NB successes.

| Component of Result A | Verdict | Closest prior work (verification) |
| --- | --- | --- |
| First-order variance of CV, Var(a+β) | Known | Austern & Zhou Thm 3 (arXiv v2 opened; AIHP 61(4) 2025 version Crossref-verified only); Li 2023 Thm 3.1 (opened) |
| d = 0 under the training loss (Prop. 8(iv) mechanism) | Known | Austern & Zhou Remark 6 (opened) |
| Exact K-fold eigenstructure; λ₂ is what NB averages | Known | Bengio & Grandvalet 2004 Lemma 7 (opened) |
| λ₂ = Var(a−β/(K−1)) to first order; expected NB on grouped folds; factor and sign condition | Plausibly new | None found |
| (J+K−1)/(J−1) for R partitions; grouped versus row reading | Plausibly new; J = K case known | Wolter Ch. 2 (UNVERIFIED, table of contents only); Nadeau & Bengio 1999 (opened); journal version 2003 (not opened) |
| Factor of about 2 for single-run K-fold | Observed empirically; explained here | Markatou et al. 2005 Table 6 (opened) |
| Out-of-sample prediction at K = 3 and 10 | Team's own | None found |

## Pooled-AUC bias has been known since 1997; its clustered 1/G law has not

**The phenomenon, its mechanism and its remedies are all published, for independent rows.**

- **Phenomenon.** Bradley separated pooling, in which ROC points from several partitions are combined into one curve, from averaging, in which each partition's AUC is computed and the AUCs are averaged. He noted that "pooling the data in this way depresses the combined index of accuracy, AUC" ([Bradley 1997, p. 1150](https://math.ucdavis.edu/~saito/data/roc/bradley-auc.pdf)).
- **Mechanism.** Parker et al. gave both halves of what Proposition 9(iii) calls the mixing bias. First, a training set's class proportions are anti-correlated with its test set's, which shifts each fold model's prior or intercept; this is the analogue of C_mix. Second, noise between fold models pulls the pooled AUC toward 0.5; this is the analogue of c₂. Their no-signal pooled AUCs fell below 0.3 ([Parker et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2211509/)).
- **Calibration view.** Forman and Scholz illustrated the same failure as a lack of calibration across folds: fold AUCs averaging 92% against 80% pooled. They also documented that WEKA's Explorer pooled by default ([Forman & Scholz 2010, Sections 2.4 and 3.2](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf)).
- **Intercept view.** Montoya Perez et al. traced the bias to the constant term of the prediction function ([Montoya Perez et al. 2019](https://arxiv.org/abs/1801.09386)).
- **Recent restatement.** Austin, Pe'er and Korem named the mechanism "distributional bias" and proposed rebalanced leave-one-out CV ([Austin et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12662204/)).
- **Remedies.** Leave-pair-out CV ([Airola et al. 2010](https://proceedings.mlr.press/v8/airola10a.html); [Airola et al. 2011](https://www.utupub.fi/items/639ce106-68e0-4b27-be21-2fe77dba9dfb/full)) and Parker's balanced stratified CV.

None of these papers gives a closed form for the bias.

**Two papers come closest on dependent data.**

- **Airola et al. (2019), spatial leave-pair-out CV.** It compares only predictions made in the same CV round and excludes neighbours of either test point. The pooling bias grew with the radius of the excluded neighbourhood: at 30 km, leave-pair-out gave 0.84 against 0.70 for leave-one-out ([Airola et al. 2019, Sections 2.4 and 4.1](https://www.utupub.fi/items/6ca8f721-0d1b-4b82-9b51-e1cf531f2365/full)). This is the only empirical hint that holding out dependence-respecting blocks enlarges the bias, and it gives no rate.
- **LeDell, Petersen and van der Laan (2015).** They treat ids as the independent units, keep each id in one fold, average the within-fold AUCs, and derive an influence curve summed within ids ([LeDell et al. 2015, Section 5, Theorem 5.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4533123/)). Their cvAUC package exposes this as `ci.pooled.cvAUC` ([cvAUC manual](https://cran.r-project.org/web/packages/cvAUC/cvAUC.pdf)). Their target is the average of the fold models' true AUCs, which depends on the data, not the full model's AUC on new clusters.

So Proposition 9(i) is LeDell's theorem with clusters as units, extended in two directions: to the pooled and BCWF estimators, and to the full-model target. This is the same step that separates Proposition 8 from Bayle et al. Read against their definitions, LeDell's estimator includes same-id pairs while their target compares independent observations. Under their asymptotics that same-cluster term falls below first order, which is why they do not discuss it.

The within-cluster versus between-cluster concordance (A_w against A_b) is established outside CV ([van Klaveren et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3897966/); [Van Oirbeek & Lesaffre 2012](https://doi.org/10.1016/j.csda.2011.11.023), metadata only). So is the clustered AUC variance ([Obuchowski 1997](https://doi.org/10.2307/2533958), metadata only).

**What the searches did not find** is the quantitative clustered theory:

- the bias has **order 1/G rather than 1/n**;
- its **leading term C_mix/(G−1) does not depend on K**, so adding folds does not remove it;
- the next term is c₂/((K−1)(G−1));
- both constants can be computed from the population;
- the **same-cluster-pair bias** is Kκ(A_w−A_b)/G for the fold-averaged estimate and κ(A_w−A_b)/G for the pooled one, and the cvAUC default inherits it;
- **BCWF-pooled** exists as a named estimator, with coverage of **0.848 to 0.938 at nominal 0.90** over the 18 cells with G ≥ 40;
- the NB ratio for fold AUCs is about 1;
- Proposition 10's identity c₂ = C_mix/2 for independent rows.

BCWF itself is a natural recombination of three known ideas: leave-pair-out CV, spatial leave-pair-out CV and between-cluster concordance. Claim its name and its evaluation, not its construction.

**Result B has more exposed weaknesses than Result A.**

- **No remainder bound.** Proposition 9(iii) is a formal second-order expansion without a remainder bound.
- **A failed registered test.** N18 failed at K = 2 under strong clustering: −0.917 observed against −1.127 predicted, an 18.6% miss against a 15% band. An exploratory rerun cut the shortfall to 5.8%.
- **An identity that does not survive clustering.** c₂ = C_mix/2 is false in general once rows are clustered: the reviewer's counterexamples give 0.454 and 0.422.
- **A small real-data effect.** On the real data, pooled AUC minus BCWF-pooled averaged −0.004 to +0.002 over 20 random event partitions.
- **A missing citation.** Proposition 9(ii) still needs a citation for d = 0 under correct specification (Neyman–Pearson optimality of the true risk score).
- **A sentence prior work contradicts.** proposals/06 says the independent-rows bias is "of order 1/n and negligible". Parker's no-signal pooled AUCs below 0.3, and Austin's logistic regression pushed below 0.5 on random data, show it is severe with few rows or a weak signal.
- **A tension with prior work over K.** Proposition 10's factor (2K−1)/(2(K−1)) is largest at K = 2. Austin et al. instead report that the effect weakens as more points are held out, with leave-one-out the worst case. The likely reason is that the severe cases are the no-signal regime, where the density of D at 0 blows up and the expansion does not apply. The preprint should say so explicitly.

| Component of Result B | Verdict | Closest prior work (verification) |
| --- | --- | --- |
| Pooled CV AUC biased low (independent rows) | Known | Bradley 1997; Parker 2007; Forman & Scholz 2010; Airola 2010, 2011; Montoya Perez 2019; Austin 2025 (all opened) |
| Two-part mechanism (prior or intercept shift, plus inter-fold noise) | Known qualitatively, no formula | Parker 2007 (opened) |
| Prop. 9(i): first order; clustered variance valid | Partly known | LeDell 2015 Thm 5.1 (opened); Obuchowski 1997 (metadata only) |
| Prop. 9(iii): mixing bias of order 1/G, K-free leading term | Plausibly new | Airola 2019 empirical hint only (opened) |
| Same-cluster-pair bias of fold-averaged AUC | Plausibly new as a CV result | Concepts in van Klaveren 2014 (opened), Van Oirbeek & Lesaffre 2012 (metadata) |
| BCWF-pooled with Obuchowski SE and coverage study | Construction partly known; evaluation not found | Airola 2010, 2019 (opened) |
| Prop. 10, c₂ = C_mix/2 and the closed-form bias | Plausibly new | None found |
| Prop. 9(ii), NB ratio for fold AUCs | Not found | None |

## Unequal clusters reuse classical variance theory; only the fold normalization looks new

**Most of Proposition 11 is textbook survey and econometric theory.**

- **CR1 leverage bias.** Bell and McCaffrey's Theorem 1 shows that the CR1 cluster-robust variance is biased low unless a leverage-like quantity is constant across clusters. For an intercept-only model, the pooled CV estimate, that quantity is m_h/n, so CR1 is biased low exactly when cluster sizes differ. Their eq. (3), specialized to an intercept with block-diagonal errors, is the first factor of the project's closed form 2 ([Bell & McCaffrey 2002, pp. 170–171](https://www150.statcan.gc.ca/n1/en/pub/12-001-x/2002002/article/9058-eng.pdf)). Computed from the 21 real event sizes in `data/pairs.csv`, with cluster variances proportional to m_h², that factor is **0.945**, which matches the project's figure exactly.
- **The same factor, seen from the cluster side.** MacKinnon and Webb give the per-cluster factor (M−N₁)/M. Even with no intra-cluster correlation, 50 clusters sized like US states rejected about 6% of the time at the 5% level ([MacKinnon & Webb, working paper, Sections 4 and 6](https://ageconsearch.umn.edu/record/274639/files/qed_wp_1314.pdf)). MacKinnon, Nielsen and Webb define cluster leverage, which equals N_g/N for an intercept, and note that CR2 is unbiased under errors with identity-proportional covariance ([MNW 2023, Sections 2.2 and 3.4](https://arxiv.org/abs/2205.03285)).
- **Consistency and asymptotic normality.** Carter, Schnepel and Steigerwald prove the cluster-robust t stays asymptotically Gaussian without equal sizes, and define the effective number of clusters G* ([Carter et al., working paper, Section 3](https://faculty.econ.ucsb.edu/~doug/researchpapers/Asymptotic%20Behavior%20of%20a%20t%20Test%20Robust%20to%20Cluster%20Heterogeneity.pdf)). Hansen and Lee supply asymptotics with heterogeneous cluster sizes ([Hansen & Lee 2019](https://arxiv.org/abs/1902.01497)). With the real sizes, G* is **13.8 if the cluster variances scale with m_h and 9.0 if they scale with m_h²**, against G = 21. Both figures were recomputed for this report. This is a ready-made reason to use t on G*−1.
- **Design effect with unequal sizes.** Kish's design effect for unequal sizes, 1 + {(CV²+1)m̄ − 1}ρ, has (CV²+1)m̄ = E[m²]/E[m] = m̃ ([Rutterford, Copas & Eldridge 2015, eq. 17](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4521133/fullTextXML)). For the events m̃ is 302.5 against m̄ = 199.2.
- **Unread originals.** Kish (1965) and Cochran (1977) could not be opened, so their exact sections are UNVERIFIED.

**The weight ω_h has a precursor.**

- **Austern and Zhou.** Their variance proof contains a point-level analogue of ω_h: a sum over folds of block size times a training-size-indexed stability term. They remark that it simplifies only with equal blocks. But their blocks differ by at most one point, so the deviation is negligible and they bound it away ([Austern & Zhou, arXiv v2, eqs. 165–169, p. 39](https://arxiv.org/abs/2001.11111)).
- **Bayle et al.** Their Theorem 8 allows general fold sizes, but centres on the fold models' average risk, so no ω_h arises ([Bayle et al., Appendix D](https://arxiv.org/abs/2007.12671)).
- **Arlot and Lerasle.** They normalize each training set by its own size, but restrict attention to regular partitions ([Arlot & Lerasle 2016](https://jmlr.org/papers/v17/14-296.html)).

No source treats fold shares that fluctuate at O_p(G^−1/2), as they do when whole clusters of unequal size are dealt at random, or gives the expansion ω_h = 1 − K²/(K−1)²·ε_j. With the real sizes, max|ω_h − 1| has median 0.17 at K = 5, which is not a technicality. The honest framing: the weight appears implicitly in Austern and Zhou for blocks that differ by one point; the team gives it explicitly for cluster-sized imbalance.

**The Nadeau–Bengio normalization is the piece the searches did not find anywhere.**

- **Every CV variance paper assumes one training size and one test size**, or equal blocks: Nadeau and Bengio ([CIRANO 99s-25, Section 1](https://camirang.cirano.qc.ca/files/publications/99s-25.pdf)), Bouckaert and Frank, Bengio and Grandvalet (footnote 3), and Bates et al.
- **Software weights by fold size only in the point estimate.** It computes each fold's mean over its own rows. scikit-learn's old `iid=True` weighted fold means by test size, but only for the point estimate, and was removed in 0.24 ([GridSearchCV 0.22](https://scikit-learn.org/0.22/modules/generated/sklearn.model_selection.GridSearchCV.html)). mlr3 offers `macro_weighted` and `micro` for the point estimate too ([mlr3 Measure](https://mlr3.mlr-org.com/reference/Measure.html)).
- **The fix is not in print.** Dividing fold residual totals by n/K before computing the NB variance, so that a partition's fold values average to the pooled estimate, was found nowhere.
- **The regime is common in practice.** In scikit-learn, `GroupKFold(shuffle=True)` balances the number of groups per fold, not the number of rows, as the user guide and the source agree. The API page contradicts both ([scikit-learn source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_split.py); [user guide](https://scikit-learn.org/stable/modules/cross_validation.html)). Anyone who repeats grouped partitions for the corrected t is therefore in exactly the regime where the normalization term arises.
- **The closest survey analogue is unread.** That is the delete-m jackknife for unequal m of Busing, Meijer and van der Leeden (1999). It was seen only as a search record and is UNVERIFIED, as is the classical ratio-estimator linearization of Woodruff (1971), seen as metadata only. Read Busing et al. before claiming the normalization result.

**Proposition 11(e) restates a known distinction in CV terms.** The informative-cluster-size literature separates the "all members" (row-weighted) target from the "typical member of a typical cluster" (cluster-weighted) target, and shows that unweighted estimating equations target the former ([Seaman, Pavlou & Copas 2014, Section 3.1](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4312901/fullTextXML); [Williamson, Datta & Satten 2003](https://doi.org/10.1111/1541-0420.00005), abstract only). None of these papers mentions cross-validation. Proposition 11(e) says that d = 0 for the row-weighted target even under misspecification, and that d ≠ 0 in general for the cluster-weighted one. It is therefore a combination of known pieces in a new setting.

**The empirical side is the weakest of the three results.**

- **The heuristic P2.** P2 omits terms of the same order as some it keeps.
- **Results near band edges.** Three of the four registered tests landed near their band edges.
- **K = 3.** It is not predicted well.
- **CR1 under log-loss.** Its mean ran about 17% low at G = 21.
- **The real data.** The normalization effect seen in the A2 designs did not carry over to the real data. The per-data-set plug-in was unreliable: in-sample residual totals carried only 28–53% of the cross-validated residual sum of squares.

The applied payoff of C is therefore a recommendation (residual totals over n/K; CR2 or t on G*−1), not a measured real-data correction.

| Component of Result C | Verdict | Closest prior work (verification) |
| --- | --- | --- |
| (a) CV − Err_new is the mean of cluster residual totals | Partly known | Bayle 2020 Thm 8 (opened); Wager 2020 (opened; equal folds) |
| ω_h and its expansion | Plausibly new as stated; structural precursor | Austern & Zhou eqs. 165–169 (opened) |
| (b) Ratio-estimator variance; Kish design effect with m̃ | Known | Rutterford 2015 eq. 17 (opened); Kish, Cochran (UNVERIFIED) |
| Unbalanced CRVE consistent; Gaussian t | Known | Carter et al. 2017; Hansen & Lee 2019 (working papers opened) |
| CR1 leverage bias (closed form 2, first factor) | Known, direct specialization | Bell & McCaffrey 2002 eq. 3, Thm 1 (opened); MacKinnon & Webb 2017; MNW 2023 |
| (c) NB limit with unequal folds, in expectation | Not found | NB, Bouckaert & Frank, Bengio & Grandvalet all assume equal sizes |
| Own-row normalization term; residual totals over n/K | Not found in CV literature | Busing et al. 1999 (UNVERIFIED, unread) is the closest survey analogue |
| (e) d = 0 for the row-weighted target | Ingredients known; CV statement not found | Seaman 2014 (opened); Williamson 2003 (abstract) |
| P2 and measured order-1/G terms at G = 21 | Not found | Bell & McCaffrey note the CR1 bias is O(1/G) |

## Practitioners who already group their folds have no error-bar method

**The lesson about point estimates is already learned, so Lemma 1 adds nothing new for practitioners.**

- **Digital health.** In a review of 62 smartphone and wearable papers, 28 used record-wise CV. Their median error was 5.6%, against 13.0% for subject-wise papers ([Saeb et al. 2017](https://doi.org/10.1093/gigascience/gix019)).
- **EEG.** Only 17 of 63 translational deep-learning EEG papers clearly avoided subject leakage. In the authors' own experiment, accuracy was 99.8% with segment-level holdout against 53.0% with subject-level holdout ([Brookshire et al. 2024](https://doi.org/10.3389/fnins.2024.1373515)).
- **Across fields.** Leakage affected 294 papers in 17 fields ([Kapoor & Narayanan 2023](https://doi.org/10.1016/j.patter.2023.100804)).

**The uncertainty step is where practice is thin.**

- **Medical imaging.** Only 1 of 221 MICCAI 2023 segmentation papers reported confidence intervals ([Christodoulou et al. 2024](https://arxiv.org/abs/2409.17763)).
- **Education.** None of 87 MOOC prediction experiments reported fold-level variability ([Gardner & Brooks 2018](https://doi.org/10.18608/jla.2018.52.7)).
- **Clinical prediction.** Only 5.9% of 152 ML prediction-model studies reported predictive performance completely ([Andaur Navarro et al. 2022](https://doi.org/10.1186/s12874-021-01469-6)).
- **Neuroimaging.** In Varoquaux's simulation, standard-error-of-the-mean bounds across folds were too small by a factor of 0.73 for leave-one-out and 0.26 for repeated splits ([Varoquaux 2018](https://doi.org/10.1016/j.neuroimage.2017.06.061)).

**Guidance tells authors to report intervals but not how to compute them for a grouped-CV metric.**

- **TRIPOD+AI** asks for performance with confidence intervals (item 23a). It never uses the words cross-validation, fold or pooling, and it refers clustered studies to TRIPOD-Cluster ([Collins et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/)).
- **TRIPOD-Cluster** recommends internal–external CV with a random-effects meta-analysis of per-cluster c statistics ([Debray et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9903176/)). That suits a few large clusters, and it targets the within-cluster AUC A_w, not the between-cluster AUC that BCWF estimates. A preprint must say so, because this is the main competing recommendation.
- **PROBAST+AI** never uses the word "cluster" ([Moons et al. 2025](https://doi.org/10.1136/bmj-2024-082505)).

**Software makes both aggregations easy and offers no clustered error bars.**

- **scikit-learn.** `cross_val_predict` warns that passing its predictions to a metric may not be valid. But the reasons it gives are unequal test sizes and non-decomposable metrics, not calibration that shifts between folds ([scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)).
- **mlr3.** Pooled ("micro") averaging is one setting away ([mlr3 Measure](https://mlr3.mlr-org.com/reference/Measure.html)).
- **mlr3inferr 0.2.2.** It packages corrected-t, Wald-CV and nested-CV intervals, and its documentation says nothing about groups ([CRAN mlr3inferr](https://cran.r-project.org/web/packages/mlr3inferr/index.html)).
- **pROC.** It gives DeLong and bootstrap intervals over rows, with no cluster argument ([pROC manual](https://cran.r-project.org/web/packages/pROC/pROC.pdf)).
- **The exception is cvAUC**, whose cluster-aware interval dates from 2015 ([cvAUC manual](https://cran.r-project.org/web/packages/cvAUC/cvAUC.pdf)).
- **Unreported aggregation.** In student modelling the AUC is computed globally, per skill or per student, papers rarely say which, and part of one celebrated improvement came from that choice ([Pelánek 2017](https://doi.org/10.1145/3099023.3099042)).

**Each result has a definite audience.**

- **Result A** serves anyone who runs the corrected t on grouped folds: mlr3inferr's `ci.cor_t` users, and neuroimaging pipelines that hold out families as in Marraffini et al. Its message has four parts:
  - grouped folds are necessary;
  - on them the correction is conservative by a predictable factor;
  - that factor is about 1 for log-loss and the AUC and about 1.2 for 0–1 error at K = 5 in the designs tested;
  - a single run overstates the variance by (2K−1)/(K−1), or 2.25 at K = 5.
- **Result B** serves anyone who calls `roc_auc_score` on `cross_val_predict` output with `GroupKFold`, or uses mlr3 micro averaging, when there are few clusters and their outcome rates differ: hospitals with different prevalence, students with different success rates. Its message: more folds will not remove the leading bias. In the worst simulated cell (G = 20, strong outcome clustering), the pooled interval covered 0.705, BCWF-pooled covered 0.910, and the wild cluster bootstrap did not rescue the pooled interval (0.695).
- **Result C** serves anyone who repeats `GroupKFold(shuffle=True)` with heterogeneous cluster sizes. Its message is a one-line normalization of the NB fold values, plus the already-known advice to use CR2 or t on G*−1.

## Claim the quantification, cite the ingredients, and lead with Result A

**A defensible novelty sentence has three features.**

- **It states the search scope.** Write "we found no statement of X in a search of Y". The searches covered papers, package code, Crossref, Europe PMC and arXiv; they did not cover Google Scholar cited-by lists, Scopus, Web of Science or theses, and several classic books were not opened.
- **It names the ingredient papers at the point of claim**, not in a related-work section three pages later.
- **It labels each result by its status.** First-order derivations, exact identities and formal expansions are kept apart: Proposition 6 is exact; Propositions 8, 9(i) and 11(a)–(e) are first-order derivations; Proposition 9(iii) is a heuristic second-order expansion; P2 is a heuristic.

**Disclose the rest of the record.** The preprint should also report every pre-registered failure, including N1, N5, N8, N8b and N18. It should disclose that the reviews so far were automated adversarial reviews, and that Jenna Yang's review is still pending. It should say plainly that the real-data effects at G = 21 are small or not reliably measured.

**Suggested one-sentence framings:**

- **Result A:** "We evaluate Bengio and Grandvalet's between-block eigenvalue to first order with clusters as units. This gives the expected Nadeau–Bengio variance on grouped folds, a conservativeness factor that equals one under the training loss, and an explanation of Markatou et al.'s factor of two."
- **Result B:** "We quantify, for grouped folds, the pooling bias that Parker et al. described for independent rows: it is of order 1/G with a leading term that does not depend on K, and an estimator that compares only within-fold, between-cluster pairs removes it."
- **Result C:** "With unequal clusters the classical cluster-robust theory carries over to the pooled CV estimate, but the Nadeau–Bengio fold values should be residual totals divided by n/K rather than each fold's own mean."

**Recommendation: make Result A the core**, meaning Proposition 6, Proposition 8(ii)–(iv), the out-of-sample K = 3 and 10 test, and Proposition 7's caution for coverage studies as a companion. Five reasons:

- It pairs an exact identity with a first-order result that has a short derivation.
- It passed a registered out-of-sample prediction.
- It explains a published 2005 observation and corrects a 2026 applied misreading.
- It is directly actionable for tools such as mlr3inferr.
- Its prior-art risk is narrowly bounded: the Nadeau–Bengio 2003 journal version, Wolter's Chapter 2, and a few cited-by lists.

**Fold Result C's NB normalization in as an extension section.** It uses the same NB machinery, and its new part is a practical fix. Cite its classical parts rather than claiming them.

**Publish Result B separately, later.** Its applied payoff is arguably the largest, because the AUC dominates medical ML. But its main term rests on a formal expansion with a failed registered test, its qualitative mechanism is well trodden, the original Part 9 literature search was the briefest of the three, and it still needs the Neyman–Pearson citation plus cited-by checks of Airola 2011, Airola 2019 and LeDell 2015.

**The strongest case for leading with B instead:** if the team wants reach rather than rigour, a short applied note on BCWF-pooled and the K-free result, with the expansion clearly labelled a heuristic, would reach more readers than Result A will.

## Concrete edits to "What is new and what is known"

The table lists each edit to `proposals/08-cv-error-bars-note.tex`. Lines 712–735 are the section; the other line numbers point elsewhere in the file. The LaTeX that follows is a drop-in replacement for the section's itemized lists, using the file's existing macros.

| # | Where | Current text (paraphrased) | Edit | Why |
| --- | --- | --- | --- | --- |
| 1 | Line 714 | One-line scope note naming proposals/07 | State both checks, what was and was not searched, and that "not found" means not found in that search | Honest scope |
| 2 | "Known", new first bullet | Nothing on the NB or K-fold variance literature | Add Nadeau & Bengio's fixed-loss surrogate and caveat, Bouckaert & Frank, Bengio & Grandvalet Lemma 7, Austern & Zhou Thm 3 and Remark 6, Li Thm 3.1, Markatou Table 6 | These are A's ingredients |
| 3 | Line 718 | Wolter Ch. 2 "including the rule that groups be formed from whole clusters" | Keep the J = K attribution; drop or flag the whole-cluster rule until Ch. 2 is read | Chapter text UNVERIFIED |
| 4 | Line 723 | Pooling bias cited to Parker, Forman, Airola 2010, 2011 | Add Bradley 1997 as the origin, Montoya Perez and Austin; describe Forman as an illustration; add LeDell as partly covering Prop. 9(i); add van Klaveren and Van Oirbeek & Lesaffre for A_w and A_b | Precise credit |
| 5 | Line 724 | Unequal-cluster classics: Cochran, Kish, BM, MW | Add Bell & McCaffrey eq. 3 as closed form 2's first factor, MNW, Carter et al., Hansen & Lee, Rutterford eq. 17, G* values, Austern & Zhou's precursor of ω_h, and the informative-cluster-size papers | Most of Prop. 11 is classical |
| 6 | Line 730 | "The Nadeau–Bengio limit … of Proposition 8(ii)" | Reword as the first-order value of Bengio & Grandvalet's λ₂, with the sign condition, the factor-1 case from d = 0, and the Markatou explanation | Accurate novelty claim |
| 7 | Line 729 | R-partition factor and grouped-versus-row reading | Keep, and add the Marraffini contrast | Shows the question is open in practice |
| 8 | Line 732 | "a leading term C_mix/G … next term c₂/((K−1)G)" | Change to C_mix/(G−1) and c₂/((K−1)(G−1)), matching the corrected Prop. 9; call (iii) a formal expansion; mention N18; say c₂ = C_mix/2 is false in general under clustering | Consistency with the review |
| 9 | Line 733 | "these await a literature check" | Replace with specific claims, the Austern & Zhou precursor, and the Busing et al. caveat | The check is now done |
| 10 | Line 507 (Part 9 theory remark) | "The mixing bias is the pooling bias of Parker, Forman seen through clustering" | "…the clustered form of the pooling bias whose mechanism Parker et al. describe for independent rows (see also Bradley; Forman and Scholz; Austin et al.)" | Credit the mechanism to Parker |
| 11 | proposals/06 lines 485–486 | "With independent rows it is of order 1/n and negligible" | "…of order 1/n, small with many rows and a clear signal but severe with few rows or weak signal, especially under leave-one-out (Parker et al. 2007; Austin et al. 2025)" | Contradicted by prior work as written |
| 12 | Line 485 (Prop. 9(ii)) | d = 0 "under correct specification (Neyman–Pearson)" | Add a citation for the AUC-optimality of the true risk score | Uncited claim |

```latex
\section{What is new and what is known}\label{sec:new}

Two literature checks (\texttt{proposals/07}, October 2026, and a second on
5 November 2026) read the works below in full unless marked, and searched
package documentation and source code, Crossref, Europe PMC and arXiv. They did
not search Google Scholar cited-by lists, Scopus, Web of Science or theses, and
could not open Kish, Cochran, Wolter's Chapter~2 or the journal version of
Nadeau and Bengio. ``Not found'' below means not found in that search.

\emph{Known.}
\begin{itemize}
\item \emph{The variance of $K$-fold CV.} Nadeau and Bengio derived
$\rho_0=n_2/n$ for independent random splits by treating each loss as fixed,
and noted that the correction can err in either direction \cite{NB};
Bouckaert and Frank applied it to repeated $K$-fold CV by analogy \cite{BF}.
Bengio and Grandvalet give the exact eigenstructure of the $K$-fold error
covariance \cite[Lemma~7]{BG}: the variance of one partition's fold means
estimates the between-block eigenvalue $\lambda_2$, while
$\Var(\CV)=\lambda_3/n$. To first order $\lambda_3=\Var(a+\beta)$
\cite[Theorem~3]{AZ}, \cite[Theorem~3.1]{Li}, and the covariance term vanishes
when the evaluation loss is the training loss \cite[Remark~6]{AZ}, which is our
$d=0$. Markatou et al.\ found the correction on one $K$-fold run about twice
the true variance for a sample mean \cite[Table~6]{Markatou}.
\item Proposition~6 at $J=K$ is the random group method of survey sampling
\cite[Chapter~2]{Wolter}.
\item Applied with clusters as units, the CV central limit theorem of Bayle
et al.~\cite{BBJM} gives Proposition~8(i) for the average of the fold models'
errors, including 0--1 loss; their Wald interval with events as units is the
cluster-robust interval without $G/(G-1)$.
\item Wager \cite{Wager} overlaps Proposition~8(i) and~(iv): CV's leading
error term does not depend on the model, and the conditional risk moves at
lower order.
\item Random-row CV is optimistic under grouping \cite{Roberts,Saeb,RR,PM},
and naive CV standard errors are too small \cite{Varoquaux,BG,NB}; the
seen-against-new-cluster distinction of Lemma~1 is in \cite{RR}, without
intervals. We found no paper that runs the nested CV of \cite{BHT} on
clustered data and reports coverage, so A2 and N2 are a small confirming
demonstration.
\item Propositions~1 to~3 are clusters-as-units extensions of Theorem~3 of
\cite{BHT}, of \cite{BG} and of \cite{BBJM}, not new theory; the design effect
is Kish's \cite{Kish}. Grouped CV with influence curves summed within clusters
\cite{LeDell}, resampling whole subjects \cite{Cai} and internal--external CV
\cite{deJong} are published. Proposition~7 is the classical design-based
against model-based distinction.
\item \emph{The pooled CV AUC.} Pooling held-out scores from different fold
models lowers the AUC \cite{Bradley}. For independent rows the mechanism,
a shift in each fold model's prior or intercept opposite to its test fold's
composition plus noise between fold models, is described in words by Parker
et al.\ \cite{Parker}, with the intercept view in \cite{Montoya} and a recent
restatement in \cite{Austin}; Forman and Scholz \cite{Forman} illustrate it as
a failure of calibration across folds. The remedies are averaging fold AUCs,
leave-pair-out CV \cite{Airola2010,Airola2011,Montoya} and rebalanced training
sets \cite{Parker,Austin}. On dependent data, spatial leave-pair-out CV
\cite{Airola2019} removes the pooling bias and reports that it grows with the
excluded neighbourhood; BCWF is its grouped $K$-fold analogue. With clusters
as units, LeDell et al.\ \cite[Section~5, Theorem~5.1]{LeDell} give the
influence-curve variance of the fold-averaged AUC for the average of the fold
models' AUCs, which is Proposition~9(i) for that estimator and target. The
clustered placement-value variance is Obuchowski's \cite{Obuchowski}, and the
within- against between-cluster concordance is in \cite{VOL,vanKlaveren}.
\item \emph{Unequal clusters.} The ratio-estimator form of
Proposition~11(b) \cite{Cochran}; Kish's design effect with
$\tilde m=\E[m^2]/\E[m]$, in the form $1+\{(\mathrm{CV}^2+1)\bar m-1\}\rho$
\cite[eq.~17]{Rutterford}; the asymptotic normality of the unbalanced
cluster-robust $t$ \cite{CSS,HL}; the leverage bias of CR1, whose
intercept-only case is the first factor of closed form~2
\cite[eq.~3 and Theorem~1]{BM}, \cite[Section~6]{MW}, \cite[Section~3.4]{MNW},
and its removal by CR2 under working independence; and the effective number of
clusters $G^*$ \cite{CSS}, 13.8 or 9.0 for the real events as the cluster
variances scale with $m_h$ or $m_h^2$. A point-level form of the weight
$\omega_h$ appears in the variance proof of \cite[eqs.~165--169]{AZ} for blocks
that differ by at most one row, where it is negligible. The row-weighted and
cluster-weighted targets of Proposition~11 are the ``all members'' and
``typical member'' estimands of the informative-cluster-size literature
\cite{Williamson,Seaman}.
\end{itemize}

\emph{Not found, and plausibly new.}
\begin{itemize}
\item \emph{Proposition~8(ii).} The first-order value of Bengio and
Grandvalet's $\lambda_2$, which is $\Var(a-\beta/(K-1))$. With Proposition~6 it
gives the expected Nadeau--Bengio variance on grouped folds for $R$
partitions, the factor $\Var(a-\beta/(K-1))/\Var(a)$ relative to $\VCR$, the
condition $\Var(\beta)>2(K-1)\Cov(a,\beta)$ for conservativeness and, from
$d=0$, a factor of 1 under log-loss. It accounts for Markatou et al.'s factor of
about 2 as $(2K-1)/(K-1)$ with $\beta=0$. The out-of-sample test at $K=3$
and~$10$ (N14, N15) is ours.
\item \emph{Proposition~6.} The factor $(J+K-1)/(J-1)$ for $R$ repeated
partitions, and its reading for grouped against row folds: the corrected $t$
handles clustering only on grouped folds (N7). An applied paper that uses the
correction on grouped folds attributes its conservativeness to the $1/(K-1)$
term \cite{Marraffini}; by Proposition~6 that term is what turns the spread of
fold means into the variance of the overall mean, and the excess comes from the
$\beta$ terms and the finite-$J$ factor.
\item The cautions for coverage studies of CV error bars: redraw the clusters
(Proposition~7, A2 against N4), and score the target precisely
(Parts~6b and~7).
\item \emph{The AUC.} A quantification of the pooling bias under grouped folds:
order $1/G$ rather than $1/n$, a leading term $\Cmix/(G-1)$ that does not
depend on $K$, a next term $c_2/((K-1)(G-1))$, constants computable from the
population, and their test (N16 to N21; N18 was not supported at $K=2$).
Proposition~9(iii) is a formal expansion without a remainder bound. Also: the
same-cluster-pair bias of the fold-averaged and pooled estimates; BCWF-pooled
as a named estimator with a coverage study; the Nadeau--Bengio ratio for fold
AUCs; and Proposition~10's $c_2=\Cmix/2$ for independent rows, which is false
in general under clustering.
\item \emph{Unequal clusters.} The weight $\omega_h$ for cluster-sized
imbalance, with its expansion $1-\frac{K^2}{(K-1)^2}\varepsilon_{j(h)}$ and its
order; the Nadeau--Bengio limit with unequal folds, in expectation; the
own-row normalization term and its removal by fold residual totals divided by
$n/K$; Proposition~11(e) as a statement about CV targets; and the measurement
of the order-$1/G$ terms at $G=21$ with the heuristic P2. The fold-value device
resembles the linearization of a ratio estimator over random groups; we have
not yet read the delete-$m$ jackknife for unequal $m$ \cite{Busing}, its
closest survey analogue, and do not claim the normalization result until we
have.
\end{itemize}
We do not claim the cluster-nested, bias-adjusted interval that the literature
check first proposed as the contribution: N1 failed, and N5 showed its
under-coverage once events were redrawn.
```

**Bibliography additions**, in the file's short style. Each was verified as noted in the research notes.

```latex
\bibitem{AZ} Austern and Zhou, \emph{Asymptotics of cross-validation}, Ann.\ Inst.\ Henri Poincar\'e Probab.\ Statist.\ \textbf{61} (2025), no.~4, 2804--2865; arXiv:2001.11111 (v2 read). % DOI 10.1214/24-AIHP1488, Crossref-verified
\bibitem{Li} Li, \emph{Asymptotics of $K$-fold cross validation}, J.\ Artificial Intelligence Res.\ \textbf{78} (2023), 491--526.
\bibitem{Bradley} Bradley, Pattern Recognition \textbf{30} (1997), no.~7, 1145--1159. [Pooled against averaged CV AUC.]
\bibitem{Austin} Austin, Pe'er and Korem, Science Advances \textbf{11} (2025), no.~48, eadx6976.
\bibitem{vanKlaveren} van Klaveren, Steyerberg, Perel and Vergouwe, BMC Med.\ Res.\ Methodol.\ \textbf{14} (2014), 5.
\bibitem{VOL} Van Oirbeek and Lesaffre, Comput.\ Statist.\ Data Anal.\ \textbf{56} (2012), no.~6, 1966--1980. % metadata only
\bibitem{CSS} Carter, Schnepel and Steigerwald, Rev.\ Econ.\ Statist.\ \textbf{99} (2017), no.~4, 698--709.
\bibitem{HL} Hansen and Lee, J.\ Econometrics \textbf{210} (2019), no.~2, 268--290.
\bibitem{MNW} MacKinnon, Nielsen and Webb, J.\ Econometrics \textbf{232} (2023), no.~2, 272--299.
\bibitem{Rutterford} Rutterford, Copas and Eldridge, Int.\ J.\ Epidemiol.\ \textbf{44} (2015), no.~3, 1051--1067.
\bibitem{Seaman} Seaman, Pavlou and Copas, Biometrics \textbf{70} (2014), no.~2, 449--456.
\bibitem{Williamson} Williamson, Datta and Satten, Biometrics \textbf{59} (2003), no.~1, 36--42. % abstract only
\bibitem{Marraffini} Marraffini et al., arXiv:2606.04010, 2026.
\bibitem{Busing} Busing, Meijer and van der Leeden, Statistics and Computing \textbf{9} (1999), no.~1, 3--8. % not yet read
```

**Fixes to existing entries:**

- MW: add *J. Appl. Econometrics* 32 (2017), no. 2, 233–254.
- BM: add pp. 169–181.
- Markatou: add *JMLR* 6 (2005), 1127–1168.
- BF: add PAKDD 2004, LNCS 3056, 3–12.
- Obuchowski: add *Biometrics* 53 (1997), no. 2, 567–578.
- Airola2011: add 55 (2011), no. 4, 1828–1844.
- Airola2019: add 33 (2019), no. 3, 730–747.
- Wager: add *JASA* 115 (2020), no. 529, 157–160. This comes from a bibliographic record only.
- NB: add a comment that content attributions were checked against the 1999 NIPS and CIRANO versions, not the 2003 journal text.
- Wolter: the page range 21–106 is unverified.

## Twenty-one open checks stand between the team and a novelty claim

| # | Check | Protects | How | Priority |
| --- | --- | --- | --- | --- |
| 1 | Jenna Yang reviews Propositions 6, 8, 9, 11, including λ₂ = Var(a−β/(K−1)) (re-derived for this report, and it holds) | All | Mathematical lead's written sign-off | Blocking |
| 2 | Manual Google Scholar cited-by searches of Airola et al. 2011 and 2019, LeDell et al. 2015, Parker et al. 2007, Forman & Scholz 2010, filtered for cluster, grouped, subject, site, patient, "pooled AUC" | B | Scholar by hand; Scholar blocks automated access | High |
| 3 | Cited-by searches of Bengio & Grandvalet 2004, Markatou et al. 2005, Austern & Zhou, Bayle et al. 2020; a keyword-filtered search inside the Nadeau & Bengio 2003 citers (grouped, cluster, "expectation", "conservative") | A | Scholar and Web of Science | High |
| 4 | Read the Nadeau & Bengio 2003 *Machine Learning* text, Sections 4–5 | A | Library access | High |
| 5 | Read the cvAUC source: does `ci.pooled.cvAUC` use all within-fold pairs (including same-id pairs) or between-id pairs only, and how does it weight folds? | B (BCWF novelty; same-cluster-pair bias) | CRAN source tarball | High |
| 6 | Read Busing, Meijer & van der Leeden 1999, delete-m jackknife for unequal m | C (normalization, n/K) | Library access | High |
| 7 | Read Wolter 2007 Ch. 2: averaging over several random divisions (R > 1), the whole-cluster rule, the page range; also Kott's delete-a-group jackknife | A (Prop. 6) | Library access | High |
| 8 | Cite the AUC-optimality of the true risk score (Neyman–Pearson) for d = 0 | B (Prop. 9(ii)) | Standard ROC text or paper | Medium |
| 9 | Reconcile Prop. 10's K-dependence (bias largest at K = 2) with Austin et al. and Parker et al. (leave-one-out worst); state the regime | B | Short simulation in the weak-signal regime | Medium |
| 10 | Read Woodruff 1971 (ratio-estimator linearization) and the relevant sections of Kish 1965 and Cochran 1977 | C | Library access | Medium |
| 11 | Open Swets & Pickett 1982, Smith et al. 2014, Perlich & Świrszcz 2011, Numminen et al. 2023, and Rosner & Glynn's clustered-AUC papers | B | Library access | Medium |
| 12 | Open arXiv 2507.22299 (cluster-based CV strategies), 2607.02394 (patient-grouped CV with cluster bootstrap), and Pohjankukka et al. 2005.14263 | B, C | arXiv | Medium |
| 13 | Check Li 2023, Arlot & Celisse 2010, Kale et al. 2011, and the full text of Rabinowicz & Rosset 2022 for unequal folds or a CV variance under clustering | A, C | Full texts | Medium |
| 14 | Survey 2023–2026 spatial-CV methods (kNNDM/CAST, design-based map accuracy) for interval proposals | A | Targeted search | Medium |
| 15 | Move the Markatou check (above) into `scripts/` with a test against the known answer at n = 1000, and decide whether to cite Markatou's numbers or the team's | A | Project convention 7 | Low (done in scratch) |
| 16 | Read Ooi et al. 2022, the protocol Marraffini et al. follow, to see where NB on grouped folds entered neuroimaging practice | A (framing) | Full text | Low |
| 17 | Check whether mlr3inferr's `ci.cor_t` and `ci.ncv` honour mlr3's group column role | Audience claim | Package source | Low |
| 18 | Check whether pROC's `cov(method = "obuchowski")` handles clustering | B (framing) | Package source | Low |
| 19 | Confirm whether a Grandvalet–Bengio follow-up report on CV hypothesis testing (about 2006) exists | A | Search of authors' pages | Low |
| 20 | Check Dietterich 1998, Demšar 2006 and Corani & Benavoli 2015 for unequal folds | C | Full texts | Low |
| 21 | Page numbers from the journal versions of MacKinnon & Webb, Carter et al., Imbens & Kolesár, Bates et al. and Wager | All | Publisher PDFs | Low |

## Conclusion

The three results have one thing in common. Each new piece sits in the gap between what CV theory assumes and what applied grouped CV actually does. Theory targets the average risk of the fold models, independent rows and equal folds. Practice reports the full model's error on new clusters, with clusters as units and folds of unequal size. Proposition 8 goes from Bayle et al.'s fold-average target to the full model. Proposition 9(i) goes from LeDell's fold-average AUC target to the full model, and 9(iii) from Parker's 1/n bias to 1/G. Proposition 11 goes from Austern and Zhou's one-row imbalance to imbalance at the scale of whole clusters. Stated that way, the preprint's thesis is honest and hard to dismiss: it does not claim new phenomena, it claims the exact or first-order accounting at the point where practitioners actually operate.

Two cautions follow. First, the Markatou result shows that the published empirical record already contains confirmations of Result A, hidden in plain sight for two decades. That is good evidence for the result, and it also means someone may have written down the algebra in a place the searches did not reach, so the cited-by checks matter more for A than its clean status suggests. Second, the most useful output for practitioners is not any one theorem but three working rules:

- hold out whole clusters before applying any variance correction;
- divide fold residual totals by n/K;
- do not pool scores across fold models when there are few clusters with differing outcome rates.

A preprint built around Result A can carry all three rules, and none of them depends on the parts of Results B and C that are still heuristic.
