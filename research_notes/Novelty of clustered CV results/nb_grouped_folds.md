# Novelty check: the Nadeau-Bengio variance on grouped (cluster-held-out) K-fold CV, Propositions 6 and 8(ii)

Scope and method (2026-11-05). This note covers the variance of CV estimates, the Nadeau-Bengio (NB) corrected variance, and grouped, blocked or leave-cluster-out CV. I read proposals/06 (sections 6c and 6e), proposals/08 (Proposition 8 and "What is new and what is known") and proposals/07 first.

I opened the full text of the following and searched each for statements that overlap the result:
- Nadeau and Bengio's NIPS 1999 version
- Bengio and Grandvalet (2004)
- Bouckaert and Frank (2004)
- Markatou et al. (2005)
- Bayle et al. (2020)
- Austern and Zhou (arXiv v2)
- Li (2023)
- Wager (arXiv)
- Bates, Hastie and Tibshirani (arXiv v4)
- Schulz-Kümpel et al. (arXiv v2)
- McAlinn and Takanashi (arXiv v3)
- Iyengar, Lam and Wang (arXiv v2)
- Marraffini et al. (arXiv 2026)
- Roberts et al. (2017)

The search tools could not open Google Scholar "cited by" lists; see the Gaps sections.

Quoting. Copyright rules limit me to one short verbatim quote, so every other overlapping statement is given as a close paraphrase with a section, page or equation locator. Formulas are restated in our notation: $a$ is the cluster loss residual, $\beta=d^\top\psi$ the cluster influence on expected loss, $c=a-\beta/(K-1)$, $J=RK$ fold values and $G$ clusters.

## Q1. Does any prior work derive the expected NB variance in terms of influence-function components such as Var(a - beta/(K-1)), with or without clusters?

### Takeaway
No. I found no paper that writes $E[\hat\sigma^2_{NB}]$ in influence-function terms, with or without clusters.

The ingredients all exist in the i.i.d. case:
- Bengio and Grandvalet give the exact eigen-structure of the K-fold error covariance. Its "between-block" eigenvalue $\lambda_2$ is precisely what the NB estimator averages over a single partition.
- Austern and Zhou (2020) and Li (2023) give the first-order (influence-function) decomposition of the CV variance, $\mathrm{Var}(a+\beta)$ in our notation.
- Nobody has evaluated $\lambda_2$ to first order, which gives $\mathrm{Var}(a-\beta/(K-1))$, or connected it to NB.

So Proposition 8(ii) looks like a new combination of known pieces, not a known result.

### Cited Findings
- **Nadeau and Bengio, NIPS 12 (1999).** They study J independent random train/test splits, not K-fold.
  - Section 2, eq. (6): the variance of the average test error is $\sigma_1(\rho+(1-\rho)/J)$, with $\rho$ the correlation between split-level test errors.
  - Section 3, "1st Method: Corrected Resampled t-Test", p. 310, eq. (7): the sample variance of the split errors has expectation $\sigma_1(1-\rho)$. So $(1/J+\rho/(1-\rho))\hat\sigma^2$ would be unbiased, but $\rho$ is unknown.
  - They replace $\rho$ by a surrogate built under a fixed-loss approximation: "let us make the approximation that L(Z_{S_j}; Z_i) depends only on Z_i and n_1". Under it the correlation is $n_2/(n_1+n_2)$, giving the $(1/J+n_2/n_1)\hat\sigma^2$ estimator.
  - They then say (paraphrase) that it over- or underestimates the variance according to whether this surrogate exceeds or falls short of the true $\rho$. The section intro says the method may have a positive or negative bias.
  - No influence-function analysis.
  - [Nadeau & Bengio, NIPS 1999, pp. 307-313](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf)
- **Nadeau and Bengio, journal version.** *Machine Learning* 52(3):239-281 (2003), DOI 10.1023/A:1024068626366. Not opened (Springer redirects to a login). The bibliographic details are confirmed only through the reference lists of Bengio & Grandvalet (2004) and Bayle et al. (2020). Content: UNVERIFIED. — [Springer DOI landing (not opened)](https://link.springer.com/article/10.1023/A:1024068626366)
- **Bengio and Grandvalet, JMLR 5 (2004).**
  - Corollary 3, eq. (7), p. 1096: the K-fold CV variance is $\theta=\sigma^2/n+(m-1)\omega/n+(n-m)\gamma/n$. Here $\sigma^2$ is the per-example loss variance, $\omega$ the covariance within a test block, $\gamma$ the covariance between blocks, and $m=n/K$.
  - Lemma 7, p. 1099: the covariance matrix has three eigenvalues. $\lambda_1=\sigma^2-\omega$ (multiplicity $n-K$). $\lambda_2=\sigma^2+(m-1)\omega-m\gamma$ (multiplicity $K-1$, on the block indicators orthogonal to $\mathbf 1$). $\lambda_3=\sigma^2+(m-1)\omega+(n-m)\gamma$ (on $\mathbf 1$), with $\mathrm{Var}[\hat\mu]=\lambda_3/n$.
  - Only $\lambda_1$ and $\lambda_2$ can be estimated without bias. Section 5 adds that repeating K-fold splits does not give independent realizations.
  - The paper never evaluates $\sigma^2,\omega,\gamma$ through an expansion of the fitted model, and never computes the expectation of the NB estimator.
  - Section 1, p. 1090: the authors stress that NB considered independent splits, while they treat K-fold.
  - [Bengio & Grandvalet 2004, JMLR 5:1089-1105](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)
- **Austern and Zhou, arXiv 2001.11111v2 (27 Jun 2020).**
  - Theorem 3, Section 2.2: the CV risk centred at the expected risk has asymptotic variance $\sigma_{cv}^2=\sigma_1^2+\sigma_2^2+2\rho$. In our notation $\sigma_1^2\leftrightarrow\mathrm{Var}(a)$; $\sigma_2^2$ is the scaled variance of the fold models' risk, $\leftrightarrow\mathrm{Var}(\beta)$; and $\rho\leftrightarrow\mathrm{Cov}(a,\beta)$.
  - Section 4.1, Proposition 3: for M-estimators, $\rho=-\mathrm{Cov}\big(\partial R(\theta^*)^\top H^{-1}\partial\Psi,\,L\big)$, which is $\mathrm{Cov}(\beta,a)$.
  - Section 3: the within-block empirical variance is consistent for $\sigma_1^2$ (Proposition 1). Estimating $\sigma_2^2$ by the usual empirical estimator needs $K\to\infty$, so they propose a swap estimator (Proposition 2).
  - The words Nadeau or Bengio do not appear. The between-fold sample variance used by NB is not analysed.
  - [Austern & Zhou, arXiv 2001.11111](https://arxiv.org/pdf/2001.11111)
  - Published version: a later paper's reference list gives *Ann. Inst. Henri Poincaré Probab. Stat.* 61 (2025) 2804-2865. UNVERIFIED; not opened. — [McAlinn & Takanashi reference list](https://arxiv.org/pdf/2511.12698)
- **Li, "Asymptotics of K-Fold Cross Validation", JAIR 78 (2023) 491-526, DOI 10.1613/jair.1.13974.**
  - Theorem 3.1, Section 3.1, pp. 499-500: K-fold CV centred at the expected out-of-sample error has variance $\sigma_\gamma^2+\sigma^2_{01,\psi}+2\lambda$ for $\sqrt n$-rate estimators. This is $\mathrm{Var}(a)+\mathrm{Var}(\beta)+2\mathrm{Cov}(a,\beta)$ in our notation, with $\psi$ = influence function times loss derivative.
  - Remark 3.3: the variance does not depend on $K$.
  - i.i.d. only. No NB, no clusters.
  - [Li 2023, JAIR](https://jair.org/index.php/jair/article/view/13974)
- **Bayle, Bayle, Janson and Mackey, NeurIPS 2020** (34th conference, per the PDF footer).
  - Section 2: the CLT is a special case of a result for independent, not necessarily identically distributed points (App. B).
  - Section 5.1 and App. K: they compare against NB's corrected repeated train-validation t (10 random 90/10 splits, factor $1/10+0.1/0.9$). It covers near nominal, while their CLT interval is narrowest (Fig. 1).
  - They give no derivation of NB's expectation, and NB is run on random splits, not K-fold.
  - [Bayle et al., arXiv 2007.12671](https://arxiv.org/pdf/2007.12671)
- **Markatou, Tian, Biswas and Hripcsak, JMLR 6 (2005) 1127-1168.** They derive moment approximations ("MA") of the CV variance for the sample mean and for regression under squared and absolute loss.
  - Section 2.1: restates NB's caveat that the bias can have either sign.
  - Section 4.1, Table 6, pp. 1147-1149: for 4-, 5- and 10-fold and LOOCV (simple mean, N(0,1), n = 100 and 1000), the NB value is about twice the MA value. The text after Table 6 notes that NB did not treat K-fold.
  - No first-order theory of why the factor is about 2.
  - [Markatou et al. 2005, JMLR](https://www.jmlr.org/papers/volume6/markatou05a/markatou05a.pdf)
- **Wager, arXiv 1909.11696.** Proposition 1: in a nonparametric setting, the first-order behaviour of K-fold CV does not depend on the method, with limit $N(0,\mathrm{Var}((Y-\mu(X))^2))$. No NB.
  - [Wager, arXiv 1909.11696](https://arxiv.org/pdf/1909.11696)
  - Published as a comment in *JASA* 115(529):157-160, DOI 10.1080/01621459.2020.1727235. Taken from search-result bibliographic records; the publisher page was not opened. — [X-MOL record](https://www.x-mol.com/paper/1241045274106204160)
- **Bates, Hastie and Tibshirani, arXiv 2104.00673v4.** Section 1 (related work), pp. 2-3: NB's and Markatou et al.'s variance estimates rely only on sample size and higher moments of the errors. In the authors' words (paraphrase), they do not address the covariance term of their Section 4.1. No expectation of NB is derived. — [BHT arXiv v4](https://arxiv.org/pdf/2104.00673)
  - JASA 119(546):1434-1445 (2024), DOI 10.1080/01621459.2023.2197686, from search-result bibliographic records (not opened at the publisher). — [record](https://academicnewsletter.sufe.edu.cn/info/443017)
- **Later work (2024-2026) that does not touch NB's expectation:**
  - Schulz-Kümpel et al. (arXiv 2409.18836v2, Jan 2025). Section 4.5 calls the NB factor a "heuristic correction factor" (eq. CRT.3). Section 5 finds Corrected Resampled-T among the best methods on large data. — [Schulz-Kümpel et al.](https://arxiv.org/pdf/2409.18836)
  - Iyengar, Lam and Wang (arXiv 2407.02754v2; NeurIPS 2024). They compare plug-in, K-fold and LOOCV bias and coverage through higher-order expansions. NB is not mentioned. — [Iyengar et al.](https://arxiv.org/pdf/2407.02754)
  - McAlinn and Takanashi (arXiv 2511.12698v3, Jan 2026). Hold-out variance bounds and the choice of K. No NB analysis. — [McAlinn & Takanashi](https://arxiv.org/pdf/2511.12698)

### Inferences
- **Proposition 8(ii) at R = 1 (one partition, i.i.d. units) is a first-order evaluation of Bengio and Grandvalet's $\lambda_2$.** This is my own algebra and should be checked.
  - From their Lemma 7, the sample variance $S^2$ of the K fold means of one partition has expectation $\lambda_2/m$. So $E[\hat\sigma^2_{NB}]=(1/K+1/(K-1))\lambda_2/m=\frac{2K-1}{K-1}\cdot\frac{\lambda_2}{n}$, while $\mathrm{Var}(\mathrm{CV})=\lambda_3/n$.
  - Put the first-order loss $e_i=a_i+\bar\beta_{T_k}$ into their $\sigma^2,\omega,\gamma$. With $n-m=(K-1)m$ this gives $\omega=\mathrm{Var}(\beta)/(n-m)$ and $\gamma=[2\mathrm{Cov}(a,\beta)+(n-2m)\mathrm{Var}(\beta)/(n-m)]/(n-m)$.
  - Hence $\lambda_2=\mathrm{Var}(a)-2\mathrm{Cov}(a,\beta)/(K-1)+\mathrm{Var}(\beta)/(K-1)^2=\mathrm{Var}(a-\beta/(K-1))$, and $\lambda_3=\mathrm{Var}(a+\beta)$.
  - The second identity is Austern and Zhou's $\sigma_{cv}^2$ and Li's $\sigma^2$. The first, about $\lambda_2$, is the quantity behind Proposition 8(ii), and none of these papers computes it.
- **Suggested framing for proposals/08.** Proposition 8(ii) is the first-order value of Bengio and Grandvalet's between-block eigenvalue, combined with Austern and Zhou's and Li's influence decomposition. It is extended to R repeated partitions (through Proposition 6, since $c_h$ is the same in every partition) and to clusters as units, and it is compared with the cluster-robust variance $\mathrm{Var}(a)$ rather than with $\mathrm{Var}(\mathrm{CV})$.
- **NB's own fixed-loss derivation is the $\beta=0$ special case in crude form.** Their surrogate assumes the loss does not depend on the training set at all. Proposition 8 replaces "no dependence" with "first-order dependence through $\beta$" and gives the resulting bias.

### Gaps
- **The 2003 journal version of Nadeau and Bengio was not opened.** It is longer than the NIPS paper and may analyse $\rho$ further, for example whether the surrogate typically understates the true $\rho$. Markatou et al. (2005, p. 1149) state that NB do not treat K-fold, which lowers but does not remove this risk.
- **Grandvalet and Bengio's possible follow-up report on hypothesis testing for CV (around 2006)** could not be found. A search for it returned nothing. UNVERIFIED that it exists.
- **Google Scholar "cited by" lists** of Nadeau and Bengio (several thousand citations), Bates et al. and Bayle et al. could not be read with the available tools. A paper buried in them (for example a thesis) could contain a first-order NB analysis.

## Q2. Is the identity in Proposition 6 (on grouped folds, the expected NB variance over random partitions equals (J+K-1)/(J-1) times the cluster-robust variance) stated anywhere?

### Takeaway
Not in that form. Three related things exist:
- NB derived their $n_2/n$ correlation under the same fixed-loss idea, for random subsampling.
- Bouckaert and Frank carried the factor over to repeated K-fold by analogy, without a derivation.
- The $J=K$ grouped case is the survey-sampling random-group method (Wolter, Chapter 2; chapter text not opened).

No source found states:
- the $(J+K-1)/(J-1)$ factor for R repeated partitions;
- the equality with the cluster-robust (random-group) variance on grouped folds;
- the row-fold reading, in which the NB variance on row folds is the naive variance times that factor, so it inherits the design-effect shortfall.

Verdict: partly known (the ingredients and the J = K case), with an exact extension not found in print.

### Cited Findings
- **Nadeau and Bengio (NIPS 1999), Section 3, p. 310.** The surrogate $\rho_0=n_2/(n_1+n_2)$ is derived by assuming the loss depends only on the test example and the training-set size, which amounts to fixed losses. It is for J independent random splits, not partitions. — [Nadeau & Bengio 1999](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf)
- **Bouckaert and Frank (PAKDD 2004, LNCS 3056, pp. 3-12, DOI 10.1007/978-3-540-24775-3_3), Section 3.3.**
  - They define the "corrected repeated k-fold cv test" with variance $(1/(kr)+n_2/n_1)\hat\sigma^2$.
  - Their only justification (paraphrase): CV is a special case of random subsampling in which the test sets within a run do not overlap, so the same correction applies.
  - No derivation. Section 5 recommends 10-times 10-fold for replicability.
  - [Bouckaert & Frank 2004 (Waikato copy)](https://researchcommons.waikato.ac.nz/handle/10289/1451)
- **Bengio and Grandvalet (2004), Section 5, p. 1099.** Multiple K-fold splits of one data set do not give independent realizations of the error vector. Their analysis covers one partition, with expectations over the data, not over partitions. — [Bengio & Grandvalet 2004](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)
- **Markatou et al. (2005), Section 4.1, Table 6.** On a single K-fold run, NB is about twice the moment-approximation variance (ratios computed from their table):

  | Fold count | n = 100 | n = 1000 | Prediction $(2K-1)/(K-1)$ |
  | --- | --- | --- | --- |
  | 4-fold | 1.99 | 2.12 | 2.33 |
  | 5-fold | 2.16 | 2.06 | 2.25 |
  | 10-fold | 2.12 | 2.03 | 2.11 |
  | LOOCV | 1.99 | 1.99 | 2.01 (K = 100) and 2.00 (K = 1000) |

  — [Markatou et al. 2005](https://www.jmlr.org/papers/volume6/markatou05a/markatou05a.pdf)
- **Wolter, *Introduction to Variance Estimation* (Springer, 2007; DOI 10.1007/978-0-387-35099-8).** Chapter 2 is "The Method of Random Groups". This is confirmed from catalogue tables of contents only; the chapter text was not opened. The theorem number and the "rule (iii)" about forming groups from whole clusters that proposals/06 attributes to it remain UNVERIFIED. — [catalogue record](https://find.library.upatras.gr/Record/978-0-387-35099-8/Details)

### Inferences
- **Markatou's Table 6 is an independent published check of Proposition 6 combined with Proposition 8(iv) at R = 1.** It is the simple mean under squared loss, so the evaluation loss is the training criterion, $\beta=0$, and the predicted factor is $(2K-1)/(K-1)$. The agreement is close at K = 10 and for LOOCV. Their values fall below the prediction at K = 4 and 5 (by about 0.1 to 0.3), probably Monte Carlo noise: the NB variances they report, such as 0.00126 at 4-fold with n = 100, imply a large SD per data set. Table 6 does not state how many replications it averages.
- **The novelty of Proposition 6 is modest.** It is the random-group estimator in CV notation, as proposals/06 already says, plus three things not found in print:
  - the exact R-partition factor;
  - the statement that the equality holds with cluster-robust variance on grouped folds and with the naive variance on row folds;
  - the consequence that NB handles clustering only when the folds are grouped.

### Gaps
- Wolter's Chapter 2 was not read, so I cannot confirm whether it treats averaging over several random divisions (the R > 1 analogue). The R-partition claim of novelty rests on not finding it in the CV papers above, not on reading Wolter.
- No search of the survey-sampling literature on repeated random groups, or of Kott's delete-a-group jackknife, was done. These are the most likely places for an R > 1 analogue.

## Q3. Has anyone shown when the NB correction is conservative, by how much, and that the factor is 1 when the evaluation loss is the training criterion?

### Takeaway
- **When and by how much: not found.**
  - NB themselves say only that the bias can go either way, depending on whether their surrogate correlation is above or below the true one.
  - Empirical papers report NB as near-nominal or somewhat wide: Bayle et al. 2020; Schulz-Kümpel et al. 2025; Markatou's factor of 2 for single-run K-fold.
  - A 2026 preprint asserts NB is conservative on a heuristic that Proposition 6 contradicts.
- **The factor-1 case: the mechanism is known, its consequence for NB is not.**
  - Known: Austern and Zhou (2020, Proposition 3 and Remark 6) show that the covariance term $\rho$, our $\mathrm{Cov}(a,\beta)$, vanishes when the evaluation loss is the training loss, because $\partial R(\theta^*)=0$. That is our $d=0$.
  - Not found: that this makes the NB variance asymptotically equal to $\frac{J+K-1}{J-1}$ times the cluster-robust variance; the conservativeness condition $\mathrm{Var}(\beta)>2(K-1)\mathrm{Cov}(a,\beta)$; the factor $\mathrm{Var}(a-\beta/(K-1))/\mathrm{Var}(a)$.

### Cited Findings
- **Nadeau and Bengio (NIPS 1999), Section 3.** The corrected estimator over- or underestimates according to whether $\rho_0>\rho$ or $\rho_0<\rho$. Only the second method (conservative Z) is designed to be conservative. — [Nadeau & Bengio 1999](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf)
- **Markatou et al. (2005), Section 2.1.** Restates that NB's corrected estimator is not exactly unbiased and may err in either direction. Section 4.1, Table 6: NB is about twice the moment approximation for K-fold (single run). — [Markatou et al. 2005](https://www.jmlr.org/papers/volume6/markatou05a/markatou05a.pdf)
- **Austern and Zhou (arXiv 2001.11111v2).** Section 4.1, Remark 6: when the model is evaluated on the loss it was trained on, the gradient of the risk at $\theta^*$ is zero, so $\rho=0$. The abstract summarises: parametric M-estimators get the "full" speed-up under the training loss, while surrogate losses or regularizers can give more or less than that. They do not connect this to NB. — [Austern & Zhou](https://arxiv.org/pdf/2001.11111)
- **Wager (arXiv 1909.11696), Proposition 1.** CV's leading-order fluctuation does not depend on the method evaluated. This is consistent with $\mathrm{Err}_{\mathrm{new}}$ moving only at second order when $d=0$. — [Wager](https://arxiv.org/pdf/1909.11696)
- **Bayle et al. (NeurIPS 2020), Section 5.1 and Fig. 1.** The corrected repeated train-validation t (NB) had coverage near 95%. Their CLT interval was narrower. — [Bayle et al.](https://arxiv.org/pdf/2007.12671)
- **Schulz-Kümpel et al. (arXiv 2409.18836v2).**
  - Section 4.5: the NB factor is heuristic.
  - Section 5: Corrected Resampled-T performed well on large data.
  - Near p. 29: the risk coverage of the CV Wald interval fell with n, but not that of Corrected Resampled-T. They attribute this to the heuristic factor.
  - [Schulz-Kümpel et al.](https://arxiv.org/pdf/2409.18836)
- **Marraffini et al. (arXiv 2606.04010v1, June 2026), Appendix B.**
  - They use NB on family-aware repeated 20-times 10-fold CV.
  - They assert, without derivation, that NB is conservative because its $1/(k-1)$ term dominates and treats the repeats as adding no information, so the effective sample size collapses toward $k-1$.
  - This is a preprint in neuroimaging, not methods work. — [Marraffini et al.](https://arxiv.org/pdf/2606.04010)
- **Bates, Hastie and Tibshirani (arXiv v4), Section 1.** Dietterich-style and NB half-sample schemes are typically far too conservative because their internal fits use under half the sample. This refers to NB's conservative Z, not the corrected t. — [BHT](https://arxiv.org/pdf/2104.00673)

### Inferences
- **What is new in Proposition 8(ii)–(iv), as far as these sources show:**
  - the explicit asymptotic factor $\frac{J+K-1}{J-1}\cdot\mathrm{Var}(a-\beta/(K-1))/\mathrm{Var}(a)$;
  - the condition $\mathrm{Var}(\beta)>2(K-1)\mathrm{Cov}(a,\beta)$ for conservativeness at large J;
  - the observation that the factor is about 1 under log-loss because $d=0$. That $d=0$ is Austern and Zhou's Remark 6, so it should be cited as the known ingredient.
  - its quantitative, out-of-sample check (K = 3 and 10 predicted from K = 5).
- **The 2026 preprint's heuristic is wrong in the fixed-loss case**, by Proposition 6: as $J\to\infty$, NB tends to the cluster-robust variance, factor $\to1$. The $1/(K-1)$ term is exactly what turns the spread of fold means into the variance of the overall mean; it does not cause conservativeness. Under Proposition 8, conservativeness comes from the $\beta$ terms and from the finite-J factor $1+K/(J-1)$. This is a citable contrast showing the question is still open in applied practice.
- **That NB on grouped (family- or subject-held-out) folds is used in practice** (Marraffini et al., following a protocol they attribute to Ooi et al. 2022) supports the relevance of Propositions 6 and 8 to applied work.

### Gaps
- Ooi et al. (2022), the protocol source cited by Marraffini et al., was not opened (UNVERIFIED).
- No paper was found that measures NB's ratio to the true variance as a function of K for a model trained by log-likelihood and scored by 0-1 loss. The team's Part 8 seems to be the only such test.

## Q4. What do the grouped, blocked, spatial and leave-cluster-out CV literatures say about variance estimation?

### Takeaway
They focus almost entirely on bias (optimism of random-row CV, extrapolation and pessimism of blocking) and on what is being estimated. Variance estimation for the grouped-CV estimate is essentially absent:
- Roberts et al. (2017) give no variance estimator for block CV.
- Rabinowicz and Rosset (2022) correct bias only.
- Spatial-CV work reports fold-to-fold spread informally.

The CV-variance papers that do allow non-i.i.d. units (Bayle et al.'s independent but not identically distributed CLT) can be applied with clusters as units, but do not discuss grouped folds or NB.

### Cited Findings
- **Roberts et al. (2017), *Ecography* 40:913-929, DOI 10.1111/ecog.02881.**
  - Recommends block CV whenever dependence structures exist, because random CV underestimates predictive error.
  - A text search of the full paper found no estimator of the variance or standard error of the block-CV estimate.
  - Uncertainty appears only informally. In the resource-selection case study (around pp. 918-919), parameter estimates varied more across blocked folds, which the authors read as a measure of uncertainty.
  - [Roberts et al. 2017 (author copy)](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf)
- **Rabinowicz and Rosset (2022), *JASA* 117(538):718-731, DOI 10.1080/01621459.2020.1801451.** Introduces a bias-corrected CV (CVc) for correlated data under squared loss. The abstract mentions no variance, standard error or interval for the CV estimate. Only the abstract page was opened. — [IDEAS/RePEc record](https://ideas.repec.org/a/taf/jnlasa/v117y2022i538p718-731.html)
- **Bates, Hastie and Tibshirani (arXiv v4), concluding discussion.** Lists adapting nested CV to dependent data as future work, citing Rabinowicz and Rosset. — [BHT](https://arxiv.org/pdf/2104.00673)
- **Bayle et al. (2020), Section 2 after Theorem 1, and App. B.** The CLT holds for independent, not necessarily identically distributed points. With clusters as the points this gives a grouped-CV Wald interval, but the paper does not discuss grouping. — [Bayle et al.](https://arxiv.org/pdf/2007.12671)
- **Bengio and Grandvalet (2004), Section 2.2, p. 1092.** For i.i.d. data, the naive variance becomes correct only as the learner stabilises, and test-error correlations vanish. No treatment of dependent data. — [Bengio & Grandvalet 2004](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)
- **Search results only (not opened, UNVERIFIED).**
  - Spatial-CV studies typically show fold-to-fold standard errors as descriptive spread.
  - A clustered-sample map-accuracy paper (de Bruin et al. 2022, *Ecological Informatics*) recommends sampling-intensity-weighted CV and treats bias, not variance.
  - [ScienceDirect: "Dealing with clustered samples for assessing map accuracy by cross-validation"](https://www.sciencedirect.com/science/article/pii/S1574954122001145)

### Inferences
- The grouped-CV variance question, meaning which error bar fits grouped folds and how NB behaves on them, has no dedicated treatment in the spatial or blocked-CV literature I could reach. Propositions 6 and 8 fill a gap there rather than duplicating it.
- Practitioners who use NB on grouped folds (Marraffini et al. 2026) do so with no theory for that case.

### Gaps
- No systematic search of 2023-2026 spatial-CV methods papers for variance or interval proposals (for example kNNDM / CAST, or design-based map accuracy). The searches I ran surfaced none with a CV variance estimator, but coverage is not exhaustive.
- Leave-one-cluster-out jackknife variances exist for estimators in clustered trials and GEE (search results only, not opened). Whether anyone has applied a cluster jackknife to the CV error itself was not established.

## Q5. Overall verdict: is Proposition 8(ii) known, partly known, or plausibly new?

### Takeaway
**Plausibly new, as a result. Its ingredients are known and should be cited explicitly.** Nothing found states the expected NB variance in terms of influence-function components, the conservativeness factor $\mathrm{Var}(a-\beta/(K-1))/\mathrm{Var}(a)$, the sign condition, or any version with grouped folds.

The closest prior work:
1. **Bengio and Grandvalet (2004), Lemma 7.** The between-block eigenvalue $\lambda_2$ governs NB's expectation; Proposition 8(ii) is its first-order value.
2. **Austern and Zhou (2020), Theorem 3, Proposition 3 and Remark 6; Li (2023), Theorem 3.1.** The influence decomposition $\mathrm{Var}(a+\beta)$, and $\rho=0$ when the loss is the training criterion.
3. **Nadeau and Bengio (1999/2003), Section 3.** A fixed-loss derivation of $\rho_0$, with the caveat that the bias can go either way.
4. **Markatou et al. (2005), Table 6.** NB is about twice the true variance for single-run K-fold on the sample mean, consistent with $(2K-1)/(K-1)$.
5. **Bouckaert and Frank (2004).** The repeated-K-fold version, by analogy and without derivation.

Proposition 6 is partly known (random groups; NB's fixed-loss surrogate). The exact R-partition factor, and its reading for grouped against row folds, were not found.

### Cited Findings
Verification status of every citation used:

| Work | Opened? | Bibliographic details checked against |
| --- | --- | --- |
| Nadeau & Bengio, NIPS 12, 1999, pp. 307-313 | Yes, scanned PDF read | PDF ([link](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf)) |
| Nadeau & Bengio, *Mach. Learn.* 52(3):239-281, 2003, DOI 10.1023/A:1024068626366 | No | Reference lists of Bengio & Grandvalet and Bayle et al. Content UNVERIFIED. |
| Bengio & Grandvalet, *JMLR* 5:1089-1105, 2004 | Yes | PDF ([link](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)) |
| Bouckaert & Frank, PAKDD 2004, LNCS 3056, pp. 3-12, DOI 10.1007/978-3-540-24775-3_3 | Yes, repository PDF | Repository record ([link](https://researchcommons.waikato.ac.nz/handle/10289/1451)) |
| Markatou, Tian, Biswas & Hripcsak, *JMLR* 6:1127-1168, 2005 | Yes | PDF ([link](https://www.jmlr.org/papers/volume6/markatou05a/markatou05a.pdf)) |
| Bayle, Bayle, Janson & Mackey, NeurIPS 2020 (vol. 33, pp. 16339-16350 per a citing paper) | Yes, arXiv 2007.12671 | arXiv PDF ([link](https://arxiv.org/pdf/2007.12671)) |
| Austern & Zhou, arXiv 2001.11111v2 | Yes | arXiv PDF ([link](https://arxiv.org/pdf/2001.11111)). Journal version (AIHP 61, 2025) UNVERIFIED. |
| Li, *JAIR* 78:491-526, 2023, DOI 10.1613/jair.1.13974 | Yes | JAIR PDF ([link](https://jair.org/index.php/jair/article/view/13974)) |
| Wager, arXiv 1909.11696 | Yes | arXiv. JASA 115(529):157-160 from a bibliographic record only. |
| Bates, Hastie & Tibshirani, arXiv 2104.00673v4 | Yes | arXiv. JASA 119(546):1434-1445, DOI 10.1080/01621459.2023.2197686, from a bibliographic record only. |
| Schulz-Kümpel et al., arXiv 2409.18836v2 | Yes | arXiv. Journal venue (DMLR) UNVERIFIED. |
| Iyengar, Lam & Wang, arXiv 2407.02754v2 | Yes | arXiv |
| McAlinn & Takanashi, arXiv 2511.12698v3 | Yes | arXiv |
| Marraffini et al., arXiv 2606.04010v1 | Yes | arXiv |
| Roberts et al., *Ecography* 40:913-929, 2017, DOI 10.1111/ecog.02881 | Yes, author copy | Author copy |
| Rabinowicz & Rosset, *JASA* 117(538):718-731, 2022, DOI 10.1080/01621459.2020.1801451 | Abstract page only | RePEc record |
| Wolter, *Introduction to Variance Estimation*, Springer, 2007, DOI 10.1007/978-0-387-35099-8, Chapter 2 | No, table of contents only | Catalogue listing. Chapter content UNVERIFIED. |

### Inferences
Suggested wording for proposals/08, section "What is new and what is known":
- **Move to "Known":**
  - Austern and Zhou's Remark 6 ($d=0$ under the training loss), as the source of Proposition 8(iv)'s mechanism.
  - Austern and Zhou's and Li's $\mathrm{Var}(a+\beta)$, as the i.i.d. analogue of the CV-variance part of Proposition 8.
  - Bengio and Grandvalet's $\lambda_2$, as the exact (non-asymptotic) quantity whose first-order value Proposition 8(ii) gives.
  - Markatou et al.'s Table 6, as prior empirical evidence of the factor of about 2 at R = 1, consistent with Proposition 6 and Proposition 8(iv).
- **Keep as "Plausibly new":**
  - the expected-NB formula in influence terms and the conservativeness factor with its sign condition;
  - the grouped-fold (clusters-as-units) version and its comparison with the cluster-robust variance;
  - the R-partition factor;
  - the out-of-sample predictive test.

### Gaps
- **Remaining risk to the novelty claim:**
  - Google Scholar "cited by" lists (not accessible here);
  - the unread 2003 journal version of Nadeau and Bengio;
  - Wolter's Chapter 2;
  - theses and technical reports, such as a possible Grandvalet and Bengio report on hypothesis testing for CV, whose existence is UNVERIFIED.
- **Recommended before claiming novelty in print:** Jenna should check the $\lambda_2=\mathrm{Var}(a-\beta/(K-1))$ algebra under Q1, and someone with library access should read NB 2003 Sections 4-5 and Wolter Chapter 2.
