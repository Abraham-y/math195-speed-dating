# Theory Track: Cross-Validation Intervals When Rows Are Clustered

First pass, 2026-09-24. Companion to the Layer A section of the speed-dating proposal. Status of each part is marked: proved, proposed, simulated, or open.

## 0. Summary

Three things are established below, in increasing order of novelty.

1. Grouped cross-validation (whole clusters held out) and random-split cross-validation estimate different quantities: the error on rows from new clusters versus the error on rows from clusters already seen. This is a one-line consequence of exchangeability (Lemma 1), and it means that no interval can fix a random-split estimate when the question is about new people.

2. Under grouped folds, the variance of the CV estimate equals the independent-rows formula of Bates, Hastie and Tibshirani (2024) plus one extra term, $(m-1)\,\delta_c/n$, where $m$ is the cluster size and $\delta_c$ is the excess covariance between per-point errors in the same cluster (Proposition 1). The usual standard error misses that term entirely, so it is too small by at least the survey-sampling design effect $1+(m-1)\rho_c$ (Corollary). A cluster-robust standard error recovers the cluster term exactly, and what it still misses is precisely the between-cluster covariance induced by shared training data, the same object the independent-rows literature studies (Proposition 2).

3. Nested cross-validation, the independent-rows fix, can be adapted to clustered data by holding out clusters at both levels and replacing its within-fold variance correction with a cluster-robust one. The resulting MSE estimate is unbiased for the MSE of a grouped CV estimate (Proposition 3); using the original correction under clustering is provably conservative. The adapted estimate is noisy at five folds, and a first simulation shows that flooring it at the cluster-robust variance gives nominal coverage where each component alone does not.

The cross-classified (dyadic) case, where every row belongs to one rater and one ratee, is set up in Section 6 with the estimators stated and the proofs left open.

## 1. Setup and notation

Rows $z_i=(x_i,y_i)$, $i=1,\dots,n$, each in exactly one cluster $g(i)\in\{1,\dots,G\}$, balanced so $n=Gm$. Two-level model: cluster effects $U_1,\dots,U_G$ are i.i.d.; given $U_g$, the rows of cluster $g$ are i.i.d. from a distribution $P_{U_g}$. Rows in different clusters are independent; rows in the same cluster are exchangeable but dependent.

A learner maps a training set $D$ to a fitted function $\hat f_D$ and is symmetric (does not depend on the order of training rows). A bounded loss $\ell(z;f)$. Folds $F_1,\dots,F_K$, assigned at random and independently of the data. Grouped folds: each cluster lies entirely in one fold, $G/K$ clusters per fold. Random-split folds: rows assigned individually, so clusters straddle folds.

Per-point CV error and the CV estimate:

$$e_i=\ell\big(z_i;\hat f_{-k(i)}\big),\qquad \widehat{\mathrm{Err}}_{\mathrm{CV}}=\frac1n\sum_{i=1}^n e_i,$$

where $\hat f_{-k}$ is trained on all folds except $F_k$. Write $\bar e_g$ for the mean of $e_i$ over cluster $g$.

Two estimands for a model trained on $D$:

- New-cluster error: $\mathrm{Err}^{\mathrm{new}}(D)=E\big[\ell(Z';\hat f_D)\mid D\big]$, where $Z'$ is a row from a fresh cluster.
- Same-cluster error: $\mathrm{Err}^{\mathrm{same}}(D)=\frac1G\sum_g E\big[\ell(Z'_g;\hat f_D)\mid D,U_g\big]$, where $Z'_g$ is a fresh row from existing cluster $g$.

For the dating data, a new cluster is a new person, so $\mathrm{Err}^{\mathrm{new}}$ is the quantity a practitioner means by "how well does this predict for someone we have not seen."

## 2. What each CV scheme estimates (proved)

Lemma 1. Under the two-level model with grouped folds and a symmetric learner, for every row $i$ in fold $k$,

$$E\big[e_i\mid D_{-k}\big]=\mathrm{Err}^{\mathrm{new}}(D_{-k}),$$

so $E\big[\widehat{\mathrm{Err}}_{\mathrm{CV}}\big]=E\big[\mathrm{Err}^{\mathrm{new}}(D')\big]$ for a training set $D'$ of $G(K-1)/K$ clusters. Under random-split folds, a held-out row's cluster typically has other rows in $D_{-k}$, and the corresponding identity holds with a same-cluster estimand at the same training size.

Proof. With grouped folds, the cluster of row $i$ is disjoint from $D_{-k}$, so $z_i$ is independent of $D_{-k}$ and has the marginal law of a row from a fresh cluster. Conditioning on $D_{-k}$ gives the first display; averaging over $i$ and $k$ gives the second. Under random splits, condition on $D_{-k}$ and on $U_{g(i)}$: the row $z_i$ is then a fresh draw from $P_{U_{g(i)}}$, which is the same-cluster estimand. $\square$

Remark. The gap $\mathrm{Err}^{\mathrm{new}}-\mathrm{Err}^{\mathrm{same}}$ is the cluster leakage. It is not signed in general, but it is nonnegative whenever the learner can exploit cluster-level structure in $x$ to absorb the cluster effect in $y$, which is the usual case with flexible learners or with covariates that vary mostly between clusters. In the simulation of Section 7, random-split CV is biased downward by about one standard deviation of the estimate.

## 3. Variance of the grouped CV estimate (proved)

By exchangeability, symmetry of the learner, and the symmetric random fold assignment, the joint law of a pair $(e_i,e_j)$, $i\neq j$, depends only on the pair's type. Define

- $\sigma^2=\mathrm{Var}(e_i)$,
- $\sigma_{cc}=\mathrm{Cov}(e_i,e_j)$ for $i,j$ in the same cluster (hence the same fold),
- $\sigma_{sf}=\mathrm{Cov}(e_i,e_j)$ for $i,j$ in different clusters but the same fold,
- $\sigma_{df}=\mathrm{Cov}(e_i,e_j)$ for $i,j$ in different folds,

and the excess within-cluster covariance $\delta_c=\sigma_{cc}-\sigma_{sf}$.

Proposition 1. Under the two-level model with grouped balanced folds,

$$\mathrm{Var}\big(\widehat{\mathrm{Err}}_{\mathrm{CV}}\big)=\frac{\sigma^2}{n}+\Big(\frac1K-\frac1n\Big)\sigma_{sf}+\Big(1-\frac1K\Big)\sigma_{df}+\frac{m-1}{n}\,\delta_c .$$

The first three terms are the independent-rows formula (the case $m=1$); the last is the cluster term.

Proof. Count ordered pairs $(i,j)$, $i\ne j$: there are $n(m-1)$ in the same cluster, $n(n/K-m)$ in the same fold but different clusters, and $n(n-n/K)$ in different folds. Then

$$\mathrm{Var}\big(\widehat{\mathrm{Err}}_{\mathrm{CV}}\big)=\frac{1}{n^2}\Big[n\sigma^2+n(m-1)\sigma_{cc}+n\Big(\frac nK-m\Big)\sigma_{sf}+n\Big(n-\frac nK\Big)\sigma_{df}\Big].$$

Substitute $\sigma_{cc}=\sigma_{sf}+\delta_c$ and simplify: $\frac{m-1}{n}\sigma_{sf}+(\frac1K-\frac mn)\sigma_{sf}=(\frac1K-\frac1n)\sigma_{sf}$. $\square$

Corollary (design effect). Let $\hat V_{\mathrm{naive}}=s^2/n$ with $s^2$ the sample variance of the $e_i$, and $V=\mathrm{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}})$. For any $e_1,\dots,e_n$ with common variance $\sigma^2$, $E[s^2]=\frac{n}{n-1}(\sigma^2-V)$, hence

$$E\big[\hat V_{\mathrm{naive}}\big]=\frac{\sigma^2-V}{n-1}\approx\frac{\sigma^2}{n},\qquad \frac{V}{E[\hat V_{\mathrm{naive}}]}\approx 1+(m-1)\rho_{cc}+\Big(\frac nK-m\Big)\rho_{sf}+\Big(n-\frac nK\Big)\rho_{df},$$

with $\rho_{\cdot}=\sigma_{\cdot}/\sigma^2$. Ignoring the two between-cluster terms, the ratio is the Kish design effect $1+(m-1)\rho_{cc}$. With $m=20$ and a within-cluster error correlation of $0.05$, the naive variance is off by a factor of about two before the independent-rows problem is even considered.

Proof of the identity. $\sum_i(e_i-\bar e)^2=\sum_i e_i^2-n\bar e^2$, and $E[\sum_i e_i^2]=n(\sigma^2+\mu^2)$, $E[n\bar e^2]=n(V+\mu^2)$. $\square$

## 4. What the cluster-robust standard error captures (proved)

Define the cluster-robust variance estimate, treating cluster means as the units:

$$\hat V_{\mathrm{CR}}=\frac{1}{G(G-1)}\sum_{g=1}^G\big(\bar e_g-\widehat{\mathrm{Err}}_{\mathrm{CV}}\big)^2 .$$

Let $v=\mathrm{Var}(\bar e_g)=\frac{\sigma^2+(m-1)\sigma_{cc}}{m}$, common to all clusters, and let $\bar c$ be the average over $g'\ne g$ of $\mathrm{Cov}(\bar e_g,\bar e_{g'})$:

$$\bar c=\frac{(G/K-1)\,\sigma_{sf}+(G-G/K)\,\sigma_{df}}{G-1}\approx\frac1K\sigma_{sf}+\Big(1-\frac1K\Big)\sigma_{df}.$$

Proposition 2. Under the conditions of Proposition 1,

$$V=\frac vG+\frac{G-1}{G}\,\bar c,\qquad E\big[\hat V_{\mathrm{CR}}\big]=\frac vG-\frac{\bar c}{G},\qquad V-E\big[\hat V_{\mathrm{CR}}\big]=\bar c .$$

Proof. The first identity is Proposition 1 rewritten at the cluster level. For the second, apply the sample-variance identity of the Corollary to the $G$ cluster means (balanced clusters make $\widehat{\mathrm{Err}}_{\mathrm{CV}}$ their mean): $E[\frac{1}{G-1}\sum_g(\bar e_g-\bar e)^2]=\frac{G}{G-1}(v-V)$, so $E[\hat V_{\mathrm{CR}}]=\frac{v-V}{G-1}=\frac{v-v/G-\frac{G-1}{G}\bar c}{G-1}=\frac vG-\frac{\bar c}{G}$. Subtract. $\square$

Reading. The bias of the cluster-robust estimate is exactly $\bar c$, the average covariance between errors in different clusters, and it contains no $m$ and no $\delta_c$. So the cluster-robust standard error removes the whole clustering shortfall and leaves exactly the shared-training-data covariance that Bates, Hastie and Tibshirani analyze for independent rows. For $m=1$ the cluster-robust and naive estimates coincide and Proposition 2 reduces to their observation. In the simulation below the remaining shortfall is about 15% of $V$.

## 5. Cluster-nested cross-validation (proposed; unbiasedness proved)

Nested CV (Bates, Hastie and Tibshirani 2024, Algorithm 2) estimates the MSE of a CV estimate as follows. For each outer fold $k$: run an inner $(K-1)$-fold CV on $D_{-k}$ to get $\hat e^{(k)}_{\mathrm{in}}$; train on $D_{-k}$ and evaluate on $F_k$ to get per-point errors and their mean $e^{(k)}_{\mathrm{out}}$; set $a_k=(\hat e^{(k)}_{\mathrm{in}}-e^{(k)}_{\mathrm{out}})^2$ and $b_k$ = an estimate of $\mathrm{Var}(e^{(k)}_{\mathrm{out}}\mid D_{-k})$, for which they use the within-fold sample variance divided by the fold size. Then $\widehat{\mathrm{MSE}}=\mathrm{mean}_k\,a_k-\mathrm{mean}_k\,b_k$, averaged over $R$ random fold assignments, and the interval is $\widehat{\mathrm{Err}}_{\mathrm{CV}}\pm z\sqrt{\widehat{\mathrm{MSE}}}$ (they add a bias adjustment omitted here).

Cluster-nested CV (proposed). Same algorithm with two changes:

1. Both the outer and the inner fold assignments are grouped by cluster.
2. The correction $b_k$ is the cluster-robust variance of the outer fold mean: with $G_k$ held-out clusters and their mean errors $\bar e_g$, $b_k=\frac{1}{G_k(G_k-1)}\sum_{g\in F_k}(\bar e_g-e^{(k)}_{\mathrm{out}})^2$.

Proposition 3. Under the two-level model with grouped folds at both levels and a symmetric learner, the unclipped cluster-nested estimate satisfies

$$E\big[\widehat{\mathrm{MSE}}_{\mathrm{CN}}\big]=E\Big[\big(\hat e^{(k)}_{\mathrm{in}}-\mathrm{Err}^{\mathrm{new}}(D_{-k})\big)^2\Big],$$

the mean squared error of a grouped $(K-1)$-fold CV estimate, computed on $G(K-1)/K$ clusters, for the new-cluster error of the model trained on those clusters.

Proof. Condition on $D_{-k}$ and the fold assignment. The held-out clusters are i.i.d. and independent of $D_{-k}$ (grouped folds), so their mean errors $\bar e_g$, $g\in F_k$, are i.i.d. given $D_{-k}$, with mean $\mathrm{Err}^{\mathrm{new}}(D_{-k})$ by Lemma 1 and some variance $v_k$. Their mean $e^{(k)}_{\mathrm{out}}$ therefore has conditional mean $\mathrm{Err}^{\mathrm{new}}(D_{-k})$ and conditional variance $v_k/G_k$, and $b_k$ is the standard unbiased estimate of $v_k/G_k$ from $G_k$ i.i.d. draws. Since $\hat e^{(k)}_{\mathrm{in}}$ is a function of $D_{-k}$, $E[a_k\mid D_{-k}]=(\hat e^{(k)}_{\mathrm{in}}-\mathrm{Err}^{\mathrm{new}}(D_{-k}))^2+v_k/G_k$. Subtract $E[b_k\mid D_{-k}]=v_k/G_k$, take expectations, and average over $k$ and the $R$ assignments. $\square$

Remark (the original correction is conservative under clustering). With grouped folds but the independent-rows $b_k$, the sample-variance identity gives $E[b^{\mathrm{iid}}_k\mid D_{-k}]\approx\sigma^2_k/n_k$, whereas the correct value is $v_k/G_k=(\sigma_k^2+(m-1)\sigma_{cc,k})/n_k$. So $\widehat{\mathrm{MSE}}$ is biased upward by about $(m-1)\sigma_{cc,k}/n_k$, the within-fold cluster covariance, and the intervals over-cover. The simulation confirms this (98% coverage at nominal 90%, intervals 1.7 times wider than needed).

Remark (variance of the estimate). Proposition 3 is about the mean. The estimate is a difference of two noisy quantities, each averaged over only $K$ folds per assignment, and it is negative in a few percent of replicates at $K=5$, $R=20$. Its misses are concentrated where it happens to be small. Since $\hat V_{\mathrm{CR}}$ is a low-variance estimate that is biased downward by exactly $\bar c\ge0$ (Proposition 2, when the between-cluster covariances are nonnegative), a natural stabilizer is the floor

$$\hat V_{\mathrm{floor}}=\max\big(\hat V_{\mathrm{CR}},\,\widehat{\mathrm{MSE}}_{\mathrm{CN}}\big).$$

This is a heuristic. A principled version would estimate $\bar c$ directly and add it to $\hat V_{\mathrm{CR}}$; see Section 8.

## 6. Cross-classified data: the dyadic case (set up; proofs open)

In the speed-dating data each row is an ordered pair (rater $r$, ratee $c$) inside a wave $w$; within a wave every rater meets every ratee of the other sex, and waves are independent. Rows sharing a rater are dependent, rows sharing a ratee are dependent, and the two rows of the same date (one from each side) are dependent through the dyad.

Estimands. Holding out whole waves gives the "new rater and new ratee" error and makes Lemma 1 go through verbatim, with waves as the independent units. Holding out raters but not ratees gives a "new rater, seen ratee" error, in which the training set still contains the held-out ratees' desirability information; it is a legitimate but different estimand, and the two should not be compared as if they estimated the same thing.

Variance. With wave-grouped folds, Proposition 1 holds with waves as clusters, so a wave-cluster-robust $\hat V_{\mathrm{CR}}$ and a wave-grouped cluster-nested CV are valid as stated in Sections 4 and 5. The practical difficulty is degrees of freedom: 21 waves in five folds leaves about four waves per fold for $b_k$.

Two-way estimator. Within a held-out fold, model the per-point errors conditional on the trained model as $e_{rc}=\mu+\alpha_r+\beta_c+\gamma_{\{r,c\}}+\varepsilon_{rc}$ with independent mean-zero components. Under this additive structure the multiway cluster-robust estimator of Cameron, Gelbach and Miller (2011), $\hat V_{R}+\hat V_{C}-\hat V_{R\cap C}$, with the dyad added as a third clustering by inclusion-exclusion, uses every row's information rather than four wave means.

Proposition 4 (proved 2026-10-15 for one direction; for Jenna's review). (i) With wave-grouped folds, the wave-cluster $b_k$ is unbiased for $\mathrm{Var}(e^{(k)}_{\mathrm{out}}\mid D_{-k})$ for any within-wave dependence. (ii) Take one balanced complete array of $R$ raters by $C$ ratees ($n=RC$ rows, one direction), with $e_{rc}=\mu+\alpha_r+\beta_c+\gamma_{rc}$ and independent mean-zero components of variances $\sigma_\alpha^2,\sigma_\beta^2,\sigma_\gamma^2$. Let $\hat V_{\mathrm{CGM}}=\hat V_R+\hat V_C-\hat V_{RC}$, each term the unbalanced cluster-robust variance of the mean with its own $G/(G-1)$ ($G=R$, $C$, and $n$). Then

$$\mathrm{Var}(\bar e)=\frac{\sigma_\alpha^2}{R}+\frac{\sigma_\beta^2}{C}+\frac{\sigma_\gamma^2}{RC},\qquad E\big[\hat V_{\mathrm{CGM}}\big]-\mathrm{Var}(\bar e)=-\,\frac{C(R-1)\,\sigma_\alpha^2+R(C-1)\,\sigma_\beta^2}{n(n-1)} .$$

So the two-way estimator is never conservative here. Its bias is of order $(\sigma_\alpha^2+\sigma_\beta^2)/n$, which is about $-2/(R+C)$ of the truth when $\sigma_\alpha=\sigma_\beta$ (about $-10\%$ at $R=C=10$). The bias is exactly computable, so it can be corrected with ANOVA estimates of $\sigma_\alpha^2,\sigma_\beta^2$.

Proof. (i) As in Proposition 3: given $D_{-k}$, the held-out waves are i.i.d., so their mean errors are i.i.d. with a common variance, and $b_k$ is the unbiased sample-variance estimate of the variance of their mean. (ii) Write $\bar e_{r\cdot}$ for rater means. Since the design is complete, $\sum_c(e_{rc}-\bar e)=C(\bar e_{r\cdot}-\bar e)$ and $\bar e_{r\cdot}-\bar e=(\alpha_r-\bar\alpha)+(\bar\gamma_{r\cdot}-\bar\gamma)$ (the $\beta$ terms cancel). Hence $E\sum_r\big(\sum_c(e_{rc}-\bar e)\big)^2=C^2(R-1)(\sigma_\alpha^2+\sigma_\gamma^2/C)$, and with the factor $R/(R-1)$ and the divisor $n^2$, $E[\hat V_R]=\sigma_\alpha^2/R+\sigma_\gamma^2/n$. By symmetry $E[\hat V_C]=\sigma_\beta^2/C+\sigma_\gamma^2/n$. For the row-level term, $E\sum_{rc}(e_{rc}-\bar e)^2=C(R-1)\sigma_\alpha^2+R(C-1)\sigma_\beta^2+(n-1)\sigma_\gamma^2$, and $\hat V_{RC}$ divides this by $n(n-1)$. The $\sigma_\gamma^2$ terms combine to $\sigma_\gamma^2/n$ exactly; what is left is the display. $\square$

Remark (two directions). In the real design each dyad gives two rows: $r\to c$ and $c\to r$. Two kinds of dependence remain outside the rater and ratee clusters. First, the two directions of a dyad share no rater and no ratee, but their pair terms correlate (dyadic reciprocity; 0.21 for liking, `scripts/srm.py`). Second, a person's rater effect and ratee effect can correlate (generalized reciprocity; about 0 here). The dyadic-robust estimator (Fafchamps and Gubert 2007; Aronow, Samii and Assenova 2015), which sums $r_ir_j$ over all pairs of rows sharing a person, includes both. Its finite-sample bias in the balanced bipartite design is not yet worked out. The exploratory dyadic design in `scripts/sim_cv_designs.py` compares all five SEs.

Remark (what the literature says; proposals/07). For this dataset the problem is nested inside events, because every person attends one wave. Aronow et al. found dyadic-robust SEs barely larger than subject-clustered ones on these same data. The two-way question matters for within-event targets, such as a new rater and a new ratee at an event already seen, and for data where people appear in several clusters.

Open questions in this section, in order of value: (a) the dyadic-robust analogue of Proposition 4(ii), with the cross-direction and actor-partner covariances; (b) the relative variance of the wave-cluster and multiway $b_k$, which decides which one to use at 21 waves; (c) whether a wave-level effect (a good evening) breaks the additive model, testable from the data; (d) the extension of Proposition 3 with the multiway $b_k$.

## 6b. Generalized reciprocity in the social relations model (proved)

This is used by `scripts/srm.py`. One event has $n$ women and $m$ men, all pairs observed. Women rating men: $x_{ij}=\mu+a_i+b_j+g_{ij}$. Men rating women: $y_{ji}=\mu'+a'_j+b'_i+g'_{ji}$. Here $\mathrm{Var}(g)=\sigma^2$, $\mathrm{Var}(g')=\sigma'^2$, $\mathrm{Cov}(g_{ij},g'_{ji})=c$ (dyadic reciprocity), and the pairs $(a_i,b'_i)$ are i.i.d. with covariance $\sigma_{ab}$ (generalized reciprocity, women's side). The estimated effects are the centred row and column means $\hat a_i=\bar x_{i\cdot}-\bar x_{\cdot\cdot}$ and $\hat b'_i=\bar y_{\cdot i}-\bar y_{\cdot\cdot}$.

Proposition 5. $E\big[\sum_i\hat a_i\hat b'_i\big]=(n-1)\big(\sigma_{ab}+c/m\big)$ and $E\big[\sum_i\hat a_i^2\big]=(n-1)\big(\sigma_a^2+\sigma^2/m\big)$. With $\hat\sigma^2=\sum\hat g^2/((n-1)(m-1))$ and $\hat c=\sum\hat g\hat g'/((n-1)(m-1))$ from the double-centred residuals, both unbiased, the moments $\sum\hat a\hat b'/(n-1)-\hat c/m$ and $\sum\hat a^2/(n-1)-\hat\sigma^2/m$ are unbiased for $\sigma_{ab}$ and $\sigma_a^2$.

Proof. $\hat a_i=(a_i-\bar a)+(\bar g_{i\cdot}-\bar g_{\cdot\cdot})$ and $\hat b'_i=(b'_i-\bar b')+(\bar g'_{\cdot i}-\bar g'_{\cdot\cdot})$, where $\bar g'_{\cdot i}$ averages the $m$ men's residuals for woman $i$. The noise pair $(\bar g_{i\cdot},\bar g'_{\cdot i})$ is independent across $i$ with covariance $m^{-2}\sum_j\mathrm{Cov}(g_{ij},g'_{ji})=c/m$. The centred cross-product of $n$ i.i.d. pairs has expectation $(n-1)$ times their covariance. For the residuals, with $H$ the centring matrix, $E\,\mathrm{tr}(H_nGH_m\,H_mG'^{\top}H_n)=\sum_{ij}c\,(1-1/n)(1-1/m)=c(n-1)(m-1)$, and the same for $\sigma^2$. $\square$

Reading. The correlation of the raw estimated effects, which is the registered K2 estimator, therefore tends to $(\sigma_{ab}+c/m)/\sqrt{(\sigma_a^2+\sigma^2/m)(\sigma_b^2+\sigma'^2/m)}$, not to the generalized reciprocity. Put in the real estimates ($\sigma_a^2\approx0.98$, $\sigma_b^2\approx0.52$, $\sigma^2\approx1.89$, $c\approx0.21\times1.89$, $m\approx13$) and a true value near 0: this gives about 0.03, which matches the observed 0.036 and the simulated mean 0.031. With missing ratings, $c/m$ becomes $c\cdot n^{\mathrm{both}}_i/(n^{\mathrm{given}}_in^{\mathrm{received}}_i)$ per person, which is what the script uses (approximate). The ratio of corrected moments is consistent but can be unstable when a corrected variance is near zero (events of 3 or 4 a side; tested in `tests/test_part3.py`).

## 6c. The Nadeau-Bengio correction on grouped folds (proved for fixed losses, 2026-10-15; for Jenna's review)

Part 4 found that the Nadeau-Bengio corrected resampled t, run with whole events in each fold, covered close to
90% with narrower intervals than our repair. The proposition below explains why. It concerns the test-sampling
part of the variance, conditioning on the losses (no refitting).

Setup:

- Losses $\ell_1,\dots,\ell_n$ are fixed, in $G$ balanced clusters of $m$ rows with $G=Kq$. Write $\bar y$ for their
  mean, $\bar c_g$ for the cluster means, $s_c^2=\frac{1}{G-1}\sum_g(\bar c_g-\bar y)^2$, and
  $s^2=\frac{1}{n-1}\sum_i(\ell_i-\bar y)^2$.
- A *grouped* partition deals the clusters at random into $K$ folds of $q$ clusters each. A *row* partition deals the
  rows at random into $K$ folds of $n/K$ rows each.
- Take $R$ independent partitions and let $e_1,\dots,e_J$ ($J=RK$) be the fold mean losses. The Nadeau-Bengio
  variance is $\hat\sigma^2_{\mathrm{NB}}=\big(\frac1J+\frac{1}{K-1}\big)S^2$, with $S^2$ the sample variance of the
  $e_j$.

Proposition 6. Averaging over the random partitions,

$$E\big[\hat\sigma^2_{\mathrm{NB}}\big]=\frac{J+K-1}{J-1}\cdot\frac{s_c^2}{G}\quad\text{(grouped)},\qquad E\big[\hat\sigma^2_{\mathrm{NB}}\big]=\frac{J+K-1}{J-1}\cdot\frac{s^2}{n}\quad\text{(rows)}.$$

With balanced clusters, $s_c^2/G$ is exactly the cluster-robust variance of $\bar y$ (the $G/(G-1)$ form of Section 4), and
$s^2/n$ is the naive variance. So on grouped folds the test-sampling part of the Nadeau-Bengio variance is the
cluster-robust variance times $1+K/(J-1)$. On row folds it is the naive variance times the same factor.

Proof.

1. Every partition's folds average to $\bar y$, so the mean of all $J$ fold means is $\bar y$ exactly. Hence
   $E[S^2]=\frac{1}{J-1}\sum_jE[(e_j-\bar y)^2]=\frac{J}{J-1}\operatorname{Var}_{\mathrm{sel}}(e_j)$.
2. A grouped fold mean is the mean of $q$ cluster means drawn without replacement from $G$. Its selection variance is
   $(1-q/G)\,s_c^2/q=(K-1)s_c^2/G$; for rows it is $(K-1)s^2/n$.
3. Multiply by $\frac1J+\frac{1}{K-1}=\frac{K-1+J}{J(K-1)}$. $\square$

Consequences.

1. **Under clustering.** Under the two-level model, $E[s_c^2/G]\approx\operatorname{Var}(\bar y)$, while $E[s^2/n]\approx\sigma^2/n$
   falls short of $\operatorname{Var}(\bar y)=\sigma^2(1+(m-1)\rho)/n$ by the Kish design effect. The correction therefore
   handles clustering only because the folds are grouped. On row folds it inherits the naive variance's shortfall,
   exactly $1+(m-1)\rho$. Registered prediction N7 tests this.
2. **One CV run ($R=1$, $J=K$).** The factor is $(2K-1)/(K-1)$, which is 2.25 at $K=5$. Applied to a single $K$-fold
   run, the correction overstates the test-sampling variance by that much.
3. **With refitting.** Fold errors also vary because the fitted model changes from fold to fold. That extra spread
   in $S^2$ is the training-set variability that the cluster-robust SE misses ($\bar c$ in Proposition 2). Conjecture,
   to be checked in N6's grid and then proved: the excess of $\hat\sigma^2_{\mathrm{NB}}$ over
   $\frac{J+K-1}{J-1}\hat V_{\mathrm{CR}}$ tracks $\bar c$.

## 7. First simulation (done; to be extended)

Design. $G=40$ clusters of $m=20$ rows, $p=10$ covariates, logistic regression fitted by IRLS with a tiny ridge, 0-1 loss, $K=5$. Data: $x_i=u_{g(i)}+v_i$ with $u_g,v_i\sim N(0,I)$ (half the covariate variance is between clusters), $y_i\sim\mathrm{Bernoulli}(\sigma(x_i^\top\beta+a_{g(i)}))$ with $a_g\sim N(0,1)$ and $\beta=(1,1,0.5,0.5,0,\dots,0)$. The truth $\mathrm{Err}^{\mathrm{new}}(D)$ for the full-data model is computed on 300 fresh clusters. 300 replicates; nested procedures average over $R=20$ fold assignments. The Bates, Hastie and Tibshirani bias adjustment is omitted. Script: `sim_cv_cluster.py`.

Point estimates. Mean $\mathrm{Err}^{\mathrm{new}}$ was 0.239. Grouped CV averaged 0.242 (bias $+0.003$, within noise); random-split CV averaged 0.232 (bias $-0.008$, about one standard deviation of the estimate), the leakage of Lemma 1's remark.

Variance. The empirical variance of the grouped CV estimate was $5.0\times10^{-4}$. The naive estimate averaged $2.3\times10^{-4}$ (ratio 2.2, the design effect of the Corollary). The cluster-robust estimate averaged $4.3\times10^{-4}$ (ratio 1.17; the remaining 15% is $\bar c$ of Proposition 2). The cluster-nested MSE estimate averaged $6.8\times10^{-4}$ against an empirical MSE of $5.6\times10^{-4}$, slightly conservative as Proposition 3 predicts for a size-$n'$ estimate; the independent-rows correction averaged $1.7\times10^{-3}$, three times too large.

Coverage of 90% intervals for $\mathrm{Err}^{\mathrm{new}}(D)$:

| Procedure | Coverage | Mean half-width |
| --- | --- | --- |
| Random-split CV, naive SE | 0.70 | 0.025 |
| Random-split CV, nested CV as published | 0.65 | 0.025 |
| Grouped CV, naive SE | 0.67 | 0.025 |
| Grouped CV, cluster-robust SE | 0.84 | 0.034 |
| Grouped CV, nested with independent-rows $b_k$ | 0.98 | 0.066 |
| Grouped CV, cluster-nested | 0.81 | 0.040 |
| Grouped CV, floor of cluster-robust and cluster-nested | 0.91 | 0.044 |
| Oracle: constant SE equal to the true root MSE | 0.91 | 0.039 |

Reading. Random-split intervals fail for two reasons at once, wrong target and too narrow, and the published nested CV does not help because the target is wrong. Grouping fixes the target; the cluster-robust SE fixes most of the width; cluster-nested has the right size on average (its mean half-width matches the oracle) but is too variable, and its misses occur exactly where it comes out small; the floor restores nominal coverage at a cost of about 12% extra width over the oracle. Increasing $R$ from 2 to 20 moved cluster-nested coverage from 0.63 to 0.81, so part of the remaining gap is Monte Carlo variance that a larger $R$ or a larger $K$ would reduce.

## 7b. First run on the real data (done)

Speed-dating data, 7,031 dates with complete ratings from 537 raters in 21 waves; logistic regression predicting a yes from the six attribute ratings and the date's position in the evening; 0-1 loss; $K=5$. Script: `layerA_real_data.py`.

| Fold scheme | CV error | Naive SE | Cluster-robust SE | Ratio |
| --- | --- | --- | --- | --- |
| Random split | 0.251 | 0.0052 | n/a | |
| Grouped by rater | 0.254 | 0.0052 | 0.0080 (rater), 0.0083 (rater and ratee two-way) | 1.5 to 1.6 |
| Grouped by wave | 0.253 | 0.0052 | 0.0080 (wave), 0.0084 (two-way) | 1.5 to 1.6 |
| Grouped by rater, cluster-nested | 0.254 | | 0.0091 | 1.8 |

Reading. The within-rater correlation of per-point errors is 0.09 with a median of 14 dates per rater, so the design effect of the Corollary predicts a variance ratio of about 2.2, and the cluster-robust estimators find 2.4 to 2.6; cluster-nested, which also picks up the between-fold term, finds 3.1. The naive 90% half-width of 0.0085 should be about 0.013 to 0.015 on this data. The leakage from random splits is small here (0.003, inside the noise), because a logistic model on the ratings alone has little cluster-level structure to exploit; it should be re-checked with trees and with rater-level covariates in Part 1. Wave-level and rater-level cluster-robust estimates agree once the unbalanced-cluster form of the estimator is used (the sum over clusters of squared within-cluster residual sums, with the $G/(G-1)$ factor), which is what the script implements; the balanced formula in Section 4 is the special case.

## 8. What to prove and run next

1. Add the Bates, Hastie and Tibshirani bias adjustment and their recommended $R$; rerun. Expected: cluster-nested alone approaches 0.88 to 0.90.
2. Replace the floor by an estimate of $\bar c$: by Proposition 2, $V=E[\hat V_{\mathrm{CR}}]+\bar c$, and $\bar c$ is a between-fold covariance that the nested structure can estimate directly from the cross-fold products $(\bar e_g-\bar e)(\bar e_{g'}-\bar e)$ averaged over clusters in different folds. Prove its unbiasedness and compare its variance with $\widehat{\mathrm{MSE}}_{\mathrm{CN}}$.
3. Unbalanced clusters: Propositions 1 and 2 extend with $m$ replaced by the appropriate weighted moments of the cluster sizes; write the general statement.
4. Prove Proposition 4 and compute the finite-sample term for the balanced bipartite design.
5. Simulate the dyadic case with the speed-dating dimensions (21 waves, 10 to 20 per side), comparing wave-cluster and multiway $b_k$.
6. Then run the whole procedure on the real data for the Part 1 models, which is Layer A of the proposal.

Items 1 to 3 are a week for someone who likes this material, with the simulation harness already written. Items 4 and 5 are the part that would be new to the literature as far as we can tell.

Status, 2026-10-15:

- Item 1 is done. The bias adjustment runs in `sim_cv_matched.py` (A2: 92.6% coverage at nominal 90% with 21 events) and `errorbars_real.py`.
- Item 4 is proved for one direction (Proposition 4(ii)); the two-direction case is open.
- Item 5 ran in `sim_cv_designs.py` (18 designs, plus the exploratory dyadic design). N2 is supported: the
  published random-row interval covers 46% to 74%. N1 is not supported: the cluster-nested, bias-adjusted interval
  covers 86.5% to 92% with the real 21 events, but 83% to 86% with 10 redrawn events and 73.5% to 82% with 40. The
  main cause is the variance of the estimate (Section 5, last remark). In 4% to 15% of data sets the MSE estimate
  is zero or negative, which gives a zero-width interval; among the rest the interval covers 84% to 98%. The floor
  of Section 5 covered 92% to 100% in every design. That is a post hoc finding, to be registered before it is tested.
- **Why the cluster-robust SE over-covers in A2 (99%).** The A2 design and the 21-event designs reuse the real events
  in every replicate, so the variation from drawing new events is absent. The cluster-robust SE still counts it,
  averaging 0.017 against a true spread of 0.010 to 0.017. With events redrawn it matches the spread, and its
  coverage falls to 84% to 94.5%. This is Proposition 2 seen from the other side: $\hat V_{\mathrm{CR}}$ estimates
  the random-design variance.
- **Part 4 results (`sim_cv_repair.py`, events redrawn in every data set).** The floor covered 89% to 96.5% in all
  15 designs with person or event effects (N3 supported). With A2's events redrawn, the cluster-robust interval
  covered 87.4% instead of 99.2%, which confirms the fixed-design explanation (N4). The cluster-nested,
  bias-adjusted interval covered 83.6% instead of 92.6% (N5). Exploratory: the Nadeau-Bengio corrected t on
  event folds covered 90% to 97.5%, with a mean half-width of 0.038 against 0.047 for the floor. Next for theory:
  derive the expectation of the Nadeau-Bengio variance under Proposition 1's covariance structure, and find when
  its $1/(K-1)$ term matches $\bar c$. Next for simulation: a pre-registered confirmatory run on fresh seeds.
- **Estimating $\bar c$ directly (item 2): not from one CV run.** The obvious estimator averages the cross-products
  $(\bar e_g-\bar e)(\bar e_{g'}-\bar e)$ over pairs of clusters in different folds. With balanced clusters,
  $\sum_{g\ne g'}(\bar e_g-\bar e)(\bar e_{g'}-\bar e)=\big(\sum_g(\bar e_g-\bar e)\big)^2-\sum_g(\bar e_g-\bar e)^2=-\sum_g(\bar e_g-\bar e)^2$. The
  sum over different-fold pairs is the same quantity minus the same-fold pairs, so it is determined by the
  cluster-robust sum and the within-fold structure, and adds no information about $\bar c$. A direct estimate needs
  variation across fold assignments or training sets. Candidates are covariances of cluster errors across the $R$
  repeated assignments, or the nested structure's outer-fold errors. This is open, for Jenna. Meanwhile Part 4
  tests the floor (N3) and other repairs empirically (`sim_cv_repair.py`).
- **The dyadic design.** Within-rater CV is biased by +0.024 for the full-data error (training-size effect), so
  every interval missed. Against the spread of the CV estimate (0.0065), the rater-clustered and ratee-clustered
  SEs fit (0.0064 and 0.0065), the two-way and dyadic-robust SEs are about 15% wide, and the naive SE is 20% narrow.
- Item 6 is done for logistic regression and LDA (`errorbars_real.py`).
- The literature check (proposals/07) changes the framing. Propositions 1 to 3 extend Bates, Hastie and Tibshirani, Bengio and Grandvalet, and Bayle et al. with clusters as the units; they are not new theory. The candidate contribution is the combined procedure and its coverage evidence at about 21 clusters.

## 9. References

- Bates, Hastie, Tibshirani. "Cross-Validation: What Does It Estimate and How Well Does It Do It?" JASA, 2024.
- Bengio, Grandvalet. "No Unbiased Estimator of the Variance of K-Fold Cross-Validation." JMLR, 2004.
- Cameron, Gelbach, Miller. "Robust Inference with Multiway Clustering." Journal of Business and Economic Statistics, 2011.
- Kish. Survey Sampling. Wiley, 1965 (the design effect).
- Roberts et al. "Cross-Validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic Structure." Ecography, 2017 (blocked CV and target mismatch, empirical).
- Kenny, Kashy, Cook. Dyadic Data Analysis. Guilford, 2006 (the social relations model).
