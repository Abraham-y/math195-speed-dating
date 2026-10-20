# Theory Track: Cross-Validation Intervals When Rows Are Clustered

First pass, 2026-09-24. Companion to the Layer A section of the speed-dating proposal. Status of each part is marked: proved, proposed, simulated, or open.

## 0. Summary

Three things are established below, in increasing order of novelty.

1. Grouped cross-validation (whole clusters held out) and random-split cross-validation estimate different quantities: the error on rows from new clusters versus the error on rows from clusters already seen. This is a one-line consequence of exchangeability (Lemma 1), and it means that no interval can fix a random-split estimate when the question is about new people.

2. Under grouped folds, the variance of the CV estimate equals the independent-rows formula of Bates, Hastie and Tibshirani (2024) plus one extra term, $(m-1)\,\delta_c/n$, where $m$ is the cluster size and $\delta_c$ is the excess covariance between per-point errors in the same cluster (Proposition 1). The usual standard error misses that term. When the between-cluster covariances are nonnegative, it is too small by at least the survey-sampling design effect $1+(m-1)\rho_c$ (Corollary). In expectation, a cluster-robust variance recovers the cluster term; its remaining bias is the between-cluster covariance $\bar c$ that shared training data induce (Proposition 2), and $\bar c$ can have either sign. Proposition 2 concerns $\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}})$. For an interval around the full model's new-cluster error, the relevant variance is $\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}}-\mathrm{Err}_{\mathrm{new}})$, which the cluster-robust variance estimates to first order (Proposition 8, section 6e).

3. Nested cross-validation, the independent-rows fix, can be adapted to clustered data by holding out clusters at both levels and replacing its within-fold variance correction with a cluster-robust one. The resulting MSE estimate is unbiased for the MSE of a grouped CV estimate (Proposition 3); using the original correction under clustering is provably conservative. The adapted estimate is noisy at five folds, and a first simulation shows that flooring it at the cluster-robust variance gives nominal coverage where each component alone does not.

The cross-classified (dyadic) case, where every row belongs to one rater and one ratee, is set up in Section 6 with the estimators stated and the proofs left open.

4. Sections 6c to 6e were added on 2026-10-15 and reviewed adversarially the same day; the review's corrections are folded in.
   - Proposition 6: on grouped folds, the Nadeau-Bengio variance is the random-group (cluster-robust) variance times $(J+K-1)/(J-1)$.
   - Proposition 7: coverage studies differ depending on whether the clusters are fixed or redrawn.
   - Proposition 8: a first-order theory of what grouped CV estimates and which error bar fits it.
   - The simulations that test them, and a statement of where the error-bar track stands.

## 1. Setup and notation

Rows $z_i=(x_i,y_i)$, $i=1,\dots,n$, each in exactly one cluster $g(i)\in\{1,\dots,G\}$, balanced so $n=Gm$. Two-level model: cluster effects $U_1,\dots,U_G$ are i.i.d.; given $U_g$, the rows of cluster $g$ are i.i.d. from a distribution $P_{U_g}$. Rows in different clusters are independent; rows in the same cluster are exchangeable but dependent.

A learner maps a training set $D$ to a fitted function $\hat f_D$ and is symmetric (does not depend on the order of training rows). A loss $\ell(z;f)$ with finite second moments (bounded in Sections 2 to 5; log-loss, used in Section 6e, is not bounded). Folds $F_1,\dots,F_K$, assigned at random and independently of the data. Grouped folds: each cluster lies entirely in one fold, $G/K$ clusters per fold. Random-split folds: rows assigned individually, so clusters straddle folds.

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

Corollary (design effect). Let $\hat V_{\mathrm{naive}}=s^2/n$ with $s^2$ the sample variance of the $e_i$, and $V=\mathrm{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}})$. For any $e_1,\dots,e_n$ with common mean and common variance $\sigma^2$, $E[s^2]=\frac{n}{n-1}(\sigma^2-V)$, hence

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

Reading. For balanced clusters, the bias of the cluster-robust variance as an estimate of $\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}})$ is exactly $\bar c$, the average covariance between errors in different clusters. It contains no $m$ or $\delta_c$ explicitly, though $\sigma_{sf}$ and $\sigma_{df}$ still depend on the cluster structure through the training sets. For $m=1$ the cluster-robust and naive estimates coincide, and Proposition 2 reduces to Bates, Hastie and Tibshirani's observation. Two corrections from the 2026-10-15 review:

1. **$\bar c$ need not be nonnegative.** By Proposition 8, $\bar c=[2\operatorname{Cov}(a,\beta)+\operatorname{Var}(\beta)]/G+o(1/G)$, which is first order and of either sign. In the review's Monte Carlo with 0-1 loss, the cluster-robust variance overstated $\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}})$ by about 25% at every $G$.
2. **$\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}})$ is not the interval's target.** For the full model's new-cluster error the relevant variance is $\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}}-\mathrm{Err}_{\mathrm{new}})$, which the cluster-robust variance estimates to first order (Proposition 8 (i)).

The proposition is exact only for balanced clusters; the unbalanced estimator the scripts use is not covered.

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

Remark (variance of the estimate). Proposition 3 is about the mean. The estimate is a difference of two noisy quantities, each averaged over only $K$ folds per assignment, and it is negative in a few percent of replicates at $K=5$, $R=20$. Its misses are concentrated where it happens to be small. $\hat V_{\mathrm{CR}}$ is a low-variance estimate, so a natural stabilizer is the floor below. The original rationale was that $\hat V_{\mathrm{CR}}$ is biased downward by $\bar c\ge0$. As the review showed, $\bar c$ can be negative, and the interval's target is $\mathrm{Err}_{\mathrm{new}}$, for which $\hat V_{\mathrm{CR}}$ is first-order right (Proposition 8). So the floor is conservative by design rather than a correction of a known bias.

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

Proposition 6. Taking expectation over the random partitions, given the losses (the $R$ partitions need not be independent),

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

Where this sits in the literature (literature check, 2026-10-15). The $J=K$ case is the random group method of
survey sampling: Wolter (2007), *Introduction to Variance Estimation*, Chapter 2 (pp. 21-106; the theorem number
2.4.4 recorded earlier could not be verified in an open copy, 2026-10-19). There, averaged over random
divisions of a fixed sample, the random-group variance equals the with-replacement variance, and in multistage
samples the groups must be formed from whole clusters (his rule iii). Nadeau and Bengio derived their
$\rho_0=n_2/n$ from the same fixed-loss approximation, but for random train/test splits. Bouckaert and Frank (2004)
plugged it into repeated $K$-fold without a derivation. Markatou et al. (2005, Table 6) found empirically that the
correction on a single $K$-fold run is about twice their variance approximation. We found no statement of three
things: the factor for $R$ repeated partitions, the link between the corrected repeated $K$-fold variance and the
random-group (cluster-robust) variance, or any analysis of the corrected t under grouped folds. The honest claim:
Proposition 6 is the random-group estimator rewritten in cross-validation notation, with a small exact extension and
a reading for grouped against row folds. The literature search did not cover Google Scholar or theses.

Consequences.

1. **Under clustering.** Under the two-level model, $E[s_c^2/G]\approx\operatorname{Var}(\bar y)$, while $E[s^2/n]\approx\sigma^2/n$
   falls short of $\operatorname{Var}(\bar y)=\sigma^2(1+(m-1)\rho)/n$ by the Kish design effect. Strictly, the realized shortfall
   is $m s_c^2/s^2$, which equals $\frac{n-1}{n-m}[1+(m-1)\rho]$ with Cochran's finite-population $\rho$. The correction therefore
   handles clustering only because the folds are grouped; on row folds it inherits the naive variance's shortfall.
   Registered prediction N7 tests this, and it held: row-fold coverage was 51% to 76%.
2. **One CV run ($R=1$, $J=K$).** The factor is $(2K-1)/(K-1)$, which is 2.25 at $K=5$. Applied to a single $K$-fold
   run, the correction overstates the test-sampling variance by that much.
3. **With refitting.** Fold errors also vary because the fitted model changes from fold to fold. The original
   conjecture was that the excess of $\hat\sigma^2_{\mathrm{NB}}$ over $\frac{J+K-1}{J-1}\hat V_{\mathrm{CR}}$ tracks $\bar c$. The review
   refuted it. By Proposition 8, that excess is about $\frac{J+K-1}{J-1}[\operatorname{Var}(\beta)/(K-1)^2-2\operatorname{Cov}(a,\beta)/(K-1)]/G$,
   while $\bar c\approx[\operatorname{Var}(\beta)+2\operatorname{Cov}(a,\beta)]/G$. In the review's Monte Carlo the two had opposite signs
   (+0.0046/G against -0.0053/G).

## 6d. Fixed against redrawn clusters in a coverage study (proved, 2026-10-15)

Proposition 7. Hold the clusters fixed, in balanced clusters of $m$ rows. Cluster $g$ has expected loss
$\mu_g$, which depends on its fixed composition. Losses are $\ell_i=\mu_{g(i)}+u_{g(i)}+v_i$, with independent
mean-zero $u_g$ (variance $\sigma_u^2$) and $v_i$ (variance $\sigma_v^2$). Let $S_\mu^2=\frac{1}{G-1}\sum_g(\mu_g-\bar\mu)^2$. Then

$$\operatorname{Var}(\bar y\mid\text{design})=\frac{\sigma_u^2}{G}+\frac{\sigma_v^2}{n},\qquad E\big[\hat V_{\mathrm{CR}}\mid\text{design}\big]=\operatorname{Var}(\bar y\mid\text{design})+\frac{S_\mu^2}{G}.$$

If instead the clusters are redrawn, with $\mu_g$ i.i.d. of variance $\sigma_\mu^2$, then
$E[\hat V_{\mathrm{CR}}]=\operatorname{Var}(\bar y)=(\sigma_\mu^2+\sigma_u^2)/G+\sigma_v^2/n$ exactly.

Proof. The cluster means are $\bar c_g=\mu_g+u_g+\bar v_g$. With the design fixed,
$E\sum_g(\bar c_g-\bar y)^2=\sum_g(\mu_g-\bar\mu)^2+(G-1)(\sigma_u^2+\sigma_v^2/m)$. Divide by $G(G-1)$. With the design random, add the
expectation of $S_\mu^2$, which is $\sigma_\mu^2$. $\square$

Reading. A coverage study that reuses the same clusters in every replicate measures the conditional variance,
while the cluster-robust variance includes the spread of the clusters' expected losses. So in such a study it
over-covers by an amount set by how different the clusters are. The proposition has no refitting: with shared
fitted models, $\bar c$ adds to the comparison (Proposition 2), and A2's events are also unbalanced. So A2's 99.2% with
the 21 real events fixed, against 87.4% with them redrawn (N4), is consistent with Proposition 7 but not
quantitatively explained by it. A fixed-design study must also state its target. In the 21-event designs, the mean cluster-robust SE was 0.017,
against an SD of the CV estimate of 0.010 to 0.017 and an SD of CV minus Err_new (the relevant target) of 0.008 to
0.013. The distinction is the classical one between design-based and model-based variance. The practical lesson
is for simulation studies of CV error bars: redraw the clusters, or the cluster-robust SE will look better than it
is.

## 6e. First-order theory: what grouped CV estimates and which error bar fits it (derived 2026-10-15; to be tested in Part 6)

Setup:

- Clusters are i.i.d. and balanced. A model is fitted by an M-estimator on a set $T$ of clusters, with the usual
  first-order expansion $\hat\theta_T-\theta^*\approx\frac{1}{|T|}\sum_{g\in T}\psi_g$; the influence functions $\psi_g$ are i.i.d.
  across clusters with mean 0.
- The evaluation loss may differ from the training criterion (here 0-1 loss with a threshold at the training
  match rate, while the model is fitted by log-likelihood). Let $A_h(\theta)$ be cluster $h$'s mean evaluation
  loss, $A(\theta)=E[A_h(\theta)]$, $a_h=A_h(\theta^*)$, $\mu=A(\theta^*)$, $d=\nabla A(\theta^*)$ and $\beta_g=d^\top\psi_g$.

Assumptions (stated after the 2026-10-15 review):

- $K$ divides $G$, and the partition is a uniform random balanced grouped one.
- $\hat\theta$ is $\sqrt G$-consistent with the stated linearization, and $\theta^*$ is interior and unique.
- Smooth losses: $A_h$ and $A$ are twice differentiable near $\theta^*$, with finite second moments.
- 0-1 loss:
  - the threshold is part of $\theta$, with influence $\psi^t_g=\bar y_g-t^*$, and $\partial A/\partial t$ is included in $d$;
  - the linear score has a bounded continuous density near the threshold.

Proposition 8 (first order, balanced grouped $K$-fold CV). To first order,

$$\widehat{\mathrm{Err}}_{\mathrm{CV}}=\bar a+\bar\beta+r_1,\qquad \mathrm{Err}_{\mathrm{new}}(\hat\theta_{\mathrm{all}})=\mu+\bar\beta+r_2,\qquad \widehat{\mathrm{Err}}_{\mathrm{CV}}-\mathrm{Err}_{\mathrm{new}}=\bar a-\mu+r_1-r_2 .$$

For smooth losses $r_1-r_2=O_p(1/G)$; this includes an $O(1/G)$ bias from the smaller training sets. For 0-1 loss
it is only $O_p(G^{-3/4})$: a cluster's loss is a step function of $\theta$, so its change is bounded by a
flip probability of order $|\hat\theta-\theta^*|$. In both cases it is $o_p(G^{-1/2})$, which is enough for (i) to (iii):

- (i) $G\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}}-\mathrm{Err}_{\mathrm{new}})\to\operatorname{Var}(a)$ (with uniform integrability), and $G\hat V_{\mathrm{CR}}\to_p\operatorname{Var}(a)$.
- (ii) The fold errors of $R$ grouped partitions are $\bar c_{F_j}+K\bar\beta/(K-1)$ with $c_h=a_h-\beta_h/(K-1)$, the same in every
  partition. So by Proposition 6, $G\,E[\hat\sigma^2_{\mathrm{NB}}]\to\frac{J+K-1}{J-1}\operatorname{Var}(a-\beta/(K-1))$. In expectation, the
  Nadeau-Bengio variance therefore exceeds the relevant variance whenever $\operatorname{Var}(\beta)>2(K-1)\operatorname{Cov}(a,\beta)$
  (sufficient, not necessary, at finite $J$).
- (iii) $\operatorname{Corr}(\widehat{\mathrm{Err}}_{\mathrm{CV}},\mathrm{Err}_{\mathrm{new}})\to\frac{\operatorname{Cov}(a,\beta)+\operatorname{Var}(\beta)}{\sqrt{\operatorname{Var}(a+\beta)\operatorname{Var}(\beta)}}$ when $\operatorname{Var}(\beta)>0$.
- (iv) Suppose three things hold:
  - the evaluation loss is the training criterion, with any penalty $o(1)$;
  - training and evaluation weight rows the same way (balanced clusters, or a correctly specified model);
  - $\theta^*$ is interior and unique.
  Then $d=0$ and $\beta\equiv0$, and the Nadeau-Bengio variance tends to $\frac{J+K-1}{J-1}$ times the cluster-robust variance.
  $\mathrm{Err}_{\mathrm{new}}$ then varies at second order, and the correlation goes to 0 at rate $G^{-1/2}$. In the review's
  Monte Carlo, $\sqrt G\cdot\operatorname{Corr}$ stayed near -1, so the correlation was -0.23 at G = 20, -0.11 at 80 and -0.05 at 640.
  With unbalanced events and a misspecified marginal model (for example an omitted random intercept), $d\ne0$ even
  for log-loss.

Derivation.

1. Condition on $D_{-k}$; the test clusters of fold $k$ are independent of $\hat\theta_T$. Then
   $\frac1q\sum_{h\in F}[A_h(\hat\theta_T)-A_h(\theta^*)]-[A(\hat\theta_T)-A(\theta^*)]$ has mean 0 and variance at most
   $E[(A_h(\hat\theta_T)-A_h(\theta^*))^2]/q$. That is $O(1/G^2)$ for smooth losses and $O(G^{-3/2})$ for 0-1 loss.
2. $A(\hat\theta_T)-A(\theta^*)=d^\top(\hat\theta_T-\theta^*)+O_p(1/G)$ (Taylor, plus the linearization remainder).
3. So fold $j$'s error is $\bar a_{F_j}+\bar\beta_{T_j}$ plus remainders. Each cluster is in $K-1$ training sets of equal size,
   so averaging over folds gives $\bar a+\bar\beta$.
4. The full model's error is $A(\hat\theta_{\mathrm{all}})=\mu+\bar\beta+O_p(1/G)$.
5. Within a partition, $\bar\beta_{T_j}-\bar\beta=-\frac{1}{K-1}(\bar\beta_{F_j}-\bar\beta)$, which gives (ii) through Proposition 6. $\square$

Relation to the literature.

- Bayle et al. (2020) target the average of the $K$ fold models' conditional test errors. They assume i.i.d.
  points and a loss-stability condition, but no smoothness. Applied with clusters as the units, their central limit
  theorem already gives (i) for that target, including 0-1 loss.
- Proposition 8's addition is the step to the full-data model. Here
  $\frac1K\sum_jA(\hat\theta_{T_j})-A(\hat\theta_{\mathrm{all}})=O_p(1/G)$, because $\frac1K\sum_j\bar\beta_{T_j}=\bar\beta$. That step needs smooth $A$, and it
  carries an $O(1/G)$ training-size bias.
- Wager (2020, arXiv 1909.11696) shows that CV's leading error term does not depend on the model and that the
  conditional risk moves at lower order. That overlaps (i) and (iv).
- What is left as plausibly new: the Nadeau-Bengio limit in (ii), and its quantitative check.

Caveats, seen in Part 5's grid:

1. Real events are unbalanced (25 to 484 dyads), so the cancellation in step 3 is only approximate.
2. Drawing 10 of 21 events without replacement adds a finite-population effect. The simple factor
   $1-10/21=0.52$ does not match the observed mean ratio of 0.75 (0.72 with the target's Monte Carlo variance
   removed), so this explanation is qualitative only.
3. At $G=21$ and 40 the mean ratio was 1.11 and 1.28 (1.04 and 1.16 corrected); Part 7 later traced most of that to
   the imprecise target.
4. Unequal fold sizes (negative exploratory pilot, seed 777, 30 partitions, real data). With fixed losses, dividing
   each fold's residual total by $n/K$ rather than by its own size restores Proposition 6 exactly for any cluster
   sizes, because the derivation only samples cluster totals without replacement. On the real models, though, that
   changed the Nadeau-Bengio ratio by at most 0.05: 1.31 to 1.35, 0.97 to 0.95, 1.48 to 1.46 and 1.45 to 1.41, with
   fold sizes varying by a CV of 0.31 to 0.35. Size imbalance is not what makes the correction conservative here,
   so this was not registered.

The review's own Monte Carlo (balanced, G = 20 to 640, 3,000 replicates each) confirmed (i) to (iii):

- (i): $G\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}}-\mathrm{Err}_{\mathrm{new}})/\operatorname{Var}(a)$ went 1.12 to 0.96 for 0-1 loss and 1.32 to 1.02 for log-loss.
- (ii): $G\,E[\hat\sigma^2_{\mathrm{NB}}]/(\text{factor}\cdot\operatorname{Var}(c))$ went 1.03 to 1.00.
- (iii): the correlation was 0.14 to 0.20 for 0-1 loss, against 0.17 predicted.

**Out-of-sample test of (ii)** (Part 8, `sim_cv_kfold.py`; N14 and N15 supported). Before running, the variance
components were estimated from Part 6b's K = 5 runs and used to predict the Nadeau-Bengio ratio at other fold
counts.

| Fold count | Predicted (0-1 loss) | Observed at G = 80, 160, 320 |
| --- | --- | --- |
| K = 10 | 1.082 | 1.071, 1.082, 1.084 |
| K = 3 | 1.452 | 1.505, 1.474, 1.463 (converging as G grows) |
| K = 2 (exploratory) | 2.112 | 2.18 to 2.25 |

Under log-loss the ratio stayed within 0.035 of 1 at K = 3, 5 and 10 (0.984 to 1.032), and was 1.03 to 1.06 at
K = 2. The K = 2 miss is 3% to 7%, as expected when half the
data is held out, where the first-order expansion is least accurate. Proposition 8 (ii) is therefore a quantitative
theory: it predicts how conservative the Nadeau-Bengio correction is at any fold count, from three variance
components.

Part 6 tests the proposition on balanced synthetic clusters over a wide range of $G$.

**Results** (2026-10-15; `sim_cv_firstorder.py`, `firstorder_check.py`).

- **Claim (ii), the Nadeau-Bengio limit, confirmed (N9).** The ratio of the Nadeau-Bengio variance to the scaled
  cluster-robust variance tends to 1 under log-loss (1.005 and 0.994 at G = 160 and 320). Under 0-1 loss it stays
  near 1.2 at every G (1.15 to 1.22). Quantitatively, the predicted ratio $\operatorname{Var}(a-\beta/4)/\operatorname{Var}(a)$, built from the
  simulated variances and covariance, matched the observed ratio at all five G in the precise run (1.193 against
  1.183, 1.196 against 1.187, 1.205 against 1.209, 1.203 against 1.206, 1.189 against 1.193), each inside the
  prediction's 90% bootstrap interval. $\operatorname{Corr}(a,\beta)$ is about -0.5.
- **Claim (iv), the correlation, confirmed (N10).** Under log-loss, $\operatorname{Corr}(\widehat{\mathrm{Err}}_{\mathrm{CV}},\mathrm{Err}_{\mathrm{new}})$ is between -0.11 and
  -0.02. Under 0-1 loss it is positive, 0.08 to 0.21.
- **Coverage confirmed (N11).** The cluster-robust interval with $t_{G-1}$ quantiles covered $\mathrm{Err}_{\mathrm{new}}$ 87.5% to 90.2% at every
  G and for both losses in the registered run, and 87.5% to 91.5% in the Part 6b replication.
- **Claim (i), the variance ratio, not confirmed at the registered precision.** N8, with a band of [0.85, 1.15], and
  N8b, with [0.9, 1.1] after removing the target's Monte Carlo variance, both failed: values ran from 0.91 to 1.17.
  Pooled over G >= 80 and both losses, the corrected ratio is 1.05 (90% bootstrap interval 0.98 to 1.11). That is
  consistent with 1. The registered bands were too tight for 400 data sets, where each ratio has about 7% sampling
  error.
- **A negative exploratory pilot.** Two fold counts (K = 5 and K = 2) were tried as a way to separate
  $\operatorname{Var}(\beta)$ and $\operatorname{Cov}(a,\beta)$, which would also give an interval for the average error. It was too noisy and
  biased to use: in 60 data sets at G = 80 (seed 999), the estimate of $\operatorname{Var}(\beta)$ had SD 2.6 times its
  true value and came out negative 22% of the time. Halving the training set probably breaks the first-order
  expansion. It was dropped and not registered.
- **A caveat for every realistic-design simulation** (A2, the design grid, the repair grid, the confirmatory grid).
  Their target was scored on 300 new events, whose own Monte Carlo SD is about 0.0045. That is 3% to 10% of
  $\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}}-\mathrm{Err}_{\mathrm{new}})$, so all reported coverages there are slightly understated, most at G = 40. With
  it removed, the cluster-robust variance ratio in the confirmatory grid was 0.55 to 0.84 at G = 10 (mean 0.72; a
  finite-population effect, since those 10 events are drawn without replacement from 21), 0.96 to 1.20 at G = 21
  (mean 1.04), and 0.89 to 1.46 at G = 40 (mean 1.16), over the four designs at each G. (An earlier version of this
  bullet quoted single designs: 0.78, 1.05, and 1.18 and 1.46.) The G = 40 excess in the realistic designs is not explained yet. Fold-size imbalance
  is too small to account for it; this is an open question.
- **Resolved by Part 7** (`sim_cv_sizes.py`). The realistic designs were rerun with G = 21 and 40 (events redrawn),
  the target scored on 2,000 new events, and fresh seeds. The cluster-robust variance ratio, corrected for the
  target's Monte Carlo variance, was 0.93 to 1.15, and coverage was:
  - 82% to 90% for the plain cluster-robust interval with normal quantiles;
  - 86% to 91% with t quantiles on G - 1 (CR2) or on G* - 1 (G* about 14 at G = 21 and 27 at G = 40);
  - 85% to 91% for the wild cluster bootstrap-t (N12 supported).
  The plain interval no longer fell below 85% at G = 40 (N13 not supported). So the earlier G = 40 shortfall was
  mostly the imprecise target plus sampling variation, not unequal event sizes.

**Where the error-bar track stands.**

1. Hold out whole clusters. This fixes what is being estimated (Lemma 1).
2. Use the cluster-robust SE with t quantiles on G - 1 (or on the effective number of clusters), or the wild
   cluster bootstrap-t. To first order this is the right variance for the full model's new-cluster error
   (Proposition 8, and Bayle et al. with clusters as units). It covered 85% to 92% in every design with a precise
   target.
3. The Nadeau-Bengio correction on grouped folds is conservative by a factor the theory predicts (Proposition 8 (ii);
   predicted and observed within about 1%). On random-row folds it fails by the design effect (Proposition 6, N7).
4. Bates, Hastie and Tibshirani's nested CV as published, with random-row folds, fails (N2). Our cluster-nested
   version is unbiased in mean (Proposition 3) but noisy.
5. For coverage studies: redraw the clusters (Proposition 7) and score the target precisely (Part 6b, Part 7).
   Otherwise the coverage numbers are distorted.
6. For the AUC, do not pool held-out scores across fold models when clusters differ in outcome rate. The pooled CV
   AUC is biased downward by a predictable amount of order 1/G (Proposition 9, Part 9). Use the between-cluster
   within-fold AUC, pooled over folds, with the Obuchowski SE and t quantiles; it covered 85% to 94% with 40 or more
   clusters (82% to 91% at 20). For AUC the
   Nadeau-Bengio correction is not conservative (ratio about 1).

## 6f. The cross-validated AUC under clustering (derived 2026-10-15; tested in Part 9)

The project reports a pooled event-CV AUC: one AUC over all held-out scores, which come from $K$ different fold
models. Its target here is the full model's AUC on new clusters, pooled over clusters. That is
$A(\theta)=P\{s_\theta(X_1)>s_\theta(X_0)\}$ for a match and a non-match from different clusters, with
$s_\theta(x)=\tilde x^\top\theta$.

Setup: Proposition 8's (balanced clusters of size $m$, $G$ clusters, $n=Gm$, uniform balanced grouped folds,
$\hat\theta_T-\theta^*\approx\frac1{|T|}\sum_{g\in T}\psi_g$, $H$ the per-row Fisher information of the logistic fit, match
rate $\pi$). Further notation:

- $F_0$ and $S_1$ are the population distribution of non-match scores and survival function of match scores at
  $\theta^*$.
- Cluster $h$'s Hajek projection is
  $a_h=\frac{1}{m\pi}\sum_{i\in h,\,y_i=1}\{F_0(s^*_i)-A^*\}+\frac{1}{m(1-\pi)}\sum_{i\in h,\,y_i=0}\{S_1(s^*_i)-A^*\}$.
- $d=\nabla A(\theta^*)$ and $\beta_h=d^\top\psi_h$.
- $s_h=\sum_{i\in h}\tilde x_i(y_i-p^*_i)$ is the cluster's score sum, and $D=s^*_i-s^*_j$ for a match–non-match pair from
  different clusters.

**Proposition 9.**

- **(i) First order.** For the pooled, fold-averaged and between-cluster within-fold (BCWF) estimates $\hat A$,
  $\hat A-A(\hat\theta_{\rm all})=\bar a+o_p(G^{-1/2})$. The Obuchowski variance of the pooled scores satisfies $G\hat V\to_p\operatorname{Var}(a)$.
  The reason: a cross-fold pair compares two models, but $A(\theta_k,\theta_{k'})$ expanded to first order sums,
  over the folds, to $\nabla A^\top(\hat\theta_{\rm all}-\theta^*)$. The intercept parts cancel because the AUC does not
  depend on location.
- **(ii) Nadeau–Bengio.** As in Proposition 8 (ii), $G\,E[\hat\sigma^2_{\rm NB}]\to\frac{J+K-1}{J-1}\operatorname{Var}(a-\beta/(K-1))$. Since $A$ does not
  depend on location or scale, $d$ has no intercept part and $d\perp\theta^*$. When the model's index direction
  maximizes the pooled AUC among linear scores, $d=0$. That holds under correct specification (Neyman–Pearson)
  and for monotone single-index truths with elliptical $x$ (Li and Duan 1989). Otherwise $d$ is first order in the
  misspecification angle. In every design computed (Part 9's components, including a hinge and a random slope),
  $|d|\le0.014$, and the predicted ratio is 0.99 to 1.003. Unlike the error rate, whose ratio is about 1.2, the
  AUC's Nadeau–Bengio correction is not conservative.
- **(iii) Second order: three biases of order $1/G$.**
  1. *Same-cluster pairs.* A within-fold AUC puts weight $f\approx(K/G)\kappa$ on match–non-match pairs from the same
     cluster, with $\kappa=E[m_{1h}m_{0h}]/(m^2\pi(1-\pi))$. Their AUC $A_w$ differs from the between-cluster $A_b$. So the
     fold-averaged estimate carries $K\kappa(A_w-A_b)/G$, and the pooled estimate $\kappa(A_w-A_b)/G$. The sign is that of
     $A_w-A_b$: positive when clusters share outcome noise, negative when they share signal.
  2. *Mixing.* In a cross-fold pair ($i$ in fold $k$, $j$ in fold $k'$), model $k$ was fitted on $j$'s cluster and model
     $k'$ on $i$'s. So the score difference is shifted by
     $\{\tilde x_i^\top H^{-1}s_{h(j)}-\tilde x_j^\top H^{-1}s_{h(i)}\}/n'$, with $n'=n(K-1)/K$. A within-fold pair from different
     clusters has no such shift. Given a match in fold $k$, a non-match from another cluster lies in another fold with
     probability $G(K-1)/(K(G-1))$. Times $1/n'$ that is $1/(m(G-1))$, so the leading bias is
     $$\frac{C_{\rm mix}}{G-1},\qquad C_{\rm mix}=\frac1m E\big[\delta(D)\{\tilde x_i^\top H^{-1}s_{h(j)}-\tilde x_j^\top H^{-1}s_{h(i)}\}\big],$$
     and it does not depend on $K$. (The first version wrote $C_{\rm mix}/G$, using a cross-fold share of $(K-1)/K$. The
     difference is $o(1/G)$ but 1% to 2.6% at the G used; review, proposals/09, E1.) With outcome clustering, $s_{h(j)}$ given a non-match $j$ points toward a lower
     intercept, and the reverse for a match, so $C_{\rm mix}<0$. Next, the two fold models of a cross-fold pair differ
     by noise $\xi=-\tilde x_j^\top(\theta_{k'}-\theta_k)$, correlated with the common error. A second-order Taylor step in the
     Heaviside function gives, per cross-fold pair,
     $E[\delta'(D)\{\operatorname{Cov}(\varepsilon_0,\xi)+\tfrac12\operatorname{Var}\xi\}]=\frac{K}{K-1}E[\delta'(D)\,\tilde x_i^\top\operatorname{Var}(\psi)\tilde x_j]/((K-1)G)$.
     Times the cross-fold share this is $c_2/((K-1)(G-1))$, with $c_2=-g'(0)$ and
     $g(t)=E[\tilde x_i^\top\operatorname{Var}(\psi)\tilde x_j\,\delta(D-t)]$. (The first version dropped the factor $K/(K-1)$ in the per-pair identity,
     although its final term was right; review E3.) In total,
     $$E[\text{pooled between-cluster}-\text{BCWF-pooled}]=\frac1{G-1}\Big(C_{\rm mix}+\frac{c_2}{K-1}\Big)+o(1/G).$$
     The next term is $\tfrac12E[\delta''(D)\,m_{ij}\operatorname{Var}_{{\rm cross},ij}]$, the mixing shift times the cross-fold noise
     variance. It scales with $K/(K-1)^2$ and is of order $1/G^2$ (review E2). The ratio biases of the two estimates
     (random pair counts in the denominator) are equal to leading order: both are $4c'/(G\mu^2)$, and they first differ
     at $O(K/G^2)$. So the difference isolates the mixing. The per-fold average carries $K$ times that ratio bias
     (confirmed by the review).
  3. *Training size.* Every CV model sees $(K-1)/K$ of the clusters, so all CV estimates carry
     $\operatorname{tr}(H_A\operatorname{Var}\psi)/(2(K-1)G)$, with $H_A$ the AUC Hessian in the slopes. This term is small because $A$ is
     flat near its maximum.

Remarks.

- The mixing bias is the pooling bias of Forman and Scholz (2010) and of Parker et al. (2007), seen through
  clustering. With independent rows it is of order $1/n$ and negligible. With clusters it is of order $1/G$, larger
  by about the design effect of the residuals. At $G=20$ with a cluster SD of 2 on the logit scale it is about one
  standard error.
- The leading term does not shrink with more folds. Only the $c_2$ part does.
- In the linear designs $c_2/C_{\rm mix}$ is close to 1/2: by exact quadrature (review E4) it is 0.50000, 0.50000, 0.50183,
  0.50027, 0.50104 and 0.50316. The binned estimates' spread of 0.49 to 0.50 was numerical noise; that estimator
  cannot resolve deviations below about 1%. With random slopes the ratio is genuinely higher, 0.52 to 0.53. Where
  $c_2=C_{\rm mix}/2$, the mixing bias is $C_{\rm mix}(2K-1)/(2(K-1))/(G-1)$. Proposition 10 proves this for independent rows; with
  clustering it is false in general (below).

**Proposition 10 (the factor 1/2; proved for independent rows, 2026-10-15; generalized 2026-10-16).** Suppose the
rows are independent, the logistic model is correctly specified, and the score densities are smooth. The covariates
need not be Gaussian. Then $c_2=C_{\rm mix}/2$ exactly, and the pooling bias of $K$-fold CV is
$$-\frac{2K-1}{2(K-1)}\cdot\frac1n\,E\big[\delta(D)\,\tilde x_i^\top H^{-1}\tilde x_j\big]$$
to leading order, plus the training-size term. (On $D=0$ the two scores are equal, so $p_i=p_j$; the factor $(1-p_i+p_j)$ in
the first version is identically 1 there.) This is a closed form for the pessimistic bias of pooled CV AUC that
Airola et al. and Parker et al. report.

*Proof* (the shorter, general version from the review, proposals/09).

1. As in step 1 below, $C_{\rm mix}=-\frac1mE[\delta(D)\,q]$ with $q=\tilde x_i^\top H^{-1}\tilde x_j$, and $c_2=-\frac1mI'(0)$ with $I(t)=E[q\,\delta(D-t)]$.
2. Under correct specification $P(y=1\mid x)$ depends on $x$ only through $s$, so $x$ is independent of $y$ given $s$. Let
   $\tilde\mu(s)=E[\tilde x\mid s]$ and $Q(a,b)=\tilde\mu(a)^\top H^{-1}\tilde\mu(b)$, which is symmetric. Then
   $I(t)=\int f_1(s+t)f_0(s)Q(s+t,s)\,ds=\int f_1(u)f_0(u-t)Q(u,u-t)\,du$.
3. Average the derivatives of the two forms at $t=0$. Because $Q$ is symmetric, the $\partial Q$ terms cancel, which leaves
   $I'(0)=\tfrac12\int(f_1'f_0-f_1f_0')Q(s,s)\,ds=\tfrac12\int f_1f_0\,(\operatorname{logit}P(y=1\mid s))'\,Q(s,s)\,ds$.
4. Under a correct logistic model $(\operatorname{logit}P)'=1$, so $I'(0)=\tfrac12E[\delta(D)q]$ and $c_2=C_{\rm mix}/2$. $\square$

*First proof* (Gaussian covariates, 2026-10-15).

1. With independent rows, $E[s_{h(j)}\mid j]=-p_j\tilde x_j$ for a non-match and $(1-p_i)\tilde x_i$ for a match, and
   $\operatorname{Var}\psi=H^{-1}/m$. So $C_{\rm mix}=-\frac1mE[\delta(D)(1-p_i+p_j)q]$, with $q=\tilde x_i^\top H^{-1}\tilde x_j$, and $c_2=-\frac1mI'(0)$, with
   $I(t)=E[q\,\delta(D-t)]$.
2. Take a direction $v$ with $v^\top\theta=1$. Integrate $I'(t)=-E[q\,\delta'(D-t)]$ by parts along $v$ in $x_i$, and separately in
   $x_j$, and average the two:
   $$I'(0)=\tfrac12E\big[\delta(D)\{\partial_{v,i}q-\partial_{v,j}q+q(\partial_v\log f_1(x_i)-\partial_v\log f_0(x_j))\}\big].$$
3. The logistic link gives $\partial_v\log f_1=(1-p)+\partial_v\log f$ and $\partial_v\log f_0=-p+\partial_v\log f$, with $f$ the covariate
   density.
4. For Gaussian $x$, choose $v=\Sigma\theta/\theta^\top\Sigma\theta$. Then $\partial_v\log f(x)=-(s-\bar s)/\sigma_s^2$, which takes the same value for
   $x_i$ and $x_j$ when $D=0$, so those terms cancel.
5. What is left is $\tfrac12\tilde v^\top H^{-1}E[\delta(D)(\tilde x_j-\tilde x_i)]=-\tfrac12\tilde v^\top H^{-1}\nabla A(\theta^*)$, which is 0 under correct
   specification.
6. So $I'(0)=\tfrac12E[\delta(D)(1-p_i+p_j)q]$, and $c_2=C_{\rm mix}/2$. $\square$

Notes on Proposition 10:

- The numbers agree. In design 0 (no clustering, correct model) exact quadrature gives 0.500000 and the binned estimate
  0.4985. With logistic, t5, Laplace, bimodal and chi-square covariates the ratio was 0.49993 to 0.49999 (review).
- Under misspecification the identity is off by $-\tilde v^\top H^{-1}\nabla A/(2m)$, which is small because $\nabla A$ is small.
- **With clustering the identity is false in general** (review, proposals/09).
  - Let $B(x)=E[s_h\mid x,y=1]-E[s_h\mid x,y=0]$. Under a single index and a correct marginal logistic model,
    $c_2=C_{\rm mix}/2$ holds whenever $B$ is affine in $\tilde x$. Independent rows give $B=\tilde x$.
  - With clusters, $B$ also carries the cluster mates' conditional score difference, which is not affine in general.
  - With a Gaussian random intercept it is nearly affine, so the ratio minus 1/2 is only $O(s_a^2)$ with a tiny
    coefficient ($1.4\times10^{-3}$ at $s_a=0.8$). That is why the project's designs give about 1/2.
  - Counterexamples:
    - a two-point intercept of $\pm3$: 0.454 by quadrature, and 0.452 to 0.456 with the project's binned code on fresh
      seeds (checked independently);
    - a Gaussian intercept with strongly shared covariates: 0.475 to 0.563;
    - a bridge-distributed intercept, whose marginal model is exactly logistic: 0.507.
  - The closed form $C_{\rm mix}(2K-1)/(2(K-1))$ is therefore an approximation under clustering. Its error is under 0.4% in
    the project's designs.
- Which target? For ranking pairs across events (an app), the pooled between-cluster AUC is right, and BCWF
  estimates it without the mixing and same-cluster biases. For ranking within an event (a matchmaker), the target is
  $A_w$, and a within-cluster AUC estimates it.

Status: (i) and (ii) are derived as in Proposition 8. (iii) is a heuristic second-order expansion, with constants
computed numerically from the population (`sim_cv_auc.py --components`, tested in `tests/test_part9.py`). It is
tested in Part 9 (N16 to N21) on fresh seeds, after two labelled pilots in which the two-term expression matched the
mixing bias to within about 5%. An adversarial review (proposals/09, 2026-10-16) confirmed the first- and
second-order terms order by order. It corrected $G$ to $G-1$ and one displayed identity, generalized Proposition 10,
and refuted the clustered conjecture. Jenna's review is still to come.

**Results** (Part 9, 2026-10-15; `sim_cv_auc.py`, 400 data sets per cell; five of six hypotheses supported).

- **The pooled CV AUC is biased downward under outcome clustering (N16 supported).** The bias was -0.004 to -0.040
  in all 12 cells, with every 90% interval below 0. Times G it is roughly constant, so it is of order $1/G$: on
  average -0.81 with a cluster SD of 2 (range -0.72 to -0.99) and -0.29 with 1 (range -0.18 to -0.47). With no
  clustering at all it is under 0.003. With covariate clustering alone it is -0.006 at G = 20 and -0.003 at G = 40,
  mostly the same-cluster-pair term.
- **The mixing term is predicted quantitatively (N17 supported).** The observed G x (pooled between-cluster minus
  BCWF-pooled) was within 1.3% to 5.1% of $C_{\rm mix}+c_2/4$ in all four designs:

  | Design (sa, su) | Predicted | Observed |
  | --- | --- | --- |
  | 1, 0 | -0.300 | -0.304 |
  | 1, 1 | -0.236 | -0.242 |
  | 2, 0 | -0.846 | -0.865 |
  | 2, 1 | -0.708 | -0.745 |

- **The fold-count structure held in three of four out-of-sample cells (N18 not supported).** At $K=10$ the
  predictions held (-0.282 against -0.281, and -0.794 against -0.783), and at $K=2$ with cluster SD 1 (-0.400 against
  -0.397). At $K=2$ with cluster SD 2 the registered run gave -0.917 against -1.127, off by 18.6% against a 15% band.
  That verdict stands. The review (proposals/09, E2) traced most of the miss to sampling error:
  - At $K=2$ the statistic is roughly minus a squared difference of two fold means, so it is heavily skewed (skewness
    -2.4). 400 data sets understate its SE.
  - A 10,000-data-set rerun on fresh seeds (exploratory, seeds 20265906 + chunk) gives -1.075 (SE 0.014) against
    -1.142 with the $G/(G-1)$ factor. That is a 5.8% shortfall; the reviewer's own rerun gives 7%.
  - The rest is the third-order term, which scales with $K/(K-1)^2$ and decays with G.
  - With exact constants and the $G/(G-1)$ factor, N17's K = 5 gaps become -1.0%, -0.8%, +0.5% and +3.7%.
- **BCWF-pooled removes the bias (N19 supported).** Its absolute bias was at most 0.11 SE in all 18 cells with
  $G\ge40$. In the cluster SD 2 cells at G = 40 and 80, the pooled bias was 0.49 to 0.87 SE.
- **Coverage (N20 supported).** The BCWF-pooled interval ($t_{G-1}$, Obuchowski SE) covered 89.9% on average over the
  18 cells with $G\ge40$, ranging from 84.8% to 93.8%. The pooled interval covered 84.1% on average in the cluster SD 2
  cells, down to 70.5% at G = 20. The wild cluster bootstrap does not rescue the pooled interval (69.5% at G = 20),
  because the problem is bias, not variance.
- **The Nadeau–Bengio ratio for AUC is about 1 (N21 supported).** It was 0.978 to 0.999 against predictions of 0.991 to
  1.003, in the linear, hinge and random-slope designs at G = 80 and 160. For the error rate it is about 1.2.
- **Exploratory.**
  - The fold-averaged estimate followed $(c_{\rm train}+K\kappa(A_w-A_b))/G$ closely where the same-cluster term dominates
    (cluster SD 0, su = 1: -0.0150 against -0.0137 at G = 20, and -0.0015 against -0.0017 at G = 160). With cluster SD
    2 it ran about 30% above prediction, which is the per-fold ratio bias that pooling counts across folds removes:
    per-fold BCWF was +0.0079 against +0.0011 for BCWF-pooled at G = 20.
  - In the realistic A2 design at 21 real events (`auc_estimators_real.py`), the pooled interval covered 77% to 87%
    and BCWF-pooled 85% to 87%, with biases of -0.004 to -0.015 against at most 0.002.
  - On the real data the four estimates agree on average over 20 event partitions: pooled minus BCWF-pooled is
    -0.004 to +0.002. So the project's headline AUCs are not materially biased on average. The gap's SD across
    partitions is 0.005 to 0.009, and on the headline folds the gap is -0.003 to -0.012.
  - **The theory, applied to our data.** Proposition 9's constants can be estimated from a single data set: the
    full-data fit, its event score sums, and the cluster-robust $\operatorname{Var}\hat\theta$ (`auc_estimators_real.py --plugin`). For
    the full logistic model the predicted mixing bias at $K=5$ is -0.0010 on `pre` and +0.0003 on `pre+date`. The
    observed means over 20 partitions are -0.0005 (SE 0.0019) and +0.0009 (SE 0.0012). Our events differ too little in
    residual match rate, once the predictors are in, for mixing to matter. The plug-in also gives any project a
    way to check whether its pooled CV AUC needs the BCWF fix.

## 7. First simulation (done; to be extended)

Design. $G=40$ clusters of $m=20$ rows, $p=10$ covariates, logistic regression fitted by IRLS with a tiny ridge, 0-1 loss, $K=5$. Data: $x_i=u_{g(i)}+v_i$ with $u_g,v_i\sim N(0,I)$ (half the covariate variance is between clusters), $y_i\sim\mathrm{Bernoulli}(\sigma(x_i^\top\beta+a_{g(i)}))$ with $a_g\sim N(0,1)$ and $\beta=(1,1,0.5,0.5,0,\dots,0)$. The truth $\mathrm{Err}^{\mathrm{new}}(D)$ for the full-data model is computed on 300 fresh clusters. 300 replicates; nested procedures average over $R=20$ fold assignments. The Bates, Hastie and Tibshirani bias adjustment is omitted. Script: `sim_cv_cluster.py`.

Point estimates. Mean $\mathrm{Err}^{\mathrm{new}}$ was 0.239. Grouped CV averaged 0.242 (bias $+0.003$, within noise); random-split CV averaged 0.232 (bias $-0.008$, about one standard deviation of the estimate), the leakage of Lemma 1's remark.

Variance. The empirical variance of the grouped CV estimate was $5.0\times10^{-4}$. The naive estimate averaged $2.3\times10^{-4}$ (ratio 2.2, the design effect of the Corollary). The cluster-robust estimate averaged $4.3\times10^{-4}$ (ratio 1.17; a later review noted that the interval's target is $\mathrm{Err}_{\mathrm{new}}$, not $\operatorname{Var}(\widehat{\mathrm{Err}}_{\mathrm{CV}})$, so this gap should not be read as $\bar c$). The cluster-nested MSE estimate averaged $6.8\times10^{-4}$ against an empirical MSE of $5.6\times10^{-4}$, slightly conservative as Proposition 3 predicts for a size-$n'$ estimate; the independent-rows correction averaged $1.7\times10^{-3}$, three times too large.

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
  averaging 0.017 against an SD of CV minus Err_new of 0.008 to 0.013 (the SD of the CV estimate alone is 0.010 to
  0.017). With events redrawn it matches the spread, and its
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
- The literature check (proposals/07) changes the framing. Propositions 1 to 3 extend Bates, Hastie and Tibshirani, Bengio and Grandvalet, and Bayle et al. with clusters as the units; they are not new theory. The candidate contribution was then the combined procedure and its coverage evidence at about 21 clusters. That is superseded (2026-10-15): the procedure under-covers with events redrawn (N1, and 83.6% in N5), while existing intervals applied at the event level reach the target (N4, N11, N12). The plausibly new piece is now Proposition 8 (ii) and its out-of-sample check (N14); see "Where the error-bar track stands" in Section 6e.

## 9. References

- Bates, Hastie, Tibshirani. "Cross-Validation: What Does It Estimate and How Well Does It Do It?" JASA, 2024.
- Bengio, Grandvalet. "No Unbiased Estimator of the Variance of K-Fold Cross-Validation." JMLR, 2004.
- Cameron, Gelbach, Miller. "Robust Inference with Multiway Clustering." Journal of Business and Economic Statistics, 2011.
- Kish. Survey Sampling. Wiley, 1965 (the design effect).
- Roberts et al. "Cross-Validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic Structure." Ecography, 2017 (blocked CV and target mismatch, empirical).
- Kenny, Kashy, Cook. Dyadic Data Analysis. Guilford, 2006 (the social relations model).
