# Adversarial review of Propositions 9 (iii) and 10 (proposals/06, section 6f)

Written 2026-10-16 by an automated reviewer (a Claude subagent), which was told to find errors and to settle the
clustered case of c2 = C_mix / 2. Its scripts and logs are in the session scratchpad, not in this repository, so the
numbers below are its own unless marked "verified". Jenna's review is still to come.

Two claims were checked independently on fresh seeds:

- **K = 2 rerun** (design 4, G = 80, 10,000 data sets, seeds 20265906 + chunk): -1.075 (SE 0.014) against -1.142
  predicted with the G/(G-1) factor. That is a 5.8% shortfall; the median is -0.52 and the skewness -2.39. Verified.
- **Two-point random intercept of +-3**, with the project's binned constants (seeds 31 to 33): c2 / C_mix is 0.452
  to 0.456. With no intercept it is 0.496 to 0.500. Verified.

## Confirmed

1. **The first-order mixing term does not depend on K.** Given a match in fold k, a non-match's cluster is in another
   fold with probability G(K-1)/(K(G-1)). Times the shift 1/n' = K/(Gm(K-1)), that is 1/(m(G-1)): K cancels.
   - The signs and the conditioning on (x_j, y_j) are right.
   - No first-order term is missing. In particular, the "fold composition x fold-model intercept" term is the same
     mixing term written another way, and random fold sizes enter only at O(G^-3/2).
2. **The second-order algebra is right.** Cov(eps0, xi) + Var(xi)/2 = (K/(K-1)) x_i' V_delta x_j. Times the
   cross-fold share, this gives c2/((K-1)G).
3. **An order-by-order check.** The linearized fold-model deviations were scaled by kappa and a polynomial in kappa
   was fitted (design 4).
   - Linear coefficient: -0.749 +- 0.011 against -0.763 predicted (K = 2, G = 80); -0.772 +- 0.008 against -0.763
     (K = 5).
   - Quadratic coefficient: -0.403 +- 0.008 against -0.382 (K = 2); -0.103 +- 0.004 against -0.096 (K = 5).
4. **The ratio biases of pooled-between and BCWF-pooled are equal to leading order.** Both are 4c'/(G mu^2), and they
   first differ at O(K/G^2). The per-fold average carries K times as much.
5. **Proposition 10's proof is correct line by line.** Exact quadrature gives c2 / C_mix = 0.500000 in design 0, and
   an independent kernel Monte Carlo gives 0.5007 (SE 0.0006).

## Errors and corrections

- **E1. The constant is C_mix / (G - 1), not C_mix / G.** The difference is o(1/G) but amounts to 1% to 2.6% at the G
  used. With exact constants and the factor, N17's gaps become -1.0%, -0.8%, +0.5% and +3.7%.
- **E2. Most of N18's miss was sampling error.**
  - At K = 2 the statistic is roughly -(a1 - a2)^2, which is heavily skewed, so 400 data sets understate its SE.
  - With 10,000 data sets the shortfall is 7% (reviewer) or 5.8% (verified), not 18.6%.
  - The rest is a third-order term, (1/2) E[delta''(D) m_ij Var_cross,ij]. It scales with K/(K-1)^2 and decays
    roughly like 1/G (cubic coefficient +0.080 +- 0.005 at G = 80, +0.011 +- 0.002 at G = 320).
  - The registered verdict (not supported) stands.
- **E3. The displayed identity in item 2 drops a factor K/(K-1).** It holds only after the cross-fold share (K-1)/K
  is applied. The final c2/((K-1)G) is right.
- **E4. "c2 / C_mix is 0.49 to 0.53" is numerical noise for the linear designs.** The exact values are 0.50000,
  0.50000, 0.50183, 0.50027, 0.50104 and 0.50316. The binned estimator cannot resolve deviations below about 1%.
  The random-slope design is genuinely above 1/2: 0.522 to 0.528.

## Gaps

- Proposition 9 (iii) is a formal expansion with no remainder bound. It assumes a smooth density for D, enough
  moments, and balanced clusters.
- **Proposition 10 needs less than stated.**
  - Gaussian covariates are not needed.
  - On D = 0 the scores are equal, so p_i = p_j and (1 - p_i + p_j) = 1. The closed form is therefore
    -((2K-1)/(2(K-1))) (1/n) E[delta(D) x_i' H^-1 x_j].
  - Shorter proof:
    1. Condition on the score s. Under correct specification x is independent of y given s, so
       I(t) = integral of f1(s+t) f0(s) Q(s+t, s) ds, with Q(a, b) = mu(a)' H^-1 mu(b) symmetric.
    2. Average the i-shifted and j-shifted derivatives: I'(0) = (1/2) integral (f1' f0 - f1 f0') Q
       = (1/2) integral f1 f0 (logit P(y | s))' Q = (1/2) integral f1 f0 Q.
  - With logistic, t5, Laplace, bimodal and chi-square covariates the ratio was 0.49993 to 0.49999.

## The clustered case: false in general

- **Lemma.** Assume a single index (y independent of x given s) and a correct marginal logistic model. Let
  B(x) = E[s_h | x, y = 1] - E[s_h | x, y = 0]. Then c2 = C_mix / 2 whenever B(x) is affine in x. Independent rows give
  B = x, which is Proposition 10. Under clustering, B also contains the cluster mates' conditional score difference,
  which is not affine in general.
- **Why the project's designs give about 1/2.** With a Gaussian intercept the mates' term is nearly affine, so the
  ratio minus 1/2 is O(sa^2) with a tiny coefficient: 1.4e-3 at sa = 0.8.
- **Counterexamples** (exact quadrature):
  - a two-point intercept of +-3 gives 0.454 (verified with the project's code: 0.452 to 0.456), and +-4 gives 0.422;
  - a Gaussian intercept with shared covariates (sa = 1, su = 3, m = 400) gives 0.475 to 0.563;
  - a bridge-distributed intercept (marginal model exactly logistic, phi = 0.9) gives 0.507.
- The closed form C_mix (2K-1)/(2(K-1)) is therefore an approximation. In the project's designs its error is under
  0.4%.

## Where the reviewer was unsure

- At K = 2 a residual shortfall of about 2% at large G is not excluded (fit a + b/G gives a = 0.043 +- 0.017).
- The ratio-bias equality was tested empirically in design 4 only.
