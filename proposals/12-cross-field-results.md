# Part 12: the clustered-CV lessons in four other fields (results, descriptive)

Run 2026-11-05. Code: `scripts/cross_field.py` and `scripts/external_data.py`, tested in `tests/test_cross_field.py`
(10 tests), committed before any run on real outcomes (884a414). Two later commits, both made before the runs they
affect, fixed a crash and added a summary column (4805b61, e865a12). Data provenance: `data/external_sources.md`.

**Status: descriptive.** The team decided on 2026-11-05 not to pre-register Part 12. There are no hypotheses or
bands. Every number below is a measurement, and the choices behind it (predictors, models, fold counts, seeds) were
fixed in the committed code before the first run. A registered confirmatory test can be added later if a preprint
needs one.

Outputs:
- `results/cross_field_summary.csv`, `results/cross_field_partitions_<dataset>.csv` and
  `results/cross_field_folds_<dataset>.csv` (Stage A);
- `results/cross_field_stageb.csv` and `results/cross_field_stageb_summary.csv` (Stage B);
- four figures, `results/cross_field_{leakage,se_ratio,pooled,coverage}.png`, copied to `report/figures/`.

## The question

Parts 4 to 11 found three problems with cross-validation on the speed-dating data, where dates come in events:
- random-row folds overstate accuracy;
- the usual error bar is too narrow;
- pooling held-out scores across fold models biases the AUC low.

Are these facts about speed dating, or about any data in which rows come in groups? Part 12 runs the same checks on
four public datasets from other fields.

## What was run

| Dataset | Field | One row is | Group | Groups | Rows | Outcome | Base rate | Spread of group rates (logit SD) |
|---|---|---|---|---|---|---|---|---|
| MIT-BIH Arrhythmia | cardiology | a heartbeat | patient | 43 | 100,645 | ventricular beat | 0.070 | 2.82 (11 patients have none) |
| OULAD | education | a course registration | course run | 22 | 28,061 | fail or withdraw | 0.453 | 0.38 |
| ACS PUMS, California 2018 | census | a worker | area (PUMA) | 265 | 195,665 | income over $50k | 0.411 | 0.64 |
| ExtraSensory: sleeping | phone sensing | a minute | user | 53 | 285,268 | asleep | 0.291 | 1.22 |
| ExtraSensory: at work | phone sensing | a minute | user | 32 | 201,727 | at main workplace | 0.168 | 1.37 |

Predictors are fixed per dataset (`scripts/external_data.py` docstring):
- heartbeats: only the spacing between beats, with no waveform;
- courses: demographics plus the first 14 days of activity;
- census: the folktables ACSIncome variables without race;
- phone sensing: the 225 sensor features.

Models, both with fixed settings and no tuning:
- logistic regression (C = 1, median imputation, standardized, missing flags);
- gradient boosting (sklearn HistGradientBoosting, 200 rounds, learning rate 0.1, 31 leaves).

**Stage A** (every dataset): 20 random grouped 5-fold partitions (whole groups held out) and 5 random-row 5-fold
partitions. Each fit uses only its training folds for imputation, scaling and the classification threshold (the
training base rate). From each partition we take:
- AUC (pooled, BCWF-pooled, fold-averaged and within-group);
- the error rate and the log loss;
- the naive and the cluster-robust standard error of each.

**Stage B** (census only, where the truth can be computed): the 265 areas are the population. Each replicate does
three things:
1. draws G = 20 or 40 areas as "the study";
2. runs grouped 5-fold CV inside them;
3. fits the full model on the study and scores it on the other 225 to 245 areas, which serves as the true answer.

There are 400 replicates for each G. The interval methods are:
- the usual (naive) interval;
- cluster-robust t with G − 1 degrees of freedom;
- the wild cluster bootstrap-t with 999 draws (error rate only).

## Results in four plain claims

### 1. Hold out people, not rows (`cross_field_leakage.png`)

Random-row CV puts some of each person's rows in training and the rest in test, so a model can score well by recognising
the person. AUC with random rows minus AUC with whole groups held out:

| Dataset | Logistic | Boosting |
|---|---|---|
| Heartbeats | 0.929 → 0.947 (+0.018) | 0.910 → 0.994 (+0.084) |
| Phone: at work | 0.853 → 0.938 (+0.085) | 0.910 → 0.997 (+0.087) |
| Phone: sleeping | 0.938 → 0.969 (+0.030) | 0.944 → 0.998 (+0.055) |
| Courses | 0.703 → 0.713 (+0.010) | 0.703 → 0.715 (+0.012) |
| Census | 0.888 → 0.888 (+0.000) | 0.903 → 0.903 (+0.000) |

Leakage needs two things:
- **groups that differ:** on the outcome rate, the census areas differ more than the course runs do (logit SD 0.64 against 0.38);
- **predictors that can pick out the group:** occupation, education and hours already explain most of the area-to-area
  difference, so a census model gains nothing from seeing other workers from the same area.

Boosting leaks more than logistic regression where the groups are very different, because it can learn a
person's signature from beat spacing or sensor readings. On heartbeats, random-row CV makes boosting look better than
logistic regression (0.994 against 0.947). Holding out whole patients reverses the order (0.910 against 0.929).

### 2. Count groups, not rows (`cross_field_se_ratio.png`)

The usual standard error treats every row as an independent observation. The cluster-robust one counts groups. The
ratio, cluster-robust over naive, for logistic regression on grouped folds:

| | Heartbeats | Phone: at work | Phone: sleeping | Courses | Census |
|---|---|---|---|---|---|
| Error rate | 22.1× | 21.2× | 18.9× | 3.3× | 1.9× |
| AUC | 10.4× | 23.3× | 23.8× | 3.5× | 2.1× |

Boosting gives ratios of the same size (error rate 2.0× to 20.4×).

The census row is the instructive one. Leakage there is zero (claim 1), so the *number* from random-row CV is
right, but its *error bar* is still half as wide as it should be. The classical design effect explains why. The
variance is inflated by about 1 + (m − 1)ρ, where:
- m is the group size (about 790 workers per area, size-weighted);
- ρ is the correlation of the losses within a group.

A ratio of 1.9 needs only ρ ≈ 0.003. For heartbeats, ρ ≈ 0.20 with about 2,500 beats per patient gives the factor
of 22.

### 3. Check against the truth (`cross_field_coverage.png`)

How often the 90% interval contained the true answer, over 400 census replicates:

| | Usual | Cluster-robust t | Wild cluster bootstrap |
|---|---|---|---|
| Error rate, G = 20 | 0.59–0.62 | 0.88–0.90 | 0.88–0.90 |
| Error rate, G = 40 | 0.54–0.56 | 0.82–0.86 | 0.82–0.86 |
| AUC, G = 20 | 0.47–0.57 | 0.83–0.88 (BCWF) | |
| AUC, G = 40 | 0.44–0.49 | 0.82–0.85 (BCWF) | |

Ranges are over the two models.

- The usual interval missed the truth about half the time, and its variance was 3 to 6 times too small.
- The cluster-robust interval came close. At G = 40 it was a little narrow: the variance of (CV estimate − truth)
  was 1.16 to 1.21 times its average estimate, net of the Monte Carlo noise in the target. Speed dating gave the
  same shortfall (proposals/06 section 6e: "about 17% below its target"). At G = 20 the ratio was 0.86 to 0.94, and
  the t quantile makes coverage close to nominal.
- The wild bootstrap gave the same coverage as the cluster-robust t to within 0.005.

### 4. Don't pool scores across fold models (`cross_field_pooled.png`)

A pooled AUC ranks a test row scored by fold model 1 against a test row scored by fold model 3. The fixed
(BCWF-pooled) version compares only rows scored by the same model, and only rows from different groups. Pooled minus
fixed, mean over 20 partitions (SE):

| Dataset | Logistic | Boosting |
|---|---|---|
| Heartbeats (43) | −0.0052 (0.0018) | −0.0088 (0.0021) |
| Phone: at work (32) | −0.0062 (0.0018) | −0.0043 (0.0016) |
| Courses (22) | −0.0029 (0.0008) | −0.0021 (0.0007) |
| Phone: sleeping (53) | −0.0008 (0.0005) | −0.0008 (0.0005) |
| Census (265) | −0.00006 (0.00003) | −0.00005 (0.00003) |

Every difference has the sign the theory predicts (Proposition 9). Its size follows a rough ranking of group
differences over the number of groups. In Stage B, the pooled AUC's bias against the truth roughly halves when G
doubles:

| | Logistic | Boosting |
|---|---|---|
| Pooled bias, G = 20 | −0.0024 | −0.0036 |
| Pooled bias, G = 40 | −0.0010 | −0.0017 |
| BCWF bias, G = 20 | −0.0014 | −0.0027 |
| BCWF bias, G = 40 | −0.0003 | −0.0012 |

The halving is consistent with order 1/G. The fixed version removes between a quarter and 70% of the bias. The remainder
is the usual cost of training each fold model on 80% of the groups.

**Honest size check.** The largest mixing bias, −0.009 on heartbeats, is about 0.4 of the honest standard error
(0.021). In these datasets it is a systematic but small correction. Claims 1 to 3 change conclusions; claim 4 rarely
will on its own.

## For the appendix, not the talk

- **Nadeau-Bengio against cluster-robust (Proposition 6).** For fixed losses, the Nadeau-Bengio variance should be
  (J + K − 1)/(J − 1) times the cluster-robust variance; the test of that formula gives a ratio near 1. With refitting
  on real data, the ratio over the 10 dataset-model cases is:
  - error rate: 1.03 to 1.50 (median 1.21);
  - log loss: 0.94 to 1.45;
  - AUC: 0.80 to 1.97 (the largest is heartbeats with logistic).

  The fixed-loss link therefore holds to within a factor of about 1.5 for the error rate. Refitting adds variance the
  formula does not see.
- **Stage B variance ratios** (variance of CV − truth over the mean estimated variance, net of the target's Monte
  Carlo variance, which is 8% to 13% of the total):
  - cluster-robust: 0.86 to 0.94 at G = 20, 1.16 to 1.21 at G = 40;
  - naive: 3.2 to 5.6.
- **Mixing against heterogeneity.** Ranking the five datasets by (group logit SD)² / G gives:
  heartbeats 0.185, at work 0.058, sleeping 0.028, courses 0.0065, census 0.0015.

  The size of the mixing term follows the same order, except that courses and sleeping swap (Spearman 0.9). Five
  points and a raw rather than residual heterogeneity measure make this exploratory only.

## Caveats

- One predictor set and one setting per model per dataset, with no tuning. The absolute AUCs are not comparable to
  published results. Heartbeats in particular uses beat spacing only.
- Courses: a student who took several course runs is not grouped across them.
- Census: unweighted rows; race excluded (convention 8).
- Phone sensing: the data were downloaded over plain HTTP with Abraham's approval. The hashes record what was
  received, not a verified original. Minutes within a user are also correlated in time, which is part of what the
  grouped folds handle.
- Stage A has 5 random-row partitions against 20 grouped ones; the leakage numbers are means over those.

## What is known and what is new

- **Claim 1.** Subject-wise CV is standard advice in activity recognition and in the inter-patient ECG literature
  (de Chazal et al.), and a 2026 OULAD paper shows the same gap. What is new here is the same design measured side by
  side on five datasets, including a census case where the gap is zero.
- **Claim 2.** The design effect is classical. What is new is measuring it for CV metrics across fields and checking
  coverage against a computable truth (claim 3).
- **Claim 4.** These are the first real-data measurements we know of of Proposition 9's mixing term (see
  proposals/10). It is real and has the predicted sign and scaling, but it is small in practice.

## For the class (draft slide wording, ISL level)

Frame: "We found three problems with cross-validation in speed dating. Are they special to speed dating? We
tried four other fields."

**Slide 1. Hold out people, not rows.**
- Claim: If the same patient or phone user is in both the training and the test folds, random-row CV rewards
  recognising the person. Heartbeats: AUC 0.91 with whole patients held out, 0.99 with random rows. Census: no change.
- Figure: `cross_field_leakage.png`.
- Note: Leakage needs groups that differ and predictors that can recognise the group. The census has the first but
  not the second.

**Slide 2. Count groups, not rows.**
- Claim: 100,000 heartbeats come from 43 people. The honest error bar is 2 to 22 times wider than the usual one.
- Figure: `cross_field_se_ratio.png`.
- Note: Even in the census, where the number is right, the error bar is half as wide as it should be. With 790
  workers per area, a within-area correlation of 0.003 is enough to double it.

**Slide 3. We checked against the truth.**
- Claim: In the census we can compute the right answer from the areas we did not use. The usual 90% interval caught
  it about half the time. The cluster-robust interval caught it 82% to 90% of the time.
- Figure: `cross_field_coverage.png`.

**Slide 4 (optional, or a footnote). Compute AUC within each fold.**
- Claim: Putting every fold's scores into one AUC compares scores from different models. It runs slightly low, by up
  to 0.009. It always goes in the same direction and shrinks as the number of groups grows.
- Figure: `cross_field_pooled.png`.

**Closing line.** Split by group, count groups for the error bar, and compute AUC within folds.
