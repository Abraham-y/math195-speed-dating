# Non-dating datasets for testing the clustered-CV claims

Scouted 2026-11-05 for the Math 195 capstone. No data files were downloaded. Pages, data dictionaries and papers were read only. Every URL and citation below was opened during scouting; anything that could not be confirmed is marked **UNVERIFIED**.

Claims to test (from `proposals/08-cv-error-bars-note.tex`, abstract and Section "The cross-validated AUC under clustering (Part 9)"):

- **(i)** Random-row CV overstates performance when groups repeat.
- **(ii)** Naive error bars on CV metrics are too narrow. Cluster-robust (CR1) and wild-cluster-bootstrap intervals are honest.
- **(iii)** Pooling held-out scores across grouped folds into one AUC is biased low when groups differ in outcome rate. The bias is of order (between-group outcome-rate heterogeneity)/G. The between-cluster-within-fold (BCWF) AUC removes it.

What makes a dataset useful here:
- a binary outcome;
- G between 20 and 500, with unequal group sizes;
- large between-group differences in base rate, which (iii) needs;
- a published paper that pools a metric over grouped folds.

---

## 1. Ranked recommendation

| Rank | Dataset (field) | Why it suits claims i to iii |
|---|---|---|
| 1 | **ExtraSensory** (mobile and wearable sensing) | 60 users with 685 to 9,706 one-minute rows each, so (i) has strong temporal leakage to find and (ii) gets unequal clusters. For (iii), label use differs sharply between users: AT_SCHOOL is used by 49 of 60 users and LOC_main_workplace by 32 of 60. The authors' own user-wise 5-fold CV sums TP, TN, FP and FN over all folds, which is the pooling practice. |
| 2 | **MIT-BIH Arrhythmia Database** (cardiology) | 48 records from 47 patients with 1,518 to 3,363 beats each (ii). The intra-patient versus inter-patient literature is a ready-made example of (i). For (iii), ventricular-beat counts range from 0 (11 records) to 992 (record 208), the most extreme base-rate heterogeneity in this list. de Chazal et al. report pooled statistics over 22 held-out patients. |
| 3 | **Open University Learning Analytics Dataset (OULAD)** (education) | 22 module-presentations of 365 to 2,498 registrations each, with students crossing modules. A 2026 paper already shows (i): student-disjoint random CV gave AUROC 0.716, leave-one-module-out 0.661. That paper reports a *pooled* cross-module AUROC, exactly the practice in (iii), with per-module AUROCs of 0.635 to 0.736 and very different calibration by module. |
| 4 | **ACS PUMS via `folktables` (ACSIncome)** (social science; a widely used ML benchmark) | A different regime: about 1.6M rows in G = 51 states, or about 265 PUMAs in California (**UNVERIFIED** count). The naive standard error is near zero while the cluster-robust one is not, which is the sharpest possible test of (ii). Because row noise is tiny, an O(1/G) pooling bias for (iii) can be measured precisely. Cross-state transfer is documented to be unpredictable. *Weaker on published practice:* no verified paper runs grouped K-fold CV with a pooled AUC on it. |

**Alternates:**
- **Sleep-EDF Expanded** (EEG). Named license (ODC-By). DeepSleepNet pools predictions over subject-wise folds. But the polysomnography (PSG) files total about 7.2 GB and need EEG feature extraction.
- **Wyscout soccer events** (sports). CC BY 4.0, 77 MB. Grouping by team or match is natural. Expect little residual between-group heterogeneity in conversion rate, which makes it a near-null case for (iii).
- **UCI HAR / HAPT** (activity recognition). CC BY 4.0. A scripted, balanced protocol, so a *negative control* for (iii). It is also the dataset Saeb et al. used to show (i).

**Caveats on the top four:**
- *ExtraSensory* has no named license. Its terms say the data are free to use "conditioned on citing our original paper". The site serves downloads over **plain HTTP only** (HTTPS refused the connection), and I found no published checksum. Under the user's security instructions, flag this before downloading.
- *ACS PUMS* also has no named open license. It is US federal data under Census Bureau terms (see section 3.4).
- *OULAD* has only G = 22 at the module-presentation level, at the lower edge of the target range, and students appear in more than one module (crossed clustering).

---

## 2. Candidate table

G is the number of groups.

| # | Dataset | Field | License / access | Download | Rows; outcome; base rate | Group var; G; sizes | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | ExtraSensory | wearables / phones | No named license; "free … conditioned on citing" Vaizman 2017a. No registration. HTTP only | features+labels zip 225,374,973 B; CV folds zip 10,803 B | ~300k labeled minutes (308,306 with accelerometer); 51 binary labels, e.g. SLEEPING 83,055 positives, FIX_walking 22,136 | user UUID; 60; 685–9,706 labeled rows | **Top 1** |
| 2 | MIT-BIH Arrhythmia | cardiology | ODC-By 1.0; open access | zip 73.5 MB (104.3 MB unzipped) | ~110,000 beats; ventricular (V) vs not; 7,130 V beats in PhysioNet table (~6.5%) | record; 48 (47 subjects; 44 non-paced); 1,518–3,363 beats | **Top 2** |
| 3 | OULAD | education | CC BY 4.0; no registration | zip 44.6 MB (studentVle.csv 432.8 MB unzipped) | 32,593 registrations (28,785 learners); fail or withdrawn vs pass or distinction; 0.473 among learners active at day 7 | module-presentation; 22; 365–2,498 (or module; 7) | **Top 3** |
| 4 | ACS PUMS 2018 (`folktables` ACSIncome) | social science / ML fairness | Census Bureau terms (no named license); folktables code MIT; no registration | `csv_pus.zip` 536 MB (all states) or `csv_pca.zip` 64 MB (CA) | 1,599,229 rows (US); income > $50k; ≈36.9% (1 − 63.1% constant-predictor accuracy) | state (`ST`), 51; or PUMA within state (≥100k people each; CA ≈265, UNVERIFIED) | **Top 4** |
| 5 | Sleep-EDF Expanded | EEG / sleep | ODC-By 1.0; open access | 8.1 GB total; SC PSG files 35.5–50.1 MB × 153 (~7.2 GB); hypnograms 1.8–10.9 KB | 30-s epochs; binary such as wake vs sleep or N3 vs rest | subject; SC 78 subjects (UNVERIFIED count), 153 nights; ages 25–101 | Alternate (size) |
| 6 | UCI HAR (and HAPT) | activity recognition | CC BY 4.0 | HAR zip 58.2 MB; HAPT zip 75.9 MB | 10,299 windows (HAPT 10,929); binarize e.g. dynamic vs static | subject; 30; ~343 per subject on average | Negative control for (iii) |
| 7 | WISDM 2019 (phone + watch) | activity recognition | CC BY 4.0 | zip 295.4 MB | 15.6M raw samples; 18 activities, 3 min each | subject; 51; balanced by design | Negative control; not needed |
| 8 | eICU-CRD Demo | ICU, multi-site | ODbL 1.0; open access | 130.4 MB zip | >2,500 unit stays; in-hospital mortality (rate not stated on page) | hospital; 20; unequal | Possible; small |
| 9 | Diabetes 130-US hospitals | hospital readmission | CC BY 4.0 | 3 MB zip (diabetic_data.csv 18.3 MB) | 101,766 encounters; readmitted < 30 days (rate UNVERIFIED) | `patient_nbr`: ~71k patients, mostly 1 encounter; no hospital id listed | Not suitable for (iii) (G huge, groups tiny) |
| 10 | PhysioNet/CinC Challenge 2019 (sepsis) | ICU time series | CC BY 4.0; open access | training data ~42 MB | hourly rows; SepsisLabel | patient; 40,336 (2 hospital systems) | Only if subsampled to G of 100–500 |
| 11 | Wyscout soccer events (Pappalardo et al. 2019) | sports | CC BY 4.0 (figshare) | events.zip 77,323,413 B; matches.zip 645,097 B; players.json 1,737,347 B; teams.json ~26.8 kB | shots ≈1.5% of 3,251,294 events; goal ≈10% of shots | team (~142, UNVERIFIED), match (1,941) or player (4,299) | Alternate (near-null for iii) |
| 12 | BCI 50-ha forest census (Condit et al. 2019) | ecology | Dryad; license not named on page (Dryad CC0 policy, UNVERIFIED); PIs ask to be told of use and "may request co-authorship" | 286.72 MB total (bci.tree.zip 48.39 MB) | 423,617 trees, 8 censuses; tree survival (binary) | species (~300, UNVERIFIED) or 20×20 m quadrat | Lead only; no grouped-CV + AUC paper found |
| x1 | ASSISTments 2009-10 | education | **Excluded**: terms require "your written agreement" (FERPA) | — | — | school, class, student | Fails "no data-use agreement" |
| x2 | NEON tick pathogen status (DP1.10092.001) | ecology | **Excluded**: since 2026-06-30 NEON requires a user account or API token; license now CC BY 4.0 | — | ticks tested per pathogen | site; only 19 sites | Fails "no registration"; G < 20 |
| x3 | ABIDE (preprocessed) | neuroimaging | **Excluded**: INDI requires NITRC registration; CC BY-NC-SA | — | ASD vs control, 1,112 subjects | site; 17 | Fails "no registration"; G < 20 |

---

## 3. Details per candidate

### 3.1 ExtraSensory (rank 1)

- **Official page:** http://extrasensory.ucsd.edu/ (HTTP only; HTTPS refused).
- **License and access:**
  - The page says: "everyone is invited to download the dataset for free and use it (conditioned on citing our original paper)."
  - The primary-data README says: "Any usage of the dataset for publications requires citing" Vaizman, Ellis and Lanckriet (2017).
  - No named license. No registration.
- **De-identification:**
  - Users are identified by random UUIDs.
  - The primary data carry only relative-location features; absolute coordinates are in a separate optional file.
  - Participants were "mostly students … from the UCSD campus". None are Stanford students.
- **Files** (sizes from HTTP HEAD; nothing downloaded):
  - `http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip`: 225,374,973 bytes (the page says 215 MB). It holds 60 `[UUID].features_labels.csv.gz` files.
  - `http://extrasensory.ucsd.edu/data/cv5Folds.zip`: 10,803 bytes. It is the authors' 5-fold partition of users.
  - `http://extrasensory.ucsd.edu/data/primary_data_files/README.txt`: column documentation.
  - Optional, not needed: original labels (970 KB), mood labels (795 KB), absolute location (2.2 MB), raw sensors (0.09–9 GB each).
- **Rows, outcome and base rate:**
  - About 300k labeled examples, one per minute.
  - Each example carries precomputed features (accelerometer, gyroscope, watch, location variability, audio MFCC summaries, phone state) and 51 cleaned binary labels coded 1, 0 or `nan` (missing).
  - Natural binary targets:

    | Label | Positive examples | Users with any |
    |---|---|---|
    | SLEEPING | 83,055 | 53 |
    | LOC_home | 152,892 | 57 |
    | AT_SCHOOL | 42,331 | 49 |
    | FIX_walking | 22,136 | 60 |
    | OR_exercise | 8,081 | 44 |
    | LOC_main_workplace | 33,944 | 32 |

- **Groups:** user, G = 60. Labeled examples per user: 685–9,706 (mean 5,139, SD 2,332). Days of participation: 2.9–28.1.
- **Between-group heterogeneity** (documentation):
  - The label table counts how many users ever applied each label. For example, LOC_main_workplace was used by 32 of 60 users, AT_SCHOOL by 49 and IN_CLASS by 20.
  - So for many labels a large share of users have a within-user base rate of exactly 0.
  - The site also says "a typical user mainly used ~15 labels", with per-user label-usage entropy of 3.9 bits on average.
  - Per-user rates are not tabulated (not computed here).
- **Papers showing the practice:**
  - Vaizman, Y., Ellis, K., Lanckriet, G. (2017). "Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches." *IEEE Pervasive Computing* 16(4):62–74. DOI 10.1109/MPRV.2017.3971131 (arXiv 1609.06354).
    - Five-fold CV in which "each fold has 48 users in the training set and the other twelve users in the test set", plus leave-one-user-out.
    - The headline metric is balanced accuracy (BA).
  - Vaizman, Y. (2018). *Behavioral Context Recognition In the Wild.* PhD dissertation, UC San Diego. https://escholarship.org/uc/item/200910xx
    - Chapter 2 reprints the 2017a paper. On p. 62 it states that users, not examples, were partitioned "to avoid having examples from the same subject appear in both the training set and the test set".
    - It also states: "At the end of the 5-fold procedure we summed up the total numbers of TP, TN, FP and FN over the entire evaluation set." That is a pooled metric over grouped folds.
  - Vaizman, Y., Weibel, N., Lanckriet, G. (2018). "Context Recognition In-the-Wild: Unified Model for Multi-Modal Sensors and Multi-Label Classification." *Proc. ACM IMWUT* 1(4):1–22. DOI 10.1145/3161192. It uses the same user folds; its metric aggregation was not checked.

### 3.2 MIT-BIH Arrhythmia Database (rank 2)

- **Official page:** https://physionet.org/content/mitdb/1.0.0/ (DOI 10.13026/C2F305).
- **License and access:** Open Data Commons Attribution License v1.0. "Anyone can access the files." No credentialing.
- **Files:**
  - ZIP via https://physionet.org/content/mitdb/get-zip/1.0.0/ (73.5 MB; 104.3 MB uncompressed).
  - Per record: `NNN.dat` (~1.9 MB signal), `NNN.hea` (~143 B header), `NNN.atr` (4.5–8.4 KB beat annotations; e.g. 100.atr 4.5 KB, 208.atr 8.4 KB).
  - The 48 `.atr` and `.hea` files alone are a few hundred KB. They are enough for RR-interval features. Morphology features need the `.dat` files.
- **Rows, outcome and base rate:**
  - About 110,000 annotated beats; 48 half-hour two-lead records at 360 Hz.
  - Binary target: ventricular ectopic beat (AAMI class V, or the `V` symbol) vs not.
  - The PhysioNet beat table has 7,130 `V` beats, about 6.5%.
  - Supraventricular ectopic beats are an alternative target.
- **Groups:**
  - Record/patient, G = 48. Records 201 and 202 are the same subject, so 47 subject groups.
  - Common practice drops the 4 paced records, leaving G = 44.
  - Beats per record: 1,518 (record 123) to 3,363 (record 215).
  - PhysioNet itself says records 201 and 202 "should be assigned to the same partition" for subject-independent evaluation.
- **Between-group heterogeneity** (documentation; table of beat types: https://physionet.org/physiobank/database/html/mitdbdir/tables.htm):
  - 11 records have zero `V` beats.
  - Records 208, 233, 200 and 106 have 992, 831, 826 and 520.
  - Selection was deliberately mixed: 23 records chosen at random and 25 "to include less common but clinically significant arrhythmias".
- **Papers showing the practice:**
  - de Chazal, P., O'Dwyer, M., Reilly, R. B. (2004). "Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features." *IEEE Trans. Biomed. Eng.* 51(7):1196–1206. DOI 10.1109/TBME.2004.827359.
    - Uses the 44 non-pacemaker records split into two sets of about 50,000 beats by record (patient-wise).
    - The selected configuration is evaluated on the held-out set with single pooled sensitivity and positive predictivity per class (VEB 77.7% / 81.9%).
    - This is the origin of the "inter-patient" paradigm.
  - Luz, E. J. da S., Schwartz, W. R., Cámara-Chávez, G., Menotti, D. (2016). "ECG-based heartbeat classification for arrhythmia detection: A survey." *Computer Methods and Programs in Biomedicine* 127:144–164. DOI 10.1016/j.cmpb.2015.12.008. It surveys intra- vs inter-patient evaluation; its specific findings were seen only in indexing summaries, so treat the claim as **UNVERIFIED** in detail.
  - No verified paper does grouped *K-fold* CV with a pooled *AUC* on MIT-BIH. The verified practice is a single patient-wise split with pooled beat statistics.

### 3.3 Open University Learning Analytics Dataset, OULAD (rank 3)

- **Official page:** https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset (DOI 10.24432/C5KK69). The OU's own page now redirects to https://research.stem.open.ac.uk/ouanalyse/ (not opened).
- **License and access:** CC BY 4.0, "for any purpose, provided that the appropriate credit is given." No registration.
- **De-identification:** k-anonymity (k = 5) with ARX. Module names, IDs and dates were obfuscated. Age and IMD band were generalized. Kuzilek et al. (2017) describe the process.
- **Files:**
  - `https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip` (44.6 MB).
  - Contents (uncompressed):

    | File | Size |
    |---|---|
    | studentVle.csv | 432.8 MB |
    | studentAssessment.csv | 5.4 MB |
    | studentInfo.csv | 3.3 MB |
    | studentRegistration.csv | 1.1 MB |
    | vle.csv | 264.3 KB |
    | courses.csv | not shown (UNVERIFIED) |
    | assessments.csv | not shown (UNVERIFIED) |

- **Rows, outcome and base rate:**
  - 32,593 student-module-presentation registrations from 28,785 learners.
  - Outcome `final_result`: Pass, Distinction, Fail, Withdrawn. The standard binarization is Fail or Withdrawn vs Pass or Distinction.
  - Lim (2026) reports non-success prevalence of 0.473 among learners still active at day 7, falling to 0.442 at day 28.
  - The overall `final_result` split is not stated in the data paper.
- **Groups:**
  - Module-presentation, G = 22, with 365 (AAA 2014J) to 2,498 (CCC 2014J) registrations (descriptive notebook at https://sergiosim.github.io/OULAD/notebooks/first_descriptive_analysis.html).
  - Or module, G = 7, with 748 (AAA) to 7,909 (BBB) students.
  - Students repeat across modules, so the clustering is crossed.
- **Between-group heterogeneity:**
  - Per-module pass or withdrawal rates are not tabulated in Kuzilek et al. (2017) (**UNVERIFIED** numerically).
  - Lim (2026) reports that "calibration and error rates varied sharply across modules": expected calibration error ran from 0.027 (CCC) to 0.357 (GGG), and per-module AUROC from 0.635 to 0.736.
  - Modules span STEM (CCC, DDD, EEE, FFF) and Social Sciences (AAA, BBB, GGG).
- **Papers:**
  - Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017). "Open University Learning Analytics dataset." *Scientific Data* 4:170171. DOI 10.1038/sdata.2017.171. The data paper.
  - Lim, M. (2026). "Student-Disjoint Cross-Module Benchmarking for Calibrated Early Warning in Online Learning." *IJACSA* 17(9):617–. DOI 10.14569/ijacsa.2026.0170958. PDF: https://thesai.org/Downloads/Volume17No9/Paper_58-Student_Disjoint_Cross_Module_Benchmarking.pdf
    - Seven leave-one-module-out folds with learners removed from training if they appear in the test module.
    - Reports a "pooled cross-module AUROC" of 0.661 (day 28, logistic regression) against 0.716 for student-disjoint random 5-fold CV.
    - This gives (i) and the pooled practice of (iii) directly.
    - Caveat: a single-author paper in a lower-tier venue.

### 3.4 ACS PUMS 2018 via `folktables`, task ACSIncome (rank 4)

- **Official pages:**
  - Package: https://github.com/socialfoundations/folktables
  - Data: https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/
- **License and access:**
  - folktables code: MIT.
  - Data: "governed by the terms of use provided by the Census Bureau" (folktables README). Those terms (https://www.census.gov/data/developers/about/terms-of-service.html) are API terms. They forbid re-identification and implying endorsement; they do not name an open license.
  - Works of the US federal government are generally not under copyright, but no page opened here states that for PUMS (**UNVERIFIED**).
  - Files are plain public downloads; no registration.
- **De-identification:** public-use microdata. PUMAs each contain at least 100,000 people (Census PUMA guidance).
- **Files** (2018 1-year person files):

  | File | Size | Use |
  |---|---|---|
  | `csv_pus.zip` | 536 MB | whole US, for state groups |
  | 52 per-state files `csv_pXX.zip` | ~549 MB together, 1.2 MB (WY) to 64 MB (CA) | alternative to `csv_pus.zip` |
  | `csv_pca.zip` | 64 MB | CA only, for PUMA groups |

- **Rows, outcome and base rate:**
  - ACSIncome (US 2018): 1,599,229 rows and 10 features.
  - The label is PINCP > $50,000. Filters: age > 16, at least 1 hour worked per week, income at least $100.
  - Constant-predictor accuracy is 63.1%, so the positive rate is about 36.9% (Ding et al. 2021, Table 1).
  - The task's default fairness group is race (`RAC1P`). We would not use it, consistent with the project's convention on race variables.
- **Groups:** state (`ST`), G = 51 (50 states + DC; PR is a separate file), very unequal in size. Or PUMA within one state: CA has about 265 (**UNVERIFIED**).
- **Between-group heterogeneity:**
  - Ding et al. (2021): "Training on one state and testing on another generally leads to unpredictable results"; the size of fairness-intervention effects "varies greatly by state". Per-state positive rates are not tabulated.
  - Gardner et al. (2023, TableShift) build an ACS Income task held out by geographic region and report that shift gaps are "strongly related to shifts in the label distribution".
  - Numeric per-state base rates: **UNVERIFIED**.
- **Papers:**
  - Ding, F., Hardt, M., Miller, J., Schmidt, L. (2021). "Retiring Adult: New Datasets for Fair Machine Learning." *NeurIPS 34*. https://proceedings.nips.cc/paper_files/paper/2021/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf. No DOI. Per-state models and cross-state transfer, reporting accuracy.
  - Gardner, J., Popovic, Z., Schmidt, L. (2023). "Benchmarking Distribution Shift in Tabular Data with TableShift." NeurIPS 2023 Datasets & Benchmarks. https://arxiv.org/abs/2312.07577. Held-out-domain accuracy.
  - Neither runs grouped *K-fold* CV with a pooled AUC; that practice is not shown for this dataset.

### 3.5 Sleep-EDF Database Expanded (alternate)

- **Official page:** https://physionet.org/content/sleep-edfx/1.0.0/ (DOI 10.13026/C2X676). ODC-By 1.0; open access.
- **Files:**
  - `sleep-cassette/*-PSG.edf`: 153 files, 35.5–50.1 MB each, about 7.2 GB.
  - `*-Hypnogram.edf`: 153 files, 1.8–10.9 KB.
  - `SC-subjects.xls`: 23.5 KB.
  - The full ZIP is 8.1 GB.
- **Content:** Sleep Cassette subjects are healthy, aged 25–101, with about two nights each. The subject count, usually given as 78, is not on the page (**UNVERIFIED**).
- **Rows and outcome:** 30-s epochs with R&K stages; binarize as wake vs sleep, or N3 (S3+S4) vs rest.
- **Heterogeneity:** the cohort spans ages 25–101. A per-subject stage-proportion table was not found (**UNVERIFIED**).
- **Paper:** Supratak, A., Dong, H., Wu, C., Guo, Y. (2017). "DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG." *IEEE TNSRE* 25(11):1998–2008. DOI 10.1109/TNSRE.2017.2721116.
  - 20-fold subject-wise CV on Sleep-EDF.
  - "we combined the predicted sleep stages from all folds and computed the performance metrics": the pooled practice.

### 3.6 UCI HAR and HAPT (negative control for iii)

- **HAR:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones. CC BY 4.0. `UCI HAR Dataset.zip` 58.2 MB. 10,299 windows from 30 subjects; the train/test split is by volunteer. DOI 10.24432/C54S4K.
- **HAPT:** https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions. CC BY 4.0. 75.9 MB. 10,929 instances, 12 activities, subject ids 1–30. DOI 10.24432/C54G7M.
- **Heterogeneity:** a scripted protocol, so activity shares should be nearly equal across subjects (inferred, not tabulated). That makes it useful as a control where the bias in (iii) should vanish.
- **Paper for (i):** Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., Kording, K. P. (2017). "The need to approximate the use-case in clinical machine learning." *GigaScience* 6(5):gix019. DOI 10.1093/gigascience/gix019.
  - Uses HAPT.
  - Record-wise CV error was about 2% against about 7–9% subject-wise.
  - Their literature review found that nearly half of relevant studies used record-wise CV.

### 3.7 WISDM 2019 (not needed)

- https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset. CC BY 4.0.
- `wisdm-dataset.zip` 295.4 MB. 51 subjects, 18 activities, 3 minutes each, so balanced by design: a second negative control. DOI 10.24432/C5HK59.
- No grouped-CV paper checked.

### 3.8 eICU-CRD Demo (possible medical alternative)

- https://physionet.org/content/eicu-crd-demo/2.0.1/. ODbL 1.0; open access; ZIP 130.4 MB. DOI 10.13026/4mxk-na84.
- More than 2,500 unit stays from 20 of the larger eICU hospitals.
- Tables include `patient`, `hospital`, `apachePatientResult` and `apachePredVar`.
- Outcome: in-hospital mortality (rate not stated on page). G = 20, the bare minimum.
- **Paper (uses the full, credentialed eICU, but the same structure):** Johnson, A. E. W., Pollard, T. J., Naumann, T. (2018). "Generalizability of predictive models for intensive care unit patients." ML4H Workshop at NeurIPS 2018. https://arxiv.org/abs/1812.02275
  - 46 hospitals split hospital-wise.
  - 5-fold CV AUROC 0.854 on the training hospitals; held-out-hospital AUROC 0.688–0.933 and SMR 0.560–1.43.
  - "the cross-validation estimate of performance consistently over-estimated performance on the external hospitals."
  - The SMR range is evidence of between-hospital outcome-rate heterogeneity that the model does not capture.

### 3.9 Diabetes 130-US hospitals (not suitable)

- https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008. CC BY 4.0. 3 MB zip (`diabetic_data.csv` 18.3 MB, `IDS_mapping.csv` 2.5 KB). DOI 10.24432/C5230J.
- 101,766 encounters. `patient_nbr` repeats, but most patients have one encounter (~71k patients, **UNVERIFIED**). No hospital id appears in the partial variable list.
- **Papers:**
  - Strack, B. et al. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." *BioMed Research International* 2014:781670. DOI 10.1155/2014/781670.
  - TableShift (above) uses it with an admission-source shift.
- Useful only to show patient-level leakage in (i).

### 3.10 PhysioNet/CinC Challenge 2019, sepsis (optional extreme case)

- https://physionet.org/content/challenge-2019/1.0.0/. CC BY 4.0; open access. Training sets A (20,336 patients) and B (20,000), about 42 MB.
- One `.psv` file per patient with hourly rows. SepsisLabel = 1 from 6 h before onset.
- Paper as listed on that page: Reyna, M. A. et al. (2020). *Critical Care Medicine* 48(2):210–217. DOI 10.1097/CCM.0000000000004145 (not opened separately).
- G is too large unless patients are subsampled. Within-patient base rates are 0 for most patients.

### 3.11 Wyscout soccer match events (sports alternate)

- **Paper:** Pappalardo, L. et al. (2019). "A public data set of spatio-temporal match events in soccer competitions." *Scientific Data* 6:236. DOI 10.1038/s41597-019-0247-7.
- **Collection:** https://figshare.com/collections/Soccer_match_event_dataset/4415000. Each item is CC BY 4.0 (figshare API).
- **Files** (figshare API):

  | File | Size |
  |---|---|
  | events.zip | 77,323,413 B |
  | matches.zip | 645,097 B |
  | players.json | 1,737,347 B |
  | teams.json | ~26.8 kB |

- **Content:** 1,941 matches, 3,251,294 events, 4,299 players in 7 competitions (2017/18 top-5 leagues, World Cup 2018, Euro 2016). Shots are about 1.5% of events; about 1 shot in 10 is a goal (Mead et al.).
- **Groups:** team, match or player.
- **Paper:** Mead, J., O'Hare, A., McMenemy, P. (2023). "Expected goals in football: Improving model performance and demonstrating value." *PLOS ONE* 18(4):e0282295. DOI 10.1371/journal.pone.0282295.
  - Uses this dataset with a *random stratified* 70/30 split and 10-fold CV, reporting test AUC 0.8. That is the random-row practice of (i).
  - It reports that player value and team Elo relate to conversion, which is modest group heterogeneity.

### 3.12 BCI 50-ha plot census (ecology lead)

- **Data:** Condit, R. et al. (2019). "Complete data from the Barro Colorado 50-ha plot: 423617 trees, 35 years." Dryad. DOI 10.15146/5xcp-0d46. https://datadryad.org/stash/dataset/doi:10.15146/5xcp-0d46
- **Files:** 286.72 MB total (`bci.tree.zip` 48.39 MB, `bci.stem.zip` 95.40 MB, …). The license is not named on the page.
- **Practice reference:** Roberts, D. R. et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography* 40(8):913–929. DOI 10.1111/ecog.02881. This is the general reference for blocked CV in ecology; it does not use this dataset.
- No ecology dataset found here meets every requirement *and* has a verified grouped-CV + AUC paper. NEON, the natural choice, now requires login.

### 3.13 Excluded

- **ASSISTments 2009-10.**
  - Data page: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
  - Terms: https://sites.google.com/site/assistmentsdata/termsofuseforusingdata. They state that "WPI is required to get your written agreement", plus no sharing and mandatory code release. That is a data-use agreement.
  - It would otherwise be ideal: school and class ids, and the knowledge-tracing literature pools AUC over student-wise folds.
- **NEON tick pathogen status (DP1.10092.001).**
  - Since 2026-06-30, "users are now required to create a user account" (https://www.neonscience.org/impact/observatory-blog/logins-required-and-data-licensing-updated).
  - Only 19 sites (NEON API product record).
- **ABIDE.**
  - http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html: "users must be logged into NITRC … and registered".
  - Licensed CC BY-NC-SA; 17 sites. The preprocessed copy (http://preprocessed-connectomes-project.org/abide/) is hosted openly but falls under the same INDI terms.

---

## 4. Files to download for the top picks (for approval)

Put each in its own new, empty directory and read it with `python -I`, per the environment rules.

| Pick | File | URL | Size |
|---|---|---|---|
| 1 ExtraSensory | `ExtraSensory.per_uuid_features_labels.zip` | http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip | 225,374,973 B (215 MiB). **Plain HTTP; no checksum found** |
| 1 ExtraSensory | `cv5Folds.zip` | http://extrasensory.ucsd.edu/data/cv5Folds.zip | 10,803 B |
| 1 ExtraSensory | `README.txt` | http://extrasensory.ucsd.edu/data/primary_data_files/README.txt | small text |
| 2 MIT-BIH | full database ZIP | https://physionet.org/content/mitdb/get-zip/1.0.0/ | 73.5 MB (104.3 MB unzipped; includes `SHA256SUMS.txt` per PhysioNet convention, UNVERIFIED for this record) |
| 2 MIT-BIH (minimal option) | 48 × `NNN.atr` + `NNN.hea` (RR-interval features only) | https://physionet.org/content/mitdb/1.0.0/ (per-file links) | ~4.5–8.4 KB per .atr, ~143 B per .hea; under 1 MB total |
| 3 OULAD | dataset ZIP | https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip | 44.6 MB (studentVle.csv is 432.8 MB uncompressed) |
| 4 ACS PUMS (PUMA groups) | `csv_pca.zip` | https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip | 64 MB |
| 4 ACS PUMS (state groups) | `csv_pus.zip` | https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pus.zip | 536 MB |

Minimum set for all four: about 225 + 74 + 45 + 64 = **about 408 MB**, using CA-only ACS. Adding the US-wide ACS file brings it to **about 944 MB**.
