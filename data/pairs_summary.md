# Pair table summary

- Input: data/clean.csv, 8368 rows, 4184 dyads
- Dyad rows: 4184 (expected 4184); matches 690, rate 0.1649 (expected 0.165)
- dyads not seen from exactly one woman and one man (should be 0): 0
- dyads whose two rows disagree on event (should be 0): 0
- dyads whose two rows disagree on match (should be 0): 0
- dyads whose two rows disagree on int_corr (should be 0): 0
- dyads whose two rows disagree on same_field (should be 0): 0
- dyads whose two rows disagree on condtn (should be 0): 0
- dyads where age_diff is not antisymmetric (should be 0): 0
- dyads where match != her yes * his yes (should be 0): 0
- dyads where our 16-interest correlation, rounded, differs from int_corr (should be 0): 0
- dyads where int_corr is present but ours is not (should be 0): 0
- dyads where same_field differs from clean.csv (should be 0): 0
- dyads where abs_age_diff changes when f and m are swapped (should be 0): 0
- dyads where int_corr changes when f and m are swapped (should be 0): 0
- dyads where int_corr17 changes when f and m are swapped (should be 0): 0
- dyads where same_field changes when f and m are swapped (should be 0): 0
- dyads where same_goal changes when f and m are swapped (should be 0): 0
- dyads where fit_to_f_prefs does not become fit_to_m_prefs when swapped (should be 0): 0
- dyads where age_diff does not change sign when swapped (should be 0): 0
- Feature set `pre`: 76 columns (6 categorical codes); complete dyads 3943 of 4184, matches among them 647
- Feature set `pre+date`: 92 columns (6 categorical codes); complete dyads 2766 of 4184, matches among them 490
- Missing values by feature, where any: f_rating_shar 576, m_rating_shar 490, f_rating_amb 362, m_rating_amb 350, f_rating_fun 183, m_rating_fun 167, f_prob 164, m_rating_intel 149, f_rating_intel 147, m_rating_sinc 147, m_prob 144, f_rating_like 135, f_rating_sinc 130, m_rating_like 105, fit_to_m_prefs 105, m_rating_attr 101, f_rating_attr 101, f_career_c 99, fit_to_f_prefs 97, abs_age_diff 94, same_field 82, f_pref_shar 81, f_exphappy 81, same_goal 79, int_corr 79, f_self_attr 67, f_self_sinc 67, f_self_intel 67, f_self_fun 67, f_self_amb 67, f_age 65, f_pref_intel 59, f_art 59, f_pref_attr 59, f_yoga 59, f_shopping 59, f_music 59, f_concerts 59, f_movies 59, f_theater 59, f_tv 59, f_reading 59, f_clubbing 59, f_gaming 59, f_hiking 59, f_pref_fun 59, f_museums 59, f_dining 59, f_exercise 59, f_tvsports 59, f_sports 59, f_go_out 59, f_date 59, f_goal 59, f_pref_sinc 59, f_pref_amb 59, f_field_cd 43, m_career_c 39, m_field_cd 39, m_self_amb 38, m_pref_shar 38, m_pref_amb 38, m_self_intel 38, m_self_fun 38, m_self_sinc 38, m_self_attr 38, m_date 38, m_age 29, m_pref_fun 29, m_gaming 20, m_goal 20, m_go_out 20, m_exphappy 20, m_sports 20, m_tvsports 20, m_exercise 20, m_dining 20, m_art 20, m_hiking 20, m_reading 20, m_clubbing 20, m_pref_attr 20, m_tv 20, m_theater 20, m_movies 20, m_concerts 20, m_music 20, m_shopping 20, m_yoga 20, m_pref_intel 20, m_pref_sinc 20, m_museums 20
- People with an interest rating outside the codebook's 1 to 10 scale (kept as recorded): 14 (values [0, 13, 14])
- int_corr17 vs int_corr: correlation 0.936

Output: data/pairs.csv with 4184 rows and 107 columns; feature sets in data/feature_sets.json.
