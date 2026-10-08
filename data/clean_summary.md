# Cleaning summary

- Raw rows: 8378; participants: 551; waves: 21
- Dropped 10 rows with no partner id (waves [5]) and 0 further rows with no decision
- Dropped 0 duplicate (rater, ratee) rows
- Blank items set to 0 where the answered items of a 100-point block already sum to 100 (PREREG.md, 2026-10-08): block 1_1: iids [129, 130, 528] (40 rows); block 2_1: iids [130] (9 rows); block 4_1: iids [528] (22 rows)
- Block 1_1 (own preferences, pre-event): converted to shares of 100. Waves 6-9: already shares of 100 in the release (every complete row within 0.1 of 100). 100-point waves: 9 people (144 rows) summed more than 2 away from 100 before rescaling
- Block 2_1 (what the opposite sex wants, pre-event): converted to shares of 100. Waves 6-9: already shares of 100 in the release (every complete row within 0.1 of 100). 100-point waves: 19 people (299 rows) summed more than 2 away from 100 before rescaling
- Block 4_1 (what same-sex peers want, pre-event): converted to shares of 100. Waves 6-9: raw 1-10 answers in the release (row sums 27 to 60). 100-point waves: 11 people (184 rows) summed more than 2 away from 100 before rescaling
- Missing rate per rating: attr 2.4%, sinc 3.3%, intel 3.5%, fun 4.2%, amb 8.5%, shar 12.7%
- Complete-case rows (all six ratings present): 7031 of 8368
- Event flags added: wave_undergrad (waves [5]), wave_yes_budget (waves [12]), wave_diff_mc (waves [13, 14]), wave_brought_item (waves [18, 19, 20, 21]); condtn kept (1 = limited choice, 2 = extensive)
- Set to missing as recording errors (PREREG.md): gaming = 14 for 5 people (78 rows), reading = 13 for 3 people (51 rows); recorded values kept in gaming_recorded, reading_recorded
- Rows whose partner is in a different wave (should be 0): 0
- Rows where match != dec * dec_o (should be 0): 0
- Dyads: 4184; seen from both sides: 4184; one side only: 0
- Dates per rater: min 5, median 16, max 22

Output: data/clean.csv with 8368 rows and 235 columns.
