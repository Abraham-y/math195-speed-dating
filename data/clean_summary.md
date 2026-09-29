# Cleaning summary

- Raw rows: 8378; participants: 551; waves: 21
- Dropped 10 rows with no decision or no partner id
- Dropped 0 duplicate (rater, ratee) rows
- Block 1_1 (own preferences, pre-event): converted to shares of 100; 144 100-point rows summed more than 2 away from 100 before rescaling
- Block 2_1 (what the opposite sex wants, pre-event): converted to shares of 100; 299 100-point rows summed more than 2 away from 100 before rescaling
- Block 4_1 (what same-sex peers want, pre-event): converted to shares of 100; 184 100-point rows summed more than 2 away from 100 before rescaling
- Missing rate per rating: attr 2.4%, sinc 3.3%, intel 3.5%, fun 4.2%, amb 8.5%, shar 12.7%
- Complete-case rows (all six ratings present): 7031 of 8368
- Event flags added: wave_undergrad (waves [5]), wave_yes_budget (waves [12]), wave_diff_mc (waves [13, 14]), wave_brought_item (waves [18, 19, 20, 21]); condtn kept (1 = limited choice, 2 = extensive)
- Rows whose partner is in a different wave (should be 0): 0
- Rows where match != dec * dec_o (should be 0): 0
- Dyads: 4184; seen from both sides: 4184; one side only: 0
- Dates per rater: min 5, median 16, max 22

Output: data/clean.csv with 8368 rows and 232 columns.
