# External datasets (downloaded 2026-11-05)

These datasets are for testing the error-bar results outside speed dating (proposals/10, proposals/11). The data live in
`data/external/<name>/` and are git-ignored. Each archive was tested (`unzip -t`), its entries were checked for absolute
or `..` paths, and it was extracted into `extracted/` in its own folder. Read them only as data (`python -I`).
Abraham approved each download on 2026-11-05.

| Folder | Source | Licence / terms | Files | SHA-256 |
|---|---|---|---|---|
| `mitdb` | MIT-BIH Arrhythmia Database 1.0.0, https://physionet.org/files/mitdb/1.0.0/ (HTTPS, served from physionet-open.s3.amazonaws.com) | ODC-By 1.0; cite Moody & Mark (2001) and Goldberger et al. (2000) | `RECORDS`, and for each of the 48 records its `.atr` (beat annotations) and `.hea` (header); 544 KB | concatenation of `*.atr *.hea RECORDS` in shell glob order: `bed08ed10e5bb5a4f00b50a27a8517af0f63b4c51d580dfb5dae555954bcd318` |
| `oulad` | Open University Learning Analytics dataset, https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip (HTTPS) | CC BY 4.0; cite Kuzilek, Hlosta & Zdrahal (2017), Scientific Data 4:170171 | `oulad.zip`, 46,748,244 bytes | `f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4` |
| `acs_pums_ca` | ACS 2018 1-year PUMS, California persons, https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip (HTTPS) | US Census Bureau terms (no named open licence; no re-identification, no implied endorsement) | `csv_pca.zip`, 67,233,615 bytes | `6194ade52892ddb0e41903f7d3b49399842f08eeebd42e6b19ee8e417affbf79` |
| `extrasensory` | ExtraSensory Dataset, http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip and http://extrasensory.ucsd.edu/data/cv5Folds.zip | free for research on condition of citing Vaizman, Ellis & Lanckriet (2017), IEEE Pervasive Computing 16(4):62-74 | `ExtraSensory.per_uuid_features_labels.zip`, 225,374,973 bytes; `cv5Folds.zip`, 10,803 bytes | `412d307bdf54a60dff81a705c2f25833889a46dca64a184e3b588ad99c69cb56`; `66183635452d8c1b250b570fba3000296bf29e4d54eda6af270d2b4f5a3b9ee6` |

ExtraSensory is served over plain HTTP only, and no checksum is published. Abraham approved the download knowing this
conflicts with the encrypted-traffic rule. Checks on what arrived:
- the byte counts match the sizes reported during scouting;
- both archives pass `unzip -t`;
- they hold only the 60 per-user `.csv.gz` files and the fold lists.

The hashes above are for what was received, not a verified original. If an HTTPS copy becomes available, compare against
them.
