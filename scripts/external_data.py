"""Loaders for the four external datasets of Part 12 (sources, licences and hashes in data/external_sources.md).
Each loader returns (X, y, g): a DataFrame of predictors, a 0/1 outcome array and a group id array, with the choices
below fixed before any model was run on these data. Read-only; the files under data/external/ are treated as data.

- mitdb: MIT-BIH Arrhythmia, one row per beat. Outcome: a ventricular ectopic beat (AAMI class V: annotation V or E).
  Predictors from the beat annotations only (no ECG signal): the RR interval before and after the beat, the mean RR of
  the 10 previous beats, the record's mean RR, and three ratios. The four paced records (102, 104, 107, 217) are left
  out, as is usual (AAMI); beats are the AAMI classes N, S, V, F, Q; each record's first and last beat are dropped.
  Group: the subject (records 201 and 202 are the same person), 43 subjects.
- oulad: Open University Learning Analytics, one row per registration still active on day 14 of the course.
  Outcome: final result Fail or Withdrawn. Predictors: demographics, registration date, clicks before the course and
  on days 0 to 14, active days 0 to 14, assessments submitted by day 14 and their mean score. Group: the
  module-presentation (22). A student can appear in several module-presentations; the grouping ignores that.
- acs: ACS 2018 1-year PUMS, California, the folktables ACSIncome filter (age over 16, income over $100, hours
  worked over 0, person weight at least 1), unweighted. Outcome: income over $50,000. Predictors: age, education,
  hours, sex, marital status, class of worker, relationship, occupation (major group: the first two digits of the
  four-digit code) and place of birth (US state or a world region band). Race (RAC1P) is left out (CLAUDE.md
  convention 8). Group: the PUMA (public-use microdata area).
- extrasensory: ExtraSensory, one row per labelled minute. Outcome: a label (`label:SLEEPING` or
  `label:LOC_main_workplace`), rows where that label was not reported dropped. Predictors: the 225 sensor and
  discrete features. Group: the user (60).
"""
import pathlib, gzip
import numpy as np, pandas as pd

EXT = pathlib.Path("data/external")
MIT_FS = 360                                            # samples per second
MIT_PACED = {"102", "104", "107", "217"}
MIT_SAME_SUBJECT = {"202": "201"}
# AAMI beat classes by MIT annotation code (N, L, R, e, j | A, a, J, S | V, E | F | /, f, Q)
MIT_BEATS = {1: "N", 2: "N", 3: "N", 34: "N", 11: "N", 8: "S", 4: "S", 7: "S", 9: "S", 5: "V", 10: "V", 6: "F",
             12: "Q", 38: "Q", 13: "Q"}
ES_LABELS = ("label:SLEEPING", "label:LOC_main_workplace")


def read_atr(path):
    """MIT-format annotation file -> list of (sample, code). SKIP moves the time by a 32-bit interval (high word
    first); NUM, SUB and CHN carry no time; AUX is followed by its bytes, padded to an even count."""
    w = np.frombuffer(pathlib.Path(path).read_bytes(), dtype="<u2"); i, t, out = 0, 0, []
    while i < len(w):
        a, v = int(w[i]) >> 10, int(w[i]) & 0x3FF; i += 1
        if a == 0 and v == 0:
            break
        if a == 59:
            t += (int(w[i]) << 16) | int(w[i + 1]); i += 2
            continue
        if a in (60, 61, 62):
            continue
        if a == 63:
            i += (v + 1) // 2
            continue
        t += v; out.append((t, a))
    return out


def mitdb(base=EXT / "mitdb"):
    rows = []
    for rec in (base / "RECORDS").read_text().split():
        if rec in MIT_PACED:
            continue
        b = [(t, MIT_BEATS[a]) for t, a in read_atr(base / f"{rec}.atr") if a in MIT_BEATS]
        t = np.array([x[0] for x in b], float) / MIT_FS; cls = np.array([x[1] for x in b])
        rr = np.diff(t)                                     # rr[i] = t[i+1] - t[i]
        pre, post = rr[:-1], rr[1:]                         # for beats 1 .. n-2
        local = pd.Series(rr).rolling(10, min_periods=1).mean().to_numpy()[:-1]   # mean of up to 10 RR ending at the beat
        d = pd.DataFrame(dict(rr_pre=pre, rr_post=post, rr_local=local, rr_record=rr.mean()))
        d["pre_over_local"] = d.rr_pre / d.rr_local; d["post_over_pre"] = d.rr_post / d.rr_pre
        d["pre_over_record"] = d.rr_pre / d.rr_record
        d["y"] = (cls[1:-1] == "V").astype(int); d["record"] = rec; d["group"] = MIT_SAME_SUBJECT.get(rec, rec)
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    return d.drop(columns=["y", "record", "group"]), d.y.to_numpy(), d.group.to_numpy()


def oulad(base=EXT / "oulad" / "extracted", day=14):
    info = pd.read_csv(base / "studentInfo.csv")
    reg = pd.read_csv(base / "studentRegistration.csv", na_values="?")
    key = ["code_module", "code_presentation", "id_student"]
    d = info.merge(reg, on=key, how="left")
    d = d[(d.date_registration.isna() | (d.date_registration <= day)) & (d.date_unregistration.isna() | (d.date_unregistration > day))]
    vle = pd.read_csv(base / "studentVle.csv", usecols=key + ["date", "sum_click"],
                      dtype=dict(code_module="category", code_presentation="category", id_student="int32", date="int16", sum_click="int32"))
    vle = vle[vle.date <= day]
    pre = vle[vle.date < 0].groupby(key, observed=True).sum_click.sum().rename("clicks_before")
    early = vle[vle.date >= 0].groupby(key, observed=True).agg(clicks_0_14=("sum_click", "sum"), active_days_0_14=("date", "nunique"))
    sub = pd.read_csv(base / "studentAssessment.csv", na_values="?").merge(pd.read_csv(base / "assessments.csv", na_values="?"), on="id_assessment")
    sub = sub[sub.date_submitted <= day].groupby(key).agg(assessments_by_14=("score", "size"), mean_score_by_14=("score", "mean"))
    for t in (pre, early, sub):
        t.index = t.index.set_names(key)
        d = d.merge(t.reset_index().astype({"code_module": str, "code_presentation": str}), on=key, how="left")
    for c in ("clicks_before", "clicks_0_14", "active_days_0_14", "assessments_by_14"):
        d[c] = d[c].fillna(0)
    d = d.reset_index(drop=True)
    y = d.final_result.isin(["Fail", "Withdrawn"]).astype(int).to_numpy()
    g = (d.code_module + "_" + d.code_presentation).to_numpy()
    X = d[["gender", "region", "highest_education", "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability",
           "date_registration", "clicks_before", "clicks_0_14", "active_days_0_14", "assessments_by_14", "mean_score_by_14"]]
    return X, y, g


def acs(path=EXT / "acs_pums_ca" / "extracted" / "psam_p06.csv"):
    cols = ["PUMA", "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "PINCP", "PWGTP"]
    d = pd.read_csv(path, usecols=cols, dtype={"OCCP": str})
    d = d[(d.AGEP > 16) & (d.PINCP > 100) & (d.WKHP > 0) & (d.PWGTP >= 1)].reset_index(drop=True)
    occ = d.OCCP.fillna("").str.strip().str.zfill(4).str[:2]
    pob = np.where(d.POBP <= 56, "US", "world_" + (d.POBP // 100).astype(int).astype(str))
    X = pd.DataFrame(dict(age=d.AGEP, education=d.SCHL, hours=d.WKHP, sex=d.SEX.astype(str), marital=d.MAR.astype(str),
                          worker_class=d.COW.astype(str), relationship=d.RELP.astype(str), occupation=occ, birthplace=pob))
    return X, (d.PINCP > 50000).astype(int).to_numpy(), d.PUMA.to_numpy()


def extrasensory(label, base=EXT / "extrasensory" / "extracted"):
    assert label in ES_LABELS
    parts = []
    for f in sorted(base.glob("*.features_labels.csv.gz")):
        with gzip.open(f) as fh:
            d = pd.read_csv(fh)
        d = d[d[label].notna()]
        feats = [c for c in d.columns if c != "timestamp" and not c.startswith("label")]
        parts.append(d[feats].copy().assign(_y=d[label].astype(int), _g=f.name.split(".")[0]))
    d = pd.concat(parts, ignore_index=True)
    return d.drop(columns=["_y", "_g"]), d._y.to_numpy(), d._g.to_numpy()


LOADERS = {"mitdb": mitdb, "oulad": oulad, "acs": acs,
           "extrasensory_sleeping": lambda: extrasensory("label:SLEEPING"),
           "extrasensory_workplace": lambda: extrasensory("label:LOC_main_workplace")}


def describe(name):
    """Rows, groups, base rate and the spread of the outcome rate across groups (no model)."""
    X, y, g = LOADERS[name]()
    r = pd.Series(y).groupby(g).agg(["mean", "size"])
    lg = np.log((r["mean"] * r["size"] + 0.5) / ((1 - r["mean"]) * r["size"] + 0.5))
    return dict(dataset=name, rows=len(y), predictors=X.shape[1], groups=len(r), group_size_min=int(r["size"].min()),
                group_size_median=float(r["size"].median()), group_size_max=int(r["size"].max()), base_rate=float(y.mean()),
                group_rate_sd=float(r["mean"].std()), group_rate_min=float(r["mean"].min()), group_rate_max=float(r["mean"].max()),
                group_logit_sd=float(lg.std()), groups_all_zero=int((r["mean"] == 0).sum()), groups_all_one=int((r["mean"] == 1).sum()))


if __name__ == "__main__":
    out = pd.DataFrame([describe(n) for n in LOADERS])
    pd.set_option("display.width", 250); print(out.to_string(index=False))
    out.to_csv("results/external_data_describe.csv", index=False)
