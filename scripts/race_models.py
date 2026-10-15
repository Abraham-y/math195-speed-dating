"""P3d: the race block (PREREG.md, Part 4; the team's decision of 2026-10-15), and the yes model in the framing of
Fisman, Iyengar, Kamenica and Simonson (2008, "Racial preferences in dating").

Input : data/pairs.csv, data/feature_sets.json, data/clean.csv
Output: results/race_models_metrics.csv, results/race_models_hypotheses.csv, results/race_models_yes.csv,
        results/race_models_rates.csv
Seeds : as scripts/part1_models.py (event folds 20261006; the full logistic model has no other random step).

P3d. The full logistic model of the P3b/P3c ladder on `pre_interactions`, with and without the block `samerace`,
`samerace` x `f_imprace`, `samerace` x `m_imprace` (feature set `pre_race`), on the same rows and the headline folds;
headline and complete-case versions (the complete-case dyads are complete on every ladder column and the race
columns). Individual race codes are not used, and race is never a response.
Exploratory, in the 2008 framing (stated same-race importance against revealed choices): by gender, on dates with all
six ratings and a stated importance, a logistic regression of the yes on the six ratings (standardized within
gender), `samerace`, and `samerace` x the rater's own `imprace` (centred); and the yes rates on same-race and
different-race dates by gender, with the log-odds difference from a logistic regression on `samerace` alone. All SEs
are the cluster sandwich by event of part1_models.coefficient_table (checked in sim_coef_cluster.py).
Run from the repository root: python scripts/race_models.py [--jobs 5]
"""
import sys, json, pathlib, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1
from sim_auc_cluster import coverage_note

RES = pathlib.Path("results")
RACE_RAW = ["samerace", "f_imprace", "m_imprace"]


def imprace_clean(v):
    """Stated importance of same race, 1 to 10; anything else (one 0) is missing (the out-of-range rule)."""
    v = pd.to_numeric(v, errors="coerce")
    return v.where(v.between(1, 10))


def yes_framing(clean):
    """By gender: the yes model with samerace and samerace x own imprace, and the raw same-race yes rates."""
    from sim_coef_cluster import coverage_note as coef_note
    clean = clean.assign(imprace=imprace_clean(clean.imprace))
    coefs, rates = [], []
    for g, name in ((0, "women"), (1, "men")):
        Z, y, ev, d = P1.yes_design(clean, g)
        keep = d.imprace.notna().to_numpy() & d.samerace.notna().to_numpy()
        imp = d.imprace - d.imprace[keep].mean()
        D = Z.assign(samerace=d.samerace, samerace_x_imprace=d.samerace * imp)[keep].reset_index(drop=True)
        t, _ = P1.coefficient_table(D, y[keep], ev[keep])
        coefs.append(t.assign(rater_gender=name, dates=int(keep.sum()), imprace_mean=float(d.imprace[keep].mean()),
                              model="yes ~ six ratings + samerace + samerace x own imprace (centred)"))
        s = clean[(clean.gender == g) & clean.samerace.notna()]
        t1, _ = P1.coefficient_table(s[["samerace"]].reset_index(drop=True).astype(float), s.dec.to_numpy(float), s.event.to_numpy())
        r = t1.set_index("term").loc["samerace"]
        rates.append(dict(rater_gender=name, dates=len(s), yes_rate_same_race=s[s.samerace == 1].dec.mean(),
                          yes_rate_different_race=s[s.samerace == 0].dec.mean(), share_same_race=s.samerace.mean(),
                          log_odds_difference=r.coef, log_odds_difference_se=r.se_cluster_event))
    se_note = coef_note("yes")
    return pd.concat(coefs, ignore_index=True).assign(se_check=se_note, fold_scheme="none (all dates)"), \
        pd.DataFrame(rates).assign(se_check=se_note, fold_scheme="none (all dates)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); args = ap.parse_args()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    headline_folds = M.event_folds(pairs.event)
    every_step = sets["pref_reference"] + sets["extended_pair"] + M.INTERACTIONS + RACE_RAW
    met = []
    for version, spec, extra in (("headline", M.HEADLINE, ()), ("complete-case", M.COMPLETE_CASE, every_step)):
        for fs in ("pre_interactions", "pre_race"):
            rows, *_ = P1.cross_validate(pairs, sets, spec, fs, ("logistic",), args.jobs, complete_extra=extra, folds=headline_folds)
            met += [dict(r, version=version) for r in rows]
            P1.log(f"{version} {fs}: AUC {rows[0]['auc']:.4f} on {rows[0]['dyads']} dyads")
    met = pd.DataFrame(met).assign(se_check=coverage_note())
    met.to_csv(RES / "race_models_metrics.csv", index=False)
    hyp = []
    for version, d in met.groupby("version", sort=False):
        assert d.dyads.nunique() == 1, "both models must use the same dyads"
        a = d.set_index("feature_set").auc
        hyp.append(dict(hypothesis="P3d", version=version, statistic=f"full logistic AUC with the race block {a.pre_race:.4f} minus without "
                        f"{a.pre_interactions:.4f} = {a.pre_race - a.pre_interactions:+.4f} ({int(d.dyads.iloc[0])} dyads)",
                        supported=bool(a.pre_race - a.pre_interactions < 0.02), fold_scheme="event", se_check=coverage_note()))
    H = pd.DataFrame(hyp); H.to_csv(RES / "race_models_hypotheses.csv", index=False)
    coefs, rates = yes_framing(pd.read_csv("data/clean.csv", low_memory=False))
    coefs.to_csv(RES / "race_models_yes.csv", index=False); rates.to_csv(RES / "race_models_rates.csv", index=False)
    print(met[["version", "feature_set", "auc", "auc_se", "dyads", "matches"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(H[["hypothesis", "version", "statistic", "supported"]].to_string(index=False))
    print(coefs[coefs.term.isin(["samerace", "samerace_x_imprace", "attr"])][["rater_gender", "term", "coef", "se_cluster_event", "z", "dates"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(rates.drop(columns=["se_check", "fold_scheme"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
