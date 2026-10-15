"""M1: the matchmaker (PREREG.md, Part 5). Could a matchmaker who pairs people one-to-one by the predicted match
probabilities do better than chance, before the date and after it?

Input : data/pairs.csv, data/feature_sets.json; results/part1_models_metrics.csv (to confirm the AUCs reproduce
        Part 1) and results/stable_matching_summary.csv (reference only)
Output: results/matchmaker_events.csv (one row per event), results/matchmaker_summary.csv,
        results/matchmaker_hypotheses.csv
Seeds : Part 1's for the predictions (event folds 20261006, inner tuning folds 20261006 + 1 + fold, random forest
        random_state 20261106 + fold); the event bootstrap 20263906 (1,000 resamples, the same resamples for every
        statistic, so the feature sets are compared on paired resamples).

- Predictions: the held-out match probabilities of the headline random forest (part1_models.cross_validate, `event`
  5-fold CV with the Part 1 folds), for `pre` and `pre+date`; each is placed by build_frame's ids (f_iid, m_iid) and
  event. The AUCs are checked against results/part1_models_metrics.csv.
- Assignment: per event, the maximum-weight one-to-one assignment (scipy linear_sum_assignment) on the women x men
  matrix of predicted probabilities. Pairs absent from the data never enter: they get weight 0 for the solver and
  are dropped from its answer, which gives the maximum-weight matching among the present pairs (the probabilities are
  >= 0, so any matching extends to a full assignment of the same weight). Every event in pairs.csv has the full
  women x men grid (4,184 = the sum over events of women x men), so no pair is absent and every event gets
  min(women, men) pairs; the code asserts nothing absent was chosen.
- Comparison: realized matches among the assigned pairs against random pairing's expectation, min(women, men) x the
  event's match rate (matches over dyads present). With a complete grid this is exactly the mean over all one-to-one
  assignments of min(women, men) pairs.
- Statistic: the ratio of totals over events (sum realized / sum expected), with a 90% percentile interval from
  1,000 bootstrap resamples of events (the assignments held fixed). M1 is supported if the `pre` ratio is below 1.3
  and the `pre+date` ratio is above 2 (point estimates, thresholds as registered).
- Exploratory: precision at k (k = 1, 3), per woman (her k partners with the highest predicted probability) and per
  man, with the lift over the base rate (the overall match rate of the dyads) and over random choice (the mean of
  each person's own match rate, which is what picking k partners at random would give); ties broken by partner iid.
  Also the most matches any one-to-one pairing could realize in each event (a maximum matching on the mutual yeses),
  and, as a reference, the share of mutual yeses in the stable matchings of scripts/stable_matching.py (built from
  the in-date liking ratings).
Run from the repository root: python scripts/matchmaker.py [--jobs 2]
"""
import sys, json, pathlib, argparse, time
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1

RES = pathlib.Path("results")
BOOT_SEED, BOOT_REPS = 20263906, 1000
FEATURE_SETS = ("pre", "pre+date")
KS = (1, 3)
RATIO_PRE_BELOW, RATIO_DATE_ABOVE = 1.3, 2.0          # M1's thresholds (PREREG.md, Part 5)


def assign(weights):
    """Maximum-weight one-to-one assignment on a rows x columns matrix of nonnegative weights, NaN where the pair is
    absent. Absent pairs get weight 0 for the solver and are dropped from its answer, so the result is the
    maximum-weight matching among the present pairs. Returns a list of (row, column)."""
    W = np.asarray(weights, float)
    present = ~np.isnan(W)
    assert (W[present] >= 0).all(), "weights must be nonnegative"
    r, c = linear_sum_assignment(np.where(present, W, 0.0), maximize=True)
    return [(int(i), int(j)) for i, j in zip(r, c) if present[i, j]]


def expected_random(women, men, matches, dyads):
    """Random pairing's expected matches: min(women, men) x the event's match rate (matches over dyads present)."""
    return min(women, men) * matches / dyads


def precision_at_k(pred, side, prob_col, k):
    """One row per person on `side` ('women' or 'men'): the share of matches among the k partners with the highest
    predicted probability (ties broken by partner iid), and the person's own match rate over all partners."""
    me, other = ("f_iid", "m_iid") if side == "women" else ("m_iid", "f_iid")
    d = pred.sort_values([me, prob_col, other], ascending=[True, False, True])
    top = d.groupby(me).head(k).groupby(me).match.mean().rename("precision")
    own = pred.groupby(me).agg(own_rate=("match", "mean"), event=("event", "first"), partners=("match", "size"))
    out = own.join(top).reset_index().rename(columns={me: "iid"})
    assert (out.partners >= k).all(), f"someone has fewer than {k} partners"
    return out


def grids(d, value):
    women, men = np.sort(d.f_iid.unique()), np.sort(d.m_iid.unique())
    return d.pivot(index="f_iid", columns="m_iid", values=value).reindex(index=women, columns=men).to_numpy(float)


def event_table(pred):
    rows = []
    for e, d in pred.groupby("event"):
        match = grids(d, "match")
        nw, nm = match.shape
        row = dict(event=int(e), women=nw, men=nm, dyads=len(d), absent_pairs=int(np.isnan(match).sum()),
                   matches=int(d.match.sum()), match_rate=d.match.mean())
        row["expected_random"] = expected_random(nw, nm, row["matches"], row["dyads"])
        best = assign(match)                                  # the most matches any one-to-one pairing could realize
        row["max_possible"] = int(sum(match[i, j] for i, j in best))
        for fs in FEATURE_SETS:
            P = grids(d, f"prob_{fs}")
            chosen = assign(P)
            assert all(not np.isnan(match[i, j]) for i, j in chosen), "an absent pair was chosen"
            row[f"assigned_{fs}"] = len(chosen)
            row[f"realized_{fs}"] = int(sum(match[i, j] for i, j in chosen))
            row[f"predicted_{fs}"] = float(sum(P[i, j] for i, j in chosen))
        rows.append(row)
    E = pd.DataFrame(rows)
    for fs in FEATURE_SETS:
        E[f"ratio_{fs}"] = E[f"realized_{fs}"] / E.expected_random
    return E


def percentile(v, lo=5, hi=95):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2); args = ap.parse_args()
    t0 = time.time()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    part1 = pd.read_csv(RES / "part1_models_metrics.csv")
    part1 = part1[part1.headline.astype(bool) & (part1.model == "rf")].set_index("feature_set").auc

    pred, aucs = None, {}
    for fs in FEATURE_SETS:
        rows, finfo, _, (prob, y, ev, X) = P1.cross_validate(pairs, sets, M.HEADLINE, fs, ("rf",), args.jobs)
        _, y2, ev2, ids = M.build_frame(pairs, sets, M.HEADLINE, fs)
        assert np.array_equal(y, y2) and np.array_equal(ev, ev2)
        rf = next(r for r in rows if r["model"] == "rf")
        aucs[fs] = (rf["auc"], rf["auc_se"], float(part1[fs]))
        P1.log(f"{fs}: random forest AUC {rf['auc']:.4f} (SE {rf['auc_se']:.4f}); Part 1 {part1[fs]:.4f}; "
               f"leaf sizes by fold {[f['rf_min_samples_leaf'] for f in finfo]}")
        assert abs(rf["auc"] - part1[fs]) < 1e-3, f"{fs}: AUC {rf['auc']:.4f} does not reproduce Part 1 ({part1[fs]:.4f})"
        p = pd.DataFrame({"event": ev, "f_iid": ids.f_iid.to_numpy(), "m_iid": ids.m_iid.to_numpy(), "match": y.astype(int),
                          f"prob_{fs}": prob["rf"]})
        pred = p if pred is None else pred.merge(p, on=["event", "f_iid", "m_iid", "match"], how="outer", validate="1:1")
    assert len(pred) == len(pairs) and pred[[f"prob_{fs}" for fs in FEATURE_SETS]].notna().all().all()

    E = event_table(pred)
    E.to_csv(RES / "matchmaker_events.csv", index=False)
    assert E.absent_pairs.sum() == 0 and (E.assigned_pre == E[["women", "men"]].min(axis=1)).all()

    # event bootstrap: the same 1,000 resamples of the 21 events for every statistic
    G = len(E)
    idx = np.random.default_rng(BOOT_SEED).integers(0, G, size=(BOOT_REPS, G))
    exp_ = E.expected_random.to_numpy()
    summ, ratio, boot = [], {}, {}
    for fs in FEATURE_SETS:
        real = E[f"realized_{fs}"].to_numpy(float)
        ratio[fs] = real.sum() / exp_.sum()
        boot[fs] = real[idx].sum(1) / exp_[idx].sum(1)
        lo, hi = percentile(boot[fs])
        assigned = int(E[f"assigned_{fs}"].sum())
        summ += [dict(section="assignment", feature_set=fs, measure="ratio of realized to random-pairing matches (ratio of totals)",
                      value=ratio[fs], ci90_low=lo, ci90_high=hi, n=G, note=f"{int(real.sum())} realized against {exp_.sum():.2f} expected over {G} events"),
                 dict(section="assignment", feature_set=fs, measure="realized matches", value=real.sum(), n=G, note=f"{assigned} assigned pairs"),
                 dict(section="assignment", feature_set=fs, measure="realized matches per event", value=real.mean(), n=G,
                      note=f"random pairing {exp_.mean():.2f} per event"),
                 dict(section="assignment", feature_set=fs, measure="share of assigned pairs that matched", value=real.sum() / assigned,
                      ci90_low=percentile(real[idx].sum(1) / E[f"assigned_{fs}"].to_numpy()[idx].sum(1))[0],
                      ci90_high=percentile(real[idx].sum(1) / E[f"assigned_{fs}"].to_numpy()[idx].sum(1))[1], n=assigned, note=""),
                 dict(section="assignment", feature_set=fs, measure="events where the assignment beat random pairing's expectation",
                      value=int((real > exp_).sum()), n=G, note=f"tied or below in {int((real <= exp_).sum())}"),
                 dict(section="check", feature_set=fs, measure="held-out AUC of the random forest (reproduced)", value=aucs[fs][0],
                      n=len(pred), note=f"SE {aucs[fs][1]:.4f}; Part 1 {aucs[fs][2]:.4f}; difference {aucs[fs][0] - aucs[fs][2]:+.1e}")]
    d = boot["pre+date"] - boot["pre"]
    summ.append(dict(section="assignment", feature_set="pre+date minus pre", measure="difference of the ratios", value=ratio["pre+date"] - ratio["pre"],
                     ci90_low=percentile(d)[0], ci90_high=percentile(d)[1], n=G, note="paired resamples"))
    held = (boot["pre"] < RATIO_PRE_BELOW) & (boot["pre+date"] > RATIO_DATE_ABOVE)
    summ.append(dict(section="assignment", feature_set="both", measure="share of resamples in which M1's two conditions both hold (exploratory)",
                     value=held.mean(), n=BOOT_REPS, note=f"`pre` below {RATIO_PRE_BELOW} in {np.mean(boot['pre'] < RATIO_PRE_BELOW):.3f}; "
                                                          f"`pre+date` above {RATIO_DATE_ABOVE} in {np.mean(boot['pre+date'] > RATIO_DATE_ABOVE):.3f}"))
    random_exp = exp_.sum()
    assigned_total = int(E.assigned_pre.sum())
    summ += [dict(section="assignment", feature_set="none", measure="random pairing's expected matches", value=random_exp, n=G,
                  note=f"share of assigned pairs {random_exp / assigned_total:.4f}"),
             dict(section="assignment", feature_set="none", measure="most matches any one-to-one pairing could realize (oracle)",
                  value=int(E.max_possible.sum()), n=G, note=f"ratio to random {E.max_possible.sum() / random_exp:.3f}; "
                                                               f"share of assigned pairs {E.max_possible.sum() / assigned_total:.4f}")]

    # precision at k (exploratory): per person, with the same event resamples
    base = pred.match.mean()
    ev_index = {e: i for i, e in enumerate(E.event)}
    for fs in FEATURE_SETS:
        for side in ("women", "men"):
            for k in KS:
                pk = precision_at_k(pred, side, f"prob_{fs}", k)
                g = pk.event.map(ev_index).to_numpy()
                s_prec = np.bincount(g, pk.precision, G); s_own = np.bincount(g, pk.own_rate, G); s_n = np.bincount(g, minlength=G)
                b_prec = s_prec[idx].sum(1) / s_n[idx].sum(1); b_own = s_own[idx].sum(1) / s_n[idx].sum(1)
                prec, own = pk.precision.mean(), pk.own_rate.mean()
                lo, hi = percentile(b_prec); llo, lhi = percentile(b_prec / b_own)
                summ.append(dict(section="precision at k", feature_set=fs, side=side, k=k, measure="precision at k (mean over people)",
                                 value=prec, ci90_low=lo, ci90_high=hi, n=len(pk), base_rate=base, lift_over_base_rate=prec / base,
                                 random_choice=own, lift_over_random_choice=prec / own, lift_over_random_choice_ci90_low=llo,
                                 lift_over_random_choice_ci90_high=lhi,
                                 note="base rate: the overall match rate of the dyads; random choice: the mean of each person's own match rate"))

    sm = pd.read_csv(RES / "stable_matching_summary.csv")
    for r in sm[sm.matching.isin(["men propose", "women propose"])].itertuples():
        summ.append(dict(section="reference", feature_set="in-date liking", measure=f"stable matching ({r.matching}): share of pairs that were mutual yeses",
                         value=r.share_mutual_yes, n=int(r.stable_pairs), note="results/stable_matching_summary.csv (scripts/stable_matching.py)"))
    S = pd.DataFrame(summ).assign(fold_scheme="event", bootstrap_seed=BOOT_SEED, bootstrap_resamples=BOOT_REPS)
    S["k"] = S.k.astype("Int64")
    S.to_csv(RES / "matchmaker_summary.csv", index=False)

    lo_p, hi_p = percentile(boot["pre"]); lo_d, hi_d = percentile(boot["pre+date"])
    H = pd.DataFrame([dict(
        hypothesis="M1", version="headline random forest, Part 1 event folds",
        statistic=f"ratio of realized to random-pairing matches: `pre` {ratio['pre']:.3f} (90% interval {lo_p:.3f} to {hi_p:.3f}), "
                  f"`pre+date` {ratio['pre+date']:.3f} (90% interval {lo_d:.3f} to {hi_d:.3f}); realized {int(E.realized_pre.sum())} and "
                  f"{int(E['realized_pre+date'].sum())} against {random_exp:.1f} expected over {G} events ({assigned_total} assigned pairs)",
        supported=bool(ratio["pre"] < RATIO_PRE_BELOW and ratio["pre+date"] > RATIO_DATE_ABOVE),
        se_check=f"percentile interval from {BOOT_REPS} bootstrap resamples of events (seed {BOOT_SEED}), assignments held fixed; "
                 "not checked by simulation (PREREG.md names no check for M1)",
        fold_scheme="event")])
    H.to_csv(RES / "matchmaker_hypotheses.csv", index=False)

    cols = ["event", "women", "men", "matches", "match_rate", "expected_random", "max_possible", "realized_pre", "realized_pre+date",
            "predicted_pre", "predicted_pre+date"]
    print(E[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    show = S[S.section != "precision at k"][["section", "feature_set", "measure", "value", "ci90_low", "ci90_high", "n", "note"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pk = S[S.section == "precision at k"][["feature_set", "side", "k", "value", "ci90_low", "ci90_high", "n", "base_rate", "lift_over_base_rate",
                                           "random_choice", "lift_over_random_choice"]]
    print("\nprecision at k (exploratory):"); print(pk.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + H[["hypothesis", "statistic", "supported"]].to_string(index=False))
    P1.log(f"done in {time.time() - t0:.0f}s; wrote matchmaker_events.csv, matchmaker_summary.csv, matchmaker_hypotheses.csv")


if __name__ == "__main__":
    main()
