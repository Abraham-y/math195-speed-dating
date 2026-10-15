"""Power and events needed (PREREG.md, Part 5; exploratory, no hypothesis).

Input : data/pairs.csv, data/feature_sets.json, data/clean.csv; results/part1_models_metrics.csv (AUC check) and
        results/part1_models_stated_vs_revealed.csv (P5 check)
Output: results/power_summary.csv (one row per comparison and effect size), results/power_check.csv
Seeds : Part 1's for the held-out predictions (event folds 20261006, inner tuning folds 20261006 + 1 + fold, random
        forest random_state 20261106 + fold); 20264006 for every random step here, a fresh generator for each: the P5
        event bootstrap (1,000 resamples), the P4 power check (500 replicates of G* events) and the SE checks at the
        real 21 events (500 replicates each).

- P3 and P4: paired held-out AUC differences under the headline specification and the Part 1 event folds: P3 is the
  full logistic regression on `pre` minus the same on `pre_individual`; P4 is the random forest minus the full
  logistic regression on `pre`. The held-out probabilities come from part1_models.cross_validate (("rf",) on `pre`
  gives both the logistic and the forest; ("logistic",) on `pre_individual`), and the AUCs are checked against
  results/part1_models_metrics.csv. SE: modeling.auc_diff_clustered_var by event (the Obuchowski variance of the
  difference of the two models' placement values, "as in C3").
- P5: the attractiveness gap, revealed minus stated share (percentage points of the 100-point allocation), by gender.
  Deviation from PREREG.md: "clustered SEs as in C3" is defined only for AUCs, so the gap's SE is the SD of an event
  bootstrap (1,000 resamples of the 21 events with replacement) that refits part1_models.yes_model's computation on
  each resample (ratings standardized within the resample, logistic regression by IRLS, revealed shares from the
  positive coefficients, stated shares from the 100-point-wave participants of the resampled events, each copy of an
  event counted). The resample that takes every event once reproduces Part 1's gaps exactly (asserted). The margin of
  the attractiveness gap over the next largest gap (what P5's verdict turns on) is reported too.
- Events needed for 80% power at alpha = 0.05 (two-sided): G* = (z_0.975 + z_0.80)^2 sigma^2 / Delta^2, with
  sigma = sqrt(21) x SE (the SE scales as 1 / sqrt(events)), for the observed difference and for Delta = 0.01 and 0.02.
  For P5, Delta = 0.01 and 0.02 are read as shares of the allocation, i.e. 1 and 2 percentage points (deviation: the
  registered values are on the AUC scale). Also the power the 21 events give for each Delta.
- Check for P4 (registered): draw ceil(G*) events with replacement (each draw a separate cluster), recompute the AUC
  difference and its clustered SE on the fixed held-out predictions, and count the Wald tests that reject at 0.05;
  500 replicates. The share should be near 0.80. Also (exploratory) the same resampling at the real 21 events for P3
  and P4, comparing the mean clustered SE with the SD of the resampled differences (a consistency check against the
  event bootstrap, not a simulation with a known truth).
Run from the repository root: python scripts/power.py [--jobs 2]
"""
import sys, json, pathlib, argparse, time
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, "scripts")
import modeling as M
import part1_models as P1
from sim_auc_cluster import coverage_note

RES = pathlib.Path("results")
SEED = 20264006
BOOT_REPS, CHECK_REPS = 1000, 500
ALPHA, POWER = 0.05, 0.80
Z_SUM = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)            # 1.959964 + 0.841621
DELTAS = (0.01, 0.02)
ATTRS = P1.ATTRS


def events_needed(se, delta, G):
    """Events for 80% power at two-sided alpha 0.05: (z_0.975 + z_0.80)^2 sigma^2 / Delta^2, sigma = sqrt(G) SE."""
    sigma = np.sqrt(G) * se
    return (Z_SUM * sigma / delta) ** 2


def power_now(se, delta):
    """Power of the two-sided Wald test at alpha 0.05 when the true difference is delta and the SE is se."""
    z = norm.ppf(1 - ALPHA / 2); r = abs(delta) / se
    return norm.cdf(r - z) + norm.cdf(-r - z)


def resample_rows(ev, events, sample):
    """Row indices and new cluster labels for a draw of events (with repetition: each draw is its own cluster)."""
    by_event = {e: np.where(ev == e)[0] for e in events}
    idx = np.concatenate([by_event[events[s]] for s in sample])
    lab = np.concatenate([np.full(len(by_event[events[s]]), i) for i, s in enumerate(sample)])
    return idx, lab


def resampling_check(a, b, y, ev, n_events, reps, seed):
    """Replicates of the AUC difference AUC(a) - AUC(b) and its clustered SE on n_events events drawn with
    replacement; a replicate rejects when |difference / SE| > z_0.975."""
    rng = np.random.default_rng(seed); events = np.unique(ev); out = []
    for r in range(reps):
        idx, lab = resample_rows(ev, events, rng.integers(0, len(events), n_events))
        d = M.auc(a[idx], y[idx]) - M.auc(b[idx], y[idx])
        se = np.sqrt(M.auc_diff_clustered_var(a[idx], b[idx], y[idx], lab))
        out.append(dict(replicate=r, difference=d, se=se, z=d / se, reject=abs(d / se) > norm.ppf(1 - ALPHA / 2)))
    return pd.DataFrame(out)


class YesGap:
    """part1_models.yes_model's revealed-minus-stated gaps, recomputed on any multiset of events."""

    def __init__(self, clean):
        shares = [f"{a}1_1_share" for a in ATTRS]
        self.events = np.sort(clean.event.unique())
        self.g = {}
        for g in (0, 1):
            d = clean[(clean.gender == g) & clean[ATTRS].notna().all(axis=1)]
            people = clean[clean.gender == g].drop_duplicates("rater")
            p100 = people[(people.pref_scale == "100pt") & people[shares].notna().all(axis=1)]
            self.g[g] = dict(R=d[ATTRS].to_numpy(float), y=d.dec.to_numpy(float),
                             rows={e: np.where(d.event.to_numpy() == e)[0] for e in self.events},
                             S=p100[shares].to_numpy(float), people={e: np.where(p100.event.to_numpy() == e)[0] for e in self.events})

    def gaps(self, g, sample):
        """(revealed share, stated share, gap) for the six attributes, on the events in `sample` (with repetition)."""
        G = self.g[g]
        rows = np.concatenate([G["rows"][e] for e in sample]); ppl = np.concatenate([G["people"][e] for e in sample])
        R = G["R"][rows]
        Z = (R - R.mean(0)) / R.std(0, ddof=1)                    # standardized within the resample, as yes_design
        w, _ = M.fit_logit(Z, G["y"][rows], ridge=1e-6, iters=500)
        pos = np.clip(w[1:], 0, None)
        revealed = 100 * pos / pos.sum(); stated = G["S"][ppl].mean(0)
        return revealed, stated, revealed - stated


def gap_statistics(gap):
    """The attractiveness gap and its margin over the next largest gap."""
    i = ATTRS.index("attr")
    return gap[i], gap[i] - np.max(np.delete(gap, i))


def summary_rows(comparison, quantity, estimate, se, se_method, unit, deltas, G, se_check, extra=None):
    rows = []
    for label, delta in [("observed", abs(estimate))] + [(f"{d}", d * unit) for d in deltas]:
        n = events_needed(se, delta, G)
        rows.append(dict(comparison=comparison, quantity=quantity, estimate=estimate, se=se, z=estimate / se, se_method=se_method,
                         events=G, sigma=np.sqrt(G) * se, delta_label=label, delta=delta, events_needed=n,
                         events_needed_ceil=int(np.ceil(n)), power_with_current_events=power_now(se, delta), se_check=se_check,
                         **(extra or {})))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2); args = ap.parse_args()
    t0 = time.time()
    pairs = pd.read_csv("data/pairs.csv"); sets = json.loads(pathlib.Path("data/feature_sets.json").read_text())
    part1 = pd.read_csv(RES / "part1_models_metrics.csv")
    part1 = part1[part1.headline.astype(bool)].set_index(["feature_set", "model"]).auc

    # held-out predictions, Part 1 seeds and folds, headline specification
    _, _, _, (prob_pre, y, ev, _) = P1.cross_validate(pairs, sets, M.HEADLINE, "pre", ("rf",), args.jobs)
    _, _, _, (prob_ind, y2, ev2, _) = P1.cross_validate(pairs, sets, M.HEADLINE, "pre_individual", ("logistic",), args.jobs)
    ids_pre, ids_ind = M.build_frame(pairs, sets, M.HEADLINE, "pre")[3], M.build_frame(pairs, sets, M.HEADLINE, "pre_individual")[3]
    assert np.array_equal(y, y2) and np.array_equal(ev, ev2) and ids_pre.equals(ids_ind), "the paired models must score the same dyads"
    s = {"logistic `pre`": prob_pre["logistic"], "random forest `pre`": prob_pre["rf"], "logistic `pre_individual`": prob_ind["logistic"]}
    ref = {"logistic `pre`": ("pre", "logistic"), "random forest `pre`": ("pre", "rf"), "logistic `pre_individual`": ("pre_individual", "logistic")}
    checks = []
    for k, v in s.items():
        a = M.auc(v, y); checks.append(dict(model=k, auc=a, auc_part1=part1[ref[k]]))
        P1.log(f"{k}: AUC {a:.4f} (Part 1 {part1[ref[k]]:.4f})")
        assert abs(a - part1[ref[k]]) < 1e-3, f"{k} does not reproduce Part 1"
    G = len(np.unique(ev))

    comps = {"P3": ("full logistic `pre` minus `pre_individual` (AUC)", s["logistic `pre`"], s["logistic `pre_individual`"]),
             "P4": ("random forest minus full logistic, `pre` (AUC)", s["random forest `pre`"], s["logistic `pre`"])}
    est = {}
    for c, (q, a, b) in comps.items():
        d = M.auc(a, y) - M.auc(b, y); se = np.sqrt(M.auc_diff_clustered_var(a, b, y, ev))
        est[c] = (q, d, se)
        P1.log(f"{c}: {q} = {d:.4f}, clustered SE {se:.4f}")

    # checks by resampling events: the registered P4 check at G*, and the SE checks at the real 21 events
    q4, d4, se4 = est["P4"]
    g_star = events_needed(se4, abs(d4), G); n_star = int(np.ceil(g_star))
    chk = []
    for c, n, role in (("P4", n_star, f"registered power check at G* = {g_star:.1f} events for the observed difference (expected rejection share 0.80)"),
                       ("P4", G, "SE check at the real 21 events (exploratory)"), ("P3", G, "SE check at the real 21 events (exploratory)")):
        q, a, b = comps[c]
        R = resampling_check(a, b, y, ev, n, CHECK_REPS, SEED)
        share = R.reject.mean()
        chk.append(dict(comparison=c, quantity=q, role=role, events_drawn=n, replicates=CHECK_REPS, seed=SEED,
                        observed_difference=est[c][1], observed_se_21_events=est[c][2],
                        rejection_share=share, rejection_share_se=np.sqrt(share * (1 - share) / CHECK_REPS),
                        normal_theory_power=power_now(est[c][2] * np.sqrt(G / n), est[c][1]),
                        mean_difference=R.difference.mean(), sd_difference=R.difference.std(ddof=1), mean_se=R.se.mean(), median_se=R.se.median(),
                        ratio_mean_se_to_sd=R.se.mean() / R.difference.std(ddof=1)))
        P1.log(f"check {c} at {n} events: rejection share {share:.3f}; mean SE / SD of differences {chk[-1]['ratio_mean_se_to_sd']:.3f}")
    C = pd.DataFrame(chk).assign(fold_scheme="event (held-out predictions fixed; events resampled)")
    C.to_csv(RES / "power_check.csv", index=False)

    rows = []
    for c in ("P3", "P4"):
        q, d, se = est[c]
        sc = C[(C.comparison == c) & (C.events_drawn == G)].iloc[0]
        note = (f"clustered SE of the difference; event-resampling consistency check at 21 events ({CHECK_REPS} replicates, seed {SEED}): "
                f"mean SE / SD of resampled differences {sc.ratio_mean_se_to_sd:.2f}. Single-AUC SEs: {coverage_note()}")
        rows += summary_rows(c, q, d, se, "modeling.auc_diff_clustered_var by event", 1.0, DELTAS, G, note,
                             dict(fold_scheme="event", seed=f"Part 1 ({M.FOLD_SEED}; forest {P1.RF_SEED} + fold)"))

    # P5: event bootstrap of the attractiveness gap, refitting the yes model on each resample
    yg = YesGap(pd.read_csv("data/clean.csv", low_memory=False))
    sr = pd.read_csv(RES / "part1_models_stated_vs_revealed.csv")
    rng = np.random.default_rng(SEED)
    samples = [yg.events[rng.integers(0, G, G)] for _ in range(BOOT_REPS)]
    for g, name in ((0, "women"), (1, "men")):
        rev, sta, gap = yg.gaps(g, yg.events)
        t = sr[sr.rater_gender == name].set_index("attribute").loc[ATTRS]
        assert np.allclose(rev, t.revealed_share, atol=1e-6) and np.allclose(sta, t.stated_share, atol=1e-6) \
            and np.allclose(gap, t.gap_revealed_minus_stated, atol=1e-6), "the identity resample must reproduce Part 1"
        g0, m0 = gap_statistics(gap)
        B = np.array([gap_statistics(yg.gaps(g, smp)[2]) for smp in samples])
        for j, (q, e0) in enumerate((("attractiveness gap, revealed minus stated share (points)", g0),
                                     ("margin of the attractiveness gap over the next largest gap (points; exploratory)", m0))):
            se = B[:, j].std(ddof=1); lo, hi = np.percentile(B[:, j], [5, 95])
            rows += summary_rows(f"P5 {name}", q, e0, se, f"event bootstrap SD ({BOOT_REPS} resamples, seed {SEED}), yes model refitted",
                                 100.0, DELTAS, G,
                                 f"bootstrap SE (deviation: PREREG's 'clustered SEs as in C3' is defined only for AUCs); 90% percentile "
                                 f"interval {lo:.2f} to {hi:.2f}; the yes model's coefficient SEs were checked in scripts/sim_coef_cluster.py",
                                 dict(fold_scheme="none (all dates)", seed=SEED, ci90_low=lo, ci90_high=hi))
        P1.log(f"P5 {name}: gap {g0:.2f} (bootstrap SE {B[:, 0].std(ddof=1):.2f}); margin {m0:.2f} (SE {B[:, 1].std(ddof=1):.2f})")
    S = pd.DataFrame(rows)
    S["delta_unit"] = np.where(S.comparison.str.startswith("P5"), "percentage points (0.01 = 1 point)", "AUC")
    S.to_csv(RES / "power_summary.csv", index=False)

    print(pd.DataFrame(checks).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(S[["comparison", "quantity", "estimate", "se", "z", "delta_label", "delta", "events_needed", "events_needed_ceil",
             "power_with_current_events"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(C[["comparison", "role", "events_drawn", "rejection_share", "rejection_share_se", "normal_theory_power", "mean_difference",
             "sd_difference", "mean_se", "ratio_mean_se_to_sd"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    P1.log(f"done in {time.time() - t0:.0f}s; wrote power_summary.csv, power_check.csv")


if __name__ == "__main__":
    main()
