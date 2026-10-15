"""Social relations model of liking (PREREG.md, Part 3: K1, K2; exploratory variance shares).

Input : data/clean.csv
Output: results/srm_estimates.csv (reciprocity correlations with 90% intervals, variance shares),
        results/srm_check.csv (simulation check of the estimators and intervals), results/srm_hypotheses.csv
Seeds : cluster bootstrap 20262606; simulation check 20262706.

At each event, every woman rates every man and every man rates every woman. For each event and direction (women
rating men; men rating women) liking = mean + actor + partner + residual, fitted by alternating means on the
observed ratings: the actor effect is how much a rater likes their dates in general, the partner effect how much a
person is liked in general, and the residual is what is specific to the pair (plus noise).
- Dyadic reciprocity (K1): the correlation over dyads of the two directions' residuals: if she likes him more
  than either of them would predict, does he like her more too?
- Generalized reciprocity (K2): the correlation over people of their actor effect (liking others) and their
  partner effect (being liked), each centred within event and direction.
- Variance shares: actor, partner and residual (relationship plus noise) by two-way moment estimators.
90% intervals: 1,000 resamples of whole events. The simulation draws liking from a social relations model with
the real data's estimates on the real design and checks the estimators' bias and the intervals' coverage.

The simulation showed the registered generalized-reciprocity estimator is biased: a person's estimated actor
effect averages the residuals of the ratings they gave, their partner effect the residuals of the ratings they
received from the same people, and those two sets of residuals are correlated by the dyadic reciprocity. The
expected covariance of the two estimated effects is therefore the true covariance plus (residual covariance) x
(dates rated both ways) / (dates given x dates received), and each variance is inflated by residual variance /
dates. `generalized_reciprocity_corrected` subtracts these terms (the round-robin correction of Kenny's social
relations model, here for the two-sided design) and is exploratory (PREREG.md change log, 2026-10-15). K2 is
decided on the registered estimator. With very small events (3 or 4 a side) the corrected variances can approach
zero and the ratio becomes unstable; on the real design its simulated bias is near zero (results/srm_check.csv).
K3 (PREREG.md, Part 4): `python scripts/srm.py dec` runs the same model on the yes decisions (0/1) and writes
results/srm_dec_*.csv. Its simulation check draws a latent social relations model with the liking variance shares
and a latent pair correlation of 0.2, thresholded at the real yes rate; since the 0/1 estimator's target differs
from the latent value, the intervals are checked for covering the mean estimate over replicates.
Run from the repository root: python scripts/srm.py [like|dec] [bootstrap resamples] [simulation replicates]
"""
import sys, pathlib
import numpy as np, pandas as pd

RES = pathlib.Path("results")
BOOT_SEED, SIM_SEED = 20262606, 20262706
KEYS = ("dyadic_reciprocity", "generalized_reciprocity", "generalized_reciprocity_corrected", "actor_share", "partner_share", "residual_share")


def additive(x, r, c, tol=1e-12, max_iter=1000):
    """liking = mu + actor[r] + partner[c] + residual on the observed cells, by alternating means. The effects are
    returned centred (mean zero over raters and over ratees); the residuals do not depend on the centring."""
    ri, ru = pd.factorize(r); ci, cu = pd.factorize(c)
    nr, nc = np.bincount(ri), np.bincount(ci)
    mu, a, b = x.mean(), np.zeros(len(ru)), np.zeros(len(cu))
    for _ in range(max_iter):
        a_new = np.bincount(ri, x - mu - b[ci]) / nr
        b_new = np.bincount(ci, x - mu - a_new[ri]) / nc
        done = max(np.abs(a_new - a).max(), np.abs(b_new - b).max()) < tol
        a, b = a_new, b_new
        if done:
            break
    e = x - mu - a[ri] - b[ci]
    return mu + a.mean() + b.mean(), pd.Series(a - a.mean(), index=ru), pd.Series(b - b.mean(), index=cu), e


def event_pieces(ev, col="like"):
    """One event: the dyads' residual pairs, sums over people for the generalized-reciprocity moments, and the
    variance-share moments of each direction."""
    eff, res, s2, mom = {}, {}, {}, []
    for g in (0, 1):
        d = ev[(ev.gender == g) & ev[col].notna()]
        if d.empty:
            continue
        mu, a, b, e = additive(d[col].to_numpy(float), d.rater.to_numpy(), d.ratee.to_numpy())
        nr, nc = d.rater.nunique(), d.ratee.nunique()
        s2[g] = np.sum(e ** 2) / max(len(e) - nr - nc + 1, 1)
        given, received = d.rater.value_counts(), d.ratee.value_counts()
        for p, v in a.items():
            eff.setdefault(p, {}).update(actor=v, actor_noise=s2[g] / given[p], given=given[p], gender=g)
        for p, v in b.items():
            eff.setdefault(p, {}).update(partner=v, partner_noise=s2[g] / received[p], received=received[p])
        for rr, cc, v in zip(d.rater, d.ratee, e):
            res[(min(rr, cc), max(rr, cc), g)] = v
        mom.append(dict(n=len(e), actor=max(a.var(ddof=1) - s2[g] / nc, 0), partner=max(b.var(ddof=1) - s2[g] / nr, 0), residual=s2[g]))
    both = [(i, j) for (i, j, g) in res if g == 0 and (i, j, 1) in res]
    dyads = np.array([[res[(i, j, 0)], res[(i, j, 1)]] for i, j in both]).reshape(-1, 2)
    n_both = pd.Series([p for ij in both for p in ij]).value_counts()
    n_w, n_m = (ev[ev.gender == 0].rater.nunique(), ev[ev.gender == 1].rater.nunique())
    cov_res = np.sum(dyads[:, 0] * dyads[:, 1]) / max(len(dyads) - n_w - n_m + 1, 1) if len(dyads) else 0.0
    people = pd.DataFrame([dict(person=p, **v) for p, v in eff.items() if "actor" in v and "partner" in v])
    people["cross_noise"] = cov_res * people.person.map(n_both).fillna(0) / (people.given * people.received)
    sums = []
    for g, q in people.groupby("gender"):                         # effects are centred within event and direction
        A, B = q.actor - q.actor.mean(), q.partner - q.partner.mean()
        sums.append([np.sum(A * B), np.sum(A * A), np.sum(B * B), len(q) - 1, q.cross_noise.sum(), q.actor_noise.sum(), q.partner_noise.sum(), len(q)])
    return people[["actor", "partner"]].to_numpy(), dyads, np.array(sums), mom


def estimates(pieces):
    P = np.vstack([p[0] for p in pieces]); D = np.vstack([p[1] for p in pieces]); S = np.vstack([p[2] for p in pieces]).sum(0)
    mom = pd.DataFrame([m for p in pieces for m in p[3]])
    w = mom.n / mom.n.sum()
    comp = {k: float((w * mom[k]).sum()) for k in ("actor", "partner", "residual")}
    tot = sum(comp.values())
    sab, saa, sbb, df, cross, an, pn, n = S
    va, vb = max(saa / df - an / n, 1e-12), max(sbb / df - pn / n, 1e-12)
    return dict(dyadic_reciprocity=np.corrcoef(D[:, 0], D[:, 1])[0, 1], generalized_reciprocity=np.corrcoef(P[:, 0], P[:, 1])[0, 1],
                generalized_reciprocity_corrected=(sab / df - cross / n) / np.sqrt(va * vb),
                actor_share=comp["actor"] / tot, partner_share=comp["partner"] / tot, residual_share=comp["residual"] / tot,
                actor_var=comp["actor"], partner_var=comp["partner"], residual_var=comp["residual"], people=len(P), dyads=len(D))


def bootstrap(pieces, B, rng, keys=KEYS):
    G = len(pieces); out = {k: [] for k in keys}
    for _ in range(B):
        s = estimates([pieces[i] for i in rng.integers(0, G, G)])
        for k in keys:
            out[k].append(s[k])
    return {k: (np.quantile(v, 0.05), np.quantile(v, 0.95)) for k, v in out.items()}


def simulate(clean, truth, rng):
    """Liking from a social relations model with the given variances and correlations, on the real design: actor
    and partner effects with correlation `generalized`, pair effects with correlation `dyadic` between directions."""
    sim = clean[["event", "gender", "rater", "ratee", "like"]].copy()
    people = np.unique(np.r_[sim.rater, sim.ratee])
    ra, rp = truth["generalized"], truth["dyadic"]
    sa, sp, sg = np.sqrt(truth["actor_var"]), np.sqrt(truth["partner_var"]), np.sqrt(truth["residual_var"])
    z1, z2 = rng.normal(size=len(people)), rng.normal(size=len(people))
    actor = dict(zip(people, sa * z1)); partner = dict(zip(people, sp * (ra * z1 + np.sqrt(1 - ra ** 2) * z2)))
    keys = sorted({(min(r, c), max(r, c)) for r, c in zip(sim.rater, sim.ratee)})
    u, v = rng.normal(size=len(keys)), rng.normal(size=len(keys))
    rel = dict(zip(keys, zip(sg * u, sg * (rp * u + np.sqrt(1 - rp ** 2) * v))))   # (lower id rates higher, higher rates lower)
    gval = np.array([rel[(min(r, c), max(r, c))][0 if r < c else 1] for r, c in zip(sim.rater, sim.ratee)])
    sim["like"] = np.where(clean.like.notna(), 6 + sim.rater.map(actor).to_numpy() + sim.ratee.map(partner).to_numpy() + gval, np.nan)
    return sim


def simulate_binary(clean, shares, rp, rate, rng):
    """0/1 decisions from a latent social relations model: actor, partner and pair parts with the given variance
    shares (no generalized reciprocity), pair parts correlated rp between directions, thresholded so the share of
    yeses on the observed dates equals rate."""
    truth = dict(generalized=0.0, dyadic=rp, actor_var=shares["actor_share"], partner_var=shares["partner_share"],
                 residual_var=shares["residual_share"])
    sim = simulate(clean.rename(columns={"dec": "like_dec"}).assign(like=clean.dec), truth, rng)
    obs = sim.like.notna()
    cut = np.quantile(sim.like[obs], 1 - rate)
    return sim.assign(dec=np.where(obs, (sim.like > cut).astype(float), np.nan))


def main_dec(B, reps):
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    pieces = [event_pieces(ev, "dec") for _, ev in clean.groupby("event")]
    est = estimates(pieces)
    ci = bootstrap(pieces, B, np.random.default_rng(BOOT_SEED))
    E = pd.DataFrame([dict(quantity=k, estimate=est[k], ci90_low=ci[k][0], ci90_high=ci[k][1], people=est["people"], dyads=est["dyads"],
                           bootstrap=B, seed=BOOT_SEED, registered=k != "generalized_reciprocity_corrected", outcome="dec") for k in KEYS])
    E.to_csv(RES / "srm_dec_estimates.csv", index=False)
    shares = estimates([event_pieces(ev, "like") for _, ev in clean.groupby("event")])
    rate = clean.dec.mean()
    rng = np.random.default_rng(SIM_SEED); sims = []
    for _ in range(reps):
        sim = simulate_binary(clean, shares, 0.2, rate, rng)
        sp = [event_pieces(ev, "dec") for _, ev in sim.groupby("event")]
        s, c = estimates(sp), bootstrap(sp, 200, rng, keys=("dyadic_reciprocity",))
        sims.append(dict(estimate=s["dyadic_reciprocity"], lo=c["dyadic_reciprocity"][0], hi=c["dyadic_reciprocity"][1]))
    S = pd.DataFrame(sims); target = S.estimate.mean()
    check = pd.DataFrame([dict(quantity="dyadic_reciprocity (0/1 decisions)", latent_pair_correlation=0.2, mean_estimate=target,
                               sd=S.estimate.std(ddof=1), coverage_90_of_mean_estimate=((S.lo <= target) & (target <= S.hi)).mean(),
                               reps=reps, bootstrap=200, seed=SIM_SEED, yes_rate=rate)])
    check.to_csv(RES / "srm_dec_check.csv", index=False)
    r = E.set_index("quantity").loc["dyadic_reciprocity"]
    hyp = pd.DataFrame([dict(hypothesis="K3", statistic=f"dyadic reciprocity of the yes decisions {r.estimate:.3f} (90% interval "
                             f"{r.ci90_low:.3f} to {r.ci90_high:.3f}), {est['dyads']} dyads", supported=bool(r.ci90_low > 0),
                             se_check=f"simulation check ({reps} replicates, latent model thresholded at the yes rate): coverage "
                             f"{check.coverage_90_of_mean_estimate.iloc[0]:.0%} of the mean estimate", fold_scheme="none (all decisions)")])
    hyp.to_csv(RES / "srm_dec_hypotheses.csv", index=False)
    print(E.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(check.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


def main():
    args = sys.argv[1:]
    col = args.pop(0) if args and not args[0].isdigit() else "like"
    B = int(args[0]) if len(args) > 0 else 1000
    reps = int(args[1]) if len(args) > 1 else 200
    if col == "dec":
        return main_dec(B, reps)
    assert col == "like", "srm.py takes like or dec"
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    pieces = [event_pieces(ev) for _, ev in clean.groupby("event")]
    est = estimates(pieces)
    ci = bootstrap(pieces, B, np.random.default_rng(BOOT_SEED))
    E = pd.DataFrame([dict(quantity=k, estimate=est[k], ci90_low=ci[k][0], ci90_high=ci[k][1], people=est["people"], dyads=est["dyads"],
                           bootstrap=B, seed=BOOT_SEED, registered=k != "generalized_reciprocity_corrected") for k in KEYS])
    E.to_csv(RES / "srm_estimates.csv", index=False)
    truth = dict(generalized=float(np.clip(est["generalized_reciprocity_corrected"], -0.99, 0.99)), dyadic=est["dyadic_reciprocity"],
                 actor_var=est["actor_var"], partner_var=est["partner_var"], residual_var=est["residual_var"])
    target = {"dyadic_reciprocity": truth["dyadic"], "generalized_reciprocity": truth["generalized"], "generalized_reciprocity_corrected": truth["generalized"]}
    rng = np.random.default_rng(SIM_SEED); chk = []
    for _ in range(reps):
        sim = simulate(clean, truth, rng)
        sp = [event_pieces(ev) for _, ev in sim.groupby("event")]
        s, c = estimates(sp), bootstrap(sp, 200, rng, keys=tuple(target))
        chk.append({**{k: s[k] for k in target}, **{f"{k}_covered": c[k][0] <= t <= c[k][1] for k, t in target.items()}})
    C = pd.DataFrame(chk)
    check = pd.DataFrame([dict(quantity=k, truth=t, mean_estimate=C[k].mean(), bias=C[k].mean() - t, sd=C[k].std(ddof=1),
                               coverage_90=C[f"{k}_covered"].mean(), reps=reps, bootstrap=200, seed=SIM_SEED,
                               actor_var=truth["actor_var"], partner_var=truth["partner_var"], residual_var=truth["residual_var"])
                          for k, t in target.items()])
    check.to_csv(RES / "srm_check.csv", index=False)
    note = "; ".join(f"{r.quantity}: bias {r.bias:+.3f}, coverage {r.coverage_90:.0%}" for r in check.itertuples())
    iv = lambda k: f"{est[k]:.3f} (90% interval {ci[k][0]:.3f} to {ci[k][1]:.3f})"
    hyp = pd.DataFrame([
        dict(hypothesis="K1", statistic=f"dyadic reciprocity {iv('dyadic_reciprocity')}, {est['dyads']} dyads", supported=bool(est["dyadic_reciprocity"] >= 0.10)),
        dict(hypothesis="K2", statistic=f"generalized reciprocity {iv('generalized_reciprocity')}, {est['people']} people "
             f"(exploratory corrected estimate {iv('generalized_reciprocity_corrected')})", supported=bool(est["generalized_reciprocity"] < 0))
    ]).assign(se_check=f"simulation check ({reps} replicates, truth = corrected estimate): {note}", fold_scheme="none (all ratings)")
    hyp.to_csv(RES / "srm_hypotheses.csv", index=False)
    print(E.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(check.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
