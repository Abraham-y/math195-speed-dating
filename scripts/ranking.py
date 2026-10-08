"""Desirability ranking with honest uncertainty (PREREG.md, Part 2: B1, B2; proposals/05, Layer B).

Input : data/clean.csv
Output: results/ranking_effects.csv (one row per ratee: desirability, rank and simultaneous 90% rank range),
        results/ranking_blocks.csv (per wave and rater gender), results/ranking_variance.csv (B2),
        results/ranking_coverage.csv (simulation check of the rank ranges), results/ranking_dates_needed.csv
        (exploratory), results/ranking_hypotheses.csv, results/ranking_ranges.png
Seeds : bootstrap 20262106; coverage simulation 20262206 (+ block); dates-needed table 20262306 + 1.

Within each wave, women rate men and men rate women, so desirability is identified only within a wave and
rater gender (the comparison graph is complete within a wave and empty across waves). For each such block:
- Rasch model: P(yes from rater i to ratee j) = logistic(b0 + desirability_j - pickiness_i), fitted by penalized
  likelihood with a ridge of 0.5 on every person effect (modeling.fit_logit), so estimates stay finite when
  someone said yes to everyone or to no one.
- 1,000 bootstrap resamples of the block's dates (rows, with replacement) give the SE of every pairwise difference
  in desirability; the simultaneous 90% rank range of ratee j (Mogstad, Romano, Shaikh and Wilhelm 2024) counts
  the ratees significantly more and less desirable than j, using the 90% quantile of the bootstrap max-t
  statistic over all pairs. Rank 1 = most desirable; span = upper - lower + 1.
B1 uses the nine blocks with exactly 10 ratees. B2 compares the SDs of rater and ratee effects in a crossed
random-effects logistic regression for each rater gender (wave fixed effects; statsmodels BinomialBayesMixedGLM).
Run from the repository root: python scripts/ranking.py [bootstrap resamples]
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")
RIDGE, LEVEL = 0.5, 0.90
BOOT_SEED, COVER_SEED, NEED_SEED = 20262106, 20262206, 20262306 + 1


def rasch(rater, ratee, y, ridge=RIDGE, R=None, J=None):
    """Penalized Rasch fit. rater, ratee: integer codes 0..R-1, 0..J-1 (R and J default to the codes seen, and are
    passed for bootstrap resamples, where someone may be missing: their effect then stays at the ridge's 0).
    Returns (b0, desirability, pickiness); desirability is centred at 0 (the intercept absorbs the shift)."""
    R = rater.max() + 1 if R is None else R
    J = ratee.max() + 1 if J is None else J
    X = np.zeros((len(y), J + R))
    X[np.arange(len(y)), ratee] = 1.0
    X[np.arange(len(y)), J + rater] = -1.0
    w, _ = M.fit_logit(X, y, ridge=ridge, iters=200)
    d, p = w[1:J + 1], w[J + 1:]
    shift = d.mean()
    return w[0] + shift, d - shift, p - shift


def rank_ranges(est, boot, level=LEVEL):
    """Simultaneous rank ranges (max-t over pairwise differences). est: (J,), boot: (B, J). Rank 1 = largest."""
    J = len(est)
    diff = est[:, None] - est[None, :]                       # d_j - d_k
    bd = boot[:, :, None] - boot[:, None, :]
    se = bd.std(axis=0, ddof=1)
    iu = np.triu_indices(J, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.abs((bd - diff)[:, iu[0], iu[1]]) / se[iu]
    c = np.quantile(np.nanmax(t, axis=1), level)
    better = ((-diff) - c * se > 0)                          # k significantly above j: d_k - d_j > c se
    worse = (diff - c * se > 0)                              # k significantly below j
    np.fill_diagonal(better, False); np.fill_diagonal(worse, False)
    lower = 1 + better.sum(axis=1)
    upper = J - worse.sum(axis=1)
    return lower, upper, c


def block_ranking(rater, ratee, y, B, rng):
    b0, d, p = rasch(rater, ratee, y)
    n = len(y); boot = np.empty((B, len(d)))
    for b in range(B):
        i = rng.integers(0, n, n)
        boot[b] = rasch(rater[i], ratee[i], y[i], R=len(p), J=len(d))[1]
    lo, up, c = rank_ranges(d, boot)
    rank = 1 + (d[None, :] > d[:, None]).sum(axis=1)
    return b0, d, p, rank, lo, up, c


def blocks(clean):
    for (w, g), b in clean.groupby(["event", "gender"]):
        ri, rater_ids = pd.factorize(b.rater, sort=True)
        ci, ratee_ids = pd.factorize(b.ratee, sort=True)
        yield int(w), int(g), ri, ci, b.dec.to_numpy(float), np.asarray(rater_ids), np.asarray(ratee_ids)


def coverage_check(clean, reps, B, rng):
    """Simulate from each 10-ratee block's fitted model; share of replicates whose ranges hold every true rank."""
    out = []
    for w, g, ri, ci, y, _, ratee_ids in blocks(clean):
        if len(ratee_ids) != 10:
            continue
        b0, d, p = rasch(ri, ci, y)
        true_rank = 1 + (d[None, :] > d[:, None]).sum(axis=1)
        hits, spans = 0, []
        for _ in range(reps):
            ys = (rng.random(len(y)) < 1 / (1 + np.exp(-(b0 + d[ci] - p[ri])))).astype(float)
            _, _, _, _, lo, up, _ = block_ranking(ri, ci, ys, B, rng)
            hits += bool(np.all((lo <= true_rank) & (true_rank <= up))); spans.append(np.median(up - lo + 1))
        out.append(dict(event=w, rater_gender="women" if g == 0 else "men", reps=reps, bootstrap=B,
                        simultaneous_coverage=hits / reps, median_span=np.median(spans)))
    return pd.DataFrame(out)


def variance_components(clean):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    rows = []
    for g, name in ((0, "women"), (1, "men")):
        d = clean[clean.gender == g][["dec", "event", "rater", "ratee"]].copy()
        m = BinomialBayesMixedGLM.from_formula("dec ~ C(event)", {"rater": "0 + C(rater)", "ratee": "0 + C(ratee)"}, d)
        r = m.fit_vb()
        sd = dict(zip(["rater", "ratee"], np.exp(np.asarray(r.vcp_mean))))
        rows.append(dict(rater_gender=name, decisions=len(d), pickiness_sd_raters=sd["rater"], desirability_sd_ratees=sd["ratee"],
                         ratio=sd["rater"] / sd["ratee"], method="BinomialBayesMixedGLM (variational Bayes), wave fixed effects"))
    return pd.DataFrame(rows)


def dates_needed(sd_rater, sd_ratee, rng, sizes=(10, 20, 40, 80, 160), sims=40, B=200, J=10):
    """Exploratory: median rank-range span for J ratees when each is rated by m raters (effects drawn from the SDs)."""
    out = []
    for m_ in sizes:
        spans = []
        for _ in range(sims):
            d, p = rng.normal(0, sd_ratee, J), rng.normal(0, sd_rater, m_)
            ri, ci = np.repeat(np.arange(m_), J), np.tile(np.arange(J), m_)
            y = (rng.random(len(ri)) < 1 / (1 + np.exp(-(d[ci] - p[ri])))).astype(float)
            _, _, _, _, lo, up, _ = block_ranking(ri, ci, y, B, rng)
            spans.append(np.median(up - lo + 1))
        out.append(dict(raters_per_ratee=m_, ratees=J, median_span=np.median(spans), sims=sims, bootstrap=B))
    return pd.DataFrame(out)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    rng = np.random.default_rng(BOOT_SEED)
    eff, blk = [], []
    for w, g, ri, ci, y, rater_ids, ratee_ids in blocks(clean):
        b0, d, p, rank, lo, up, c = block_ranking(ri, ci, y, B, rng)
        rg = "women" if g == 0 else "men"
        eff += [dict(event=w, rater_gender=rg, ratee=int(ratee_ids[j]), desirability=d[j], rank=int(rank[j]), rank_lower=int(lo[j]),
                     rank_upper=int(up[j]), span=int(up[j] - lo[j] + 1), ratees_in_block=len(ratee_ids), yes_rate_received=y[ci == j].mean(),
                     bootstrap=B, seed=BOOT_SEED) for j in range(len(ratee_ids))]
        blk.append(dict(event=w, rater_gender=rg, ratees=len(ratee_ids), raters=len(rater_ids), dates=len(y), critical_value=c,
                        median_span=float(np.median(up - lo + 1)), desirability_sd=d.std(ddof=1), pickiness_sd=p.std(ddof=1)))
    E, Bk = pd.DataFrame(eff), pd.DataFrame(blk)
    E.to_csv(RES / "ranking_effects.csv", index=False); Bk.to_csv(RES / "ranking_blocks.csv", index=False)
    ten = E[E.ratees_in_block == 10]
    b1 = float(np.median(ten.span))
    V = variance_components(clean); V.to_csv(RES / "ranking_variance.csv", index=False)
    cov = coverage_check(clean, reps=100, B=200, rng=np.random.default_rng(COVER_SEED)); cov.to_csv(RES / "ranking_coverage.csv", index=False)
    need = dates_needed(V.pickiness_sd_raters.mean(), V.desirability_sd_ratees.mean(), np.random.default_rng(NEED_SEED))
    need.to_csv(RES / "ranking_dates_needed.csv", index=False)
    hyp = pd.DataFrame([
        dict(hypothesis="B1", version="the nine wave-and-gender blocks with 10 ratees", statistic=f"median span {b1:.1f} positions over {len(ten)} ratees "
             f"(simultaneous 90% rank ranges; check: coverage {cov.simultaneous_coverage.min():.2f} to {cov.simultaneous_coverage.max():.2f})",
             supported=bool(b1 >= 4)),
        dict(hypothesis="B2", version="both rater genders", statistic="; ".join(f"{r.rater_gender} rating: pickiness SD {r.pickiness_sd_raters:.3f}, "
             f"desirability SD {r.desirability_sd_ratees:.3f}" for r in V.itertuples()),
             supported=bool((V.pickiness_sd_raters > V.desirability_sd_ratees).all()))]).assign(seed=BOOT_SEED, fold_scheme="none (all dates)")
    hyp.to_csv(RES / "ranking_hypotheses.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5), layout="constrained", sharex=True)
    for ax, ((w, rg), g) in zip(axes.flat, ten.groupby(["event", "rater_gender"])):
        g = g.sort_values("rank")
        yy = np.arange(len(g))[::-1]
        ax.hlines(yy, g.rank_lower, g.rank_upper, color="#2a78d6", lw=2.5, alpha=0.5)
        ax.plot(g["rank"], yy, "o", color="#2a78d6", ms=4)
        ax.set_title(f"wave {w}, rated by {rg}", fontsize=8, loc="left"); ax.set_yticks([]); ax.set_xticks(range(1, 11))
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(f"Simultaneous 90% rank ranges, blocks of 10 ratees (median span {b1:.0f} of 10 positions)", x=0.01, ha="left", fontsize=10)
    fig.supxlabel("Desirability rank (1 = most desirable)", fontsize=9)
    fig.savefig(RES / "ranking_ranges.png", dpi=200); plt.close(fig)

    print(f"{len(Bk)} blocks; bootstrap {B} (seed {BOOT_SEED}); median span in the 10-ratee blocks {b1:.1f}")
    print(Bk.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(V.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(cov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(need.to_string(index=False))
    print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
