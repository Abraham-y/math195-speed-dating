"""Multiple-testing control on the reported coefficient tables (PREREG.md, Part 5; exploratory, no hypothesis).

Input : results/part1_models_coefficients.csv (headline full logistic models, `pre` and `pre+date`),
        results/part1_models_stated_vs_revealed.csv (the yes models, P5)
Output: results/multiple_testing_summary.csv (one row per family), results/multiple_testing_terms.csv (one row per test)
No random operations.

p-values: two-sided normal, p = 2 (1 - Phi(|z|)), with z = coefficient / cluster-robust SE by event. The match-model
families take every estimable term other than the intercept from the headline table (spec missing=median_flags),
including the identified sum of the separated missing-code terms; the two separated terms themselves have no
estimate and are left out. The yes-model family has the six standardized ratings for each gender (z = std_coef /
se_cluster_event). A fourth family pools all of them.
Per family: m; the number with p < 0.05; Benjamini-Hochberg rejections at q = 0.10 (step-up: reject the i smallest
for the largest i with p_(i) <= i q / m); Benjamini-Yekutieli rejections at q = 0.10 (the same with q / sum_{j<=m} 1/j,
valid under any dependence); Storey's pi0 with lambda = 0.5 (#{p > 0.5} / (m (1 - 0.5)), capped at 1); and the expected
number of false findings among p < 0.05, pi0 x m x 0.05.
Caution: the coefficient SEs are too narrow by the check in scripts/sim_coef_cluster.py (see se_check), so these
p-values are too small and the counts of discoveries are upper bounds.
Run from the repository root: python scripts/multiple_testing.py
"""
import sys, pathlib
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, "scripts")
import modeling as M

RES = pathlib.Path("results")
Q, LAMBDA, ALPHA = 0.10, 0.5, 0.05


def two_sided_p(z):
    return 2 * norm.sf(np.abs(np.asarray(z, float)))


def bh_adjusted(p):
    """Benjamini-Hochberg adjusted p-values: min over j >= i of m p_(j) / j, capped at 1, in the input order."""
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p, kind="stable")
    adj = np.minimum.accumulate((m * p[o] / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(adj, 1.0)
    return out


def by_adjusted(p):
    """Benjamini-Yekutieli adjusted p-values: the BH ones times c(m) = sum_{j<=m} 1/j, capped at 1."""
    m = len(p)
    return np.minimum(bh_adjusted(p) * np.sum(1 / np.arange(1, m + 1)), 1.0)


def bh_reject(p, q):
    """Step-up rejections: the hypotheses with the i smallest p for the largest i with p_(i) <= i q / m."""
    return bh_adjusted(p) <= q


def by_reject(p, q):
    return by_adjusted(p) <= q


def storey_pi0(p, lam=LAMBDA):
    """Storey's estimate of the share of true nulls: #{p > lam} / (m (1 - lam)). Returns (capped at 1, raw)."""
    p = np.asarray(p, float)
    raw = np.mean(p > lam) / (1 - lam)
    return min(1.0, raw), raw


def families():
    """{family: tests}, with the short label each family's terms carry in the pooled family."""
    c = pd.read_csv(RES / "part1_models_coefficients.csv")
    c = c[c.spec == M.HEADLINE.label]
    out, short = {}, {}
    for fs in ("pre", "pre+date"):
        d = c[(c.feature_set == fs) & (c.term != "const") & c.estimable.astype(bool) & np.isfinite(c.z)]
        fam = f"match model coefficients, `{fs}` (headline)"
        out[fam] = pd.DataFrame({"term": d.term, "estimate": d.coef, "se": d.se_cluster_event, "z": d.z,
                                 "source": "results/part1_models_coefficients.csv", "se_check": d.se_check})
        short[fam] = f"match `{fs}`"
    s = pd.read_csv(RES / "part1_models_stated_vs_revealed.csv")
    fam = "yes models, six ratings by gender (P5)"
    out[fam] = pd.DataFrame({"term": s.rater_gender + ": " + s.attribute, "estimate": s.std_coef, "se": s.se_cluster_event,
                             "z": s.std_coef / s.se_cluster_event, "source": "results/part1_models_stated_vs_revealed.csv",
                             "se_check": s.se_check})
    short[fam] = "yes"
    out["all of the above, pooled"] = pd.concat([v.assign(term=short[k] + " | " + v.term) for k, v in out.items()], ignore_index=True)
    return out


def main():
    terms, summ = [], []
    for fam, d in families().items():
        d = d.reset_index(drop=True).copy()
        d["p"] = two_sided_p(d.z)
        m = len(d)
        d["rank"] = d.p.rank(method="first").astype(int)
        d["bh_threshold"] = d["rank"] * Q / m
        d["bh_adjusted_p"], d["by_adjusted_p"] = bh_adjusted(d.p), by_adjusted(d.p)
        d["p_below_0.05"] = d.p < ALPHA
        d["reject_bh_q0.10"], d["reject_by_q0.10"] = bh_reject(d.p, Q), by_reject(d.p, Q)
        pi0, raw = storey_pi0(d.p)
        n05 = int(d["p_below_0.05"].sum())
        exp_false = pi0 * m * ALPHA
        survivors = d[d["reject_bh_q0.10"]].sort_values("p")
        summ.append(dict(family=fam, m=m, p_below_005=n05, bh_rejections_q010=int(d["reject_bh_q0.10"].sum()),
                         by_rejections_q010=int(d["reject_by_q0.10"].sum()), storey_pi0=pi0, storey_pi0_raw=raw,
                         p_above_lambda=int((d.p > LAMBDA).sum()), expected_false_among_p_below_005=exp_false,
                         expected_false_share_among_p_below_005=exp_false / n05 if n05 else np.nan,
                         bh_survivors="; ".join(survivors.term), q=Q, storey_lambda=LAMBDA,
                         source=";".join(d.source.unique()), se_check=" | ".join(d.se_check.unique())))
        terms.append(d.assign(family=fam).sort_values("p"))
    T = pd.concat(terms, ignore_index=True)
    T = T[["family", "term", "estimate", "se", "z", "p", "rank", "bh_threshold", "bh_adjusted_p", "by_adjusted_p", "p_below_0.05",
           "reject_bh_q0.10", "reject_by_q0.10", "source", "se_check"]]
    S = pd.DataFrame(summ).assign(fold_scheme="none (all dyads / all dates)", random_operations="none")
    T.assign(fold_scheme="none (all dyads / all dates)").to_csv(RES / "multiple_testing_terms.csv", index=False)
    S.to_csv(RES / "multiple_testing_summary.csv", index=False)
    print(S[["family", "m", "p_below_005", "bh_rejections_q010", "by_rejections_q010", "storey_pi0", "storey_pi0_raw",
             "expected_false_among_p_below_005"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for r in S.itertuples():
        print(f"\n{r.family}: BH survivors at q = {Q}: {r.bh_survivors or 'none'}")
    print("\nterms with p < 0.05:")
    print(T[T["p_below_0.05"] & ~T.family.str.startswith("all")][["family", "term", "estimate", "se", "z", "p", "bh_adjusted_p",
                                                                    "reject_bh_q0.10", "reject_by_q0.10"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
