"""N16 to N20 (PREREG.md, Part 9): the cross-validated AUC when rows are clustered.

Input : none (synthetic data, Part 6's generator extended)
Output: results/sim_cv_auc_rows.csv (one row per data set), results/sim_cv_auc_summary.csv (per design and G),
        results/sim_cv_auc_nb.csv (the Nadeau-Bengio runs), results/sim_cv_auc_kfold.csv (the mixing bias at K = 2 and 10),
        results/sim_cv_auc_components.csv (population constants,
        computed before any CV run with `--components`), results/sim_cv_auc_hypotheses.csv
Seeds : components 20265006 + design; coverage cells 20265106 + 10 x design + cell; other fold counts 20265206 + 10 x
        design + K index; Nadeau-Bengio 20265306 + 10 x design + cell.

Data: G clusters of m = 40 rows, x = su * u_g + v_i (u_g, v_i independent N(0, I_3)), and
y ~ Bernoulli(logistic(-1.5 + x'(1, 0.5, 0) + sa * a_g)), a_g ~ N(0, 1) (`linear`). Two misspecified versions for the
Nadeau-Bengio test: `hinge` adds 1.5 max(x_3, 0) - 1 to the logit, `slope` makes the x_1 slope 1 + b_g with
b_g ~ N(0, 1). The model is a logistic regression on x; its score is the linear predictor.

The target, AUC_new, is the full-data model's AUC on 20,000 new clusters, pooled (a match from one event against a
non-match from another, which is how an app would rank pairs across events). Its Monte Carlo variance (Obuchowski on
the new sample) is recorded. Event 5-fold CV (clusters dealt into folds) gives four estimates:
- pooled: one AUC over all held-out scores, mixing the five fold models (what the project reports);
- fold-averaged: the mean of the five within-fold AUCs;
- between-cluster within-fold (BCWF): in each fold, the AUC over match / non-match pairs from different clusters,
  averaged over folds; and its pooled form (BCWF-pooled: those pairs' concordances summed over folds, over their count);
- pooled between-cluster: the pooled AUC without same-cluster pairs (to separate the two biases).
Intervals (90%): each estimate plus or minus t(G - 1) times the Obuchowski clustered SE of the pooled scores; for the
pooled estimate also normal quantiles and the wild cluster bootstrap-t on the AUC's per-row linearization.
Nadeau-Bengio (`--nb`): R = 12 grouped partitions (J = 60) of within-fold AUCs, against (J+K-1)/(J-1) times the
Obuchowski variance. Proposition 9 (proposals/06, section 6f) predicts the ratio Var(a - beta/4)/Var(a) from the
population constants of `--components`.
Mixing at other fold counts (`--kfold`): designs 2 and 4 at G = 80 with K = 2 and 10, the statistic
G x (pooled between-cluster AUC - BCWF-pooled), against Proposition 9 (iii)'s C_mix + c2 / (K - 1).
Run from the repository root: python scripts/sim_cv_auc.py [--components | --kfold | --nb] [replicates] [workers]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib, time, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "scripts")
import modeling as M
import sim_cv_sizes as SZ

RES = pathlib.Path("results")
M_ROWS, K, R_NB, NEW_CLUSTERS, B_WILD = 40, 5, 12, 20000, 999
SEED_COMP, SEED_COV, SEED_K, SEED_NB = 20265006, 20265106, 20265206, 20265306
K_DESIGNS, K_VALUES, K_G = (2, 4), (2, 10), 80
DESIGNS = [dict(kind="linear", sa=sa, su=su) for sa in (0.0, 1.0, 2.0) for su in (0.0, 1.0)]        # 0 to 5
NB_DESIGNS = [dict(kind=k, sa=1.0, su=1.0) for k in ("linear", "hinge", "slope")]                   # 6 to 8
GS, NB_GS = (20, 40, 80, 160), (80, 160)


def draw(G, rng, kind, sa, su):
    g = np.repeat(np.arange(G), M_ROWS)
    X = su * rng.normal(size=(G, 3))[g] + rng.normal(size=(G * M_ROWS, 3))
    eta = -1.5 + X[:, 0] + 0.5 * X[:, 1] + sa * rng.normal(size=G)[g]
    if kind == "hinge":
        eta = eta + 1.5 * np.maximum(X[:, 2], 0) - 1.0
    elif kind == "slope":
        eta = eta + rng.normal(size=G)[g] * X[:, 0]
    return X, (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float), g


# ---------------------------------------------------------------- AUC pieces

def within_cluster_counts(score, y, g):
    """Per cluster: concordant match / non-match pairs inside the cluster (ties half) and the number of such pairs."""
    d = pd.DataFrame(dict(s=np.asarray(score, float), y=np.asarray(y), g=np.asarray(g)))
    d["r"] = d.groupby("g").s.rank()
    agg = d.assign(rp=d.r * d.y).groupby("g").agg(m1=("y", "sum"), m=("y", "size"), rp=("rp", "sum"))
    m1 = agg.m1.to_numpy(float); m0 = agg.m.to_numpy(float) - m1
    return agg.rp.to_numpy() - m1 * (m1 + 1) / 2, m1 * m0


def between_cluster_counts(score, y, g):
    """Concordant match / non-match pairs from different clusters (ties half), and the number of such pairs."""
    y = np.asarray(y); n1 = (y == 1).sum(); n0 = len(y) - n1
    conc_w, pairs_w = within_cluster_counts(score, y, g)
    return M.auc(score, y) * n1 * n0 - conc_w.sum(), n1 * n0 - pairs_w.sum()


def between_cluster_auc(score, y, g):
    """AUC over match / non-match pairs from different clusters."""
    conc, pairs = between_cluster_counts(score, y, g)
    return conc / pairs


def auc_linearization(score, y):
    """Per-row pseudo-values u with mean(u) = 0 whose cluster-robust variance of the mean equals the Obuchowski
    variance (modeling.auc_clustered_var): (v10 - A) n / M for matches, (v01 - A) n / N for non-matches."""
    y = np.asarray(y); v10, v01 = M.placement_values(score, y); A = v10.mean()
    u = np.empty(len(y)); n = len(y)
    u[y == 1] = (v10 - A) * n / len(v10); u[y == 0] = (v01 - A) * n / len(v01)
    return u


def grouped(G, rng, k=K):
    to = np.empty(G, int); to[rng.permutation(G)] = np.arange(G) % k
    return to


def cv_scores(X, y, g, fold_of_cluster):
    f = fold_of_cluster[g]; s = np.empty(len(y))
    for k in range(fold_of_cluster.max() + 1):
        te = f == k
        w, _ = M.fit_logit(X[~te], y[~te])
        s[te] = w[0] + X[te] @ w[1:]
    return s, f


def estimates(s, y, g, f):
    folds = np.unique(f)
    within = [M.auc(s[f == k], y[f == k]) for k in folds]
    bc = np.array([between_cluster_counts(s[f == k], y[f == k], g[f == k]) for k in folds])
    return dict(pooled=M.auc(s, y), fold_averaged=float(np.mean(within)), bcwf=float(np.mean(bc[:, 0] / bc[:, 1])),
                bcwf_pooled=float(bc[:, 0].sum() / bc[:, 1].sum()), pooled_between=between_cluster_auc(s, y, g),
                v_obu=M.auc_clustered_var(s, y, g))


# ---------------------------------------------------------------- population constants (before any CV run)

def pair_density_term(s, y, F1, F0, bins=400):
    """E[delta(s_i - s_j) F1_i . F0_j] over a match i and a non-match j drawn independently (rows of F1 for matches,
    of F0 for non-matches), by score bins at quantiles."""
    y = np.asarray(y); edges = np.quantile(s, np.linspace(0.0005, 0.9995, bins + 1))
    b = np.clip(np.searchsorted(edges, s) - 1, 0, bins - 1); i1, i0 = y == 1, y == 0
    by_bin = lambda F, m: np.stack([np.bincount(b[m], F[m, c], minlength=bins) for c in range(F.shape[1])], 1)
    return float(np.sum(np.sum(by_bin(F1, i1) * by_bin(F0, i0), 1) / np.diff(edges)) / (i1.sum() * i0.sum()))


def pair_density_slope(s, y, F1, F0, W, bins=300, kmax=6):
    """g'(0) for g(t) = E[F1_i' W F0_j delta(s_i - s_j - t)], by uniform score bins and a least-squares slope over
    shifts of -kmax to kmax bins."""
    y = np.asarray(y); i1, i0 = y == 1, y == 0
    lo, hi = np.quantile(s, [0.001, 0.999]); wb = (hi - lo) / bins
    ub = np.floor((s - lo) / wb).astype(int); ok = (ub >= 0) & (ub < bins)
    A1 = np.stack([np.bincount(ub[ok & i1], F1[ok & i1, c], minlength=bins) for c in range(F1.shape[1])], 1) / i1.sum()
    A0 = np.stack([np.bincount(ub[ok & i0], F0[ok & i0, c], minlength=bins) for c in range(F0.shape[1])], 1) / i0.sum()
    ks = np.arange(-kmax, kmax + 1)
    gk = [np.sum((A1[max(k, 0):bins + min(k, 0)] @ W) * A0[max(-k, 0):bins - max(k, 0)]) / wb for k in ks]
    return float(np.polyfit(ks * wb, gk, 1)[0])


def components(d):
    """theta*, AUC*, the within- and between-cluster AUCs, kappa = E[m1 m0] / (m^2 pi (1 - pi)), the AUC gradient at
    theta*, and Var(a), Var(beta), Cov(a, beta) of Proposition 9, from large samples of design d."""
    des = (DESIGNS + NB_DESIGNS)[d]; rng = np.random.default_rng(SEED_COMP + d)
    X, y, g = draw(40000, rng, **des); w, _ = M.fit_logit(X, y)                       # 1.6M rows for theta*
    Xb, yb, gb = draw(40000, rng, **des); sb = Xb @ w[1:]
    A_all = M.auc(sb, yb); A_b = between_cluster_auc(sb, yb, gb)
    conc, pairs = within_cluster_counts(sb, yb, gb); A_w = conc.sum() / pairs.sum()
    pi = yb.mean(); kappa = pairs.mean() / (M_ROWS ** 2 * pi * (1 - pi))
    grad = np.zeros(4); h = 0.05                                                       # common random numbers
    for j in range(1, 4):
        e = np.zeros(4); e[j] = h
        grad[j] = (M.auc(Xb @ (w + e)[1:], yb) - M.auc(Xb @ (w - e)[1:], yb)) / (2 * h)
    Xc, yc, gc = draw(20000, rng, **des); s = Xc @ w[1:]
    neg, pos = np.sort(sb[yb == 0]), np.sort(sb[yb == 1])
    F0 = (np.searchsorted(neg, s, "left") + np.searchsorted(neg, s, "right")) / 2 / len(neg)
    S1 = 1 - (np.searchsorted(pos, s, "left") + np.searchsorted(pos, s, "right")) / 2 / len(pos)
    a = pd.Series(np.where(yc == 1, (F0 - A_all) / (M_ROWS * pi), (S1 - A_all) / (M_ROWS * (1 - pi)))).groupby(gc).sum().to_numpy()
    Xt = np.c_[np.ones(len(yc)), Xc]; p = 1 / (1 + np.exp(-Xt @ w))
    H = (Xt * (p * (1 - p))[:, None]).T @ Xt / len(yc)
    psi = pd.DataFrame(Xt * (yc - p)[:, None]).groupby(gc).sum().to_numpy() / M_ROWS @ np.linalg.inv(H).T
    beta = psi @ grad
    va, vb, cab = a.var(), beta.var(), np.cov(a, beta)[0, 1]
    # Second order (proposals/06, section 6f). Mixing: in a cross-fold pair each side's model was fitted on the other
    # side's cluster, which shifts the score difference by [x_i' H^-1 s_h(j) - x_j' H^-1 s_h(i)] / n' (s_h the
    # cluster's score sum x(y - p) at theta*); cross-fold pairs are (K-1)/K of the between-cluster pairs, so the
    # leading bias is E[delta(D) (...)] / n = C_mix / G, free of K. delta(D) is estimated with score bins.
    Hinv = np.linalg.inv(H)
    sh = pd.DataFrame(Xt * (yc - p)[:, None]).groupby(gc).transform("sum").to_numpy() @ Hinv.T     # H^-1 s_h(row)
    c_mix = (pair_density_term(s, yc, Xt, sh) - pair_density_term(s, yc, sh, Xt)) / M_ROWS
    # The next term: the two fold models of a cross-fold pair differ by noise of variance 2K Var(psi) / ((K-1)^2 G),
    # correlated with the shared part; together they add E[delta'(D) x_i' Var(psi) x_j] / ((K-1) G) =
    # -g'(0) / ((K-1) G), g(t) = E[x_i' Var(psi) x_j delta(D - t)].
    c2 = -pair_density_slope(s, yc, Xt, Xt, np.cov(psi.T))
    # Training size: every CV model sees (K-1)/K of the clusters, so E[A(theta_k)] - E[A(theta_all)] =
    # tr(H_A Var(psi)) / (2 G (K-1)) = c_T / G, with H_A the AUC Hessian in the slopes (common random numbers).
    hh = 0.1; HA = np.zeros((3, 3)); A0 = M.auc(Xb @ w[1:], yb)
    for j, k in itertools.product(range(3), range(3)):
        if k < j:
            continue
        ej, ek = np.eye(3)[j] * hh, np.eye(3)[k] * hh
        if j == k:
            HA[j, j] = (M.auc(Xb @ (w[1:] + ej), yb) - 2 * A0 + M.auc(Xb @ (w[1:] - ej), yb)) / hh ** 2
        else:
            HA[j, k] = HA[k, j] = (M.auc(Xb @ (w[1:] + ej + ek), yb) - M.auc(Xb @ (w[1:] + ej - ek), yb)
                                   - M.auc(Xb @ (w[1:] - ej + ek), yb) + M.auc(Xb @ (w[1:] - ej - ek), yb)) / (4 * hh ** 2)
    c_train = float(np.trace(HA @ np.cov(psi[:, 1:].T)) / (2 * (K - 1)))
    c_within = kappa * (A_w - A_b)
    return dict(design=d, **des, theta=" ".join(f"{v:.4f}" for v in w), auc_star=A_all, auc_between=A_b, auc_within=A_w,
                kappa=kappa, match_rate=pi, grad_norm=float(np.linalg.norm(grad[1:])), grad_dot_theta=float(grad[1:] @ w[1:]),
                var_a=va, var_beta=vb, cov_a_beta=cab, predicted_nb_ratio_K5=1 - 2 * cab / (4 * va) + vb / (16 * va),
                c_mix=c_mix, c_mix2=c2, c_train=c_train, c_within=c_within,
                pred_G_mix_K2=c_mix + c2, pred_G_mix_K5=c_mix + c2 / 4, pred_G_mix_K10=c_mix + c2 / 9,
                pred_G_bias_bcwf=c_train, pred_G_bias_pooled_between=c_train + c_mix + c2 / 4,
                pred_G_bias_pooled=c_train + c_mix + c2 / 4 + c_within, pred_G_bias_fold_averaged=c_train + K * c_within,
                seed=SEED_COMP + d)


# ---------------------------------------------------------------- coverage cells

def one(G, rng, des):
    X, y, g = draw(G, rng, **des)
    s, f = cv_scores(X, y, g, grouped(G, rng))
    out = dict(G=G, **estimates(s, y, g, f))
    out["wild_q"] = SZ.wild_cluster_t_quantile(auc_linearization(s, y), g, B_WILD, rng)
    w, _ = M.fit_logit(X, y)
    Xn, yn, gn = draw(NEW_CLUSTERS, rng, **des); sn = Xn @ w[1:]
    out.update(auc_new=M.auc(sn, yn), v_mc=M.auc_clustered_var(sn, yn, gn), match_rate=y.mean())
    return out


def run_cell(args):
    d, G, reps = args
    seed = SEED_COV + 10 * d + GS.index(G); rng = np.random.default_rng(seed); t0 = time.time()
    rows = [dict(one(G, rng, DESIGNS[d]), design=d, **DESIGNS[d], rep=r, seed=seed) for r in range(reps)]
    print(f"design {d} {DESIGNS[d]} G {G}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


ESTIMATES = ("pooled", "fold_averaged", "bcwf", "bcwf_pooled", "pooled_between")


def summarize(D):
    out = []
    for (d, G), x in D.groupby(["design", "G"]):
        q = stats.t.ppf(0.95, G - 1); se = np.sqrt(x.v_obu); vmc = x.v_mc.mean()
        row = dict(design=d, kind=x.kind.iloc[0], sa=x.sa.iloc[0], su=x.su.iloc[0], G=G, reps=len(x), mean_auc_new=x.auc_new.mean(),
                   mean_se_obu=se.mean(), mc_share=vmc / (x.pooled - x.auc_new).var(ddof=1), seed=int(x.seed.iloc[0]))
        for e in ESTIMATES:
            diff = x[e] - x.auc_new
            row[f"bias_{e}"] = diff.mean(); row[f"bias_se_{e}"] = diff.std(ddof=1) / np.sqrt(len(x))
            row[f"cov_t_{e}"] = float(np.mean(np.abs(diff) <= q * se))
            row[f"ratio_corrected_{e}"] = (diff.var(ddof=1) - vmc) / x.v_obu.mean()
        row["cov_z_pooled"] = float(np.mean(np.abs(x.pooled - x.auc_new) <= 1.645 * se))
        row["cov_wild_pooled"] = float(np.mean(np.abs(x.pooled - x.auc_new) <= x.wild_q * se))
        row["G_x_bias_pooled"] = G * row["bias_pooled"]
        mix = G * (x.pooled_between - x.bcwf_pooled)
        row["G_x_mix"] = mix.mean(); row["G_x_mix_se"] = mix.std(ddof=1) / np.sqrt(len(x))
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- the mixing bias at other fold counts

def run_k(args):
    """G x (pooled between-cluster AUC - BCWF-pooled) with K folds; no target is needed for this difference."""
    d, k, reps = args
    seed = SEED_K + 10 * d + K_VALUES.index(k); rng = np.random.default_rng(seed); t0 = time.time(); rows = []
    for r in range(reps):
        X, y, g = draw(K_G, rng, **DESIGNS[d])
        s, f = cv_scores(X, y, g, grouped(K_G, rng, k))
        e = estimates(s, y, g, f)
        rows.append(dict(design=d, **DESIGNS[d], K=k, G=K_G, rep=r, seed=seed, G_x_mix=K_G * (e["pooled_between"] - e["bcwf_pooled"]),
                         pooled=e["pooled"], bcwf_pooled=e["bcwf_pooled"], pooled_between=e["pooled_between"]))
    print(f"K {k} design {d}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


# ---------------------------------------------------------------- Nadeau-Bengio on fold AUCs

def one_nb(G, rng, des):
    X, y, g = draw(G, rng, **des)
    s, f = cv_scores(X, y, g, grouped(G, rng))
    v = M.auc_clustered_var(s, y, g); fa, fb = [], []
    for _ in range(R_NB):
        s2, f2 = cv_scores(X, y, g, grouped(G, rng))
        for k in range(K):
            te = f2 == k
            fa.append(M.auc(s2[te], y[te])); fb.append(between_cluster_auc(s2[te], y[te], g[te]))
    J = len(fa); factor = (J + K - 1) / (J - 1)
    nb = lambda v_: (1 / J + 1 / (K - 1)) * np.var(v_, ddof=1)
    return dict(G=G, v_obu=v, ratio_within=nb(fa) / (factor * v), ratio_bcwf=nb(fb) / (factor * v))


def run_nb(args):
    d, G, reps = args
    seed = SEED_NB + 10 * d + NB_GS.index(G); rng = np.random.default_rng(seed); t0 = time.time()
    rows = [dict(one_nb(G, rng, NB_DESIGNS[d - len(DESIGNS)]), design=d, **NB_DESIGNS[d - len(DESIGNS)], rep=r, seed=seed) for r in range(reps)]
    print(f"NB design {d} G {G}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(args[1]) if len(args) > 1 else 4
    if "--components" in sys.argv:
        with Pool(workers) as pool:
            C = pd.DataFrame(pool.map(components, range(len(DESIGNS) + len(NB_DESIGNS))))
        C.to_csv(RES / "sim_cv_auc_components.csv", index=False)
        with pd.option_context("display.width", 250):
            print(C.drop(columns=["theta", "seed"]).round(5).to_string(index=False))
        return
    if "--kfold" in sys.argv:
        reps = int(args[0]) if args else 400
        with Pool(workers) as pool:
            parts = pool.map(run_k, [(d, k, reps) for d in K_DESIGNS for k in K_VALUES], chunksize=1)
        D = pd.DataFrame([r for p in parts for r in p]); D.to_csv(RES / "sim_cv_auc_kfold_rows.csv", index=False)
        C = pd.read_csv(RES / "sim_cv_auc_components.csv").set_index("design")
        S = D.groupby(["design", "K"]).agg(reps=("rep", "size"), G_x_mix=("G_x_mix", "mean"),
                                           G_x_mix_se=("G_x_mix", lambda v: v.std(ddof=1) / np.sqrt(len(v))), seed=("seed", "first")).reset_index()
        S["predicted"] = [C.loc[d, f"pred_G_mix_K{k}"] for d, k in zip(S.design, S.K)]
        S["relative_error"] = S.G_x_mix / S.predicted - 1
        S.to_csv(RES / "sim_cv_auc_kfold.csv", index=False)
        print(S.round(4).to_string(index=False))
        return
    if "--nb" in sys.argv:
        reps = int(args[0]) if args else 200
        tasks = [(d, G, reps) for d in range(len(DESIGNS), len(DESIGNS) + len(NB_DESIGNS)) for G in sorted(NB_GS, reverse=True)]
        with Pool(workers) as pool:
            parts = pool.map(run_nb, tasks, chunksize=1)
        D = pd.DataFrame([r for p in parts for r in p]); D.to_csv(RES / "sim_cv_auc_nb_rows.csv", index=False)
        S = D.groupby(["design", "kind", "G"]).agg(reps=("rep", "size"), ratio_within=("ratio_within", "mean"),
                                                   ratio_within_se=("ratio_within", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                                   ratio_bcwf=("ratio_bcwf", "mean"), seed=("seed", "first")).reset_index()
        C = pd.read_csv(RES / "sim_cv_auc_components.csv").set_index("design")
        S["predicted"] = S.design.map(C.predicted_nb_ratio_K5)
        S.to_csv(RES / "sim_cv_auc_nb.csv", index=False)
        print(S.round(4).to_string(index=False))
        return
    reps = int(args[0]) if args else 400
    tasks = [(d, G, reps) for d in range(len(DESIGNS)) for G in sorted(GS, reverse=True)]
    with Pool(workers) as pool:
        parts = pool.map(run_cell, tasks, chunksize=1)
    D = pd.DataFrame([r for p in parts for r in p]).sort_values(["design", "G", "rep"])
    D.to_csv(RES / "sim_cv_auc_rows.csv", index=False)
    S = summarize(D); S.to_csv(RES / "sim_cv_auc_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        cols = ["design", "sa", "su", "G"] + [f"bias_{e}" for e in ESTIMATES] + ["cov_z_pooled", "cov_t_pooled", "cov_wild_pooled", "cov_t_bcwf", "mc_share"]
        print(S[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
