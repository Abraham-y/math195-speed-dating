"""N22 to N25 (PREREG.md, Part 11; to be registered): grouped CV with unequal cluster sizes (proposals/06, section 6g,
Proposition 11).

Input : data/pairs.csv (the 21 real event sizes; the A2 truth model and the real events)
Output: results/sim_cv_unequal_predictions.csv (`--predictions`: population constants, P1 and P2, computed before any
        CV run, plus exploratory predictions), results/sim_cv_unequal_rows.csv (synthetic designs, one row per data set),
        results/sim_cv_unequal_a2_rows.csv (A2 designs, one row per data set), results/sim_cv_unequal_summary.csv
        (hypothesis inputs and exploratory quantities, flagged), results/sim_cv_unequal_hypotheses.csv.
        `--quick` writes the same names with the suffix _quick.
Seeds : population constants 20267006, P2 surrogate 20267016 (the exploratory predictions continue the same stream after
        the registered ones), the population sample that scores Err_new 20267026; synthetic design d (0: S21f, 1: S84r),
        chunk c: 20267106 + 100 d + c; A2 design d (0, 1), chunk c: 20267306 + 100 d + c, its intercept calibrated once
        with 20267306 + 100 d + 99. `--quick` uses scratch seeds: each of these minus 19269306 (997700, 997710,
        997720, 997800 + 100 d + c, 998000 + 100 d + c).

Proposition 11: with unequal cluster sizes, the pooled grouped CV estimate minus the full model's row-weighted
new-cluster error is (1/n) sum_h a_h to first order (a_h the cluster's loss-residual total), the unbalanced
cluster-robust variance estimates its variance, and E[NB] / ((J+K-1)/(J-1) E[V_CR]) tends to Var(a - beta/(K-1)) / Var(a)
with cluster totals (P1; a limit in expectation, not in probability at fixed J). P2 is the linear representation of
Proposition 11 (f) (scripts/prop11_linear.py), which keeps the actual sizes and partitions; it is a first-order
heuristic that omits the pooled estimate's partition-to-partition variation, the test cluster's gradient deviation,
curvature and flip noise (proposals/06, section 6g).

Synthetic designs (Part 6's generator: x = u_g + v_i, y ~ Bernoulli(logistic(-1.5 + x'(1, 0.5, 0) + a_g)), a logistic
regression on x), with cluster sizes from the real events:
- S21f: the 21 real event sizes, as they are; 2,000 data sets;
- S84r: 84 sizes drawn with replacement from the real ones; 1,000 data sets.
Per data set: one grouped 5-fold run (pooled CV, unbalanced cluster-robust variance V_CR); the Nadeau-Bengio variance at
K = 3, 5 and 10 from R = 20, 12 and 6 grouped partitions (J = 60) with three fold values: each fold's mean over its own
rows ("own", as the project's scripts), its residual total about the partition's pooled CV divided by n/K ("res"), and
that residual value plus the partition's pooled CV ("shift"; own - shift is the normalization part and shift - res the
pooled estimate's partition-to-partition part); for 0-1 loss at the training positive rate and for log-loss. Err_new is
the full model's exact row-weighted population loss (a fixed sample of 400,000 rows from the row marginal, P(y = 1 | x)
by quadrature), so it carries no Monte Carlo noise across data sets. The plug-in P2 (exploratory in these designs) uses
the data set's in-sample residual totals and beta_hat_h = d_hat' H^-1 S_h + d_t (Y_h - m_h t), d_hat from the
kernel-smoothed 0-1 loss (Silverman bandwidth).
A2 designs: the A2 truth (scripts/sim_cv_matched.py) with 21 events redrawn with replacement (fresh people), person SD
0.5 and event SD 0, and person SD 1.0 and event SD 0.4; 500 data sets each; event 5-fold CV, K = 5, R = 12 (J = 60),
V_CR from one 5-fold run; Err_new on 2,000 new events with its Monte Carlo variance; the per-data-set plug-in P2 with
beta_hat from the explicit influence and from the jackknife, -(n - m_h)[A_s(fit without event h) - A_s(full fit)], A_s
the smoothed 0-1 loss.
Hypotheses (predicted values and bands are the constants below; the predictions are copied from `--predictions`):
- N22 (discriminates P2 from P1): S21f, 0-1 loss, mean own-row ratio. At K = 10 within BAND_N22_K10 of P2; at K = 5
  within BAND_N22_K5 of P2 and below the midpoint of P1 and P2. Both parts are needed.
- N23 (approach to the first-order limit; not discriminating): S84r, 0-1 loss, the mean own-row ratio within BAND_N23
  of P1 at K = 5 and 10.
- N24 (calibration of the real-data plug-in): in both A2 designs, mean(observed own-row ratio - its plug-in P2,
  jackknife beta) within BAND_N24, and the mean paired own-row minus residual-total ratio at least MIN_N24_GAP.
- N25 (log-loss, S21f): the residual-total ratio within BAND_N25 of 1 at K = 5 and 10, and the own-row ratio within
  BAND_N25 of P2 at K = 5 and 10.
Exploratory (computed and saved, not in the hypotheses table): K = 3; the paired own - res and the shift decomposition
in the synthetic designs; Var(CV - Err_new) / mean V_CR and G Var(CV - Err_new); the per-data-set plug-ins in the
synthetic designs; the second-order log-loss surrogate (`--predictions`).
Run from the repository root: python scripts/sim_cv_unequal.py [--predictions | --verdicts] [--quick] [workers]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib, time, functools
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modeling as M
import prop11_linear as P

RES = pathlib.Path("results")
SEED_CONST, SEED_SURR, SEED_POP, SEED_SYN, SEED_A2 = 20267006, 20267016, 20267026, 20267106, 20267306
QUICK_SHIFT = 19269306                                   # quick mode: every seed minus this (scratch seeds 997700 and up)
K_CV, CHUNK = 5, 50
KS_R = {3: 20, 5: 12, 10: 6}                             # K -> R (J = 60); K = 3 is exploratory
KS_TEST = (5, 10)                                        # fold counts in the hypotheses
SYN = {"S21f": dict(G=21, resample=False, reps=2000), "S84r": dict(G=84, resample=True, reps=1000)}
A2D = {"A2 person 0.5, event 0": dict(person=0.5, event=0.0, reps=500), "A2 person 1.0, event 0.4": dict(person=1.0, event=0.4, reps=500)}
QUICK_REPS = {"S21f": 2, "S84r": 2, "A2 person 0.5, event 0": 2, "A2 person 1.0, event 0.4": 2}
BETA, B0 = np.array([1.0, 0.5, 0.0]), -1.5
NEW_EVENTS_A2, POP_ROWS = 2000, 400_000
# ---- Hypotheses N22 to N25. Predicted values copied from the `--predictions` run of 2026-11-03 (seeds 20267006 and
# 20267016, 40,000 draws): P1 1.2597 and 1.1072; P2 for S21f, own rows, 0-1 loss 1.2174 and 0.9982, log-loss 1.0646 and 0.9835.
PRED_P1_01 = {5: 1.260, 10: 1.107}                       # P1, 0-1 loss, any design with the real size law
PRED_P2_01_S21F = {5: 1.217, 10: 0.998}                  # P2, S21f, 0-1 loss, own-row fold means
PRED_P2_LOG_S21F = {5: 1.065, 10: 0.984}                 # P2, S21f, log-loss, own-row fold means
BAND_N22_K10, BAND_N22_K5 = 0.04, 0.03                   # N22: |obs - P2| at K = 10 and at K = 5
N22_K5_MIDPOINT = (PRED_P1_01[5] + PRED_P2_01_S21F[5]) / 2   # N22: obs at K = 5 must also be below this (1.2385)
BAND_N23 = 0.03                                          # N23: |obs - P1| at K = 5 and 10 (S84r)
BAND_N24, MIN_N24_GAP = 0.08, 0.10                       # N24: |mean(own - plug-in P2)|; mean(own - res) at least this
BAND_N25 = 0.03                                          # N25: |res - 1| and |own - P2| at K = 5 and 10 (S21f, log-loss)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(40)


def real_sizes():
    s = pd.read_csv("data/pairs.csv").groupby("event").size().to_numpy()
    assert len(s) == 21 and s.sum() == 4184
    return np.sort(s)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def pi_of(X):
    """P(y = 1 | x) in the synthetic design: E over a ~ N(0, 1) of logistic(-1.5 + x'beta + a), by Gauss-Hermite."""
    eta = B0 + X @ BETA
    return np.concatenate([(sigmoid(e[:, None] + np.sqrt(2) * _GH_X[None, :]) @ _GH_W) / np.sqrt(np.pi)
                           for e in np.array_split(eta, max(1, len(eta) // 250_000))])


def draw(sizes, rng):
    G = len(sizes); g = np.repeat(np.arange(G), sizes)
    X = rng.normal(size=(G, 3))[g] + rng.normal(size=(len(g), 3))
    eta = B0 + X @ BETA + rng.normal(size=G)[g]
    return X, (rng.random(len(eta)) < sigmoid(eta)).astype(float), g


def losses(w, rate, X, y):
    """Per-row 0-1 loss (predict 1 when the probability exceeds the training positive rate) and log-loss."""
    p = np.clip(M.predict_logit(w, X), 1e-6, 1 - 1e-6)
    return ((p > rate).astype(float) != y).astype(float), -(y * np.log(p) + (1 - y) * np.log(1 - p))


class Population:
    """A fixed sample of x from the row marginal N(0, 2 I) with pi(x): the row-weighted population losses A(w, t)."""
    def __init__(self, n, rng):
        self.X = rng.normal(size=(n, 3)) * np.sqrt(2); self.pi = pi_of(self.X)

    def A(self, w, t):
        p = np.clip(M.predict_logit(w, self.X), 1e-6, 1 - 1e-6); pi = self.pi
        return np.mean(pi + (1 - 2 * pi) * (p > t)), np.mean(-(pi * np.log(p) + (1 - pi) * np.log(1 - p)))


def fold_losses(X, y, fold):
    l01, lll = np.empty(len(y)), np.empty(len(y))
    for k in np.unique(fold):
        te = fold == k
        w, _ = M.fit_logit(X[~te], y[~te])
        l01[te], lll[te] = losses(w, y[~te].mean(), X[te], y[te])
    return l01, lll


def fold_stats(l, f, K):
    """For one partition's per-row CV losses l and fold labels f, three fold values: the own-row mean; the residual total
    about the partition's pooled CV divided by n/K ("res"); and res plus the pooled CV ("shift")."""
    n, c = len(l), l.mean()
    res = [(l[f == k] - c).sum() * K / n for k in range(K)]
    return [l[f == k].mean() for k in range(K)], res, [c + r for r in res]


def nb_ratio(fold_values, K, vcr):
    """Nadeau-Bengio variance (1/J + 1/(K-1)) S^2 over the J fold values, divided by (J+K-1)/(J-1) V_CR."""
    J = len(fold_values)
    return (1 / J + 1 / (K - 1)) * np.var(fold_values, ddof=1) / ((J + K - 1) / (J - 1) * vcr)


def smoothed_gradient(p, t, y, Xb):
    """Gradient of the kernel-smoothed empirical 0-1 loss mean(y + (1 - 2y) Phi(z/h)), z = logit p - logit t, with
    respect to the coefficients and the threshold (Gaussian kernel, Silverman bandwidth)."""
    n = len(y); z = np.log(p / (1 - p)) - np.log(t / (1 - t))
    h = 0.9 * min(z.std(), (np.quantile(z, 0.75) - np.quantile(z, 0.25)) / 1.34) * n ** -0.2
    k = norm.pdf(z / h) / h
    return ((1 - 2 * y) * k) @ Xb / n, -np.sum((1 - 2 * y) * k) / n / (t * (1 - t)), h


def plugin_totals(X, y, g):
    """One data set's plug-in totals for the full fit (threshold at the training rate): in-sample loss-residual totals
    (0-1 and log) and beta_hat_h = d_hat' H^-1 S_h + d_t (Y_h - m_h t) for the 0-1 loss."""
    w, _ = M.fit_logit(X, y); n = len(y); t = y.mean(); G = g.max() + 1
    Xb = np.hstack([np.ones((n, 1)), X]); p = np.clip(M.predict_logit(w, X), 1e-9, 1 - 1e-9)
    l01 = ((p > t).astype(float) != y).astype(float); lll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    tot = lambda v: np.bincount(g, weights=v, minlength=G)
    m = np.bincount(g, minlength=G).astype(float)
    H = (Xb * (p * (1 - p))[:, None]).T @ Xb / n
    S = np.stack([tot(Xb[:, j] * (y - p)) for j in range(Xb.shape[1])], 1)
    dw, dt, h = smoothed_gradient(p, t, y, Xb)
    return dict(m=m, a01=tot(l01) - m * l01.mean(), all=tot(lll) - m * lll.mean(), beta=S @ np.linalg.solve(H, dw) + dt * (tot(y) - m * t),
                w=w, t=t, h=h)


def plugin_p2(m, a, b, rng, ks_r, D=100):
    """Partition average of the linear representation with fixed totals (a, b): E V_NB / ((J+K-1)/(J-1) E V_CR)."""
    G = len(m)
    st = P.linear_stats(m, np.broadcast_to(a, (D, G)).copy(), np.broadcast_to(b, (D, G)).copy(), rng, K_CV, ks_r)
    return {(v, K): st[f"vnb_{v}_K{K}"].mean() / (st[f"fac_K{K}"][0] * st["vcr"].mean()) for K in ks_r for v in ("own", "res")}


# ---------------------------------------------------------------- population constants and predictions

def population_constants(rng, quick=False):
    """theta*, t*, mu, d = grad A01 at (theta*, t*) (exact on the boundary, with a finite-difference check), H, and the
    per-row and same-cluster covariances of (0-1 loss, b, log-loss, score x~(y - p*)) at theta*, with
    b = d_w' H^-1 x~ (y - p*) + d_t (y - t*). Same-cluster covariances use clusters of MR rows and conditional expectations
    given (x, a_g), ANOVA-corrected. The score columns (used only by the exploratory second-order surrogate) are computed
    from the same draws, so the first three columns do not depend on them."""
    NP, NB, NFD, NC, MR, CH = (300_000, 200_000, 300_000, 4_000, 100, 1_000) if quick else (3_000_000, 2_000_000, 2_000_000, 40_000, 250, 2_000)
    Xp = rng.normal(size=(NP, 3)) * np.sqrt(2); pip = pi_of(Xp)
    w, _ = M.fit_logit(Xp, pip, ridge=0.0); t = pip.mean()
    Xb = np.hstack([np.ones((NP, 1)), Xp]); p = sigmoid(Xb @ w)
    H = (Xb * (p * (1 - p))[:, None]).T @ Xb / NP
    mu01 = np.mean(pip + (1 - 2 * pip) * (p > t)); mull = np.mean(-(pip * np.log(p) + (1 - pip) * np.log(1 - p)))
    wv = w[1:]; s0 = np.log(t / (1 - t)) - w[0]; nw2 = wv @ wv
    fz0 = norm.pdf(s0 / np.sqrt(2 * nw2)) / np.sqrt(2 * nw2)
    xi = rng.normal(size=(NB, 3)) * np.sqrt(2)
    xb = s0 * wv / nw2 + xi - np.outer(xi @ wv, wv) / nw2               # x given w0 + w'x = logit t
    gb = 1 - 2 * pi_of(xb)
    d_w = fz0 * (gb @ np.hstack([np.ones((NB, 1)), xb])) / NB; d_t = -fz0 * gb.mean() / (t * (1 - t))
    pop = Population(NFD, rng); eps = 0.02
    fd = [(pop.A(w + eps * np.eye(4)[j], t)[0] - pop.A(w - eps * np.eye(4)[j], t)[0]) / (2 * eps) for j in range(4)]
    fd_t = (pop.A(w, t + 0.002)[0] - pop.A(w, t - 0.002)[0]) / 0.004
    gvec = np.linalg.solve(H, d_w); q_ = 7
    cms, within, rowc, Vsq, Vsum, nrow = [], np.zeros((q_, q_)), np.zeros((q_, q_)), np.zeros((q_, q_)), np.zeros(q_), 0
    for _ in range(0, NC, CH):
        u = rng.normal(size=(CH, 3)); a = rng.normal(size=CH)
        x = (u[:, None, :] + rng.normal(size=(CH, MR, 3))).reshape(-1, 3)
        q = sigmoid(B0 + x @ BETA + np.repeat(a, MR))
        xt = np.hstack([np.ones((len(x), 1)), x]); ps = sigmoid(xt @ w); pred1 = ps > t
        lam = xt @ gvec + d_t; lg = np.log(ps / (1 - ps)); sgn = np.where(pred1, -1.0, 1.0); qq = q * (1 - q)
        V = np.hstack([np.stack([np.where(pred1, 1 - q, q), xt @ gvec * (q - ps) + d_t * (q - t), -(q * np.log(ps) + (1 - q) * np.log(1 - ps))], 1),
                       xt * (q - ps)[:, None]])
        coef = np.hstack([np.stack([sgn, lam, -lg], 1), xt])            # each quantity = coef * y + const, given (x, a)
        rowc += (coef * qq[:, None]).T @ coef
        Vr = V.reshape(CH, MR, q_); cm = Vr.mean(1); cms.append(cm); dev = Vr - cm[:, None, :]
        within += np.einsum("cri,crj->ij", dev, dev); Vsq += V.T @ V; Vsum += V.sum(0); nrow += len(V)
    cm = np.concatenate(cms); mean = Vsum / nrow
    cc = np.cov(cm.T) - within / (NC * (MR - 1)) / MR
    row = Vsq / nrow - np.outer(mean, mean) + rowc / nrow
    return dict(theta=w, t=t, mu01=mu01, mull=mull, d_w=d_w, d_t=d_t, d_w_fd=np.array(fd), d_t_fd=fd_t, H=H, row=row, cc=cc)


def predictions(quick=False):
    shift = QUICK_SHIFT if quick else 0
    t0 = time.time()
    C = population_constants(np.random.default_rng(SEED_CONST - shift), quick)
    rng = np.random.default_rng(SEED_SURR - shift); D = 4000 if quick else 40000
    sizes = real_sizes(); mtilde = (sizes ** 2).sum() / sizes.sum(); mbar = sizes.mean()
    row3, cc3 = C["row"][:3, :3], C["cc"][:3, :3]
    comp = {"0-1": (row3[:2, :2], cc3[:2, :2]),
            "log": (np.array([[row3[2, 2], 0], [0, 1e-12]]), np.array([[cc3[2, 2], 0], [0, 1e-13]]))}
    draws = {"S21f": lambda d, r: np.broadcast_to(sizes, (d, 21)).copy(), "S84r": lambda d, r: r.choice(sizes, (d, 84))}
    rows = []
    # registered predictions: K = 5 and 10 (this part of the stream is what the constants above were copied from)
    for des, sz in draws.items():
        for loss, (row, cc) in comp.items():
            sm = P.summarize(P.surrogate(sz, row, cc, rng, D=D))
            G = 21 if des == "S21f" else 84
            for K in P.KS_R:
                p1, V = P.p1(row, cc, mtilde, K)
                rows.append(dict(role="registered", method="P2 linear (Gaussian totals)", design=des, loss=loss, K=K, G=G, mbar=mbar, mtilde=mtilde,
                                 P1=p1 if loss == "0-1" else 1.0, P2_own=sm[f"meanratio_own_K{K}"], P2_res=sm[f"meanratio_res_K{K}"],
                                 P2_own_ratio_of_means=sm[f"ratioofmeans_own_K{K}"], P2_res_ratio_of_means=sm[f"ratioofmeans_res_K{K}"],
                                 P2_var_ratio=sm["var_ratio"], P1_G_var_diff=V[0, 0] / mbar, P2_G_var_diff=G * sm["var_diff"], draws=D))
    # exploratory, continuing the same stream: K = 3, and the second-order log-loss surrogate at K = 3, 5, 10
    for des, sz in draws.items():
        G = 21 if des == "S21f" else 84
        for loss, (row, cc) in comp.items():
            sm = P.summarize(P.surrogate(sz, row, cc, rng, D=D, ks_r={3: 20}), ks=(3,))
            p1, V = P.p1(row, cc, mtilde, 3)
            rows.append(dict(role="exploratory", method="P2 linear (Gaussian totals)", design=des, loss=loss, K=3, G=G, mbar=mbar, mtilde=mtilde,
                             P1=p1 if loss == "0-1" else 1.0, P2_own=sm["meanratio_own_K3"], P2_res=sm["meanratio_res_K3"],
                             P2_own_ratio_of_means=sm["ratioofmeans_own_K3"], P2_res_ratio_of_means=sm["ratioofmeans_res_K3"],
                             P2_var_ratio=sm["var_ratio"], P1_G_var_diff=V[0, 0] / mbar, P2_G_var_diff=G * sm["var_diff"], draws=D))
        idx = [2, 3, 4, 5, 6]                                           # log-loss and the score
        D2 = 1000 if quick else 10000
        st = P.second_order_surrogate(sz, C["row"][np.ix_(idx, idx)], C["cc"][np.ix_(idx, idx)], np.linalg.inv(C["H"]), rng, D=D2, ks_r=KS_R)
        diff = st["cv"] - st["err"]
        for K in KS_R:
            rows.append(dict(role="exploratory", method="second-order log-loss surrogate (Gaussian totals and scores)", design=des, loss="log", K=K, G=G,
                             mbar=mbar, mtilde=mtilde, P1=1.0, P2_own=st[f"ratio_own_K{K}"].mean(),
                             P2_own_ratio_of_means=st[f"vnb_own_K{K}"].mean() / (st[f"fac_K{K}"][0] * st["vcr"].mean()),
                             P2_var_ratio=diff.var(ddof=1) / st["vcr"].mean(), P2_G_var_diff=G * diff.var(ddof=1), draws=D2))
    Pdf = pd.DataFrame(rows).assign(seed_constants=SEED_CONST - shift, seed_surrogate=SEED_SURR - shift)
    consts = {"theta_star": C["theta"], "t_star": C["t"], "mu_01": C["mu01"], "mu_log": C["mull"], "d_w": C["d_w"], "d_w_finite_difference": C["d_w_fd"],
              "d_t": C["d_t"], "d_t_finite_difference": C["d_t_fd"], "row_cov (0-1, b, log)": row3, "same_cluster_cov (0-1, b, log)": cc3}
    for k, v in consts.items():
        Pdf[k] = np.array2string(np.atleast_1d(np.round(v, 6)).ravel(), separator=", ", max_line_width=10 ** 6)
    out = RES / f"sim_cv_unequal_predictions{'_quick' if quick else ''}.csv"
    Pdf.to_csv(out, index=False)
    for k, v in consts.items():
        print(f"{k}: {np.round(v, 5).tolist()}")
    cols = ["role", "design", "loss", "K", "P1", "P2_own", "P2_res", "P2_own_ratio_of_means", "P2_res_ratio_of_means", "P2_var_ratio", "P1_G_var_diff", "P2_G_var_diff"]
    with pd.option_context("display.width", 250):
        for meth, d in Pdf.groupby("method", sort=False):
            print(meth); print(d[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")


# ---------------------------------------------------------------- simulation

def one_syn(design, rng, pop, sizes):
    spec = SYN[design]; G = spec["G"]
    m = rng.choice(sizes, G) if spec["resample"] else sizes.copy()
    X, y, g = draw(m, rng); n = len(y)
    l01, lll = fold_losses(X, y, P.deal(1, G, K_CV, rng)[0][g])
    out = dict(design=design, G=G, n=n, mtilde=float((m ** 2).sum() / n), cv01=l01.mean(), cvll=lll.mean(),
               vcr01=M.cluster_robust_var_of_mean(l01, g), vcrll=M.cluster_robust_var_of_mean(lll, g))
    for K, R in KS_R.items():
        vals = {(v, ls): [] for v in ("own", "res", "shift") for ls in ("01", "ll")}
        for _ in range(R):
            f = P.deal(1, G, K, rng)[0][g]
            for ls, lv in zip(("01", "ll"), fold_losses(X, y, f)):
                for v, fv in zip(("own", "res", "shift"), fold_stats(lv, f, K)):
                    vals[(v, ls)] += fv
        for (v, ls), fv in vals.items():
            out[f"ratio_{v}_{ls}_K{K}"] = nb_ratio(fv, K, out[f"vcr{ls}"])
    w, _ = M.fit_logit(X, y)
    out["en01"], out["enll"] = pop.A(w, y.mean())
    T = plugin_totals(X, y, g); sub = np.random.default_rng(rng.integers(2 ** 32))
    for K in KS_R:
        out[f"p1plug_01_K{K}"] = np.sum((T["a01"] - T["beta"] / (K - 1)) ** 2) / np.sum(T["a01"] ** 2)
    for ls, a, b in (("01", T["a01"], T["beta"]), ("ll", T["all"], 0 * T["all"])):
        for (v, K), r in plugin_p2(m.astype(float), a, b, sub, KS_R).items():
            out[f"p2plug_{v}_{ls}_K{K}"] = r
    return out


@functools.lru_cache(maxsize=None)
def a2_truth():
    import sim_cv_repair as SR
    return SR.truth_model()


@functools.lru_cache(maxsize=None)
def a2_intercept(person, event, seed):
    import sim_cv_matched as A2
    pairs, X0, f0, m0, e0, beta_all, _ = a2_truth()
    sd = {"woman": person, "man": person, "event": event}
    return A2.calibrate_intercept(beta_all[0], beta_all[1:], sd, X0, f0, m0, e0, pairs.match.mean(), np.random.default_rng(seed))


def jackknife_beta(X, y, g):
    """beta_hat_h = -(n - m_h) [A_s(fit without cluster h) - A_s(full fit)], A_s the smoothed empirical 0-1 loss on all rows."""
    w, _ = M.fit_logit(X, y); t = y.mean(); n = len(y)
    p = np.clip(M.predict_logit(w, X), 1e-9, 1 - 1e-9); z = np.log(p / (1 - p)) - np.log(t / (1 - t))
    h = 0.9 * min(z.std(), (np.quantile(z, 0.75) - np.quantile(z, 0.25)) / 1.34) * n ** -0.2
    As = lambda pp, tt: np.mean(y + (1 - 2 * y) * norm.cdf((np.log(pp / (1 - pp)) - np.log(tt / (1 - tt))) / h))
    A0 = As(p, t); G = g.max() + 1; m = np.bincount(g, minlength=G); out = np.empty(G)
    for k in range(G):
        tr = g != k
        wk, _ = M.fit_logit(X[tr], y[tr], w0=w)
        out[k] = -(n - m[k]) * (As(np.clip(M.predict_logit(wk, X), 1e-9, 1 - 1e-9), y[tr].mean()) - A0)
    return out


def one_a2(design, rng, intercept):
    import sim_cv_matched as A2, sim_cv_repair as SR, sim_cv_sizes as SZ
    spec = A2D[design]; sd = {"woman": spec["person"], "man": spec["person"], "event": spec["event"]}
    pairs, X0, f0, m0, e0, beta_all, _ = a2_truth(); K, R = K_CV, KS_R[K_CV]
    X, f, m, e = SR.redraw_events(X0, f0, m0, e0, 21, rng)
    y = A2.draw(intercept, beta_all[1:], sd, X, f, m, e, rng)
    Xn, fn, mn, en = SZ.new_events(pairs, X0, f0, m0, e0, NEW_EVENTS_A2, rng)
    ln = A2.loss(X, y, Xn, A2.draw(intercept, beta_all[1:], sd, Xn, fn, mn, en, rng))
    G = e.max() + 1; sizes = np.bincount(e).astype(float)
    eg = A2.cv(X, y, P.deal(1, G, K, rng)[0][e]); vcr = M.cluster_robust_var_of_mean(eg, e)
    vals = {"own": [], "res": [], "shift": []}
    for _ in range(R):
        fo = P.deal(1, G, K, rng)[0][e]
        for v, fv in zip(("own", "res", "shift"), fold_stats(A2.cv(X, y, fo), fo, K)):
            vals[v] += fv
    out = dict(design=design, person_sd=spec["person"], event_sd=spec["event"], intercept=intercept, n=len(y),
               mtilde=float((sizes ** 2).sum() / len(y)), cv=eg.mean(), vcr=vcr, err_new=ln.mean(), v_mc=M.cluster_robust_var_of_mean(ln, en),
               **{f"ratio_{v}": nb_ratio(fv, K, vcr) for v, fv in vals.items()})
    T = plugin_totals(X, y, e); bj = jackknife_beta(X, y, e); sub = np.random.default_rng(rng.integers(2 ** 32))
    for tag, b in (("infl", T["beta"]), ("jk", bj)):
        out[f"p1plug_{tag}"] = np.sum((T["a01"] - b / (K - 1)) ** 2) / np.sum(T["a01"] ** 2)
        pp = plugin_p2(sizes, T["a01"], b, sub, {K: R})
        out[f"p2plug_own_{tag}"], out[f"p2plug_res_{tag}"] = pp[("own", K)], pp[("res", K)]
    out["corr_beta_infl_jk"] = np.corrcoef(T["beta"], bj)[0, 1]
    return out


def run_chunk(task):
    kind, design, chunk, reps, shift = task
    t0 = time.time()
    if kind == "syn":
        d = list(SYN).index(design); seed = SEED_SYN + 100 * d + chunk - shift
        rng = np.random.default_rng(seed); pop = Population(POP_ROWS, np.random.default_rng(SEED_POP - shift)); sizes = real_sizes()
        rows = [dict(one_syn(design, rng, pop, sizes), rep=chunk * CHUNK + r, seed=seed) for r in range(reps)]
    else:
        d = list(A2D).index(design); base = SEED_A2 + 100 * d - shift; seed = base + chunk
        icpt = a2_intercept(A2D[design]["person"], A2D[design]["event"], base + 99)
        rng = np.random.default_rng(seed)
        rows = [dict(one_a2(design, rng, icpt), rep=chunk * CHUNK + r, seed=seed) for r in range(reps)]
    print(f"{design} chunk {chunk}: {reps} data sets in {(time.time() - t0) / 60:.1f} min", flush=True)
    return kind, rows


# ---------------------------------------------------------------- summaries and verdicts

def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def summarize(S, A):
    """Per design, loss and K: the mean ratios (own, res, shift) with SEs, the paired differences, the plug-ins and the
    variance quantities. `role` says whether a row feeds a hypothesis or is exploratory."""
    rows = []
    for des, d in S.groupby("design"):
        for ls in ("01", "ll"):
            diff = d[f"cv{ls}"] - d[f"en{ls}"]
            for K in KS_R:
                r = dict(design=des, loss="0-1" if ls == "01" else "log", K=K, role="hypothesis input" if K in KS_TEST and (des == "S21f" or ls == "01")
                         else "exploratory", reps=len(d), seed=f"{int(d.seed.min())} to {int(d.seed.max())}")
                for v in ("own", "res", "shift"):
                    r[f"ratio_{v}"], r[f"ratio_{v}_se"] = mean_se(d[f"ratio_{v}_{ls}_K{K}"])
                r["ratio_of_means_own"] = float((d[f"ratio_own_{ls}_K{K}"] * d[f"vcr{ls}"]).mean() / d[f"vcr{ls}"].mean())
                r["own_minus_res"], r["own_minus_res_se"] = mean_se(d[f"ratio_own_{ls}_K{K}"] - d[f"ratio_res_{ls}_K{K}"])
                r["partition_part"], r["partition_part_se"] = mean_se(d[f"ratio_shift_{ls}_K{K}"] - d[f"ratio_res_{ls}_K{K}"])
                r["normalization_part"], r["normalization_part_se"] = mean_se(d[f"ratio_own_{ls}_K{K}"] - d[f"ratio_shift_{ls}_K{K}"])
                for v in ("own", "res"):
                    r[f"plugin_p2_{v}"] = float(d[f"p2plug_{v}_{ls}_K{K}"].mean())
                r.update(var_ratio=float(diff.var(ddof=1) / d[f"vcr{ls}"].mean()) if len(d) > 1 else np.nan,
                         G_var_diff=float(d.G.iloc[0] * diff.var(ddof=1)) if len(d) > 1 else np.nan, G_mean_vcr=float(d.G.iloc[0] * d[f"vcr{ls}"].mean()))
                rows.append(r)
    for des, d in A.groupby("design"):
        r = dict(design=des, loss="0-1", K=K_CV, role="hypothesis input", reps=len(d), seed=f"{int(d.seed.min())} to {int(d.seed.max())}")
        for v in ("own", "res", "shift"):
            r[f"ratio_{v}"], r[f"ratio_{v}_se"] = mean_se(d[f"ratio_{v}"])
        r["own_minus_res"], r["own_minus_res_se"] = mean_se(d.ratio_own - d.ratio_res)
        r["partition_part"], r["partition_part_se"] = mean_se(d.ratio_shift - d.ratio_res)
        r["normalization_part"], r["normalization_part_se"] = mean_se(d.ratio_own - d.ratio_shift)
        for v in ("own", "res"):
            r[f"plugin_p2_{v}"] = float(d[f"p2plug_{v}_jk"].mean())
        r["own_minus_plugin_p2"], r["own_minus_plugin_p2_se"] = mean_se(d.ratio_own - d.p2plug_own_jk)
        if len(d) > 2:
            vb = d.err_new.var(ddof=1) - d.v_mc.mean(); cab = np.cov(d.cv, d.err_new)[0, 1] - vb; va = d.cv.var(ddof=1) - vb - 2 * cab
            r["P1_moment_decomposition"] = float((va - 2 * cab / (K_CV - 1) + vb / (K_CV - 1) ** 2) / va)
            r["var_ratio"] = float(((d.cv - d.err_new).var(ddof=1) - d.v_mc.mean()) / d.vcr.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def verdicts(Sm):
    s = Sm.set_index(["design", "loss", "K"])
    g = lambda des, ls, K, c: float(s.loc[(des, ls, K), c])
    se = lambda des, ls, K: f" (SE {g(des, ls, K, 'ratio_own_se'):.3f})"
    seeds = lambda des: str(s.loc[(des, "0-1", K_CV), "seed"])
    o22 = {K: g("S21f", "0-1", K, "ratio_own") for K in KS_TEST}
    n22 = (abs(o22[10] - PRED_P2_01_S21F[10]) <= BAND_N22_K10 and abs(o22[5] - PRED_P2_01_S21F[5]) <= BAND_N22_K5 and o22[5] < N22_K5_MIDPOINT)
    o23 = {K: g("S84r", "0-1", K, "ratio_own") for K in KS_TEST}
    o24 = {des: (g(des, "0-1", K_CV, "own_minus_plugin_p2"), g(des, "0-1", K_CV, "own_minus_res")) for des in A2D}
    r25 = {K: g("S21f", "log", K, "ratio_res") for K in KS_TEST}; o25 = {K: g("S21f", "log", K, "ratio_own") for K in KS_TEST}
    rows = [
        dict(hypothesis="N22", statistic=(
            f"S21f (21 real sizes), 0-1 loss, mean own-row NB ratio over data sets: K = 10 {o22[10]:.3f}{se('S21f', '0-1', 10)} against "
            f"P2 {PRED_P2_01_S21F[10]:.3f} (band {BAND_N22_K10}; P1 {PRED_P1_01[10]:.3f}); K = 5 {o22[5]:.3f}{se('S21f', '0-1', 5)} against "
            f"P2 {PRED_P2_01_S21F[5]:.3f} (band {BAND_N22_K5}) and below the P1/P2 midpoint {N22_K5_MIDPOINT:.4f}"), supported=bool(n22), seed=seeds("S21f")),
        dict(hypothesis="N23", statistic="S84r (84 resampled sizes), 0-1 loss, mean own-row NB ratio over data sets: " + "; ".join(
            f"K = {K} {o23[K]:.3f}{se('S84r', '0-1', K)} against P1 {PRED_P1_01[K]:.3f}" for K in KS_TEST) + f" (band {BAND_N23})",
             supported=bool(all(abs(o23[K] - PRED_P1_01[K]) <= BAND_N23 for K in KS_TEST)), seed=seeds("S84r")),
        dict(hypothesis="N24", statistic="A2 truth, 21 redrawn events, K = 5: " + "; ".join(
            f"{des}: mean(own-row ratio - per-data-set plug-in P2, jackknife beta) {o24[des][0]:+.3f} (band {BAND_N24}), mean paired own-row minus "
            f"residual-total ratio {o24[des][1]:.3f} (at least {MIN_N24_GAP})" for des in A2D),
             supported=bool(all(abs(v[0]) <= BAND_N24 and v[1] >= MIN_N24_GAP for v in o24.values())), seed="; ".join(seeds(des) for des in A2D)),
        dict(hypothesis="N25", statistic="S21f, log-loss, mean NB ratios over data sets: residual-total " + "; ".join(f"K = {K} {r25[K]:.3f}" for K in KS_TEST)
             + f" against 1; own-row " + "; ".join(f"K = {K} {o25[K]:.3f} against P2 {PRED_P2_LOG_S21F[K]:.3f}" for K in KS_TEST) + f" (band {BAND_N25})",
             supported=bool(all(abs(r25[K] - 1) <= BAND_N25 and abs(o25[K] - PRED_P2_LOG_S21F[K]) <= BAND_N25 for K in KS_TEST)), seed=seeds("S21f")),
    ]
    return pd.DataFrame(rows).assign(fold_scheme="grouped (cluster) K-fold; V_CR from one 5-fold run; NB from R grouped partitions with J = 60")


def write_outputs(S, A, sfx):
    S.to_csv(RES / f"sim_cv_unequal_rows{sfx}.csv", index=False); A.to_csv(RES / f"sim_cv_unequal_a2_rows{sfx}.csv", index=False)
    Sm = summarize(S, A); Sm.to_csv(RES / f"sim_cv_unequal_summary{sfx}.csv", index=False)
    H = verdicts(Sm); H.to_csv(RES / f"sim_cv_unequal_hypotheses{sfx}.csv", index=False)
    cols = ["design", "loss", "K", "role", "reps", "ratio_own", "ratio_own_se", "ratio_res", "own_minus_res", "partition_part", "plugin_p2_own",
            "var_ratio", "G_var_diff", "own_minus_plugin_p2", "P1_moment_decomposition"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(Sm[[c for c in cols if c in Sm]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(H[["hypothesis", "statistic", "supported"]].to_string(index=False))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv; sfx = "_quick" if quick else ""; shift = QUICK_SHIFT if quick else 0
    workers = int(args[0]) if args else 4
    if "--predictions" in sys.argv:
        return predictions(quick)
    if "--verdicts" in sys.argv:
        return write_outputs(pd.read_csv(RES / f"sim_cv_unequal_rows{sfx}.csv"), pd.read_csv(RES / f"sim_cv_unequal_a2_rows{sfx}.csv"), sfx)
    tasks = []
    for kind, specs in (("syn", SYN), ("a2", A2D)):
        for des, spec in specs.items():
            reps = QUICK_REPS[des] if quick else spec["reps"]
            tasks += [(kind, des, c, min(CHUNK, reps - c * CHUNK), shift) for c in range((reps + CHUNK - 1) // CHUNK)]
    tasks.sort(key=lambda t: {"S84r": 0}.get(t[1], 1 if t[0] == "a2" else 2))
    t0 = time.time()
    with Pool(workers) as pool:
        parts = pool.map(run_chunk, tasks, chunksize=1)
    S = pd.DataFrame([r for k, p in parts if k == "syn" for r in p]).sort_values(["design", "rep"])
    A = pd.DataFrame([r for k, p in parts if k == "a2" for r in p]).sort_values(["design", "rep"])
    print(f"{len(S) + len(A)} data sets in {(time.time() - t0) / 60:.1f} min")
    write_outputs(S, A, sfx)


if __name__ == "__main__":
    main()
