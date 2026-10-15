"""C3: chemistry, a low-rank taste-by-trait term added to W1's within-event yes model (PREREG.md, Part 5).

Input : data/clean.csv; results/yes_extensions_within_event.csv (W1's AUC, which d = 0 must reproduce)
Output: results/chemistry_metrics.csv (held-out AUCs for d = 0, 1, 2 and the paired differences),
        results/chemistry_folds.csv (per outer fold: inner AUC of each lambda, the chosen lambda, convergence),
        results/chemistry_hypotheses.csv
Seeds : within-rater folds 20262006 (W1's, yes_extensions.within_event); inner within-rater folds 20262006 + 1 + k in
        outer fold k (the inner-seed rule of part1_models.tune_inner, which PREREG does not fix for C3); random
        starts 20263606, one SeedSequence per fit: [20263606, d, outer fold, inner fold (4 = the outer fit), lambda index].

Model. One row per date from one side: logit P(yes_ij) = mu + a_i + b_j + u_i'v_j. a_i is rater i's choosiness and b_j
ratee j's general appeal (W1); u_i (rater i's tastes) and v_j (ratee j's traits) are d-vectors, so u_i'v_j is the part
of the yes specific to the pair that a rank-d taste-by-trait term can describe. d = 0 is W1 exactly.
Fitting: block coordinate descent on the penalized negative log-likelihood
    sum over dates of [log(1 + e^eta) - y eta] + 1/2 (|a|^2 + |b|^2) + lambda/2 (sum_i |u_i|^2 + sum_j |v_j|^2):
  (1) mu, a, b by ridge-1 IRLS with u_i'v_j as an offset (unpenalized intercept, W1's fit when d = 0);
  (2) each u_i by a ridge-lambda logistic regression on its rater's training dates: features v_j, offset
      mu + a_i + b_j, no intercept;
  (3) each v_j likewise on its ratee's training dates, features u_i.
Each block is solved to convergence by Newton steps, halved when a step would raise that block's loss, so every block
update lowers the penalized loss. Rounds stop when one lowers it by less than 1e-9 of its value, or after 25 rounds.
Starts: u and v entries N(0, 0.5^2), 3 starts per fit, the one with the lowest penalized training loss kept.
lambda (one value for u and v) in {1, 3, 10}: the value with the best pooled inner AUC over 4 within-rater folds of the
outer training part, ties to the smaller value (as part1_models.tune_inner).
Statistic (C3): AUC(d = 1) - AUC(d = 0) on the same held-out predictions, with the clustered (Obuchowski) variance of
the difference of the two models' placement values by event (modeling.auc_diff_clustered_var). Supported if < 0.01.
C3 is decided on the 25-round cap; the same fits with a 200-round cap are reported beside it (exploratory), since
with lambda = 1 or 3 the penalized loss is still falling slowly at round 25.
Run from the repository root: OMP_NUM_THREADS=1 python scripts/chemistry.py [--jobs 2] [--starts 3] [--rounds 25] [--rounds-check 200]
"""
import os, sys, time, pathlib, argparse
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import expit
sys.path.insert(0, "scripts")
import modeling as M
import yes_extensions as YE

RES = pathlib.Path("results")
WITHIN_SEED, CHEM_SEED = YE.WITHIN_SEED, 20263606
LAMBDAS, DIMS, K_OUTER, K_INNER = (1.0, 3.0, 10.0), (0, 1, 2), 5, 4
RIDGE_AB, START_SD, TOL = 1.0, 0.5, 1e-9


# ---------------------------------------------------------------- pieces

def nll(y, eta):
    """Per-row negative log-likelihood of a logistic model, log(1 + e^eta) - y eta."""
    return np.logaddexp(0, eta) - y * eta


def with_intercept(X):
    n = X.shape[0]
    return sp.hstack([sp.csr_matrix(np.ones((n, 1))), X], format="csr") if sp.issparse(X) else np.hstack([np.ones((n, 1)), X])


def fit_logit_offset(X, y, offset=0.0, ridge=1.0, iters=200, w0=None, tol=1e-8):
    """Ridge-logistic regression with an unpenalized intercept and a fixed offset, by IRLS: the Newton steps of
    modeling.fit_logit (which it equals when the offset is zero), with the step halved if it would raise the penalized
    loss (a safeguard that W1's fit never triggers). X dense or scipy sparse. Returns coefficients, intercept first."""
    Xb = with_intercept(X); sparse = sp.issparse(Xb)
    n, p = Xb.shape
    off = np.broadcast_to(np.asarray(offset, float), (n,))
    w = np.zeros(p) if w0 is None else np.array(w0, float)
    pen = np.full(p, ridge); pen[0] = 0
    loss = lambda w: float(np.sum(nll(y, off + Xb @ w)) + 0.5 * np.sum(pen * w ** 2))
    cur = loss(w)
    for _ in range(iters):
        eta = np.clip(off + Xb @ w, -30, 30); mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu); g = Xb.T @ (y - mu) - pen * w
        if sparse:
            H = (Xb.T @ Xb.multiply(W[:, None]) + sp.diags(pen)).tocsc()
            step = spsolve(H, g)
        else:
            step = np.linalg.solve((Xb * W[:, None]).T @ Xb + np.diag(pen), g)
        t = 1.0
        while t > 1e-6 and loss(w + t * step) > cur + 1e-12 * abs(cur):
            t /= 2
        w = w + t * step; cur = loss(w)
        if np.max(np.abs(t * step)) < tol:
            break
    return w


def factor_step(g, X, off, y, F, lam, iters=50, tol=1e-10):
    """Update each row of F (one d-vector per group g, e.g. per rater) by Newton steps on that group's penalized loss
    sum over its rows of nll(y, off + F_g . X_row) + lam/2 |F_g|^2, halving a group's step when it would raise that
    group's loss. Groups without rows go to 0."""
    G, d = F.shape; F = F.copy()
    def group_loss(F):
        return np.bincount(g, nll(y, off + np.sum(F[g] * X, 1)), minlength=G) + lam / 2 * np.sum(F ** 2, 1)
    cur = group_loss(F)
    for _ in range(iters):
        p = expit(off + np.sum(F[g] * X, 1)); W, r = p * (1 - p), y - p
        grad = np.column_stack([np.bincount(g, X[:, a] * r, minlength=G) for a in range(d)]) - lam * F
        H = np.empty((G, d, d))
        for a in range(d):
            for b in range(a, d):
                H[:, a, b] = H[:, b, a] = np.bincount(g, W * X[:, a] * X[:, b], minlength=G)
        H += lam * np.eye(d)
        step = np.linalg.solve(H, grad[..., None])[..., 0]
        t = np.ones(G)
        for _ in range(40):
            new = group_loss(F + t[:, None] * step)
            bad = new > cur + 1e-12 * np.abs(cur)
            if not bad.any():
                break
            t[bad] /= 2
        F += t[:, None] * step; cur = group_loss(F)
        if np.max(np.abs(t[:, None] * step)) < tol:
            break
    return F


def penalized_loss(Xb, y, ri, ci, w, U, V, lam, ridge=RIDGE_AB):
    eta = Xb @ w + (np.sum(U[ri] * V[ci], 1) if U.shape[1] else 0.0)
    return float(np.sum(nll(y, eta)) + ridge / 2 * np.sum(w[1:] ** 2) + lam / 2 * (np.sum(U ** 2) + np.sum(V ** 2)))


def fit_chemistry(X, ri, ci, y, d, lam, rng, n_raters, n_ratees, starts=3, rounds=25, ridge=RIDGE_AB, trace=False):
    """Fit logit P(yes) = mu + a_i + b_j + u_i'v_j on the rows given (X the rater and ratee indicators of those rows,
    ri / ci their rater and ratee indices). d = 0 is W1's ridge-1 fit. For d >= 1, block coordinate descent from
    `starts` random starts; the fit with the lowest penalized loss is returned as a dict (w, U, V, loss, rounds,
    converged, and with trace=True the penalized loss after every block update)."""
    if d == 0:
        w = fit_logit_offset(X, y, 0.0, ridge, iters=200)
        Z = np.zeros((n_raters, 0)), np.zeros((n_ratees, 0))
        return dict(w=w, U=Z[0], V=Z[1], loss=penalized_loss(with_intercept(X), y, ri, ci, w, *Z, 0.0, ridge), rounds=0,
                    converged=True, trace=[])
    Xb = with_intercept(X); best = None
    for _ in range(starts):
        U, V = rng.normal(0, START_SD, (n_raters, d)), rng.normal(0, START_SD, (n_ratees, d))
        w, tr = None, []
        prev = np.inf; done = False
        for r in range(1, rounds + 1):
            w = fit_logit_offset(X, y, np.sum(U[ri] * V[ci], 1), ridge, iters=200, w0=w)
            if trace: tr.append(penalized_loss(Xb, y, ri, ci, w, U, V, lam, ridge))
            base = Xb @ w
            U = factor_step(ri, V[ci], base, y, U, lam)
            if trace: tr.append(penalized_loss(Xb, y, ri, ci, w, U, V, lam, ridge))
            V = factor_step(ci, U[ri], base, y, V, lam)
            cur = penalized_loss(Xb, y, ri, ci, w, U, V, lam, ridge)
            if trace: tr.append(cur)
            if prev - cur < TOL * abs(cur):
                done = True; break
            prev = cur
        fit = dict(w=w, U=U, V=V, loss=cur, rounds=r, converged=done, trace=tr)
        if best is None or fit["loss"] < best["loss"]:
            best = fit
    return best


def predict(fit, X, ri, ci):
    eta = with_intercept(X) @ fit["w"] + (np.sum(fit["U"][ri] * fit["V"][ci], 1) if fit["U"].shape[1] else 0.0)
    return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


# ---------------------------------------------------------------- cross-validation

def setup(clean):
    """W1's data, design and within-rater folds (yes_extensions.within_event, seed 20262006)."""
    d = clean[["rater", "ratee", "event", "gender", "dec"]].reset_index(drop=True)
    X = sp.csr_matrix(YE.indicator_design(d.rater.to_numpy(), d.ratee.to_numpy()))
    folds = YE.within_rater_folds(d.rater.to_numpy(), K_OUTER, np.random.default_rng(WITHIN_SEED))
    ri, ci = pd.factorize(d.rater)[0], pd.factorize(d.ratee)[0]
    return d, X, d.dec.to_numpy(float), ri, ci, folds


def outer_fold(k, d_dim, X, y, ri, ci, rater, folds, starts, rounds):
    """One outer fold for one d: lambda chosen on inner within-rater folds, then the fit on the whole training part."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    nr, nc = ri.max() + 1, ci.max() + 1
    tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
    info = dict(d=d_dim, outer_fold=k)
    if d_dim == 0:
        lam, li = 0.0, 0
    else:
        inner = YE.within_rater_folds(rater[tr], K_INNER, np.random.default_rng(WITHIN_SEED + 1 + k))
        scores = {}
        for li, lam in enumerate(LAMBDAS):
            p = np.empty(len(tr))
            for j in range(K_INNER):
                a, b = tr[inner != j], tr[inner == j]
                f = fit_chemistry(X[a], ri[a], ci[a], y[a], d_dim, lam, np.random.default_rng([CHEM_SEED, d_dim, k, j, li]), nr, nc, starts, rounds)
                p[inner == j] = predict(f, X[b], ri[b], ci[b])
            scores[lam] = M.auc(p, y[tr])
            info[f"inner_auc_lambda_{lam:g}"] = scores[lam]
        lam = max(LAMBDAS, key=lambda v: scores[v])                         # ties go to the earlier (smaller) value
        li = LAMBDAS.index(lam)
    fit = fit_chemistry(X[tr], ri[tr], ci[tr], y[tr], d_dim, lam, np.random.default_rng([CHEM_SEED, d_dim, k, K_INNER, li]), nr, nc, starts, rounds)
    uv = np.sum(fit["U"][ri[tr]] * fit["V"][ci[tr]], 1) if d_dim else np.zeros(len(tr))
    info.update(lambda_chosen=lam, rounds=fit["rounds"], converged=fit["converged"], train_penalized_loss=fit["loss"],
                sd_uv_train=float(uv.std()), seconds=time.time() - t0)
    return te, predict(fit, X[te], ri[te], ci[te]), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=25); ap.add_argument("--rounds-check", type=int, default=200)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    t0 = time.time()
    clean = pd.read_csv("data/clean.csv", low_memory=False)
    d, X, y, ri, ci, folds = setup(clean)
    rater, ev = d.rater.to_numpy(), d.event.to_numpy()
    runs = [(dd, args.rounds) for dd in DIMS] + [(dd, args.rounds_check) for dd in DIMS if dd]     # registered cap, then the check
    tasks = [(k, dd, cap) for dd, cap in runs for k in range(K_OUTER)]
    out = Parallel(n_jobs=args.jobs)(delayed(outer_fold)(k, dd, X, y, ri, ci, rater, folds, args.starts, cap) for k, dd, cap in tasks)
    pred = {r: np.empty(len(y)) for r in runs}; info = []
    for (k, dd, cap), (te, p, inf) in zip(tasks, out):
        pred[(dd, cap)][te] = p; info.append(dict(inf, round_cap=cap))
    F = pd.DataFrame(info).assign(fold_scheme="within-rater random folds", fold_seed=WITHIN_SEED, start_seed=CHEM_SEED, starts=args.starts)
    F.to_csv(RES / "chemistry_folds.csv", index=False)

    w1 = pd.read_csv(RES / "yes_extensions_within_event.csv").auc.iloc[0]
    p_w1 = pred[(0, args.rounds)]
    a0 = M.auc(p_w1, y)
    assert abs(a0 - w1) < 1e-6, f"d = 0 should reproduce W1's AUC {w1:.6f}, got {a0:.6f}"
    rows = []
    for label, keep in (("all decisions", np.ones(len(y), bool)), ("women's decisions", d.gender.to_numpy() == 0), ("men's decisions", d.gender.to_numpy() == 1)):
        for dd, cap in runs:
            p0, p = p_w1[keep], pred[(dd, cap)][keep]
            diff = M.auc(p, y[keep]) - M.auc(p0, y[keep])
            se_d = np.sqrt(M.auc_diff_clustered_var(p, p0, y[keep], ev[keep])) if dd else 0.0
            f = F[(F.d == dd) & (F.round_cap == cap)]
            rows.append(dict(decisions=label, d=dd, round_cap=cap if dd else "", model="W1: mu + a_i + b_j" if dd == 0 else f"mu + a_i + b_j + u_i'v_j, rank {dd}",
                             role="registered" if (dd in (0, 1) and cap == args.rounds) else "exploratory",
                             dates=int(keep.sum()), auc=M.auc(p, y[keep]),
                             auc_se=np.sqrt(M.auc_clustered_var(p, y[keep], ev[keep])), auc_minus_d0=diff, diff_se=se_d,
                             diff_ci90_low=diff - 1.645 * se_d, diff_ci90_high=diff + 1.645 * se_d,
                             lambdas_by_fold=";".join(f"{v:g}" for v in f.lambda_chosen) if dd else "",
                             fits_converged=f"{int(f.converged.sum())}/{K_OUTER}" if dd else "",
                             fold_scheme="within-rater random folds", fold_seed=WITHIN_SEED, start_seed=CHEM_SEED if dd else ""))
    Mt = pd.DataFrame(rows)
    Mt.to_csv(RES / "chemistry_metrics.csv", index=False)

    from sim_auc_cluster import coverage_note
    al = Mt[Mt.decisions == "all decisions"].set_index(["d", "round_cap"])
    r1, r2 = al.loc[(1, args.rounds)], al.loc[(2, args.rounds)]
    c1 = al.loc[(1, args.rounds_check)]
    a1 = r1.auc
    se_note = ("paired variance: the Obuchowski clustered AUC variance (checked in scripts/sim_auc_cluster.py) applied to the "
               "differences of the two models' placement values, a linear combination, so the same quadratic form applies "
               "(known answers in tests/test_part5_core.py: 0 for identical models, the single-model variance against a constant). "
               + coverage_note())
    hyp = pd.DataFrame([dict(hypothesis="C3", version=f"all decisions, d = 1 against d = 0, {args.rounds}-round cap",
                             statistic=f"held-out AUC d = 1 {a1:.4f} minus d = 0 (W1) {a0:.4f} = {a1 - a0:+.4f} (SE {r1.diff_se:.4f}, "
                                       f"90% interval {r1.diff_ci90_low:+.4f} to {r1.diff_ci90_high:+.4f}); lambda by fold {r1.lambdas_by_fold}; "
                                       f"{args.rounds_check}-round check: {c1.auc:.4f}, {c1.auc_minus_d0:+.4f} (SE {c1.diff_se:.4f}); "
                                       f"exploratory d = 2: {r2.auc:.4f}, {r2.auc_minus_d0:+.4f} (SE {r2.diff_se:.4f})",
                             supported=bool(a1 - a0 < 0.01), se_check=se_note,
                             fold_scheme=f"within-rater random folds (seed {WITHIN_SEED}), {K_OUTER} outer, {K_INNER} inner for lambda")])
    hyp.to_csv(RES / "chemistry_hypotheses.csv", index=False)
    print(f"d = 0 reproduces W1: AUC {a0:.6f} (W1 file {w1:.6f})")
    with pd.option_context("display.width", 250):
        print(F.drop(columns=["fold_scheme", "fold_seed", "start_seed"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(Mt.drop(columns=["fold_scheme", "fold_seed", "start_seed", "model"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(hyp[["hypothesis", "statistic", "supported"]].to_string(index=False))
    print(f"total {time.time() - t0:.0f}s with {args.jobs} processes")


if __name__ == "__main__":
    main()
