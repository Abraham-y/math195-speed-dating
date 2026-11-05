"""Part 12: the error-bar results outside speed dating. Four public datasets from cardiology, education, census income
and phone sensing (scripts/external_data.py; sources in data/external_sources.md). Descriptive: no formal
pre-registration (the team's decision of 2026-11-05); this script was committed before any run on real outcomes,
except Stage B's pilot on scratch seeds.

Input : data/external/* through scripts/external_data.py
Output: results/cross_field_partitions.csv   one row per dataset, model, scheme and partition (Stage A)
        results/cross_field_folds.csv        fold values for the Nadeau-Bengio ratio (Stage A, exploratory)
        results/cross_field_summary.csv      Stage A per dataset and model
        results/cross_field_stageb.csv       Stage B, one row per replicate; _stageb_summary.csv per G and model
        (`--pilot`: Stage B with scratch seeds, to results/cross_field_stageb_pilot*.csv)
Seeds : grouped partitions [20268006, dataset index, r]; random-row partitions [20268106, dataset index, r]; Stage B
        replicates [20268306, G, replicate] (pilot: [996000, G, replicate]); wild bootstrap draws from the same
        generator after the replicate's draws.

Models, both with defaults and no tuning, as an applied paper would use them:
- logistic regression (scikit-learn, L2 with C = 1, lbfgs) on the training fold's median-imputed, standardized
  predictors, with a missing flag for every column that has missing values;
- gradient boosting (HistGradientBoostingClassifier: 200 iterations, learning rate 0.1, 31 leaves, no early
  stopping) on the raw predictors, missing values handled natively.
Categories are one-hot coded on the whole data set (codes held by fewer than 10 rows merged); no outcome is used.
Thresholds for the error rate: the training fold's outcome rate.

Stage A (each data set as it is): R = 20 random grouped 5-fold partitions (groups dealt at random, as GroupKFold
with shuffling) and 5 random-row partitions. Per partition: the pooled AUC, BCWF-pooled (pairs from different groups
within a fold), the pooled between-group AUC, the fold-averaged and within-group AUCs, the error rate and log-loss,
and their naive (independent rows) and cluster-robust SEs (Obuchowski by group for the AUC; DeLong-equivalent with
every row its own cluster for the naive AUC SE).
Stage B (census income only, a real-data population): the 265 PUMAs are the population. Each replicate draws G PUMAs
(G = 20 or 40) as the study, runs grouped 5-fold CV on them, and scores the full model (fitted on the whole study) on
all other PUMAs: AUC_new and Err_new. Coverage of nominal 90% intervals and bias against those targets.
Run from the repository root: python scripts/cross_field.py [--stage a|b|both] [--pilot] [--jobs 8]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pathlib, time, argparse, warnings
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
sys.path.insert(0, "scripts")
import modeling as M
import external_data as E
from sim_cv_auc import estimates, within_cluster_counts
from sim_cv_sizes import wild_cluster_t_quantile

RES = pathlib.Path("results")
DATASETS = ["mitdb", "oulad", "acs", "extrasensory_sleeping", "extrasensory_workplace"]
MODELS = ("logistic", "boosting")
K, R_GROUPED, R_ROW, MIN_CODE = 5, 20, 5, 10
SEED_GROUPED, SEED_ROW, SEED_B, SEED_B_PILOT = 20268006, 20268106, 20268306, 996000
B_SIZES, B_REPS, B_REPS_PILOT, B_WILD = (20, 40), 400, 100, 999
Z = stats.norm.ppf(0.95)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- design and models

def design(X):
    """Numeric matrix (float32, NaN kept) with one-hot categories (codes held by fewer than MIN_CODE rows merged into
    'other'); uses no outcome."""
    cols = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):          # text columns (pandas 3 stores them as `str`, not object)
            v = X[c].astype(str); counts = v.value_counts()
            v = v.where(v.map(counts) >= MIN_CODE, "other")
            levels = sorted(v.unique())
            cols += [(v == lev).astype(np.float32).rename(f"{c}={lev}") for lev in levels[1:]]
        else:
            cols.append(pd.to_numeric(X[c], errors="coerce").astype(np.float32))
    D = pd.concat(cols, axis=1)
    return D.to_numpy(np.float32), list(D.columns)


def prepare(D, tr):
    """Training-fold median imputation with missing flags, then standardization by the training rows."""
    nan_cols = np.where(np.isnan(D[tr]).any(axis=0) | np.isnan(D).any(axis=0))[0]
    med = np.nanmedian(D[tr], axis=0); med = np.where(np.isnan(med), 0, med)
    F = np.where(np.isnan(D), med, D)
    F = np.hstack([F, np.isnan(D[:, nan_cols]).astype(np.float32)])
    mu, sd = F[tr].mean(0), F[tr].std(0); sd[sd == 0] = 1
    return (F - mu) / sd


def fit_predict(model, D, y, tr, te, seed):
    warnings.filterwarnings("ignore")
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression
        F = prepare(D, tr)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(F[tr], y[tr])
        return m.predict_proba(F[te])[:, 1]
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False,
                                       random_state=seed).fit(D[tr], y[tr])
    return m.predict_proba(D[te])[:, 1]


def fold_task(model, D, y, fold, k, seed):
    tr, te = fold != k, fold == k
    return np.where(te)[0], fit_predict(model, D, y, tr, te, seed), float(y[tr].mean())


# ---------------------------------------------------------------- metrics

def naive_auc_var(s, y):
    return M.auc_clustered_var(s, y, np.arange(len(y)))


def metrics(p, thr, y, g, f):
    """Everything Stage A records for one partition's held-out probabilities."""
    e = estimates(p, y, g, f)
    err = ((p >= thr) != y).astype(float)
    ll = -(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1)))
    conc_w, pairs_w = within_cluster_counts(p, y, g)
    out = dict(auc_pooled=e["pooled"], auc_bcwf_pooled=e["bcwf_pooled"], auc_pooled_between=e["pooled_between"],
               auc_fold_averaged=e["fold_averaged"], auc_within=conc_w.sum() / pairs_w.sum() if pairs_w.sum() else np.nan,
               auc_se_naive=np.sqrt(naive_auc_var(p, y)), auc_se_cluster=np.sqrt(e["v_obu"]),
               error_rate=err.mean(), error_se_naive=err.std(ddof=1) / np.sqrt(len(err)),
               error_se_cluster=np.sqrt(M.cluster_robust_var_of_mean(err, g)),
               log_loss=ll.mean(), log_loss_se_naive=ll.std(ddof=1) / np.sqrt(len(ll)),
               log_loss_se_cluster=np.sqrt(M.cluster_robust_var_of_mean(ll, g)))
    folds = [dict(fold=k, error_rate=err[f == k].mean(), log_loss=ll[f == k].mean(),
                  auc=M.auc(p[f == k], y[f == k]) if 0 < y[f == k].mean() < 1 else np.nan, rows=int((f == k).sum()))
             for k in np.unique(f)]
    return out, folds


def grouped_partition(g, rng):
    groups = np.unique(g); to = np.empty(len(groups), int); to[rng.permutation(len(groups))] = np.arange(len(groups)) % K
    return to[np.searchsorted(groups, g)]


def row_partition(n, rng):
    f = np.empty(n, int); f[rng.permutation(n)] = np.arange(n) % K
    return f


# ---------------------------------------------------------------- Stage A

def stage_a(jobs):
    parts, folds_out = [], []
    for di, name in enumerate(DATASETS):
        X, y, g = E.LOADERS[name](); D, _ = design(X); y = y.astype(float)
        plan = [("grouped", r, grouped_partition(g, np.random.default_rng([SEED_GROUPED, di, r]))) for r in range(R_GROUPED)]
        plan += [("random-row", r, row_partition(len(y), np.random.default_rng([SEED_ROW, di, r]))) for r in range(R_ROW)]
        tasks = [(model, scheme, r, k) for model in MODELS for scheme, r, _ in plan for k in range(K)]
        fmap = {(scheme, r): f for scheme, r, f in plan}
        log(f"{name}: {len(y)} rows, {len(np.unique(g))} groups, {D.shape[1]} columns; {len(tasks)} fits")
        res = Parallel(n_jobs=jobs, max_nbytes="1M")(delayed(fold_task)(model, D, y, fmap[(scheme, r)], k, 20268206 + k)
                                                       for model, scheme, r, k in tasks)
        acc = {}
        for (model, scheme, r, k), (idx, p, thr) in zip(tasks, res):
            a = acc.setdefault((model, scheme, r), dict(p=np.full(len(y), np.nan), thr=np.full(len(y), np.nan)))
            a["p"][idx] = p; a["thr"][idx] = thr
        for (model, scheme, r), a in acc.items():
            assert not np.isnan(a["p"]).any()
            out, fv = metrics(a["p"], a["thr"], y, g, fmap[(scheme, r)])
            seed = f"[{SEED_GROUPED if scheme == 'grouped' else SEED_ROW}, {di}, {r}]"
            parts.append(dict(dataset=name, model=model, scheme=scheme, partition=r, rows=len(y), groups=len(np.unique(g)), seed=seed, **out))
            folds_out += [dict(dataset=name, model=model, scheme=scheme, partition=r, **v) for v in fv]
        log(f"  done {name}")
    P = pd.DataFrame(parts); P.to_csv(RES / "cross_field_partitions.csv", index=False)
    pd.DataFrame(folds_out).to_csv(RES / "cross_field_folds.csv", index=False)
    return P


def nb_ratio(fv, vcr):
    """Nadeau-Bengio variance from J fold values over (J+K-1)/(J-1) times the mean cluster-robust variance."""
    v = np.asarray(fv, float); v = v[~np.isnan(v)]; J = len(v)
    return (1 / J + 1 / (K - 1)) * v.var(ddof=1) / ((J + K - 1) / (J - 1) * vcr)


def summary_a(P, F):
    rows = []
    for (name, model), d in P.groupby(["dataset", "model"]):
        gr, rw = d[d.scheme == "grouped"], d[d.scheme == "random-row"]
        mix = gr.auc_pooled_between - gr.auc_bcwf_pooled; gap = gr.auc_pooled - gr.auc_bcwf_pooled
        fg = F[(F.dataset == name) & (F.model == model) & (F.scheme == "grouped")]
        rows.append(dict(
            dataset=name, model=model, rows=int(d.rows.iloc[0]), groups=int(d.groups.iloc[0]),
            auc_grouped=gr.auc_pooled.mean(), auc_random_row=rw.auc_pooled.mean(), leakage_auc=rw.auc_pooled.mean() - gr.auc_pooled.mean(),
            leakage_error=gr.error_rate.mean() - rw.error_rate.mean(),
            error_se_ratio=(gr.error_se_cluster / gr.error_se_naive).mean(), auc_se_ratio=(gr.auc_se_cluster / gr.auc_se_naive).mean(),
            log_loss_se_ratio=(gr.log_loss_se_cluster / gr.log_loss_se_naive).mean(),
            mixing=mix.mean(), mixing_se=mix.std(ddof=1) / np.sqrt(len(mix)), mixing_sd=mix.std(ddof=1),
            pooled_minus_bcwf=gap.mean(), pooled_minus_bcwf_se=gap.std(ddof=1) / np.sqrt(len(gap)),
            auc_bcwf_pooled=gr.auc_bcwf_pooled.mean(), auc_fold_averaged=gr.auc_fold_averaged.mean(), auc_within=gr.auc_within.mean(),
            auc_se_cluster=gr.auc_se_cluster.mean(),
            nb_ratio_error=nb_ratio(fg.error_rate, (gr.error_se_cluster ** 2).mean()),
            nb_ratio_log_loss=nb_ratio(fg.log_loss, (gr.log_loss_se_cluster ** 2).mean()),
            nb_ratio_auc=nb_ratio(fg.auc, (gr.auc_se_cluster ** 2).mean()),
            partitions_grouped=len(gr), partitions_row=len(rw)))
    S = pd.DataFrame(rows); S.to_csv(RES / "cross_field_summary.csv", index=False)
    return S


# ---------------------------------------------------------------- Stage B

def stage_b_rep(G, rep, D, y, g, seed0):
    rng = np.random.default_rng([seed0, G, rep])
    pumas = np.unique(g); study = rng.choice(pumas, G, replace=False)
    s = np.isin(g, study); Ds, ys, gs = D[s], y[s], g[s]
    f = grouped_partition(gs, rng)
    out = []
    for model in MODELS:
        p, thr = np.empty(len(ys)), np.empty(len(ys))
        for k in range(K):
            tr, te = f != k, f == k
            p[te] = fit_predict(model, Ds, ys, tr, te, 20268206 + k); thr[te] = ys[tr].mean()
        e = estimates(p, ys, gs, f)
        err = ((p >= thr) != ys).astype(float)
        full = np.ones(len(ys), bool)
        pn = fit_predict(model, np.vstack([Ds, D[~s]]), np.concatenate([ys, y[~s]]), np.r_[full, np.zeros((~s).sum(), bool)],
                         np.r_[np.zeros(len(ys), bool), np.ones((~s).sum(), bool)], 20268206)
        yn, gn = y[~s], g[~s]
        err_new = ((pn >= ys.mean()) != yn).astype(float)
        q_wild = wild_cluster_t_quantile(err, gs, B_WILD, rng)
        out.append(dict(G=G, replicate=rep, model=model, seed=f"[{seed0}, {G}, {rep}]", study_rows=len(ys),
                        auc_pooled=e["pooled"], auc_bcwf_pooled=e["bcwf_pooled"], auc_se_cluster=np.sqrt(e["v_obu"]),
                        auc_se_naive=np.sqrt(naive_auc_var(p, ys)), auc_new=M.auc(pn, yn), auc_new_mc_var=M.auc_clustered_var(pn, yn, gn),
                        error_rate=err.mean(), error_se_naive=err.std(ddof=1) / np.sqrt(len(err)),
                        error_se_cluster=np.sqrt(M.cluster_robust_var_of_mean(err, gs)), wild_q=q_wild,
                        err_new=err_new.mean(), err_new_mc_var=M.cluster_robust_var_of_mean(err_new, gn)))
    return out


def stage_b(jobs, pilot):
    X, y, g = E.LOADERS["acs"](); D, _ = design(X); y = y.astype(float)
    reps, seed0 = (B_REPS_PILOT, SEED_B_PILOT) if pilot else (B_REPS, SEED_B)
    log(f"Stage B{' pilot' if pilot else ''}: {reps} replicates at G = {B_SIZES}")
    res = Parallel(n_jobs=jobs, max_nbytes="1M")(delayed(stage_b_rep)(G, r, D, y, g, seed0) for G in B_SIZES for r in range(reps))
    B = pd.DataFrame([row for rep in res for row in rep])
    B.to_csv(RES / f"cross_field_stageb{'_pilot' if pilot else ''}.csv", index=False)
    return B


def summary_b(B, pilot=False):
    rows = []
    for (G, model), d in B.groupby(["G", "model"]):
        t = stats.t.ppf(0.95, G - 1)
        cov = lambda est, se, q, target: float(np.mean(np.abs(est - target) <= q * se))
        rows.append(dict(
            G=G, model=model, replicates=len(d),
            auc_new_mean=d.auc_new.mean(), bias_pooled=(d.auc_pooled - d.auc_new).mean(), bias_pooled_se=(d.auc_pooled - d.auc_new).std(ddof=1) / np.sqrt(len(d)),
            bias_bcwf=(d.auc_bcwf_pooled - d.auc_new).mean(), bias_bcwf_se=(d.auc_bcwf_pooled - d.auc_new).std(ddof=1) / np.sqrt(len(d)),
            auc_se_cluster_mean=d.auc_se_cluster.mean(),
            cover_auc_naive=cov(d.auc_pooled, d.auc_se_naive, Z, d.auc_new), cover_auc_pooled_t=cov(d.auc_pooled, d.auc_se_cluster, t, d.auc_new),
            cover_auc_bcwf_t=cov(d.auc_bcwf_pooled, d.auc_se_cluster, t, d.auc_new),
            cover_err_naive=cov(d.error_rate, d.error_se_naive, Z, d.err_new), cover_err_cluster_t=cov(d.error_rate, d.error_se_cluster, t, d.err_new),
            cover_err_wild=cov(d.error_rate, d.error_se_cluster, d.wild_q, d.err_new),
            width_ratio_err=(d.error_se_cluster / d.error_se_naive).mean(),
            mc_share_auc=d.auc_new_mc_var.mean() / (d.auc_pooled - d.auc_new).var(ddof=1),
            mc_share_err=d.err_new_mc_var.mean() / (d.error_rate - d.err_new).var(ddof=1)))
    S = pd.DataFrame(rows); S.to_csv(RES / f"cross_field_stageb{'_pilot' if pilot else ''}_summary.csv", index=False)
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="both", choices=("a", "b", "both"))
    ap.add_argument("--pilot", action="store_true"); ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    if args.pilot:
        print(summary_b(stage_b(args.jobs, True), True).to_string(index=False)); return
    if args.stage in ("a", "both"):
        SA = summary_a(stage_a(args.jobs), pd.read_csv(RES / "cross_field_folds.csv")); print(SA.to_string(index=False))
    if args.stage in ("b", "both"):
        SB = summary_b(stage_b(args.jobs, False)); print(SB.to_string(index=False))


if __name__ == "__main__":
    main()
