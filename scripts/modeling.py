"""Shared pieces for the Part 1 models, implementing the model-building entry of PREREG.md.

- event_folds: whole events assigned to K folds at random with a recorded seed.
- Spec and build_frame: one analysis specification (missing values, preference fit, out-of-range values,
  survey format) applied to the pair table, giving the raw design for a feature set.
- standardize_by_format: stated preferences and the alignment terms rescaled within survey format, from all
  participants' answers (no outcomes).
- Prep: the preparation refitted inside each training fold (skewness rule, median filling, rare codes,
  one-hot coding, splines, standardizing), in the order PREREG.md fixes.
- fit_logit, bic, stepwise: logistic regression by IRLS and forward / backward / both-ways selection by BIC
  over blocks of columns.
- cluster_robust_var_of_mean, metric_table, auc, auc_clustered_var, r_squared_and_rmse: event-held-out metrics with cluster-robust
  error bars (convention 6); the clustered AUC variance is Obuchowski (1997).
Every function here has a known-answer test in tests/test_modeling.py.
"""
from dataclasses import dataclass, replace
import numpy as np, pandas as pd

FOLD_SEED = 20261006
INTERACTIONS = ["age_diff_sq", "int_x_f_pref_shar", "int_x_m_pref_shar"]   # P3c in PREREG.md
WAVES_RATED_1_TO_10 = (6, 7, 8, 9)                                         # the waves that rated preferences 1 to 10
RARE_CODE_MIN = 10           # field and career codes held by fewer participants are merged into "other"
SKEW_LIMIT = 1.0


# ---------------------------------------------------------------- folds

def event_folds(events, k=5, seed=FOLD_SEED):
    """Map each event to a fold 0..k-1: a seeded random order of the events, dealt out in turn."""
    ev = np.sort(np.unique(events))
    order = np.random.default_rng(seed).permutation(len(ev))
    return {int(ev[i]): int(j % k) for j, i in enumerate(order)}


# ---------------------------------------------------------------- specification and raw design

@dataclass(frozen=True)
class Spec:
    missing: str = "median_flags"      # median_flags | median_no_date_flags | complete
    fit: str = "B"                     # B (compatibility term) | A (plain weighted mean)
    out_of_range: str = "rule"         # rule (recording errors missing) | recorded
    pref_format: str = "standardized"  # standardized | unstandardized

    @property
    def label(self):
        return f"missing={self.missing}; fit={self.fit}; out_of_range={self.out_of_range}; format={self.pref_format}"


HEADLINE = Spec()
COMPLETE_CASE = replace(HEADLINE, missing="complete")


def grid_specs():
    """The 24 combinations of the robustness grid (PREREG.md), headline first."""
    out = [HEADLINE]
    for m in ("median_flags", "median_no_date_flags", "complete"):
        for f in ("B", "A"):
            for o in ("rule", "recorded"):
                for p in ("standardized", "unstandardized"):
                    s = Spec(m, f, o, p)
                    if s != HEADLINE:
                        out.append(s)
    return out


def standardize_by_format(df, person_cols, dyad_cols, fmt_col="pref_scale"):
    """Rescale to mean 0, SD 1 within each survey format. Person columns (f_x and m_x pairs) use one row per
    participant, pooling women and men; dyad columns use one row per dyad. Uses no outcomes."""
    out = df.copy()
    for col in person_cols:                      # e.g. f_pref_attr; its partner column is m_pref_attr
        stem = col[2:]
        people = pd.concat([df[["f_iid", fmt_col, f"f_{stem}"]].drop_duplicates("f_iid").set_axis(["iid", "fmt", "v"], axis=1),
                            df[["m_iid", fmt_col, f"m_{stem}"]].drop_duplicates("m_iid").set_axis(["iid", "fmt", "v"], axis=1)])
        stats = people.groupby("fmt").v.agg(["mean", "std"])
        mu, sd = df[fmt_col].map(stats["mean"]), df[fmt_col].map(stats["std"])
        out[col] = (df[col] - mu) / sd
    for col in dyad_cols:
        stats = df.groupby(fmt_col)[col].agg(["mean", "std"])
        out[col] = (df[col] - df[fmt_col].map(stats["mean"])) / df[fmt_col].map(stats["std"])
    return out


def build_frame(pairs, sets, spec, feature_set, exclude_events=(), complete_extra=()):
    """Raw design for one specification and feature set: 'pre', 'pre+date', 'pre_individual', or the P3b/P3c
    ladder steps 'pre_extended' (plus the five extended pair features) and 'pre_interactions' (plus the three
    declared terms). exclude_events drops whole waves first (the waves 6 to 9 and wave 12 checks). Under the
    complete-case specification a dyad must be complete on every column of `pre+date` the models use, plus any
    complete_extra columns (the complete-case ladder passes the shared-interests shares, which P3c's terms use).
    Returns (X, y, events, ids) where ids holds f_iid and m_iid; columns keep their canonical names."""
    df = pairs[~pairs.event.isin(exclude_events)].copy()
    subs = {}
    if spec.out_of_range == "recorded":
        subs.update(sets["alternatives"]["out_of_range_as_recorded"])
    if spec.fit == "A":
        subs.update(sets["alternatives"]["preference_fit_A"])
    for canon, alt in subs.items():
        df[canon] = df[alt]
    if spec.pref_format == "standardized":
        person = [c for c in sets["format_standardized"] if c.startswith("f_pref_")]
        person += ["m" + c[1:] for c in person]
        dyad = [c for c in sets["format_standardized"] if not c.startswith(("f_pref_", "m_pref_"))]
        df = standardize_by_format(df, person, dyad)
    df["age_diff_sq"] = (df.m_age - df.f_age) ** 2
    df["int_x_f_pref_shar"] = df.int_corr * df.f_pref_shar
    df["int_x_m_pref_shar"] = df.int_corr * df.m_pref_shar
    cols = {"pre": sets["pre"], "pre+date": sets["pre+date"], "pre_individual": sets["pre_individual"],
            "pre_extended": sets["pre"] + sets.get("extended_pair", []),
            "pre_interactions": sets["pre"] + sets.get("extended_pair", []) + INTERACTIONS}[feature_set]
    cols = [c for c in cols if c not in sets["pref_reference"]]
    if spec.missing == "complete":
        full = [c for c in sets["pre+date"] if c not in sets["pref_reference"]] + list(complete_extra)
        df = df[df[full].notna().all(axis=1)]
    else:
        cols = cols + sets["flags_pre"]
        if spec.missing == "median_flags" and feature_set == "pre+date":
            cols = cols + sets["flags_date"]
    return (df[cols].reset_index(drop=True), df["match"].to_numpy(float), df["event"].to_numpy(int),
            df[["f_iid", "m_iid"]].reset_index(drop=True))


# ---------------------------------------------------------------- preparation inside a fold

def skew(x):
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std() == 0:
        return 0.0
    z = (x - x.mean()) / x.std()
    return float((z ** 3).mean())


def natural_spline_basis(x, knots):
    """Natural cubic spline basis without the constant (ESL 5.2.1): x, then N_{k+2} = d_k - d_{K-1}, k = 1..K-2."""
    x = np.asarray(x, float); K = len(knots)
    d = lambda k: (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[-1], 0, None) ** 3) / (knots[-1] - knots[k])
    return np.column_stack([x] + [d(k) - d(K - 2) for k in range(K - 2)])


class Prep:
    """Preparation fitted on a training fold and applied unchanged to the held-out events (PREREG.md order:
    skewness rule, median filling, rare-code merging and one-hot coding, splines, standardizing)."""

    def __init__(self, categorical, binary, person_cols, merge_codes=("field_cd", "career_c"), splines=(), sides=("f", "m")):
        self.categorical, self.binary, self.person_cols = list(categorical), set(binary), set(person_cols)
        self.merge_codes, self.splines, self.sides = tuple(merge_codes), tuple(splines), tuple(sides)

    def fit(self, X, ids):
        self.numeric = [c for c in X.columns if c not in self.categorical and c not in self.binary]
        self.bins = [c for c in X.columns if c in self.binary]
        self.tf = {}
        for c in self.numeric:                                   # skewness rule
            v = X[c].to_numpy(float)
            if X[c].nunique(dropna=True) <= 2:
                continue
            if c in self.person_cols:                            # one value per participant
                side = c.split("_")[0]
                v = pd.concat([X[c], ids[f"{side}_iid"]], axis=1).drop_duplicates(f"{side}_iid")[c].to_numpy(float)
            s = skew(v)
            if s > SKEW_LIMIT:
                self.tf[c] = ("right", np.nanmin(X[c]), np.nanmax(X[c]))
            elif s < -SKEW_LIMIT:
                self.tf[c] = ("left", np.nanmin(X[c]), np.nanmax(X[c]))
        T = self._transform_skew(X)
        self.medians = {c: float(np.nanmedian(T[c])) for c in self.numeric}
        self.bin_medians = {c: float(np.nanmedian(X[c])) if X[c].notna().any() else 0.0 for c in self.bins}
        self.levels = {}
        for c in self.categorical:                               # rare codes, then one-hot levels
            codes = X[c]
            if any(c.endswith(m) for m in self.merge_codes):
                stem = c.split("_", 1)[1]
                held = (pd.concat([pd.DataFrame({"iid": ids[f"{s}_iid"].to_numpy(), "code": X[f"{s}_{stem}"].to_numpy()})
                                   for s in self.sides if f"{s}_{stem}" in X.columns])
                        .drop_duplicates("iid").code.value_counts())       # each participant counted once
                rare = set(held.index[held < RARE_CODE_MIN])
            else:
                rare = set()
            lab = self._labels(codes, rare)
            counts = lab.value_counts()
            self.levels[c] = (rare, [l for l in counts.index if l != counts.index[0]], counts.index[0])
        F = self._filled(T)
        self.knots = {}
        for c in self.splines:
            if c in F.columns:
                q = np.unique(np.quantile(F[c], [0, 1 / 3, 2 / 3, 1]))
                if len(q) == 4:
                    self.knots[c] = q
        D = self._design(F, X)
        # Columns identical in the training fold (e.g. two missing-code dummies held by the same people) are all
        # kept: the data cannot divide their combined effect, and fit_logit's ridge divides it equally (PREREG.md,
        # second set of fixes of 2026-10-08).
        self.mu = {c: D[c].mean() for c in D.columns if not c.startswith(("cat:", "bin:"))}
        self.sd = {c: (D[c].std() or 1.0) for c in self.mu}
        self.columns = list(D.columns)
        return self

    def _transform_skew(self, X):
        T = X.copy()
        for c, (kind, lo, hi) in self.tf.items():
            v = T[c].clip(lo, hi)
            T[c] = np.log1p(v - lo) if kind == "right" else -np.log1p(hi - v)
        return T

    @staticmethod
    def _labels(codes, rare):
        lab = codes.map(lambda v: "missing" if pd.isna(v) else ("other" if v in rare else str(int(v))))
        return lab

    def _filled(self, T):
        F = T.copy()
        for c in self.numeric:
            F[c] = F[c].fillna(self.medians[c])
        return F

    def _design(self, F, X):
        parts = {}
        for c in self.numeric:
            if c in self.knots:
                B = natural_spline_basis(F[c], self.knots[c])
                for j in range(B.shape[1]):
                    parts[f"{c}" if j == 0 else f"{c}:ns{j}"] = B[:, j]
            else:
                parts[c] = F[c].to_numpy(float)
        for c in self.bins:
            parts[f"bin:{c}"] = X[c].fillna(self.bin_medians[c]).to_numpy(float)
        for c in self.categorical:
            rare, keep, _ = self.levels[c]
            lab = self._labels(X[c], rare)
            known = set(keep) | {self.levels[c][2]}
            lab = lab.where(lab.isin(known), "other" if "other" in known else self.levels[c][2])
            for l in keep:
                parts[f"cat:{c}={l}"] = (lab == l).to_numpy(float)
        return pd.DataFrame(parts)

    def transform(self, X):
        D = self._design(self._filled(self._transform_skew(X)), X)
        for c in self.columns:
            if c not in D.columns:
                D[c] = 0.0
        D = D[self.columns]
        for c in self.mu:
            D[c] = (D[c] - self.mu[c]) / self.sd[c]
        return D

    def blocks(self):
        """Stepwise blocks: each numeric predictor with its spline columns, each binary column, each categorical."""
        out = {}
        for j, c in enumerate(self.columns):
            if c.startswith("cat:"):
                key = c[4:].split("=")[0]
            elif c.startswith("bin:"):
                key = c[4:]
            else:
                key = c.split(":ns")[0]
            out.setdefault(key, []).append(j)
        return out


# ---------------------------------------------------------------- logistic regression, BIC, stepwise

def fit_logit(X, y, ridge=1e-6, iters=60, w0=None):
    """Logistic regression with intercept by IRLS (a tiny ridge keeps separated fits finite).
    Returns (coefficients with intercept first, log-likelihood)."""
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(p + 1) if w0 is None else w0.copy()
    pen = np.full(p + 1, ridge); pen[0] = 0
    for _ in range(iters):
        eta = np.clip(Xb @ w, -30, 30); mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        H = (Xb * W[:, None]).T @ Xb + np.diag(pen)
        step = np.linalg.solve(H, Xb.T @ (y - mu) - pen * w)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(Xb @ w, -30, 30)
    ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
    return w, ll


def predict_logit(w, X):
    return 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((X.shape[0], 1)), X]) @ w, -30, 30)))


def bic(ll, k, n):
    return -2 * ll + k * np.log(n)


def stepwise(X, y, blocks, direction):
    """Forward (from the intercept), backward (from all blocks) or both-ways (from the intercept) selection over
    blocks of columns, stopping when no single addition or removal lowers BIC. Returns the selected block names."""
    n = len(y)
    names = list(blocks)
    def score(sel):
        cols = [j for b in sel for j in blocks[b]]
        _, ll = fit_logit(X[:, cols], y) if cols else fit_logit(np.empty((n, 0)), y)
        return bic(ll, len(cols) + 1, n)
    sel = list(names) if direction == "backward" else []
    best = score(sel)
    while True:
        moves = []
        if direction in ("forward", "both"):
            moves += [("add", b) for b in names if b not in sel]
        if direction in ("backward", "both"):
            moves += [("drop", b) for b in sel]
        if not moves:
            return sel
        scored = [(score(sel + [b] if m == "add" else [s for s in sel if s != b]), m, b) for m, b in moves]
        s, m, b = min(scored, key=lambda t: t[0])
        if s >= best - 1e-9:
            return sel
        best = s
        sel = sel + [b] if m == "add" else [x for x in sel if x != b]


# ---------------------------------------------------------------- metrics and error bars

def cluster_robust_var_of_mean(u, g):
    """Variance of mean(u) with clusters g: G/(G-1) * sum over clusters of (within-cluster sum of residuals)^2 / n^2
    (the unbalanced form in convention 6)."""
    u = np.asarray(u, float); n = len(u); res = u - u.mean()
    tot = pd.Series(res).groupby(np.asarray(g)).sum().to_numpy()
    G = len(tot)
    return G / (G - 1) * np.sum(tot ** 2) / n ** 2


def auc(score, y):
    """Probability a random match outranks a random non-match (ties count half)."""
    score, y = np.asarray(score, float), np.asarray(y)
    r = pd.Series(score).rank().to_numpy()
    n1 = (y == 1).sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def placement_values(score, y):
    """Per-positive V10 (share of negatives scored below it, ties half) and per-negative V01 (share of positives
    scored above it)."""
    score, y = np.asarray(score, float), np.asarray(y)
    pos, neg = np.sort(score[y == 1]), np.sort(score[y == 0])
    s1, s0 = score[y == 1], score[y == 0]
    v10 = (np.searchsorted(neg, s1, "left") + np.searchsorted(neg, s1, "right")) / 2 / len(neg)
    v01 = 1 - (np.searchsorted(pos, s0, "left") + np.searchsorted(pos, s0, "right")) / 2 / len(pos)
    return v10, v01


def auc_clustered_var(score, y, g):
    """Obuchowski (1997) variance of the AUC when observations come in clusters g."""
    score, y, g = np.asarray(score, float), np.asarray(y), np.asarray(g)
    v10, v01 = placement_values(score, y)
    A = v10.mean(); M, N = len(v10), len(v01)
    cl = np.unique(g)
    m_k = pd.Series(y == 1).groupby(g).sum().reindex(cl, fill_value=0).to_numpy(float)
    n_k = pd.Series(y == 0).groupby(g).sum().reindex(cl, fill_value=0).to_numpy(float)
    V10 = pd.Series(v10).groupby(g[y == 1]).sum().reindex(cl, fill_value=0).to_numpy()
    V01 = pd.Series(v01).groupby(g[y == 0]).sum().reindex(cl, fill_value=0).to_numpy()
    K = len(cl)
    a, b = V10 - m_k * A, V01 - n_k * A
    S10 = K / ((K - 1) * M) * np.sum(a ** 2)
    S01 = K / ((K - 1) * N) * np.sum(b ** 2)
    S11 = K / (K - 1) * np.sum(a * b)
    return S10 / M + S01 / N + 2 * S11 / (M * N)


def r_squared_and_rmse(y, pred, g):
    """Held-out R-squared and RMSE with cluster-robust SEs (clusters g = events). R-squared = 1 - A / B with A the
    mean squared error and B the mean squared deviation of y from its mean; each date contributes the linearized
    residual -(e_i - A) / B + A (d_i - B) / B^2, with e_i its squared error and d_i its squared deviation, to the
    cluster-robust variance of a mean (convention 6). RMSE = sqrt(A), SE by the delta method."""
    y, pred, g = np.asarray(y, float), np.asarray(pred, float), np.asarray(g)
    e, d = (y - pred) ** 2, (y - y.mean()) ** 2
    A, B = e.mean(), d.mean()
    z = -(e - A) / B + A * (d - B) / B ** 2
    se_a = np.sqrt(cluster_robust_var_of_mean(e, g))
    return dict(r_squared=1 - A / B, r_squared_se=np.sqrt(cluster_robust_var_of_mean(1 - A / B + z, g)),
                rmse=np.sqrt(A), rmse_se=se_a / (2 * np.sqrt(A)))


def metric_table(prob, y, threshold, g):
    """Event-held-out metrics with cluster-robust standard errors (clusters g = events). prob and threshold are
    per dyad (the threshold of the fold that scored it). Each metric is written as a mean of per-dyad
    contributions whose residuals are the linearized ones, so the cluster-robust formula applies: sensitivity is
    a ratio (correct matches over matches), so a matched dyad's residual is (correct - sensitivity) * n / n1 and
    an unmatched dyad's is 0; specificity likewise; balanced accuracy averages the two."""
    prob, y, threshold, g = map(np.asarray, (prob, y, threshold, g))
    pred = (prob > threshold).astype(float)
    n, n1 = len(y), (y == 1).sum(); n0 = n - n1
    correct = (pred == y).astype(float)
    sens, spec = correct[y == 1].mean(), correct[y == 0].mean()
    r_sens = np.where(y == 1, (correct - sens) * n / n1, 0.0)
    r_spec = np.where(y == 0, (correct - spec) * n / n0, 0.0)
    contrib = {
        "error_rate": 1 - correct,
        "sensitivity": sens + r_sens,
        "specificity": spec + r_spec,
        "balanced_accuracy": (sens + spec) / 2 + (r_sens + r_spec) / 2,
    }
    out = {k: (v.mean(), np.sqrt(cluster_robust_var_of_mean(v, g))) for k, v in contrib.items()}
    out["auc"] = (auc(prob, y), np.sqrt(auc_clustered_var(prob, y, g)))
    return out
