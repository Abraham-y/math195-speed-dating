import numpy as np, pandas as pd, sys, os, io, contextlib
_buf = io.StringIO()
sys.path.insert(0, "scripts")
from sim_cv_cluster import fit_logit, loss01, cv_point, cr_var_of_mean, grouped_folds, random_folds
rng = np.random.default_rng(7)
df = pd.read_csv('data/speed_dating.csv', encoding='latin-1')
feats = ['attr','sinc','intel','fun','amb','shar','order']
d = df.dropna(subset=feats+['dec','pid']).copy()
X = ((d[feats]-d[feats].mean())/d[feats].std()).values; y = d.dec.values.astype(float)
rater = d.iid.values; ratee = d.pid.values.astype(int); wave = d.wave.values
n = len(y); K = 5; z = 1.645
_tee = contextlib.redirect_stdout(_buf); _tee.__enter__()
print(f"rows {n}, raters {len(np.unique(rater))}, waves {len(np.unique(wave))}, yes-rate {y.mean():.3f}")

def relabel(a): 
    u, inv = np.unique(a, return_inverse=True); return inv
r_id, c_id, w_id = relabel(rater), relabel(ratee), relabel(wave)

def twoway_var(e, r, c):
    # Cameron-Gelbach-Miller: V_r + V_c - V_{r∩c} (intersection = single rows here)
    return cr_var_of_mean(e, r) + cr_var_of_mean(e, c) - e.var(ddof=1)/len(e)

def nested_floor(X, y, g, K, R=20):
    a_l, b_l = [], []
    for _ in range(R):
        f = grouped_folds(g, K, rng)
        for k in range(K):
            out = f==k; gin = g[~out]
            fin = grouped_folds(relabel(gin), K-1, rng)
            e_in = cv_point(X[~out], y[~out], fin, K-1).mean()
            w = fit_logit(X[~out], y[~out]); eo = loss01(w, X[out], y[out])
            a_l.append((e_in-eo.mean())**2); b_l.append(cr_var_of_mean(eo, g[out]))
    return np.mean(a_l)-np.mean(b_l)

res = {}
# random split
fr = random_folds(n, K, rng); er = cv_point(X, y, fr, K)
res['random split'] = (er.mean(), np.sqrt(er.var(ddof=1)/n), None)
# grouped by rater
fg = grouped_folds(r_id, K, rng); eg = cv_point(X, y, fg, K)
se_n = np.sqrt(eg.var(ddof=1)/n); se_r = np.sqrt(cr_var_of_mean(eg, r_id)); se_2 = np.sqrt(max(twoway_var(eg, r_id, c_id),0))
mse_cn = nested_floor(X, y, r_id, K)
res['grouped by rater'] = (eg.mean(), se_n, dict(rater_CR=se_r, twoway_CR=se_2, cluster_nested=np.sqrt(max(mse_cn,0)), floor=np.sqrt(max(se_r**2, mse_cn))))
# grouped by wave
fw = grouped_folds(w_id, K, rng); ew = cv_point(X, y, fw, K)
res['grouped by wave'] = (ew.mean(), np.sqrt(ew.var(ddof=1)/n), dict(wave_CR=np.sqrt(cr_var_of_mean(ew, w_id)), twoway_CR=np.sqrt(max(twoway_var(ew, r_id, c_id),0))))
for k,(m,sn,extra) in res.items():
    print(f"\n{k:18s} CV error {m:.4f}   naive SE {sn:.4f} (half-width {z*sn:.4f})")
    if extra:
        for nm,v in extra.items(): print(f"{'':18s}   {nm:15s} SE {v:.4f}  ratio to naive {v/sn:.2f}  half-width {z*v:.4f}")
# within-rater and within-ratee correlation of per-point errors (design effect inputs)
def icc(e, g):
    m = pd.Series(e).groupby(g).transform('mean').values; sizes = pd.Series(e).groupby(g).transform('size').values
    # pairwise within-cluster covariance estimate
    tot = 0; cnt = 0
    for gg in np.unique(g):
        ee = e[g==gg]-e.mean(); s = ee.sum(); q = (ee**2).sum(); tot += s*s-q; cnt += len(ee)*(len(ee)-1)
    return tot/cnt/e.var()
print(f"\nwithin-rater error correlation {icc(eg, r_id):.3f}, within-ratee {icc(eg, c_id):.3f}, median dates per rater {np.median(np.bincount(r_id)):.0f}")
# stated vs revealed weights (Part 1 quick look): revealed = standardized logistic coefficients
w = fit_logit(X, y)
stated = d.groupby('iid')[['attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1']].first()
st = stated.div(stated.sum(axis=1), axis=0).mean()*100
print("\nstated weights (mean %, harmonized):", st.round(1).to_dict())
rev = pd.Series(w[1:7], index=['attr','sinc','intel','fun','amb','shar']); rev = (rev.clip(lower=0)/rev.clip(lower=0).sum()*100)
print("revealed weights (% of positive std. logit coefs):", rev.round(1).to_dict())

_tee.__exit__(None, None, None); os.makedirs('results', exist_ok=True)
open('results/layerA_real_data.txt','w').write(_buf.getvalue()); print(_buf.getvalue()); print('wrote results/layerA_real_data.txt')
