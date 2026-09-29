import numpy as np, time
rng = np.random.default_rng(20260924)

def sigmoid(t): return 1/(1+np.exp(-t))

def fit_logit(X, y, iters=25, ridge=1e-3):
    n,p = X.shape
    Xb = np.hstack([np.ones((n,1)), X]); w = np.zeros(p+1)
    for _ in range(iters):
        mu = sigmoid(Xb@w); W = mu*(1-mu)
        grad = Xb.T@(y-mu) - ridge*w
        H = (Xb*W[:,None]).T@Xb + ridge*np.eye(p+1)
        step = np.linalg.solve(H, grad); w += step
        if np.max(np.abs(step))<1e-7: break
    return w

def loss01(w, X, y):
    return ((np.hstack([np.ones((X.shape[0],1)),X])@w > 0) != (y>0.5)).astype(float)

def gen(G, m, p, beta, tau_a, tau_x, rng):
    a = rng.normal(0, tau_a, G); u = rng.normal(0, tau_x, (G,p))
    g = np.repeat(np.arange(G), m)
    X = u[g] + rng.normal(size=(G*m, p))
    y = (rng.random(G*m) < sigmoid(X@beta + a[g])).astype(float)
    return X, y, g

def grouped_folds(g, K, rng):
    G = g.max()+1; perm = rng.permutation(G); cl2fold = np.empty(G, int)
    cl2fold[perm] = np.arange(G) % K
    return cl2fold[g]

def random_folds(n, K, rng):
    return rng.permutation(n) % K

def cv_point(X, y, folds, K):
    e = np.empty(len(y))
    for k in range(K):
        tr = folds!=k; w = fit_logit(X[tr], y[tr]); e[~tr] = loss01(w, X[~tr], y[~tr])
    return e

def cr_var_of_mean(e, g):
    # cluster-robust variance of the mean of e, clusters g (balanced or not)
    # sandwich estimator for the variance of the overall mean: sum over clusters of the squared
    # within-cluster sum of residuals, G/(G-1) small-sample factor. Reduces to the balanced formula when sizes are equal.
    n = len(e); res = e - e.mean(); cl = np.unique(g); G = len(cl)
    tot = np.array([res[g==c].sum() for c in cl])
    return G/(G-1) * np.sum(tot**2) / n**2

def nested(X, y, g, K, fold_fn, rng, R=20):
    a_iid, b_iid, b_cr = [], [], []
    for r in range(R):
        folds = fold_fn(rng)
        for k in range(K):
            outer = folds==k; Xin, yin, gin = X[~outer], y[~outer], g[~outer]
            fin = fold_fn_inner(gin, len(yin), K-1, rng, fold_fn)
            ein = cv_point(Xin, yin, fin, K-1); e_in = ein.mean()
            w = fit_logit(Xin, yin); eo = loss01(w, X[outer], y[outer]); e_out = eo.mean()
            a_iid.append((e_in-e_out)**2); nk = outer.sum()
            b_iid.append(eo.var(ddof=1)/nk); b_cr.append(cr_var_of_mean(eo, g[outer]))
    a = np.mean(a_iid)
    return a-np.mean(b_iid), a-np.mean(b_cr)

def fold_fn_inner(gin, nin, Kin, rng, fold_fn):
    return grouped_folds(gin, Kin, rng) if fold_fn.__name__=='gf' else random_folds(nin, Kin, rng)

def run(G=40, m=20, p=10, tau_a=1.0, tau_x=1.0, K=5, reps=300, seed=1):
    rng = np.random.default_rng(seed)
    beta = np.array([1.0,1.0,0.5,0.5,0,0,0,0,0,0])[:p]
    rows = []
    t0=time.time()
    for rep in range(reps):
        X, y, g = gen(G, m, p, beta, tau_a, tau_x, rng)
        n = len(y)
        # truth: new-cluster error of the full-data model
        w_full = fit_logit(X, y)
        Xt, yt, gt = gen(300, m, p, beta, tau_a, tau_x, rng)
        err_new = loss01(w_full, Xt, yt).mean()
        # grouped CV
        def gf(r): return grouped_folds(g, K, r)
        def rf(r): return random_folds(n, K, r)
        fg = gf(rng); eg = cv_point(X, y, fg, K); E_g = eg.mean()
        se_naive = np.sqrt(eg.var(ddof=1)/n); se_cr = np.sqrt(cr_var_of_mean(eg, g))
        mse_g_iid, mse_g_cr = nested(X, y, g, K, gf, rng)
        # random-split CV (BHT as published)
        fr = rf(rng); er = cv_point(X, y, fr, K); E_r = er.mean()
        se_r_naive = np.sqrt(er.var(ddof=1)/n)
        mse_r_iid, _ = nested(X, y, g, K, rf, rng)
        rows.append([err_new, E_g, se_naive, se_cr, np.sqrt(max(mse_g_iid,0)), np.sqrt(max(mse_g_cr,0)), E_r, se_r_naive, np.sqrt(max(mse_r_iid,0)), mse_g_cr, mse_g_iid])
        if rep==4: print(f"~{(time.time()-t0)/5*reps/60:.1f} min expected")
    return np.array(rows)

if __name__=='__main__':
    import sys
    R = run(reps=int(sys.argv[1]) if len(sys.argv)>1 else 300)
    import os; os.makedirs('results', exist_ok=True); np.save('results/sim_cv_cluster_rows.npy', R)
    err, Eg, sn, scr, sni, scn, Er, srn, srni, raw_cn, raw_ni = R.T
    z=1.645
    def cov(E, se, target): return np.mean(np.abs(E-target) <= z*se)
    print("n=800 (G=40 clusters x m=20), p=10, K=5, reps=", len(R))
    print(f"true Err_new (mean over reps)      : {err.mean():.4f}  sd across training sets {err.std():.4f}")
    print(f"grouped CV estimate: mean {Eg.mean():.4f}, sd {Eg.std():.4f}  (bias vs Err_new {Eg.mean()-err.mean():+.4f})")
    print(f"random  CV estimate: mean {Er.mean():.4f}, sd {Er.std():.4f}  (bias vs Err_new {Er.mean()-err.mean():+.4f})")
    V = Eg.var()
    print(f"Var(grouped CV) empirical         : {V:.3e}")
    print(f"mean naive SE^2                   : {np.mean(sn**2):.3e}  ratio {V/np.mean(sn**2):.2f}")
    print(f"mean cluster-robust SE^2          : {np.mean(scr**2):.3e}  ratio {V/np.mean(scr**2):.2f}")
    print(f"mean nested(iid b) MSE, grouped   : {np.mean(sni**2):.3e}")
    print(f"mean cluster-nested MSE, grouped  : {np.mean(scn**2):.3e}")
    print(f"empirical MSE of grouped CV vs Err_new: {np.mean((Eg-err)**2):.3e}")
    print(f"cluster-nested raw MSE: mean {raw_cn.mean():.3e}, sd {raw_cn.std():.3e}, frac<=0 {np.mean(raw_cn<=0):.2f}; nested iid-b raw: mean {raw_ni.mean():.3e}, sd {raw_ni.std():.3e}")
    print("--- 90% interval coverage of Err_new (the new-cluster error of the full-data model) ---")
    for name,E,se in [("grouped CV, naive SE",Eg,sn),("grouped CV, cluster-robust SE",Eg,scr),
                      ("grouped CV, nested (iid b_k)",Eg,sni),("grouped CV, cluster-nested",Eg,scn),
                      ("random CV, naive SE",Er,srn),("random CV, BHT nested",Er,srni)]:
        print(f"{name:34s}: coverage {cov(E,se,err):.3f}   mean half-width {z*se.mean():.4f}")
    print("--- coverage of E[Err_new] (average over training sets) ---")
    mu = err.mean()
    for name,E,se in [("grouped CV, naive SE",Eg,sn),("grouped CV, cluster-robust SE",Eg,scr),
                      ("grouped CV, nested (iid b_k)",Eg,sni),("grouped CV, cluster-nested",Eg,scn)]:
        print(f"{name:34s}: coverage {cov(E,se,mu):.3f}")
