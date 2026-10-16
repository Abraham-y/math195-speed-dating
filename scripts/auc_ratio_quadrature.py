"""Exact (quadrature) values of Proposition 9's C_mix and c2, independent of the binned estimator in sim_cv_auc.py.

Written by the adversarial reviewer of proposals/06 section 6f (proposals/09, 2026-10-16) and adopted here; checked
against the binned constants (C_mix within 1%) and against Proposition 10 (ratio exactly 1/2 for independent rows)
in tests/test_part9.py.
Input : none. Output: results/auc_ratio_quadrature.csv. Nothing is random.

Design family: x = su u_g + v_i (u, v ~ N(0, I_3)), logit P(y = 1 | x, a) = b0 + beta'x + a_g, with a_g Gaussian (sd sa),
two-point (+-a) or bridge-distributed (Wang and Louis 2003). Because x is Gaussian and a is independent of x, the
marginal P(y | x) depends on x only through z = e'x / sqrt(1 + su^2), e = beta / |beta|. C_mix and c2 are invariant to a
linear reparametrization of x, and with block-diagonal H and Var(s_h) only the (1, z) block contributes at D = 0, so
everything reduces to one-dimensional integrals in z, with the latent (a, u_e) of the cluster mates integrated by
nested quadrature.
  C_mix = (1/m) E[delta(D) {x_i' H^-1 E[s_h | j] - x_j' H^-1 E[s_h | i]}],   D = t1 (z_i - z_j)
  c2    = -g'(0),  g(t) = E[x_i' Var(psi) x_j delta(D - t)],  Var(psi) = H^-1 Var(s_h) H^-1 / m^2
g'(0) is computed by differentiating the z-integral directly (no integration by parts).
Cases: the six linear designs of Part 9 and the counterexamples to c2 = C_mix / 2 under clustering (proposals/09).
Run from the repository root: python scripts/auc_ratio_quadrature.py
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

sig = lambda t: 1 / (1 + np.exp(-t))


def gauss_nodes(n, sd):
    x, w = hermegauss(n)
    return x * sd, w / w.sum()


def law(kind, par, n=161):
    """(nodes, weights) of the random-intercept law."""
    if kind == "none" or par == 0:
        return np.array([0.0]), np.array([1.0])
    if kind == "gauss":
        return gauss_nodes(n, par)
    if kind == "two":          # +-par with probability 1/2
        return np.array([-par, par]), np.array([0.5, 0.5])
    if kind == "bridge":       # Wang & Louis (2003); par = phi in (0, 1); marginal logit = phi * conditional
        phi = par; L = 80 / phi; a = np.linspace(-L, L, 40001)
        f = np.sin(phi * np.pi) / (2 * np.pi * (np.cosh(np.clip(phi * a, -700, 700)) + np.cos(phi * np.pi)))
        w = f * (a[1] - a[0]); return a, w / w.sum()
    if kind == "t3":           # scaled t with 3 df (heavy tails), sd = par
        from scipy import stats
        q = np.linspace(-60, 60, 24001); f = stats.t.pdf(q, 3); a = q * par / np.sqrt(3.0)
        w = f / f.sum(); return a, w
    raise ValueError(kind)


def compute(b0=-1.5, beta=(1.0, 0.5, 0.0), su=0.0, akind="gauss", apar=0.0, m=40, nz=6001, zmax=9.0, nu=481, nv=81,
            details=False):
    b = float(np.linalg.norm(beta)); sx = np.sqrt(1 + su ** 2)
    z = np.linspace(-zmax, zmax, nz); dz = z[1] - z[0]
    wz = np.full(nz, dz); wz[0] = wz[-1] = dz / 2                       # trapezoid (integrands are smooth, decay fast)
    phiz = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    A, WA = law(akind, apar)
    eta = b0 + b * sx * z                                               # conditional logit minus a, as function of z
    S = sig(eta[:, None] + A[None, :])                                  # nz x na
    h = S @ WA                                                          # true marginal P(y=1 | z)
    hp = (S * (1 - S)) @ WA * b * sx                                    # dh/dz
    # theta* = (t0, t1) in (1, z) coordinates: E[(1, z)(h - sig(t0 + t1 z))] = 0
    t = np.array([b0 * 0.8, b * sx * 0.8])
    for _ in range(100):
        p = sig(t[0] + t[1] * z); r = h - p; ww = p * (1 - p)
        g = np.array([np.sum(wz * phiz * r), np.sum(wz * phiz * z * r)])
        J = np.array([[np.sum(wz * phiz * ww), np.sum(wz * phiz * ww * z)], [np.sum(wz * phiz * ww * z), np.sum(wz * phiz * ww * z * z)]])
        st = np.linalg.solve(J, g); t += st
        if np.max(np.abs(st)) < 1e-14:
            break
    t0, t1 = t; p = sig(t0 + t1 * z); mu = np.stack([np.ones(nz), z], 1)
    E = lambda f: np.sum((wz * phiz)[:, None, None] * f, 0)
    H = E(mu[:, :, None] * mu[:, None, :] * (p * (1 - p))[:, None, None])
    B0 = E(mu[:, :, None] * mu[:, None, :] * (h - 2 * h * p + p ** 2)[:, None, None])
    pi = np.sum(wz * phiz * h)
    # cluster mates: phi(a, u_e) = E_v[(1, z_l)(sig(b0 + b(su u_e + v_e) + a) - p*(z_l))], z_l = (su u_e + v_e)/sx
    if su > 0:
        U = np.linspace(-12, 12, nu); WU = np.exp(-U ** 2 / 2); WU /= WU.sum()
    else:
        U, WU = np.array([0.0]), np.array([1.0])
    V, WV = gauss_nodes(nv, 1.0)
    zl = (su * U[:, None] + V[None, :]) / sx                            # nu x nv
    el = b0 + b * (su * U[:, None] + V[None, :])
    resid = sig(el[None, :, :] + A[:, None, None]) - sig(t0 + t1 * zl)[None]   # na x nu x nv
    Phi0 = resid @ WV; Phi1 = (resid * zl[None]) @ WV                    # na x nu
    Phi = np.stack([Phi0, Phi1], -1)                                     # na x nu x 2
    C = np.einsum("a,u,aui,auj->ij", WA, WU, Phi, Phi)
    meanPhi = np.einsum("a,u,aui->i", WA, WU, Phi)                       # should be ~0
    # posterior: a | (z, y) ∝ WA * S^y (1-S)^(1-y);  u_e | z ~ N(su z / sx, 1/sx^2)
    if su > 0:
        # weights of U nodes under N(su z/sx, 1/sx^2): reweight standard-normal nodes by likelihood ratio
        mz = su * z / sx; sd = 1 / sx
        PU = np.exp(-(U[None, :] - mz[:, None]) ** 2 / (2 * sd ** 2)); PU /= PU.sum(1, keepdims=True)   # nz x nu, uniform grid
    else:
        PU = np.ones((nz, 1))
    # posterior weights in log space (1 - S computed as sig(-.) to avoid underflow)
    LA = np.log(np.maximum(WA, 1e-300))[None, :]
    lp1 = LA - np.logaddexp(0, -(eta[:, None] + A[None, :])); lp0 = LA - np.logaddexp(0, eta[:, None] + A[None, :])
    Pa1 = np.exp(lp1 - lp1.max(1, keepdims=True)); Pa1 /= Pa1.sum(1, keepdims=True)
    Pa0 = np.exp(lp0 - lp0.max(1, keepdims=True)); Pa0 /= Pa0.sum(1, keepdims=True)
    r1 = np.einsum("za,zu,aui->zi", Pa1, PU, Phi); r0 = np.einsum("za,zu,aui->zi", Pa0, PU, Phi)
    Hi = np.linalg.inv(H)
    Vs = m * B0 + m * (m - 1) * C
    Vpsi = Hi @ Vs @ Hi / m ** 2
    f1 = phiz * h / pi; f0 = phiz * (1 - h) / (1 - pi); f1p = phiz * (-z * h + hp) / pi
    q = np.einsum("zi,ij,zj->z", mu, Hi, mu)
    # C_mix pieces: own row and cluster mates
    own = (q * (0 - p) - q * (1 - p))                                    # x_i'H^-1 x_j (-p_j) - x_j' H^-1 x_i (1 - p_i)
    mate = (m - 1) * (np.einsum("zi,ij,zj->z", mu, Hi, r0) - np.einsum("zi,ij,zj->z", mu, Hi, r1))
    Cown = np.sum(wz * f1 * f0 * own) / t1 / m; Cmate = np.sum(wz * f1 * f0 * mate) / t1 / m
    Cmix = Cown + Cmate
    kap = np.einsum("zi,ij,zj->z", mu, Vpsi, mu); dkap = np.einsum("i,ij,zj->z", np.array([0.0, 1.0]), Vpsi, mu)
    gp = np.sum(wz * (f1p * f0 * kap + f1 * f0 * dkap)) / t1 ** 2
    c2 = -gp
    # split c2 into own (B0) and mate (C) parts
    Vown = Hi @ (m * B0) @ Hi / m ** 2; Vmate = Hi @ (m * (m - 1) * C) @ Hi / m ** 2
    c2p = lambda Vp: -np.sum(wz * (f1p * f0 * np.einsum("zi,ij,zj->z", mu, Vp, mu) + f1 * f0 * np.einsum("i,ij,zj->z", np.array([0.0, 1.0]), Vp, mu))) / t1 ** 2
    c2own, c2mate = c2p(Vown), c2p(Vmate)
    # AUC gradient along z-slope direction is 0 by construction (1-D score); the IBP identity's lambda(z):
    lam = hp / (h * (1 - h)) / t1
    out = dict(b0=b0, su=su, akind=akind, apar=apar, m=m, t0=t0, t1=t1, pi=pi, C_mix=Cmix, c2=c2, ratio=c2 / Cmix,
               C_own=Cown, C_mate=Cmate, c2_own=c2own, c2_mate=c2mate,
               ratio_own=c2own / Cown, ratio_mate=(c2mate / Cmate if Cmate != 0 else np.nan),
               lam_min=float(np.min(lam[(f1 * f0) > 1e-6 * (f1 * f0).max()])), lam_max=float(np.max(lam[(f1 * f0) > 1e-6 * (f1 * f0).max()])),
               max_abs_score_mean=float(np.max(np.abs(meanPhi))), Var_psi00=Vpsi[0, 0])
    if details:
        out.update(z=z, h=h, p=p, r0=r0, r1=r1, H=H, B0=B0, C=C, Vpsi=Vpsi, f1=f1, f0=f0, mu=mu, Hi=Hi, wz=wz)
    return out


CASES = ([dict(label=f"design {d}: linear, sa {sa:g}, su {su:g}", su=su, akind="gauss", apar=sa)
          for d, (sa, su) in enumerate([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.0), (2.0, 1.0)])]
         + [dict(label="two-point intercept +-3", akind="two", apar=3.0),
            dict(label="two-point intercept +-4", akind="two", apar=4.0),
            dict(label="Gaussian intercept, shared covariates (sa 1, su 3, m 400, b0 0)", su=3.0, akind="gauss", apar=1.0, m=400,
                 beta=(0.5, 0.25, 0.0), b0=0.0),
            dict(label="Gaussian intercept, shared covariates (sa 1, su 3, m 400, b0 -4)", su=3.0, akind="gauss", apar=1.0, m=400,
                 beta=(0.5, 0.25, 0.0), b0=-4.0),
            dict(label="Gaussian intercept, shared covariates (sa 1, su 2, m 40)", su=2.0, akind="gauss", apar=1.0),
            dict(label="bridge intercept, phi 0.9 (marginal model exactly logistic)", akind="bridge", apar=0.9)])


def main():
    import pandas as pd
    rows = []
    for c in CASES:
        kw = {k: v for k, v in c.items() if k != "label"}
        r = compute(**kw); rows.append(dict(case=c["label"], **{k: r[k] for k in ("b0", "su", "akind", "apar", "m", "C_mix", "c2", "ratio",
                                                                              "C_own", "C_mate", "ratio_own")}))
        print(f"{c['label']}: C_mix {r['C_mix']:+.6f}, c2 {r['c2']:+.6f}, ratio {r['ratio']:.5f}", flush=True)
    pd.DataFrame(rows).to_csv("results/auc_ratio_quadrature.csv", index=False)


if __name__ == "__main__":
    main()
