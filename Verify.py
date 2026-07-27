"""
Numerical verification of the proposed theorems for damped MW, d = 2.

Construction.  Two experts.  A sign sequence sigma_t in {+1,-1} is piecewise
constant with m' flips at equal spacing.  Losses are

    l_t(1) = 1/2 - eps * sigma_t ,   l_t(2) = 1/2 + eps * sigma_t ,  eps in (0,1/2]

so the loss gap is g_t = l_t(1) - l_t(2) = -2 eps sigma_t, the comparator is the
currently better expert, and

    P_T = 2 m'        (independent of eps)
    V_T = 2 eps m'    (proportional to eps)

Damped MW keeps D_t = beta D_{t-1} + g_t with beta = 1-gamma, and plays
p_t(1) = sigmoid(-eta D_{t-1}).

CLAIMS TO CHECK
  (K)  recovery time after a flip is k* = ceil( ln 2 / ln(1/beta) ), independent
       of eps and of eta
  (T2) for eta -> infinity, R_T = 2 eps m' k*  exactly, i.e. R_T = V_T * k*
  (T1) for every finite eta, R_T >= (eps) * m' * k*   [ = (a/2) m' k*, a = 2 eps ]
  (S)  at fixed P_T, R_T / (2 eps T) -> 0 as eps -> 0 while the eps = 1/2
       instance keeps R_T >= m' k* / 2 : the ratio is unbounded
"""

import numpy as np


def run(eps, gamma, m_flips, seg_len, eta):
    """Returns (dynamic regret, P_T, V_T)."""
    beta = 1.0 - gamma
    T = (m_flips + 1) * seg_len
    sign = np.repeat(np.arange(m_flips + 1) % 2 * 2 - 1, seg_len)  # +-1, m' flips
    l1 = 0.5 - eps * sign
    l2 = 0.5 + eps * sign
    g = l1 - l2                                   # = -2 eps sign
    D = 0.0
    reg = 0.0
    for t in range(T):
        if np.isinf(eta):
            p1 = 1.0 if D < 0 else (0.0 if D > 0 else 0.5)
        else:
            p1 = 1.0 / (1.0 + np.exp(np.clip(eta * D, -500, 500)))
        best = 1 if l1[t] <= l2[t] else 2
        mix = p1 * l1[t] + (1 - p1) * l2[t]
        reg += mix - (l1[t] if best == 1 else l2[t])
        D = beta * D + g[t]
    best_idx = np.where(l1 <= l2, 1, 2)
    P_T = 2.0 * np.sum(best_idx[1:] != best_idx[:-1])
    f = np.column_stack([l1, l2])
    V_T = float(np.abs(np.diff(f, axis=0)).max(axis=1).sum())
    return reg, P_T, V_T


def kstar(gamma):
    return int(np.ceil(np.log(2.0) / np.log(1.0 / (1.0 - gamma))))


print("=" * 74)
print("(K) and (T2): eta = infinity,  predicted R_T = 2 eps m' k*")
print("=" * 74)
print(f"{'gamma':>7} {'k*':>4} {'eps':>6} {'m_flips':>8} {'R_T':>10} {'2 eps m k*':>11} {'V_T*k*':>9} {'ratio':>7}")
ok = True
for gamma in [0.01, 0.02, 0.05, 0.1]:
    ks = kstar(gamma)
    for eps in [0.5, 0.1, 0.02]:
        for mf in [4, 10]:
            seg = max(400, int(20 / gamma))
            R, P, V = run(eps, gamma, mf, seg, np.inf)
            pred = 2 * eps * mf * ks
            r = R / pred
            ok &= abs(r - 1) < 0.02
            print(f"{gamma:7.3f} {ks:4d} {eps:6.2f} {mf:8d} {R:10.4f} {pred:11.4f} {V*ks:9.3f} {r:7.4f}")
print("closed form matches to <2% in every cell:", ok)

print()
print("=" * 74)
print("(T1): lower bound R_T >= eps * m' * k*  for FINITE eta")
print("=" * 74)
print(f"{'gamma':>7} {'eps':>6} {'eta':>8} {'R_T':>10} {'bound':>10} {'holds':>6}")
allhold = True
for gamma in [0.02, 0.05]:
    ks = kstar(gamma)
    for eps in [0.5, 0.1]:
        for eta in [0.5, 2.0, 10.0, 50.0]:
            R, P, V = run(eps, gamma, 10, max(400, int(20 / gamma)), eta)
            bound = eps * 10 * ks
            h = R >= bound
            allhold &= h
            print(f"{gamma:7.3f} {eps:6.2f} {eta:8.1f} {R:10.4f} {bound:10.4f} {str(h):>6}")
print("lower bound holds in every cell:", allhold)

print()
print("=" * 74)
print("(S) separation at FIXED P_T: vary eps only")
print("=" * 74)
gamma, mf, eta = 0.02, 10, 20.0
seg = 1000
ks = kstar(gamma)
print(f"gamma={gamma}  k*={ks}  m'={mf}  T={(mf+1)*seg}   P_T is the same in every row")
print(f"{'eps':>8} {'P_T':>6} {'V_T':>9} {'R_T':>10} {'R_T/V_T':>9} {'R(0.5)/R(eps)':>14}")
base = None
for eps in [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]:
    R, P, V = run(eps, gamma, mf, seg, eta)
    if base is None: base = R
    print(f"{eps:8.3f} {P:6.0f} {V:9.3f} {R:10.4f} {R/V:9.3f} {base/R:14.1f}")
print()
print("P_T identical in every row; R_T spans the column above.")
print("=> no function of (P_T, T) can bound R_T within a universal constant.")
