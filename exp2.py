"""
Experiment 3 -- phase diagram.

Independently sweep the number of switches m (which sets P_T = 2m) and the
per-switch displacement delta (which sets V_T).  For each cell record the
oracle-tuned regret of the forgetting method (damped MW) and of the switching
methods (Fixed-Share, Hedge-restart), and locate the boundary where the winner
changes.
"""

import numpy as np, json
from exp_drift import run_damped, run_fixed_share, run_restart
from exp_locality import make_locality_stream

RNG = np.random.default_rng


def stream(T, d, m, delta, seed, sigma=0.15):
    rng = RNG(seed)
    x = np.linspace(0.0, 1.0, d)
    theta = np.empty(T)
    b = np.linspace(0, T, m + 1).astype(int)
    th = rng.uniform(0.2, 0.8)
    for k in range(m):
        theta[b[k]:b[k + 1]] = th
        th += delta * (1 if rng.random() < 0.5 else -1)
        if th < 0: th = -th
        if th > 1: th = 2 - th
    f = np.abs(x[None, :] - theta[:, None])
    loss = np.clip(f + sigma * rng.standard_normal((T, d)), 0.0, 1.0)
    best = np.argmin(f, axis=1)
    P_T = 2.0 * np.sum(best[1:] != best[:-1])
    V_T = float(np.abs(np.diff(f, axis=0)).max(axis=1).sum())
    return loss, best, P_T, V_T


def main():
    T, d, seeds = 10000, 30, 3
    eta0 = np.sqrt(8.0 * np.log(d) / T)
    sc = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    gg = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    ag = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    bg = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

    def cross(p, s):
        A, B = np.meshgrid(p, s, indexing="ij")
        return A.ravel(), B.ravel() * eta0

    gv, ge = cross(gg, sc); av, ae = cross(ag, sc); bv, be = cross(bg, sc)

    ms = [10, 20, 40, 80, 160]
    ds = [0.04, 0.09, 0.20, 0.45, 0.70]
    rows = []
    for m in ms:
        for delta in ds:
            D, F, R, P, V = [], [], [], [], []
            for sd in range(seeds):
                loss, best, P_T, V_T = stream(T, d, m, delta, seed=101 * sd + 3 * m + int(1000 * delta))
                D.append(run_damped(loss, best, gv, ge).min())
                F.append(run_fixed_share(loss, best, av, ae).min())
                R.append(run_restart(loss, best, bv, be).min())
                P.append(P_T); V.append(V_T)
            row = {"m": m, "delta": delta, "P_T": float(np.mean(P)), "V_T": float(np.mean(V)),
                   "damped": float(np.mean(D)), "fixed_share": float(np.mean(F)),
                   "restart": float(np.mean(R))}
            row["switch_best"] = min(row["fixed_share"], row["restart"])
            row["ratio"] = row["damped"] / row["switch_best"]
            rows.append(row)
            print(f"m={m:4d} d={delta:<5g} P_T={row['P_T']:5.0f} V_T={row['V_T']:7.2f} "
                  f"damped={row['damped']:7.1f} switch={row['switch_best']:7.1f} "
                  f"ratio={row['ratio']:5.2f} {'FORGET' if row['ratio'] < 1 else 'SWITCH'}",
                  flush=True)
    json.dump(rows, open("phase.json", "w"), indent=1)

    r = np.array([x["ratio"] for x in rows])
    vp = np.array([x["V_T"] / max(x["P_T"], 1) for x in rows])
    print("\nratio vs V_T/P_T, log-log fit:")
    A = np.column_stack([np.ones(len(r)), np.log(vp)])
    beta, *_ = np.linalg.lstsq(A, np.log(r), rcond=None)
    res = np.log(r) - A @ beta
    R2 = 1 - res @ res / ((np.log(r) - np.log(r).mean()) @ (np.log(r) - np.log(r).mean()))
    print(f"  log ratio = {beta[0]:.3f} + {beta[1]:.3f} log(V_T/P_T),  R2 = {R2:.3f}")
    print(f"  crossover at V_T/P_T = {np.exp(-beta[0]/beta[1]):.4f}")


if __name__ == "__main__":
    main()
