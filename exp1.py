"""
Experiment 2 -- the mechanism.

Holds the number of comparator switches (hence the path length P_T) FIXED and
varies only the *locality* of each switch: after a switch the latent target
moves by +/- delta.  Small delta = the previous concept is still partially
informative about the new one; large delta = the past is worthless.

If damping wins on gradual drift merely because P_T is small, this sweep should
show nothing.  If it wins because a finite memory horizon exploits a partially
informative past, the crossover should appear here, at constant P_T.
"""

import numpy as np, json
from exp_drift import (run_damped, run_fixed_share, run_restart, run_hedge,
                       run_adanormalhedge, run_two_layer)

RNG = np.random.default_rng


def make_locality_stream(T, d, m, delta, seed, sigma=0.15):
    rng = RNG(seed)
    x = np.linspace(0.0, 1.0, d)
    theta = np.empty(T)
    bounds = np.linspace(0, T, m + 1).astype(int)
    th = rng.uniform(0.2, 0.8)
    for k in range(m):
        theta[bounds[k]:bounds[k + 1]] = th
        step = delta * (1 if rng.random() < 0.5 else -1)
        th += step
        if th < 0: th = -th
        if th > 1: th = 2 - th
    mean = np.abs(x[None, :] - theta[:, None])
    loss = np.clip(mean + sigma * rng.standard_normal((T, d)), 0.0, 1.0)
    best = np.argmin(mean, axis=1)
    P_T = 2.0 * np.sum(best[1:] != best[:-1])
    return loss, best, P_T


def main():
    T, d, m, seeds = 10000, 30, 40, 5
    eta0 = np.sqrt(8.0 * np.log(d) / T)
    eta_scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    gamma_grid = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    alpha_grid = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    block_grid = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

    def cross(par, sc):
        A, B = np.meshgrid(par, sc, indexing="ij")
        return A.ravel(), B.ravel() * eta0

    gv, ge = cross(gamma_grid, eta_scales)
    av, ae = cross(alpha_grid, eta_scales)
    bv, be = cross(block_grid, eta_scales)

    deltas = [0.04, 0.06, 0.09, 0.13, 0.20, 0.30, 0.45, 0.70]
    rows = []
    for delta in deltas:
        acc = {k: [] for k in ["Damped MW", "Fixed-Share", "Hedge-restart",
                               "Hedge (static)", "AdaNormalHedge", "Two-layer ensemble"]}
        pts, gstar = [], []
        for sd in range(seeds):
            loss, best, P_T = make_locality_stream(T, d, m, delta, seed=31 * sd + 5)
            pts.append(P_T)
            r = run_damped(loss, best, gv, ge)
            acc["Damped MW"].append(r.min()); gstar.append(gv[int(np.argmin(r))])
            acc["Fixed-Share"].append(run_fixed_share(loss, best, av, ae).min())
            acc["Hedge-restart"].append(run_restart(loss, best, bv, be).min())
            acc["Hedge (static)"].append(run_hedge(loss, best, eta0))
            acc["AdaNormalHedge"].append(run_adanormalhedge(loss, best))
            acc["Two-layer ensemble"].append(run_two_layer(
                loss, best, gamma_grid,
                np.sqrt(8 * np.log(d) * np.maximum(gamma_grid, 1.0 / T))))
        row = {"delta": delta, "P_T": float(np.mean(pts)),
               "gamma_star": float(np.median(gstar))}
        for k, v in acc.items():
            row[k] = float(np.mean(v)); row[k + "_sd"] = float(np.std(v))
        rows.append(row)
        print(f"delta={delta:<5g} P_T={row['P_T']:5.0f} " +
              "  ".join(f"{k}={row[k]:7.1f}" for k in acc), flush=True)

    json.dump(rows, open("locality.json", "w"), indent=1)


if __name__ == "__main__":
    main()
