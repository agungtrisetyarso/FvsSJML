"""
Damped multiplicative weights vs. non-stationary expert baselines.

Setting: prediction with expert advice, d experts placed on [0,1].
A latent target theta_t moves over time; expert i suffers
    loss_t(i) = clip(|x_i - theta_t| + N(0,sigma), 0, 1).
Comparator sequence u_t = e_{i*(t)} with i*(t) = argmin_i |x_i - theta_t|
(the *expected*-loss minimiser, so the comparator is not noise-chasing).
Path length P_T = sum_t ||u_t - u_{t-1}||_1 = 2 * (#switches).

Regimes:
  gradual : theta follows a reflected random walk with step s  -> many small switches
  abrupt  : theta is piecewise constant with K jumps           -> few large switches

All parameterised methods are ORACLE-TUNED per configuration (best over a grid),
which is deliberately generous to the baselines.
"""

import numpy as np
import json, time

RNG = np.random.default_rng

# ---------------------------------------------------------------- environment

def make_stream(T, d, regime, drift, seed, sigma=0.15):
    rng = RNG(seed)
    x = np.linspace(0.0, 1.0, d)
    theta = np.empty(T)
    if regime == "gradual":
        th = rng.uniform()
        for t in range(T):
            th += drift * rng.standard_normal()
            if th < 0: th = -th
            if th > 1: th = 2 - th
            theta[t] = th
    elif regime == "abrupt":
        K = max(1, int(drift))
        cuts = np.sort(rng.choice(np.arange(1, T), size=K - 1, replace=False)) if K > 1 else np.array([], int)
        bounds = np.concatenate(([0], cuts, [T]))
        vals = rng.uniform(size=K)
        for k in range(K):
            theta[bounds[k]:bounds[k + 1]] = vals[k]
    else:
        raise ValueError(regime)

    mean = np.abs(x[None, :] - theta[:, None])          # (T,d) expected losses
    loss = np.clip(mean + sigma * rng.standard_normal((T, d)), 0.0, 1.0)
    best = np.argmin(mean, axis=1)                       # comparator indices
    P_T = 2.0 * np.sum(best[1:] != best[:-1])
    return loss, best, P_T


# ---------------------------------------------------------------- algorithms
# Each returns cumulative dynamic regret for a *batch* of parameter settings.

def run_damped(loss, best, gammas, etas):
    """L_t = (1-g) L_{t-1} + l_t ;  p propto exp(-eta L_t).  Memory horizon 1/g."""
    T, d = loss.shape
    P = len(gammas)
    L = np.zeros((P, d))
    reg = np.zeros(P)
    g = gammas[:, None]; e = etas[:, None]
    for t in range(T):
        z = -e * L
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        lt = loss[t]
        reg += p @ lt - lt[best[t]]
        L = (1.0 - g) * L + lt[None, :]
    return reg


def run_fixed_share(loss, best, alphas, etas):
    T, d = loss.shape
    P = len(alphas)
    w = np.full((P, d), 1.0 / d)
    reg = np.zeros(P)
    a = alphas[:, None]; e = etas[:, None]
    for t in range(T):
        p = w / w.sum(axis=1, keepdims=True)
        lt = loss[t]
        reg += p @ lt - lt[best[t]]
        w = p * np.exp(-e * lt[None, :])
        w /= w.sum(axis=1, keepdims=True)
        w = (1.0 - a) * w + a / d
    return reg


def run_restart(loss, best, blocks, etas):
    T, d = loss.shape
    P = len(blocks)
    L = np.zeros((P, d))
    reg = np.zeros(P)
    e = etas[:, None]
    for t in range(T):
        reset = (t % blocks) == 0
        L[reset] = 0.0
        z = -e * L
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        lt = loss[t]
        reg += p @ lt - lt[best[t]]
        L += lt[None, :]
    return reg


def run_hedge(loss, best, eta):
    T, d = loss.shape
    L = np.zeros(d); reg = 0.0
    for t in range(T):
        z = -eta * L; z -= z.max()
        p = np.exp(z); p /= p.sum()
        lt = loss[t]
        reg += p @ lt - lt[best[t]]
        L += lt
    return reg


def run_adanormalhedge(loss, best):
    """Luo & Schapire (2015), parameter-free."""
    T, d = loss.shape
    R = np.zeros(d); C = np.zeros(d); reg = 0.0

    def phi(r, c):
        rp = np.maximum(r, 0.0)
        return np.exp(rp * rp / (3.0 * np.maximum(c, 1e-12)))

    for t in range(T):
        w = 0.5 * (phi(R + 1.0, C + 1.0) - phi(R - 1.0, C + 1.0))
        w = np.maximum(w, 0.0)
        s = w.sum()
        p = np.full(d, 1.0 / d) if s <= 0 else w / s
        lt = loss[t]
        mix = p @ lt
        reg += mix - lt[best[t]]
        r = mix - lt
        R += r; C += np.abs(r)
    return reg


def run_two_layer(loss, best, gammas, etas):
    """Ader-style two-layer ensemble: base = damped MW over a grid of horizons,
    meta = Hedge over base learners.  Cost: K base learners, K*d memory."""
    T, d = loss.shape
    K = len(gammas)
    L = np.zeros((K, d))
    meta = np.zeros(K)
    eta_meta = np.sqrt(8.0 * np.log(max(K, 2)) / T)
    reg = 0.0
    g = gammas[:, None]; e = etas[:, None]
    for t in range(T):
        z = -e * L
        z -= z.max(axis=1, keepdims=True)
        pb = np.exp(z); pb /= pb.sum(axis=1, keepdims=True)     # (K,d)
        zm = -eta_meta * meta; zm -= zm.max()
        q = np.exp(zm); q /= q.sum()
        p = q @ pb
        lt = loss[t]
        reg += p @ lt - lt[best[t]]
        meta += pb @ lt
        L = (1.0 - g) * L + lt[None, :]
    return reg


# ---------------------------------------------------------------- experiment

def main():
    T, d, seeds = 10000, 30, 5
    eta_scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    eta0 = np.sqrt(8.0 * np.log(d) / T)

    gamma_grid = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    alpha_grid = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    block_grid = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

    def cross(par, sc):
        A, B = np.meshgrid(par, sc, indexing="ij")
        return A.ravel(), B.ravel() * eta0

    gam_v, gam_e = cross(gamma_grid, eta_scales)
    alp_v, alp_e = cross(alpha_grid, eta_scales)
    blk_v, blk_e = cross(block_grid, eta_scales)

    configs = [("gradual", s) for s in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]] + \
              [("abrupt", k) for k in [2, 5, 10, 25, 50, 100]]

    rows = []
    for regime, drift in configs:
        acc = {k: [] for k in ["Damped MW", "Fixed-Share", "Hedge-restart",
                               "Hedge (static)", "AdaNormalHedge", "Two-layer ensemble"]}
        best_gamma, pts = [], []
        for sd in range(seeds):
            loss, best, P_T = make_stream(T, d, regime, drift, seed=1000 * sd + 7)
            pts.append(P_T)

            r = run_damped(loss, best, gam_v, gam_e)
            acc["Damped MW"].append(r.min())
            best_gamma.append(gam_v[int(np.argmin(r))])

            acc["Fixed-Share"].append(run_fixed_share(loss, best, alp_v, alp_e).min())
            acc["Hedge-restart"].append(run_restart(loss, best, blk_v, blk_e).min())
            acc["Hedge (static)"].append(run_hedge(loss, best, eta0))
            acc["AdaNormalHedge"].append(run_adanormalhedge(loss, best))
            acc["Two-layer ensemble"].append(
                run_two_layer(loss, best, gamma_grid,
                              np.sqrt(8.0 * np.log(d) * np.maximum(gamma_grid, 1.0 / T))))

        row = {"regime": regime, "drift": drift,
               "P_T": float(np.mean(pts)),
               "gamma_star": float(np.median(best_gamma))}
        for k, v in acc.items():
            row[k] = float(np.mean(v))
            row[k + "_sd"] = float(np.std(v))
        rows.append(row)
        print(f"{regime:8s} drift={drift:<8g} P_T={row['P_T']:7.0f}  " +
              "  ".join(f"{k}={row[k]:7.1f}" for k in acc), flush=True)

    with open("results.json", "w") as f:
        json.dump(rows, f, indent=1)

    # ---- cost measurement (per-round wall clock, single seed) ----
    loss, best, _ = make_stream(2000, d, "gradual", 1e-3, seed=0)
    cost = {}
    t0 = time.perf_counter(); run_damped(loss, best, np.array([1e-2]), np.array([eta0])); cost["Damped MW"] = time.perf_counter() - t0
    t0 = time.perf_counter(); run_fixed_share(loss, best, np.array([1e-3]), np.array([eta0])); cost["Fixed-Share"] = time.perf_counter() - t0
    t0 = time.perf_counter(); run_adanormalhedge(loss, best); cost["AdaNormalHedge"] = time.perf_counter() - t0
    t0 = time.perf_counter(); run_two_layer(loss, best, gamma_grid, np.sqrt(8*np.log(d)*np.maximum(gamma_grid, 1/2000))); cost["Two-layer ensemble"] = time.perf_counter() - t0
    with open("cost.json", "w") as f:
        json.dump(cost, f, indent=1)
    print(cost)


if __name__ == "__main__":
    main()
