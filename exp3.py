"""
Does function variation V_T (rather than path length P_T) govern damped MW?

V_T = sum_t || f_t - f_{t-1} ||_inf , f_t = expected loss vector at round t.
P_T = sum_t || u_t - u_{t-1} ||_1  , u_t = e_{argmin f_t}.

We recompute both complexity measures for every stream used in Experiments 1
and 2, and regress log(regret) on log(V_T) and log(P_T) for each method.
"""

import numpy as np, json
from exp_drift import make_stream
from exp_locality import make_locality_stream

RNG = np.random.default_rng


def measures(T, d, theta_getter, seed):
    x = np.linspace(0.0, 1.0, d)
    loss, best, P_T = theta_getter(seed)
    return loss, best, P_T


def variation_from_stream(loss, best, d, mean_fn):
    raise NotImplementedError


def recompute(kind, arg, seed, T=10000, d=30, sigma=0.15):
    """Rebuild the expected-loss path for a stream and return (P_T, V_T)."""
    rng = RNG(seed)
    x = np.linspace(0.0, 1.0, d)
    if kind == "gradual":
        th = rng.uniform(); theta = np.empty(T)
        for t in range(T):
            th += arg * rng.standard_normal()
            if th < 0: th = -th
            if th > 1: th = 2 - th
            theta[t] = th
    elif kind == "abrupt":
        K = max(1, int(arg)); theta = np.empty(T)
        cuts = np.sort(rng.choice(np.arange(1, T), size=K - 1, replace=False)) if K > 1 else np.array([], int)
        b = np.concatenate(([0], cuts, [T])); vals = rng.uniform(size=K)
        for k in range(K):
            theta[b[k]:b[k + 1]] = vals[k]
    else:  # locality
        m = 40; theta = np.empty(T)
        b = np.linspace(0, T, m + 1).astype(int)
        th = rng.uniform(0.2, 0.8)
        for k in range(m):
            theta[b[k]:b[k + 1]] = th
            th += arg * (1 if rng.random() < 0.5 else -1)
            if th < 0: th = -th
            if th > 1: th = 2 - th
    f = np.abs(x[None, :] - theta[:, None])
    V_T = float(np.abs(np.diff(f, axis=0)).max(axis=1).sum())
    best = np.argmin(f, axis=1)
    P_T = float(2.0 * np.sum(best[1:] != best[:-1]))
    return P_T, V_T


def main():
    out = []
    for kind, args, seeds in [
        ("gradual", [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [1000 * s + 7 for s in range(5)]),
        ("abrupt", [2, 5, 10, 25, 50, 100], [1000 * s + 7 for s in range(5)]),
        ("locality", [0.04, 0.06, 0.09, 0.13, 0.20, 0.30, 0.45, 0.70], [31 * s + 5 for s in range(5)]),
    ]:
        for a in args:
            P = []; V = []
            for sd in seeds:
                p, v = recompute(kind, a, sd)
                P.append(p); V.append(v)
            out.append({"kind": kind, "arg": a, "P_T": float(np.mean(P)), "V_T": float(np.mean(V))})
            print(f"{kind:9s} arg={a:<7g} P_T={np.mean(P):7.0f}  V_T={np.mean(V):9.1f}")

    json.dump(out, open("variation.json", "w"), indent=1)

    # --- regressions -------------------------------------------------------
    r1 = json.load(open("results.json"))
    r2 = json.load(open("locality.json"))
    key = {}
    for row in out:
        key[(row["kind"], row["arg"])] = row
    recs = []
    for row in r1:
        k = key[(row["regime"], row["drift"])]
        recs.append((k["P_T"], k["V_T"], row))
    for row in r2:
        k = key[("locality", row["delta"])]
        recs.append((k["P_T"], k["V_T"], row))

    print("\nlog-log regressions over all 20 configurations (P_T,V_T > 0):")
    print(f"{'method':22s} {'R2 on log P_T':>14s} {'R2 on log V_T':>14s} {'R2 both':>9s}")
    for m in ["Damped MW", "Fixed-Share", "Hedge-restart", "Two-layer ensemble"]:
        P = np.array([p for p, v, r in recs]); V = np.array([v for p, v, r in recs])
        Y = np.array([r[m] for p, v, r in recs])
        ok = (P > 0) & (V > 0) & (Y > 0)
        lp, lv, ly = np.log(P[ok]), np.log(V[ok]), np.log(Y[ok])

        def r2(X):
            X = np.column_stack([np.ones(len(ly))] + X)
            beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
            res = ly - X @ beta
            return 1 - res @ res / ((ly - ly.mean()) @ (ly - ly.mean())), beta

        a, _ = r2([lp]); b, bb = r2([lv]); c, _ = r2([lp, lv])
        print(f"{m:22s} {a:14.3f} {b:14.3f} {c:9.3f}   (V_T exponent {bb[1]:.2f})")


if __name__ == "__main__":
    main()
