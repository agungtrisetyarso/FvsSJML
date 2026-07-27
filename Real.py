"""
Real-stream validation of the forget-vs-switch phase diagram.
Run in Colab:  !pip install river scipy scikit-posthocs -q

WHAT CHANGED FROM THE PREVIOUS RUN
----------------------------------
The previous script compared *classifiers* (HT/HAT/ARF/SRP/NB/LR) on two
streams. That measures which drift-adaptive classifier is best; it does not
test this paper's claim, which is about *meta-algorithms that combine experts*.

Here the base learners are held FIXED as a shared expert pool, and the thing
being compared is the meta-algorithm sitting on top: damped MW (ours),
Fixed-Share, Hedge-with-restarts, DWM-style decay, static Hedge, and a
two-layer ensemble.  Every meta-algorithm sees the identical expert losses, so
the comparison is controlled.

We additionally ESTIMATE P_T and V_T on each real stream from the experts'
windowed loss vectors, then test whether a stream's position in the (P_T, V_T)
plane predicts which family wins -- which is the actual validation of Fig. 3.
"""

import numpy as np, itertools, json, warnings
warnings.filterwarnings("ignore")

from river import datasets, tree, naive_bayes, linear_model, neighbors, preprocessing, optim
from river.datasets import synth

# ---------------------------------------------------------------- expert pool

class Restricted:
    """An expert that only sees one feature -- deliberately weak, so that which
    expert is best actually changes when the concept moves."""
    def __init__(self, key, model): self.k, self.m = key, model
    def _f(self, x): return {self.k: x[self.k]}
    def predict_one(self, x):
        return self.m.predict_one(self._f(x)) if self.k in x else None
    def learn_one(self, x, y):
        if self.k in x: self.m.learn_one(self._f(x), y)


def make_weak_experts(keys):
    """Canonical prediction-with-expert-advice pool: many weak specialists."""
    return {f"f:{k}": Restricted(k, naive_bayes.GaussianNB()) for k in keys}


def make_experts():
    """Fixed, diverse pool. NOTE the LogisticRegression construction: River's
    first positional argument is `optimizer`, and optimizers must be optim
    objects, not strings. Passing a string is what produced
    `'str' object has no attribute 'look_ahead'` in the previous run."""
    return {
        "HT":    tree.HoeffdingTreeClassifier(),
        "HAT":   tree.HoeffdingAdaptiveTreeClassifier(seed=0),
        "NB":    naive_bayes.GaussianNB(),
        "LR":    preprocessing.StandardScaler() | linear_model.LogisticRegression(
                     optimizer=optim.SGD(0.01)),          # <-- correct form
        "LR-fast": preprocessing.StandardScaler() | linear_model.LogisticRegression(
                     optimizer=optim.SGD(0.1)),
        "kNN":   preprocessing.StandardScaler() | neighbors.KNNClassifier(n_neighbors=5),
    }

# ------------------------------------------------------------ meta-algorithms

class Meta:
    def __init__(self, d): self.d = d
    def weights(self): raise NotImplementedError
    def update(self, loss): raise NotImplementedError

class DampedMW(Meta):
    name = "Damped MW"
    def __init__(self, d, gamma, eta):
        super().__init__(d); self.g, self.e = gamma, eta; self.L = np.zeros(d)
    def weights(self):
        z = -self.e * self.L; z -= z.max(); w = np.exp(z); return w / w.sum()
    def update(self, loss): self.L = (1 - self.g) * self.L + loss

class FixedShare(Meta):
    name = "Fixed-Share"
    def __init__(self, d, alpha, eta):
        super().__init__(d); self.a, self.e = alpha, eta; self.w = np.full(d, 1 / d)
    def weights(self): return self.w / self.w.sum()
    def update(self, loss):
        w = self.weights() * np.exp(-self.e * loss); w /= w.sum()
        self.w = (1 - self.a) * w + self.a / self.d

class HedgeRestart(Meta):
    name = "Hedge-restart"
    def __init__(self, d, B, eta):
        super().__init__(d); self.B, self.e = B, eta; self.L = np.zeros(d); self.t = 0
    def weights(self):
        z = -self.e * self.L; z -= z.max(); w = np.exp(z); return w / w.sum()
    def update(self, loss):
        self.t += 1
        if self.t % self.B == 0: self.L = np.zeros(self.d)
        else: self.L += loss

class Hedge(DampedMW):
    name = "Hedge (static)"
    def __init__(self, d, eta): super().__init__(d, 0.0, eta)

class DWMDecay(Meta):
    """Kolter & Maloof-style multiplicative decay on the weights themselves
    (rather than on cumulative loss). The direct ancestor of our method."""
    name = "DWM-decay"
    def __init__(self, d, beta):
        super().__init__(d); self.b = beta; self.w = np.ones(d)
    def weights(self): return self.w / self.w.sum()
    def update(self, loss):
        self.w *= self.b ** loss
        self.w = np.maximum(self.w / self.w.max(), 1e-8)

class TwoLayer(Meta):
    name = "Two-layer ensemble"
    def __init__(self, d, gammas, etas, eta_meta):
        super().__init__(d)
        self.base = [DampedMW(d, g, e) for g, e in zip(gammas, etas)]
        self.meta = np.zeros(len(self.base)); self.em = eta_meta
    def weights(self):
        z = -self.em * self.meta; z -= z.max(); q = np.exp(z); q /= q.sum()
        return sum(qi * b.weights() for qi, b in zip(q, self.base))
    def update(self, loss):
        for i, b in enumerate(self.base):
            self.meta[i] += b.weights() @ loss
            b.update(loss)

# ------------------------------------------------------------------ the run

def run_stream(stream, n_max, seed=0):
    experts = make_experts()
    names = list(experts)
    d = len(names)
    eta0 = np.sqrt(8 * np.log(d) / n_max)

    metas = []
    for g in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        for s in [1.0, 4.0]:
            metas.append(DampedMW(d, g, s * eta0))
    for a in [1e-4, 1e-3, 1e-2]:
        for s in [1.0, 4.0]:
            metas.append(FixedShare(d, a, s * eta0))
    for B in [200, 1000, 5000]:
        for s in [1.0, 4.0]:
            metas.append(HedgeRestart(d, B, s * eta0))
    for b in [0.9, 0.99, 0.999]:
        metas.append(DWMDecay(d, b))
    metas.append(Hedge(d, eta0))
    gg = np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    metas.append(TwoLayer(d, gg, np.sqrt(8 * np.log(d) * gg), np.sqrt(8 * np.log(6) / n_max)))

    correct = np.zeros(len(metas))
    mixloss = np.zeros(len(metas))   # <p_t, l_t>: the quantity the theory bounds
    expert_correct = np.zeros(d)
    loss_hist = []
    classes = set()

    for t, (x, y) in enumerate(stream):
        if t >= n_max: break
        classes.add(y)
        preds, loss = [], np.zeros(d)
        for i, nm in enumerate(names):
            p = experts[nm].predict_one(x)
            preds.append(p)
            loss[i] = 0.0 if p == y else 1.0
        expert_correct += 1 - loss
        loss_hist.append(loss)

        for m, meta in enumerate(metas):
            w = meta.weights()
            mixloss[m] += w @ loss
            score = {}
            for i, p in enumerate(preds):
                if p is not None: score[p] = score.get(p, 0.0) + w[i]
            if score and max(score, key=score.get) == y: correct[m] += 1

        for meta in metas: meta.update(loss)
        for nm in names: experts[nm].learn_one(x, y)

    n = min(t + 1, n_max)
    L = np.array(loss_hist)

    # --- estimate P_T and V_T from windowed expert losses ------------------
    W = 500
    if len(L) > 2 * W:
        k = np.ones(W) / W
        f = np.vstack([np.convolve(L[:, i], k, mode="valid") for i in range(d)]).T
        best = np.argmin(f, axis=1)
        P_T = float(2 * np.sum(best[1:] != best[:-1]))
        V_T = float(np.abs(np.diff(f, axis=0)).max(axis=1).sum())
    else:
        P_T = V_T = float("nan")

    acc, mix = {}, {}
    for m, meta in enumerate(metas):
        acc.setdefault(meta.name, []).append(correct[m] / n)
        mix.setdefault(meta.name, []).append(mixloss[m] / n)
    best_acc = {k: max(v) for k, v in acc.items()}          # oracle-tuned
    best_mix = {k: min(v) for k, v in mix.items()}
    best_acc["best single expert"] = float(expert_correct.max() / n)
    best_mix["best single expert"] = float(L.mean(axis=0).min()) if len(L) else float("nan")
    return best_acc, best_mix, P_T, V_T, n


def chain(a, b, switch):
    """True abrupt drift. River's ConceptDriftStream computes a sigmoid
    exp(-4(t-position)/width) and raises OverflowError for narrow widths, so
    small-width abrupt streams cannot be produced with it."""
    for t, xy in enumerate(a):
        if t >= switch: break
        yield xy
    for xy in b:
        yield xy


def stream_suite():
    """>= 10 streams: Friedman/Nemenyi over 2 datasets is not interpretable."""
    S = {}
    # abrupt: direct concatenation.  gradual: ConceptDriftStream with wide width.
    S["SEA-abrupt"] = lambda: chain(synth.SEA(variant=0, seed=1),
                                    synth.SEA(variant=3, seed=1), 10000)
    S["AGRAWAL-abrupt"] = lambda: chain(synth.Agrawal(classification_function=0, seed=1),
                                        synth.Agrawal(classification_function=4, seed=1), 10000)
    for w, tag in [(2000, "moderate"), (15000, "gradual")]:
        S[f"SEA-{tag}"] = synth.ConceptDriftStream(
            stream=synth.SEA(variant=0, seed=1), drift_stream=synth.SEA(variant=3, seed=1),
            position=10000, width=w, seed=1)
        S[f"AGRAWAL-{tag}"] = synth.ConceptDriftStream(
            stream=synth.Agrawal(classification_function=0, seed=1),
            drift_stream=synth.Agrawal(classification_function=4, seed=1),
            position=10000, width=w, seed=1)
    for mag in [0.001, 0.01, 0.1]:
        S[f"Hyperplane-{mag}"] = synth.Hyperplane(seed=1, n_features=10,
                                                  n_drift_features=5, mag_change=mag)
    S["Elec2"] = datasets.Elec2()
    # Insects: River pulls these from Google Drive links that expire -- the 404s
    # in the previous run were that, not a bug in your code. Re-enable if they work.
    for v in ["abrupt_balanced", "gradual_balanced", "incremental_balanced"]:
        try:
            ds = datasets.Insects(variant=v); ds.download()
            S[f"Insects-{v}"] = ds
        except Exception as e:
            print(f"  [skip] Insects/{v}: {e}")
    return S


if __name__ == "__main__":
    N_MAX = 20000
    rows = []
    for name, st in stream_suite().items():
        try:
            if callable(st): st = st()      # abrupt streams are lazy generators
            acc, mix, P_T, V_T, n = run_stream(st, N_MAX)
            row = {"stream": name, "n": n, "P_T": P_T, "V_T": V_T,
                   **{f"acc:{k}": v for k, v in acc.items()},
                   **{f"loss:{k}": v for k, v in mix.items()}}
            rows.append(row)
            print(f"{name:22s} n={n:6d} P_T={P_T:7.0f} V_T={V_T:7.2f} " +
                  " ".join(f"{k}={v:.4f}" for k, v in mix.items()), flush=True)
        except Exception as e:
            print(f"{name:22s} FAILED: {type(e).__name__}: {e}", flush=True)
    json.dump(rows, open("real_streams.json", "w"), indent=1)

    # --- Friedman + Nemenyi, only if enough streams -------------------------
    import pandas as pd
    df = pd.DataFrame(rows).set_index("stream")
    meth = [c for c in df.columns if c.startswith("loss:")]
    print(f"\n{len(df)} streams x {len(meth)} methods")
    if len(df) < 10:
        print("WARNING: Friedman/Nemenyi over <10 datasets is not interpretable "
              "(Demsar 2006). Add streams before reporting a CD diagram.")
    else:
        from scipy.stats import friedmanchisquare
        stat, p = friedmanchisquare(*[df[m].values for m in meth])
        print(f"Friedman chi2={stat:.3f}  p={p:.4g}")
        try:
            import scikit_posthocs as sp
            print(sp.posthoc_nemenyi_friedman(df[meth].values).round(4))
        except ImportError:
            print("pip install scikit-posthocs for the Nemenyi matrix")
    print("\nAverage ranks (1 = best):")
    print(df[meth].rank(axis=1, ascending=True).mean().sort_values())
    df.to_csv("real_streams.csv")
