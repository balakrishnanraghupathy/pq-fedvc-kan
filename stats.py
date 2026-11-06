"""Corrected resampled t-test of Nadeau and Bengio (Eq. 18) with Holm adjustment, proposed model vs. every baseline."""
import os, json, numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("PQFEDVCKAN_RESULTS", os.path.join(ROOT, "results"))


def reps(g, m, k="auc", n=5): return np.array([json.load(open(f"{R}/{g}/{m}_r{r}.json"))[k] for r in range(n)])


def corrected_t(a, b, ratio):
    """ratio = n_test / n_train (0.25 for an 80/20 split)."""
    d = a - b; k = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + ratio) * v) if v > 0 else np.inf
    return d.mean(), t, 2 * stats.t.sf(abs(t), k - 1)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for i, j in enumerate(o): run = max(run, min(1, (m - i) * ps[j])); adj[j] = run
    return adj


if __name__ == "__main__":
    out = {}
    for g, base in [("fed", ["LR", "MLP", "VQC", "Fed-KAN-pub"]), ("uci_fed", ["LR", "MLP", "VQC", "Fed-KAN-pub", "KACQ-DCNN"]),
                    ("uci_central", ["LR", "MLP", "VQC", "Fed-KAN-pub", "KACQ-DCNN"])]:
        for met in ["auc", "bacc", "mcc"]:
            rows = [corrected_t(reps(g, "VC-KAN", met), reps(g, b, met), 0.25) for b in base]
            adj = holm(np.array([r[2] for r in rows]))
            out[f"{g}_{met}"] = {b: dict(diff=float(r[0]), t=float(r[1]), p=float(r[2]), p_holm=float(a)) for b, r, a in zip(base, rows, adj)}
            print(g, met, {b: (round(r[0], 4), round(float(a), 4)) for b, r, a in zip(base, rows, adj)})
    json.dump(out, open(f"{R}/stats.json", "w"), indent=1)
