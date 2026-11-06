"""Mean and SD of every metric per experiment group, written to results/summary.json."""
import os, re, json, glob, numpy as np
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("PQFEDVCKAN_RESULTS", os.path.join(ROOT, "results"))
K = ["auc", "prauc", "brier", "sens", "spec", "bacc", "acc", "prec", "f1", "mcc"]


def load(g):
    d = defaultdict(list)
    for f in sorted(glob.glob(f"{R}/{g}/*.json")):
        d[re.sub(r"_r\d+$", "", os.path.basename(f)[:-5])].append(json.load(open(f)))
    return d


def summ(d):
    return {n: {k: (float(np.mean([r[k] for r in L])), float(np.std([r[k] for r in L], ddof=1)) if len(L) > 1 else 0.0) for k in K}
               | {"n": len(L), "params": L[0].get("params"), "t_round": float(np.mean([r.get("t_train_round", 0) for r in L]))}
            for n, L in d.items()}


if __name__ == "__main__":
    out = {}
    for g in ["fed", "uci_fed", "uci_central", "ablation", "dp", "poison", "hetero"]:
        out[g] = summ(load(g)); print("==", g)
        for n, s in out[g].items():
            print(f"{n:18s} n={s['n']} P={s['params']} " + " ".join(f"{k}={s[k][0]:.4f}" for k in ["auc", "prauc", "brier", "bacc", "acc", "f1", "mcc"]))
    json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
