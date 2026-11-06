"""ROC curves and reliability diagram of the first repetition (Fig. 4(b), 4(c)), written to results/roc.json."""
import os, sys, json, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pqfedvckan.data import brfss_split
from pqfedvckan.models import build
from pqfedvckan.train import predict
from sklearn.metrics import roc_curve, precision_recall_curve
R = os.environ.get("PQFEDVCKAN_RESULTS", os.path.join(ROOT, "results"))

if __name__ == "__main__":
    torch.set_num_threads(1)
    hosp, test, hid = brfss_split(0); out = {}
    for n in ["LR", "MLP", "VQC", "Spline-KAN", "Fed-KAN-pub", "VC-KAN"]:
        m = build(n); m.load_state_dict(torch.load(f"{R}/models/fed_{n}_r0.pt")); p = predict(m, test[0])
        f, t, _ = roc_curve(test[1], p); g = np.linspace(0, 1, 101); out[n] = dict(fpr=g.tolist(), tpr=np.interp(g, f, t).tolist())
        pr, rc, _ = precision_recall_curve(test[1], p); rg = np.linspace(0, 1, 101)
        out[n]["rec"] = rg.tolist(); out[n]["prec"] = np.interp(rg, rc[::-1], pr[::-1]).tolist()
        b = np.clip((p * 10).astype(int), 0, 9)  # ten risk bins, bins with more than 50 records
        out[n]["cal"] = [[float(p[b == i].mean()), float(test[1][b == i].mean()), int((b == i).sum())] for i in range(10) if (b == i).sum() > 50]
    json.dump(out, open(f"{R}/roc.json", "w")); print("ok")
