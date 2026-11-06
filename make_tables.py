"""Prints the values of Tables 5-9 of the paper from the JSON files in results/."""
import os, re, json, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("PQFEDVCKAN_RESULTS", os.path.join(ROOT, "results"))
NAMES = [("LR", "LR"), ("MLP", "MLP"), ("VQC", "VQC"), ("Fed-KAN", "Fed-KAN-pub"), ("KACQ-DCNN", "KACQ-DCNN"), ("PQ-FedVC-KAN", "VC-KAN")]


def L(g):
    out = {}
    for f in sorted(glob.glob(f"{R}/{g}/*.json")):
        out.setdefault(re.sub(r"_r\d+$", "", os.path.basename(f)[:-5]), []).append(json.load(open(f)))
    return out


def m(rs, k): return float(np.mean([r[k] for r in rs]))
def s(rs, k): return float(np.std([r[k] for r in rs], ddof=1))


if __name__ == "__main__":
    fed, uf, uc, ab = L("fed"), L("uci_fed"), L("uci_central"), L("ablation")
    st = json.load(open(f"{R}/stats.json"))
    print("Table 5. BRFSS federation, mean +- SD over five repetitions")
    print(f"{'Method':14s} {'P':>6s} {'ROC-AUC':>14s} {'PR-AUC':>7s} {'Brier':>7s} {'Se':>6s} {'Sp':>6s} {'BA':>14s} {'MCC':>6s} {'pHolm':>6s}")
    for lab, k in NAMES:
        if k not in fed: continue
        rs = fed[k]; ph = st["fed_auc"].get(k, {}).get("p_holm")
        print(f"{lab:14s} {rs[0]['params']:6d} {m(rs,'auc'):.3f} +- {s(rs,'auc'):.3f} {m(rs,'prauc'):7.3f} {m(rs,'brier'):7.4f} {m(rs,'sens'):6.3f} "
              f"{m(rs,'spec'):6.3f} {m(rs,'bacc'):.3f} +- {s(rs,'bacc'):.3f} {m(rs,'mcc'):6.3f} {'' if ph is None else f'{ph:6.3f}'}")
    print("\nTable 6. UCI four-site cohort, means over five repetitions (federated | centralized)")
    for lab, k in NAMES:
        a, c = uf[k], uc[k]
        print(f"{lab:14s} {a[0]['params']:8d} " + " ".join(f"{m(a, x):.3f}" for x in ("auc", "prauc", "bacc", "acc", "f1", "mcc", "brier"))
              + " | " + " ".join(f"{m(c, x):.3f}" for x in ("auc", "bacc", "mcc")))
    print("\nTable 7. Ablation on BRFSS, mean +- SD over three repetitions")
    full = fed["VC-KAN"][:3]
    for lab, rs in (("Full model", full), ("A1 spline twin", fed["Spline-KAN"][:3]), ("A2 circuit only", ab["Circuit-KAN"]),
                    ("A3 Fourier twin", ab["Fourier-KAN"]), ("A4 L = 1", ab["VC-KAN-L1"]), ("A5 L = 3", ab["VC-KAN-L3"]),
                    ("A6 no residual", ab["VC-KAN-nores"]), ("A7 fixed gate", ab["VC-KAN-fixgate"]), ("A8 plain FedAvg", ab["VC-KAN-fedavg"])):
        print(f"{lab:16s} {rs[0]['params']:5d} {m(rs,'auc'):.3f} +- {s(rs,'auc'):.3f} {m(rs,'prauc'):.3f} {m(rs,'bacc'):.3f} {m(rs,'mcc'):.3f} "
              f"{m(rs,'auc') - m(full,'auc'):+.4f}")
    print("\nTable 8. Privacy attacks (membership-inference AUC; binary-predictor recovery of gradient inversion)")
    mia = json.load(open(f"{R}/attacks/mia.json"))
    inv = {}
    for f in sorted(glob.glob(f"{R}/attacks/inv_*.json")): inv.update(json.load(open(f)))
    for lab, k in NAMES:
        if k not in mia: continue
        row = [inv.get(f"{k}_{t}", {}).get("bin_recovered") for t in ("b1_nodp", "b1_dp4", "b8_nodp", "b8_dp4")]
        print(f"{lab:14s} {mia[k]:.3f} " + " ".join("  -  " if v is None else f"{v:.3f}" for v in row))
    print(f"{'Majority guess':14s} 0.500 " + " ".join(f"{inv[f'VC-KAN_{t}']['prior_guess']:.3f}" for t in ("b1_nodp", "b1_dp4", "b8_nodp", "b8_dp4")))
    ch = json.load(open(f"{R}/attacks/channel.json"))
    print("\nTable 9. Post-quantum channel:", ch["counts"], "\n  ms:", {k: round(v, 2) for k, v in ch["ms"].items()}, "\n  bytes:", ch["sizes"])
