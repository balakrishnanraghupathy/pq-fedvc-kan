"""Runs every federated experiment reported in the paper.

Usage:  python scripts/run_experiments.py <phase> [--workers 2] [--results results]

Phases, in the order they must be run:
  tune      learning-rate selection on the first repetition (BRFSS and UCI), by aggregated validation log-loss
  fkp_tune  the same for Fed-KAN with its published architecture
  main      Tables 5 and 6: LR, MLP, VQC, spline twin, VC-KAN (5 repetitions) and KACQ-DCNN on UCI
  fkp       Fed-KAN (published architecture), 5 repetitions
  extra     Table 7 ablation, Fig. 6 differential privacy, Fig. 7 poisoning, Fig. 9(b) partitions

Each job writes one JSON file to <results>/<group>/<name>.json; finished jobs are skipped on a rerun.
"""
import os, sys, json, time, argparse, traceback, multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.environ.get("PQFEDVCKAN_RESULTS", os.path.join(ROOT, "results"))
LRS = [1e-3, 3e-3, 1e-2, 3e-2]
MAIN = ["LR", "MLP", "VQC", "Spline-KAN", "VC-KAN"]
ABL = ["Circuit-KAN", "Fourier-KAN", "VC-KAN-L1", "VC-KAN-L3", "VC-KAN-nores", "VC-KAN-fixgate"]


def path(g, n): return f"{RES}/{g}/{n}.json"
def done(g, n): return os.path.exists(path(g, n))


def save(g, n, obj):
    os.makedirs(f"{RES}/{g}", exist_ok=True)
    json.dump(obj, open(path(g, n), "w"))


def tuned(group, model):
    """Learning rate with the lowest aggregated validation log-loss; test data are not used."""
    best = None
    for lr in LRS:
        p = path(group, f"{model}_lr{lr}")
        if os.path.exists(p):
            v = json.load(open(p))["val_ll"]
            if best is None or v < best[0]: best = (v, lr)
    return best[1] if best else 1e-2


def _uci_test(tes):
    import numpy as np
    Xt = np.concatenate([t[0] for t in tes]); yt = np.concatenate([t[1] for t in tes])
    hid = np.concatenate([[k] * len(t[1]) for k, t in enumerate(tes)])
    return Xt, yt, hid


def run(job):
    import torch
    torch.set_num_threads(1)
    from pqfedvckan.data import brfss_split, uci_split
    from pqfedvckan.models import build
    from pqfedvckan.train import fed_run, central_run
    g, n = job["group"], job["name"]
    if done(g, n): return
    t0 = time.time()
    try:
        kind = job["kind"]
        if kind == "brfss_fed":
            hosp, test, hid = brfss_split(job["rep"], job.get("scheme", "age"), alpha=job.get("alpha"))
            dp = job.get("dp")
            if dp:
                from opacus.accountants.utils import get_noise_multiplier
                sigma = get_noise_multiplier(target_epsilon=dp["eps"], target_delta=1e-5, sample_rate=dp["q"],
                                             steps=dp["R"] * dp["steps"], accountant="rdp")
                dp = dict(steps=dp["steps"], q=dp["q"], C=1.0, sigma=sigma)
            model, res, p = fed_run(lambda: build(job["model"]), hosp, test, R=job.get("R", 40), E=job.get("E", 1), lr=job["lr"],
                                    agg=job.get("agg", "norm"), seed=job["rep"], channel=job.get("channel", False), dp=dp,
                                    attackers=job.get("attackers", ()), attack=job.get("attack"), server_opt=job.get("server_opt"),
                                    curve_every=job.get("curve_every", 0), test_hids=hid)
            if dp: res["sigma"] = dp["sigma"]
            if job.get("save_model"):
                os.makedirs(f"{RES}/models", exist_ok=True); torch.save(model.state_dict(), f"{RES}/models/{g}_{n}.pt")
        elif kind in ("uci_fed", "uci_central"):
            hosp, tes = uci_split(job["rep"])
            Xt, yt, hid = _uci_test(tes)
            if job["model"] == "KACQ-DCNN":
                from pqfedvckan.stack import stack_catboost
                hosp, (Xt, yt) = stack_catboost(hosp, (Xt, yt), hid, job["rep"], iters=300)
            n_in = Xt.shape[1] - (1 if job["model"] == "KACQ-DCNN" else 0)
            if kind == "uci_fed":
                model, res, p = fed_run(lambda: build(job["model"], n_in=n_in), hosp, (Xt, yt), R=job.get("R", 40), E=job.get("E", 2),
                                        bs=32, lr=job["lr"], seed=job["rep"], test_hids=hid)
            else:
                from sklearn.metrics import roc_auc_score
                model, res, p = central_run(lambda: build(job["model"], n_in=n_in), hosp, (Xt, yt), epochs=150, lr=job["lr"],
                                            bs=32, seed=job["rep"], patience=15)
                res["per_hosp_auc"] = [float(roc_auc_score(yt[hid == k], p[hid == k])) for k in range(4)]
        res.update(job=job, wall=time.time() - t0)
        save(g, n, res)
        print(f"[done] {g}/{n} auc={res['auc']:.4f} {time.time() - t0:.0f}s", flush=True)
    except Exception:
        print(f"[fail] {g}/{n}\n" + traceback.format_exc(), flush=True)


def jobs_phase(phase):
    J = []
    if phase == "tune":
        for m in MAIN:
            for lr in LRS:
                J.append(dict(group="tune", name=f"{m}_lr{lr}", kind="brfss_fed", model=m, rep=0, lr=lr))
                J.append(dict(group="tune_uci", name=f"{m}_lr{lr}", kind="uci_fed", model=m, rep=0, lr=lr))
    if phase == "fkp_tune":
        for lr in LRS:
            J.append(dict(group="tune", name=f"Fed-KAN-pub_lr{lr}", kind="brfss_fed", model="Fed-KAN-pub", rep=0, lr=lr))
            J.append(dict(group="tune_uci", name=f"Fed-KAN-pub_lr{lr}", kind="uci_fed", model="Fed-KAN-pub", rep=0, lr=lr))
    if phase == "main":
        lr = {m: tuned("tune", m) for m in MAIN}; lru = {m: tuned("tune_uci", m) for m in MAIN}
        lru["KACQ-DCNN"] = 1e-3  # learning rate of the authors' released configuration
        for rep in range(5):
            for m in MAIN:
                J.append(dict(group="fed", name=f"{m}_r{rep}", kind="brfss_fed", model=m, rep=rep, lr=lr[m], curve_every=2,
                              channel=(m == "VC-KAN" and rep == 0), save_model=True))
            for m in MAIN + ["KACQ-DCNN"]:
                J.append(dict(group="uci_fed", name=f"{m}_r{rep}", kind="uci_fed", model=m, rep=rep, lr=lru[m]))
                J.append(dict(group="uci_central", name=f"{m}_r{rep}", kind="uci_central", model=m, rep=rep, lr=lru[m]))
    if phase == "fkp":
        lr = tuned("tune", "Fed-KAN-pub"); lru = tuned("tune_uci", "Fed-KAN-pub")
        for rep in range(5):
            J.append(dict(group="fed", name=f"Fed-KAN-pub_r{rep}", kind="brfss_fed", model="Fed-KAN-pub", rep=rep, lr=lr, curve_every=2, save_model=True))
            J.append(dict(group="uci_fed", name=f"Fed-KAN-pub_r{rep}", kind="uci_fed", model="Fed-KAN-pub", rep=rep, lr=lru))
            J.append(dict(group="uci_central", name=f"Fed-KAN-pub_r{rep}", kind="uci_central", model="Fed-KAN-pub", rep=rep, lr=lru))
    if phase == "extra":
        lr = {m: tuned("tune", m) for m in MAIN}
        for rep in range(3):  # Table 7
            for a in ABL: J.append(dict(group="ablation", name=f"{a}_r{rep}", kind="brfss_fed", model=a, rep=rep, lr=lr["VC-KAN"]))
            J.append(dict(group="ablation", name=f"VC-KAN-fedavg_r{rep}", kind="brfss_fed", model="VC-KAN", rep=rep, lr=lr["VC-KAN"], agg="fedavg"))
        for rep in range(2):  # Fig. 6: DPQML = DP-VQC, FedCVR = DP-MLP with an adaptive server optimizer
            for eps in [1, 4, 8]:
                d = dict(eps=eps, q=0.01, steps=100, R=20)
                J.append(dict(group="dp", name=f"VC-KAN_e{eps}_r{rep}", kind="brfss_fed", model="VC-KAN", rep=rep, lr=lr["VC-KAN"], R=20, dp=d))
                J.append(dict(group="dp", name=f"DPQML_e{eps}_r{rep}", kind="brfss_fed", model="VQC", rep=rep, lr=lr["VQC"], R=20, dp=d, agg="fedavg"))
                J.append(dict(group="dp", name=f"FedCVR_e{eps}_r{rep}", kind="brfss_fed", model="MLP", rep=rep, lr=lr["MLP"], R=20, dp=d, agg="fedavg", server_opt="adam"))
        for rep in range(2):  # Fig. 7: attacking hospitals H6 (one attacker) or H3, H6, H7 (three attackers)
            for agg in ["fedavg", "norm"]:
                J.append(dict(group="poison", name=f"clean_{agg}_r{rep}", kind="brfss_fed", model="VC-KAN", rep=rep, lr=lr["VC-KAN"], R=20, agg=agg))
                for att, ks in [("signflip", (5,)), ("signflip", (2, 5, 6)), ("scale", (2, 5, 6)), ("labelflip", (2, 5, 6))]:
                    J.append(dict(group="poison", name=f"{att}{len(ks)}_{agg}_r{rep}", kind="brfss_fed", model="VC-KAN", rep=rep,
                                  lr=lr["VC-KAN"], R=20, agg=agg, attack=att, attackers=ks))
        for rep in range(2):  # Fig. 9(b)
            J.append(dict(group="hetero", name=f"iid_r{rep}", kind="brfss_fed", model="VC-KAN", rep=rep, lr=lr["VC-KAN"], scheme="iid"))
            for a in [0.5, 1.0]:
                J.append(dict(group="hetero", name=f"dir{a}_r{rep}", kind="brfss_fed", model="VC-KAN", rep=rep, lr=lr["VC-KAN"], scheme="dir", alpha=a))
    return J


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "fkp_tune", "main", "fkp", "extra"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--results", default=None, help="output folder (default: ./results)")
    a = ap.parse_args()
    if a.results: RES = os.environ["PQFEDVCKAN_RESULTS"] = os.path.abspath(a.results)
    J = [j for j in jobs_phase(a.phase) if not done(j["group"], j["name"])]
    print(f"{a.phase}: {len(J)} jobs -> {RES}", flush=True)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        pool.map(run, J, chunksize=1)
    print(f"{a.phase} finished", flush=True)
