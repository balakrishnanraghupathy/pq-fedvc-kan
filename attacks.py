"""Privacy and security experiments (Section IV-E, IV-G, IV-H).

Usage (after `run_experiments.py main` and `fkp`, which save the BRFSS models of every repetition):
  python scripts/attacks.py mia                      # Table 8, membership inference
  python scripts/attacks.py inv VC-KAN,MLP           # Table 8, gradient inversion
  python scripts/attacks.py inv Fed-KAN-pub
  python scripts/attacks.py channel 500              # Table 9, tampering/forgery/replay tests and timing
  python scripts/attacks.py meas                     # Fig. 9(a), finite shots and depolarizing noise
"""
import os, sys, json, time, struct, hashlib, numpy as np, torch, torch.nn.functional as Fn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pqfedvckan.data import brfss_split, load_brfss
from pqfedvckan.models import build
from pqfedvckan.train import predict
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(1)
RES = os.environ.get("PQFEDVCKAN_RESULTS", os.path.join(ROOT, "results"))
BIN = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17]
ORD = [3, 6, 13, 14, 15, 18, 19, 20]


def load_model(name, rep=0):
    m = build(name); m.load_state_dict(torch.load(f"{RES}/models/fed_{name}_r{rep}.pt")); m.eval(); return m


def mia():
    out = {}
    hosp, test, _ = brfss_split(0)
    rng = np.random.RandomState(0)
    Xtr = np.concatenate([h[0] for h in hosp]); ytr = np.concatenate([h[1] for h in hosp])
    i = rng.choice(len(ytr), 10000, replace=False); j = rng.choice(len(test[1]), 10000, replace=False)
    for name in ["LR", "MLP", "VQC", "Spline-KAN", "Fed-KAN-pub", "VC-KAN"]:
        m = load_model(name)
        def loss(X, y):
            p = np.clip(predict(m, X), 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        s = np.r_[-loss(Xtr[i], ytr[i]), -loss(test[0][j], test[1][j])]
        lab = np.r_[np.ones(10000), np.zeros(10000)]
        out[name] = float(roc_auc_score(lab, s))
    return out


def grads_of(m, X, y):
    loss = Fn.binary_cross_entropy_with_logits(m(X), y)
    return torch.autograd.grad(loss, list(m.parameters()), create_graph=True)


def inversion(name, batch=1, dp_sigma=None, n_trials=20, iters=200, seed=0):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = load_model(name)
    for p in m.parameters(): p.requires_grad_(True)
    hosp, _, _ = brfss_split(0)
    X = np.concatenate([h[0] for h in hosp]); y = np.concatenate([h[1] for h in hosp])
    Xall, _, _, _ = load_brfss()
    prior = (Xall[:, BIN].mean(0) >= 0.5).astype(float)
    rb, ro, pb = [], [], []
    for t in range(n_trials):
        idx = rng.choice(len(y), batch, replace=False)
        xt, yt = torch.as_tensor(X[idx]), torch.as_tensor(y[idx])
        # observed (possibly privatised) update
        gs = []
        for b in range(batch):
            g = torch.autograd.grad(Fn.binary_cross_entropy_with_logits(m(xt[b:b + 1]), yt[b:b + 1]), list(m.parameters()))
            gf = torch.cat([q.flatten() for q in g]).detach()
            if dp_sigma is not None: gf = gf * min(1.0, 1.0 / float(gf.norm() + 1e-12))
            gs.append(gf)
        gobs = torch.stack(gs).sum(0)
        if dp_sigma is not None: gobs = gobs + torch.randn_like(gobs) * dp_sigma
        gobs = gobs / batch
        z = torch.full((batch, 21), 0.0, requires_grad=True)  # logits of dummy inputs
        opt = torch.optim.Adam([z], lr=0.1)
        for it in range(iters):
            xd = torch.sigmoid(z)
            gd = torch.cat([q.flatten() for q in grads_of(m, xd, yt)])
            loss = 1 - Fn.cosine_similarity(gd, gobs, dim=0)
            opt.zero_grad(); loss.backward(); opt.step()
        xr = torch.sigmoid(z).detach().numpy(); xo = X[idx]
        # optimal matching of reconstructed and true records within the batch
        cost = np.array([[np.mean((xr[a, BIN] >= 0.5) != (xo[b, BIN] >= 0.5)) for b in range(batch)] for a in range(batch)])
        ra, cb = linear_sum_assignment(cost)
        for a, b in zip(ra, cb):
            rb.append(1 - cost[a, b]); ro.append(np.mean(np.abs(xr[a, ORD] - xo[b, ORD])))
            pb.append(np.mean(prior == (xo[b, BIN] >= 0.5)))
    return dict(bin_recovered=float(np.mean(rb)), ord_mae=float(np.mean(ro)), prior_guess=float(np.mean(pb)), trials=n_trials, batch=batch)


def channel_tests(n=1000):
    from pqfedvckan import pqc
    S = pqc.Server(); C = pqc.Client(0, S); rogue = pqc.Client(1, pqc.Server())  # rogue key not registered at S
    S.register(1, b"\x00" * len(C.pk))  # hospital 1 registered with a different key
    payload = os.urandom(4 * 3177)
    res = {k: 0 for k in ["valid_accepted", "ciphertext_tamper_rejected", "header_tamper_rejected", "kem_tamper_rejected",
                          "forgery_rejected", "replay_rejected", "stale_rejected"]}
    T = {k: [] for k in ["encaps", "aes_enc", "sign", "verify", "decaps", "aes_dec", "send_total", "receive_total"]}
    from kyber_py.ml_kem import ML_KEM_768
    from dilithium_py.ml_dsa import ML_DSA_65
    for r in range(n):
        t0 = time.perf_counter(); m = C.secure_send(payload, r); T["send_total"].append(time.perf_counter() - t0)
        k, rr, nonce, ct, c, sig = m
        # tampered variants (checked before the genuine one is accepted)
        cbad = bytearray(c); cbad[np.random.randint(len(c))] ^= 1
        res["ciphertext_tamper_rejected"] += S.secure_receive((k, rr, nonce, ct, bytes(cbad), sig), r) is None
        res["header_tamper_rejected"] += S.secure_receive((k, rr + 1, nonce, ct, c, sig), r + 1) is None
        ctb = bytearray(ct); ctb[np.random.randint(len(ct))] ^= 1
        res["kem_tamper_rejected"] += S.secure_receive((k, rr, nonce, bytes(ctb), c, sig), r) is None
        f = rogue.secure_send(payload, r); f = (1,) + f[1:]
        res["forgery_rejected"] += S.secure_receive(f, r) is None
        t0 = time.perf_counter(); out = S.secure_receive(m, r); T["receive_total"].append(time.perf_counter() - t0)
        res["valid_accepted"] += out == payload
        res["replay_rejected"] += S.secure_receive(m, r) is None
        m2 = C.secure_send(payload, r)
        res["stale_rejected"] += S.secure_receive(m2, r + 1) is None
        if r < 100:
            t = time.perf_counter(); ss, ct2 = ML_KEM_768.encaps(S.ek); T["encaps"].append(time.perf_counter() - t)
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            key = os.urandom(32); t = time.perf_counter(); cc = AESGCM(key).encrypt(nonce, payload, b"h"); T["aes_enc"].append(time.perf_counter() - t)
            t = time.perf_counter(); AESGCM(key).decrypt(nonce, cc, b"h"); T["aes_dec"].append(time.perf_counter() - t)
            d = hashlib.sha256(cc).digest()
            t = time.perf_counter(); sg = ML_DSA_65.sign(C.sk, d); T["sign"].append(time.perf_counter() - t)
            t = time.perf_counter(); ML_DSA_65.verify(C.pk, d, sg); T["verify"].append(time.perf_counter() - t)
            t = time.perf_counter(); ML_KEM_768.decaps(S.dk, ct2); T["decaps"].append(time.perf_counter() - t)
    sizes = dict(ct=len(ct), sig=len(sig), nonce=12, tag=16, header=8, payload=len(payload), msg=pqc.msg_size(m))
    return dict(counts=res, n=n, ms={k: 1e3 * float(np.mean(v)) for k, v in T.items()}, sizes=sizes)


def measurement():
    out = {}
    from pqfedvckan.models import KANLayer
    for rep in range(5):
        hosp, test, _ = brfss_split(rep)
        m = load_model("VC-KAN", rep)
        for shots in [None, 512, 128, 32, 16]:
            for dep in [0.0, 0.01, 0.05]:
                for l in (m.l1, m.l2): l.shots, l.depol = shots, dep
                torch.manual_seed(rep)
                p = predict(m, test[0])
                out.setdefault(f"{shots}_{dep}", []).append(float(roc_auc_score(test[1], p)))
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    os.makedirs(f"{RES}/attacks", exist_ok=True)
    if what == "mia":
        json.dump(mia(), open(f"{RES}/attacks/mia.json", "w"))
    elif what == "inv":
        from opacus.accountants.utils import get_noise_multiplier
        sig4 = get_noise_multiplier(target_epsilon=4, target_delta=1e-5, sample_rate=0.01, steps=2000, accountant="rdp")
        R = {}
        for name in sys.argv[2].split(","):
            for b in (1, 8):
                R[f"{name}_b{b}_nodp"] = inversion(name, b)
                R[f"{name}_b{b}_dp4"] = inversion(name, b, dp_sigma=sig4)
                print(name, b, R[f"{name}_b{b}_nodp"], R[f"{name}_b{b}_dp4"], flush=True)
        json.dump(R, open(f"{RES}/attacks/inv_{sys.argv[2].replace(',', '_')}.json", "w"))
    elif what == "channel":
        json.dump(channel_tests(int(sys.argv[2]) if len(sys.argv) > 2 else 1000), open(f"{RES}/attacks/channel.json", "w"))
    elif what == "meas":
        json.dump(measurement(), open(f"{RES}/attacks/measurement.json", "w"))
    print("ok", what)
