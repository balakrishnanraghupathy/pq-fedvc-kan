"""Federated training engine: local Adam / DP-SGD, norm-bounded FedAvg, FedAdam, PQ channel, threshold calibration, metrics."""
import time, copy, io, numpy as np, torch, torch.nn.functional as Fn
from torch.func import functional_call, vmap, grad
from sklearn.metrics import roc_auc_score, average_precision_score
from . import pqc

GRID = np.linspace(0.005, 0.995, 199)


def flat(model): return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_flat(model, v):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel(); p.copy_(v[i:i + n].view_as(p)); i += n


@torch.no_grad()
def predict(model, X, bs=8192):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(torch.as_tensor(X[i:i + bs]))).numpy())
    model.train(); return np.concatenate(out)


def metrics(p, y, tau):
    yh = (p >= tau).astype(int); y = y.astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(auc=float(roc_auc_score(y, p)), prauc=float(average_precision_score(y, p)), brier=float(np.mean((p - y) ** 2)),
                sens=sens, spec=spec, bacc=(sens + spec) / 2, prec=prec, f1=2 * prec * sens / max(prec + sens, 1e-12),
                acc=(tp + tn) / len(y), mcc=(tp * tn - fp * fn) / den if den > 0 else 0.0, tau=float(tau))


def youden_from_counts(models_p_y):
    """models_p_y: list of (p_val, y_val) per hospital -> tau* maximising aggregated Youden J (only counts are shared)."""
    TP = np.zeros(len(GRID)); TN = np.zeros(len(GRID)); P = N = 0
    for p, y in models_p_y:
        P += (y == 1).sum(); N += (y == 0).sum()
        TP += np.array([((p >= t) & (y == 1)).sum() for t in GRID]); TN += np.array([((p < t) & (y == 0)).sum() for t in GRID])
    J = TP / P + TN / N - 1
    return float(GRID[int(np.argmax(J))])


def local_train(model, X, y, lr, epochs, bs, gen, label_flip=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(1 - y if label_flip else y)
    n = len(yt)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = Fn.binary_cross_entropy_with_logits(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()


def dp_local_train(model, X, y, lr, steps, q, C, sigma, gen):
    """DP-SGD with Poisson sampling, per-record clipping and Gaussian noise (Eq. 11); Adam on the privatised gradient."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    names = list(params)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def loss_one(pr, x, t):
        out = functional_call(model, pr, (x.unsqueeze(0),))
        return Fn.binary_cross_entropy_with_logits(out, t.unsqueeze(0))

    g1 = vmap(grad(loss_one), in_dims=(None, 0, 0))
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y); n = len(yt)
    for _ in range(steps):
        mask = torch.rand(n, generator=gen) < q
        idx = torch.nonzero(mask).flatten()
        pr = {k: v.detach() for k, v in model.named_parameters()}
        if len(idx) == 0:
            summed = {k: torch.zeros_like(v) for k, v in pr.items()}
        else:
            gs = g1(pr, Xt[idx], yt[idx])
            norms = torch.sqrt(sum((gs[k].reshape(len(idx), -1) ** 2).sum(1) for k in names))
            fac = (C / (norms + 1e-12)).clamp(max=1.0)
            summed = {k: (gs[k] * fac.view(-1, *[1] * (gs[k].dim() - 1))).sum(0) for k in names}
        opt.zero_grad()
        for k, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=gen) * sigma * C
            p.grad = (summed[k] + noise) / (q * n)
        opt.step()


class Channel:
    """Post-quantum secured and authenticated update channel (ML-KEM-768 + HKDF + AES-256-GCM + ML-DSA-65)."""

    def __init__(self, K):
        self.server = pqc.Server(); self.clients = [pqc.Client(k, self.server) for k in range(K)]
        self.t_client = []; self.t_server = []; self.bytes = []

    def send_and_receive(self, k, r, vec):
        payload = vec.numpy().astype(np.float32).tobytes()
        t0 = time.perf_counter(); msg = self.clients[k].secure_send(payload, r); t1 = time.perf_counter()
        out = self.server.secure_receive(msg, r); t2 = time.perf_counter()
        self.t_client.append(t1 - t0); self.t_server.append(t2 - t1); self.bytes.append(pqc.msg_size(msg))
        assert out is not None
        return torch.from_numpy(np.frombuffer(out, dtype=np.float32).copy())


def fed_run(build_fn, hosp, test, R=40, E=1, lr=1e-2, bs=512, agg="norm", seed=0, channel=False, dp=None,
            attackers=(), attack=None, server_opt=None, curve_every=0, test_hids=None, log_val=True):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = build_fn(); K = len(hosp)
    n_k = np.array([len(h[1]) for h in hosp], float); w_k = n_k / n_k.sum()
    ch = Channel(K) if channel else None
    curve = []; t_train = 0.0; t_agg = 0.0
    if server_opt == "adam":
        m = torch.zeros_like(flat(model)); v = torch.zeros_like(m); b1, b2, eta_s, tau_s = 0.9, 0.99, 0.01, 1e-3
    for r in range(R):
        theta = flat(model); deltas = []
        t0 = time.perf_counter()
        for k, (Xk, yk, _, _) in enumerate(hosp):
            local = copy.deepcopy(model)
            if dp is not None:
                dp_local_train(local, Xk, yk, lr, dp["steps"], dp["q"], dp["C"], dp["sigma"], gen)
            else:
                local_train(local, Xk, yk, lr, E, bs, gen, label_flip=(attack == "labelflip" and k in attackers))
            d = flat(local) - theta
            if k in attackers:
                if attack == "signflip": d = -5 * d
                elif attack == "scale": d = 10 * d
            deltas.append(d)
        t_train += time.perf_counter() - t0
        t1 = time.perf_counter()
        if ch is not None:
            deltas = [ch.send_and_receive(k, r, d) for k, d in enumerate(deltas)]
        norms = torch.stack([d.norm() for d in deltas])
        if agg == "norm":
            Cr = norms.median()
            deltas = [d * min(1.0, float(Cr / (nn_ + 1e-12))) for d, nn_ in zip(deltas, norms)]
        upd = sum(float(w) * d for w, d in zip(w_k, deltas))
        if server_opt == "adam":
            m = b1 * m + (1 - b1) * upd; v = b2 * v + (1 - b2) * upd ** 2
            upd = eta_s * m / (v.sqrt() + tau_s)
        set_flat(model, theta + upd)
        t_agg += time.perf_counter() - t1
        if curve_every and ((r + 1) % curve_every == 0 or r == 0):
            p = predict(model, test[0]); curve.append(dict(round=r + 1, auc=float(roc_auc_score(test[1], p))))
    # threshold from aggregated validation counts
    tau = youden_from_counts([(predict(model, h[2]), h[3]) for h in hosp])
    p = predict(model, test[0]); res = metrics(p, test[1], tau)
    val_ll = float(sum(w * Fn.binary_cross_entropy(torch.as_tensor(predict(model, h[2])).clamp(1e-6, 1 - 1e-6), torch.as_tensor(h[3])).item()
                       for w, h in zip(n_k / n_k.sum(), hosp)))
    res.update(curve=curve, t_train_round=t_train / R, t_agg_round=t_agg / R, params=sum(p_.numel() for p_ in model.parameters()), val_ll=val_ll)
    if test_hids is not None:
        res["per_hosp_auc"] = [float(roc_auc_score(test[1][test_hids == k], p[test_hids == k])) if len(np.unique(test[1][test_hids == k])) > 1 else float("nan") for k in range(K)]
    if ch is not None:
        res.update(t_client_ms=1e3 * np.mean(ch.t_client), t_server_ms=1e3 * np.mean(ch.t_server), msg_bytes=float(np.mean(ch.bytes)))
    return model, res, p


def central_run(build_fn, hosp, test, epochs=40, lr=1e-2, bs=512, seed=0, patience=5):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    Xtr = np.concatenate([h[0] for h in hosp]); ytr = np.concatenate([h[1] for h in hosp])
    Xva = np.concatenate([h[2] for h in hosp]); yva = np.concatenate([h[3] for h in hosp])
    model = build_fn(); best = (1e9, None); bad = 0
    for ep in range(epochs):
        local_train(model, Xtr, ytr, lr, 1, bs, gen)
        pv = predict(model, Xva); ll = float(Fn.binary_cross_entropy(torch.as_tensor(pv).clamp(1e-6, 1 - 1e-6), torch.as_tensor(yva)))
        if ll < best[0] - 1e-5: best = (ll, copy.deepcopy(model.state_dict())); bad = 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best[1])
    tau = youden_from_counts([(predict(model, Xva), yva)])
    p = predict(model, test[0]); res = metrics(p, test[1], tau); res.update(params=sum(p_.numel() for p_ in model.parameters()), val_ll=best[0])
    return model, res, p
