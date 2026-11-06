"""Minimal example: three simulated hospitals, VC-KAN, median norm-bounded FedAvg, post-quantum channel,
optional DP-SGD, and threshold calibration from aggregated counts. Runs in about a minute on a laptop CPU.

    python examples/quickstart.py          # without differential privacy
    python examples/quickstart.py --dp 4   # DP-SGD with epsilon = 4, delta = 1e-5
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pqfedvckan import build, n_params, fed_run

ap = argparse.ArgumentParser(); ap.add_argument("--dp", type=float, default=None); ap.add_argument("--rounds", type=int, default=5)
a = ap.parse_args()
rng = np.random.RandomState(0)


def hospital(n, shift):
    """Synthetic risk-factor records in [0, 1]^21 with a nonlinear risk; prevalence differs between hospitals."""
    X = rng.rand(n, 21).astype(np.float32); X[:, 18] = np.clip(X[:, 18] + shift, 0, 1)   # column 18 plays the role of age
    logit = 4 * X[:, 18] + 2 * np.sin(3 * X[:, 0]) + 1.5 * X[:, 1] * X[:, 2] - 4.5 + rng.randn(n) * 0.5
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    k = int(0.9 * n); return X[:k], y[:k], X[k:], y[k:]


hosp = [hospital(3000, s) for s in (-0.3, 0.0, 0.3)]
test = [hospital(1000, s) for s in (-0.3, 0.0, 0.3)]
test = (np.concatenate([t[0] for t in test]), np.concatenate([t[1] for t in test]))

dp = None
if a.dp:
    from opacus.accountants.utils import get_noise_multiplier
    q, steps = 0.02, 20
    dp = dict(steps=steps, q=q, C=1.0, sigma=get_noise_multiplier(target_epsilon=a.dp, target_delta=1e-5, sample_rate=q,
                                                                  steps=a.rounds * steps, accountant="rdp"))
    print(f"DP-SGD noise multiplier for epsilon = {a.dp}: {dp['sigma']:.3f}")

print("VC-KAN parameters:", n_params(build("VC-KAN", 21)))
model, res, _ = fed_run(lambda: build("VC-KAN", 21), hosp, test, R=a.rounds, lr=1e-2, agg="norm", channel=True, dp=dp)
print({k: round(float(res[k]), 4) for k in ("auc", "prauc", "bacc", "mcc", "brier", "tau")})
print(f"upload per hospital and round: {res['msg_bytes']:.0f} bytes; client {res['t_client_ms']:.1f} ms, server {res['t_server_ms']:.1f} ms")
