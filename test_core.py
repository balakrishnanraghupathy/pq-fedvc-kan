"""Unit tests: parameter counts, B-spline basis, Proposition 1, threshold calibration, post-quantum channel."""
import os, sys, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pqfedvckan.models import build, n_params, bspline_basis, KANLayer
from pqfedvckan.train import youden_from_counts


def test_parameter_counts():
    assert n_params(build("VC-KAN", 21)) == 3177       # BRFSS network 21-8-1
    assert n_params(build("VC-KAN", 19)) == 2889       # UCI network 19-8-1
    assert n_params(build("Spline-KAN", 21)) == 3177   # parameter-matched classical twin


def test_bspline_partition_of_unity():
    x = torch.rand(1000, 3) * 0.999
    B = bspline_basis(x, G=5)
    assert B.shape == (1000, 3, 8)
    assert torch.allclose(B.sum(-1), torch.ones(1000, 3), atol=1e-5)


def test_proposition1_spectrum():
    """With geometric frequencies 2*pi*2^(l-1), the circuit output contains only harmonics 0..2^L - 1 of the unit period."""
    torch.manual_seed(0)
    for L in (1, 2, 3):
        layer = KANLayer(1, 1, L=L, path="circuit", residual=False)
        with torch.no_grad():
            layer.w.copy_(torch.tensor([2 * math.pi * 2 ** l for l in range(L)]).view(1, 1, L))
        x = torch.linspace(0, 1, 513)[:-1].unsqueeze(1)
        y = layer.circuit(x).detach().numpy().ravel()
        F = np.abs(np.fft.rfft(y)) / len(y)
        assert F[2 ** L:].max() < 1e-6, (L, F[:2 ** L + 2])


def test_youden_from_counts():
    rng = np.random.RandomState(0)
    y = (rng.rand(4000) < 0.1).astype(float); p = np.clip(0.05 + 0.5 * y + 0.1 * rng.randn(4000), 0, 1)
    tau = youden_from_counts([(p[:2000], y[:2000]), (p[2000:], y[2000:])])
    assert 0.1 < tau < 0.5


def test_channel_accepts_valid_and_rejects_attacks():
    from pqfedvckan import pqc
    S = pqc.Server(); C = pqc.Client(0, S)
    u = os.urandom(4 * 3177)
    m = C.secure_send(u, 3)
    assert pqc.msg_size(m) == 17141
    k, r, nonce, ct, c, sig = m
    bad = bytearray(c); bad[10] ^= 1
    assert S.secure_receive((k, r, nonce, ct, bytes(bad), sig), 3) is None     # ciphertext tampering
    assert S.secure_receive(m, 4) is None                                       # stale round
    assert S.secure_receive(m, 3) == u                                          # genuine update
    assert S.secure_receive(m, 3) is None                                       # replay
    rogue = pqc.Client(0, pqc.Server())
    assert S.secure_receive(rogue.secure_send(u, 3), 3) is None                 # forgery with an unregistered key
