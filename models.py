"""Models of PQ-FedVC-KAN.

VC-KAN (proposed): every edge is  lambda * phi_Q(x) + (1 - lambda) * phi_S(x) + w * SiLU(x)   (Eq. 2)
  phi_S  cubic B-spline on G uniform intervals over [0, 1]                              (Eq. 3)
  phi_Q  single-qubit data re-uploading circuit with L layers, read out as c * <Z>        (Eqs. 4-5)
Baselines: LR, MLP (21-64-32-1), dressed 4-qubit VQC, Fed-KAN with its published architecture,
KACQ-DCNN re-implementation; ablation variants (spline twin, circuit-only, Fourier twin, L = 1 / 3, no residual, fixed gate).
"""
import math, torch, torch.nn as nn, torch.nn.functional as Fn

PI = math.pi


def bspline_basis(x, G, k=3):
    """Cubic B-spline basis on G uniform intervals over [0,1]; x (..., n) -> (..., n, G+k)."""
    h = 1.0 / G
    grid = torch.arange(-k, G + k + 1, dtype=x.dtype, device=x.device) * h  # G+2k+1 knots
    x = x.unsqueeze(-1)
    B = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
    for p in range(1, k + 1):
        B = (x - grid[:-(p + 1)]) / (grid[p:-1] - grid[:-(p + 1)]) * B[..., :-1] + \
            (grid[p + 1:] - x) / (grid[p + 1:] - grid[1:(-p)]) * B[..., 1:]
    return B


class KANLayer(nn.Module):
    """Generic KAN layer. path: 'vc' (gated spline + circuit), 'spline', 'circuit', 'fourier', 'vcfourier'."""

    def __init__(self, n_in, n_out, G=5, L=2, path="vc", residual=True, gate="learn", shots=None, depol=0.0):
        super().__init__()
        self.n_in, self.n_out, self.G, self.L, self.path, self.residual, self.gate_mode = n_in, n_out, G, L, path, residual, gate
        self.shots, self.depol = shots, depol
        s = (n_out, n_in)
        use_spline = path in ("vc", "spline", "vcfourier")
        use_circ = path in ("vc", "circuit")
        use_four = path in ("fourier", "vcfourier")
        if use_spline:
            self.coef = nn.Parameter(torch.randn(n_out, n_in, G + 3) * 0.1)
        if use_circ:
            # geometric initial frequencies pi*2^(l-1)
            self.w = nn.Parameter(torch.tensor([PI * 2 ** l for l in range(L)]).repeat(n_out, n_in, 1) * (1 + 0.1 * torch.randn(n_out, n_in, L)))
            self.b = nn.Parameter(0.1 * torch.randn(n_out, n_in, L))
            self.phi = nn.Parameter(torch.rand(n_out, n_in, L) * 2 * PI)
            self.theta = nn.Parameter(torch.rand(n_out, n_in) * PI)
            self.c = nn.Parameter(0.3 * torch.randn(n_out, n_in))
        if use_four:  # equal-size trigonometric series: a0 + sum_{k=1..3}(a_k cos k w x + b_k sin k w x), learnable w
            self.fw = nn.Parameter(torch.full(s, PI) * (1 + 0.1 * torch.randn(s)))
            self.fab = nn.Parameter(0.1 * torch.randn(n_out, n_in, 7))
        if path in ("vc", "vcfourier") and gate == "learn":
            self.rho = nn.Parameter(torch.zeros(s))
        if residual:
            self.rw = nn.Parameter(torch.randn(s) / math.sqrt(n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def circuit(self, x):
        # x: (B, n_in) -> <Z> per edge (B, n_out, n_in); single-qubit statevector, two complex amplitudes
        xb = x[:, None, :]
        ar = torch.ones_like(xb.expand(-1, self.n_out, -1)); ai = torch.zeros_like(ar)
        br = torch.zeros_like(ar); bi = torch.zeros_like(ar)
        for l in range(self.L):
            ang = (self.w[..., l] * xb + self.b[..., l]) / 2
            c, s = torch.cos(ang), torch.sin(ang)
            ar, ai, br, bi = c * ar - s * br, c * ai - s * bi, s * ar + c * br, s * ai + c * bi  # RY
            p = self.phi[..., l] / 2
            cp, sp = torch.cos(p), torch.sin(p)
            ar, ai = ar * cp + ai * sp, ai * cp - ar * sp  # e^{-i p}
            br, bi = br * cp - bi * sp, bi * cp + br * sp  # e^{+i p}
        t = self.theta / 2
        c, s = torch.cos(t), torch.sin(t)
        ar, ai, br, bi = c * ar - s * br, c * ai - s * bi, s * ar + c * br, s * ai + c * bi
        z = ar ** 2 + ai ** 2 - br ** 2 - bi ** 2
        if self.depol > 0: z = z * (1 - self.depol)
        if self.shots:
            pr = ((1 + z) / 2).clamp(0, 1)
            z = 2 * torch.distributions.Binomial(self.shots, pr).sample() / self.shots - 1
        return self.c * z

    def fourier(self, x):
        xb = x[:, None, :]
        ab = self.fab
        out = ab[..., 0]
        for k in range(1, 4):
            out = out + ab[..., 2 * k - 1] * torch.cos(k * self.fw * xb) + ab[..., 2 * k] * torch.sin(k * self.fw * xb)
        return out

    def edge(self, x):
        p = self.path
        if p in ("spline", "vc", "vcfourier"):
            Bs = bspline_basis(x, self.G)  # (B, n_in, G+3)
            sp = torch.einsum("bik,oik->boi", Bs, self.coef)
        if p == "spline": e = sp
        elif p == "circuit": e = self.circuit(x)
        elif p == "fourier": e = self.fourier(x)
        else:
            q = self.circuit(x) if p == "vc" else self.fourier(x)
            lam = torch.sigmoid(self.rho) if self.gate_mode == "learn" else 0.5
            e = lam * q + (1 - lam) * sp
        if self.residual:
            e = e + self.rw * Fn.silu(x)[:, None, :]
        return e  # (B, n_out, n_in)

    def forward(self, x):
        return self.edge(x).sum(-1) + self.bias


class KAN(nn.Module):
    def __init__(self, n_in=21, hidden=8, **kw):
        super().__init__()
        self.l1 = KANLayer(n_in, hidden, **kw)
        self.l2 = KANLayer(hidden, 1, **kw)

    def forward(self, x):
        return self.l2(torch.sigmoid(self.l1(x))).squeeze(-1)


class FedKANPub(nn.Module):
    """Fed-KAN as published in [22]: spline KAN hidden widths 2-4-8, grid 5, cubic splines, dropout 0.5, linear output."""
    def __init__(self, n_in=21, widths=(2, 4, 8), G=5):
        super().__init__()
        dims = (n_in,) + tuple(widths)
        self.layers = nn.ModuleList([KANLayer(a, b, G=G, path="spline") for a, b in zip(dims[:-1], dims[1:])])
        self.drop = nn.Dropout(0.5); self.out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        for l in self.layers: x = torch.sigmoid(l(x))
        return self.out(self.drop(x)).squeeze(-1)


def VCKAN(n_in=21, L=2, **kw): return KAN(n_in, 8, G=5, L=L, path="vc", **kw)
def SplineKAN(n_in=21, L=2): return KAN(n_in, 8, G=5 + 3 * L + 3, L=L, path="spline")  # parameter-matched twin
def CircuitKAN(n_in=21, L=2): return KAN(n_in, 8, L=L, path="circuit")
def FourierKAN(n_in=21): return KAN(n_in, 8, G=5, path="vcfourier")  # equal-size classical Fourier path in place of circuit


class LR(nn.Module):
    def __init__(self, n_in=21):
        super().__init__(); self.f = nn.Linear(n_in, 1)
    def forward(self, x): return self.f(x).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n_in=21):
        super().__init__(); self.f = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.f(x).squeeze(-1)


# ---------------------------------------------------------------- multi-qubit statevector utilities (4 qubits)
def _apply_1q(state, U, q, n):
    # state (B, 2^n) complex; U (2,2) complex or (B,2,2)
    B = state.shape[0]
    st = state.reshape(B, *([2] * n))
    st = torch.movedim(st, q + 1, -1)
    st = st @ U.transpose(-1, -2) if U.dim() == 2 else torch.einsum("b...j,bij->b...i", st, U)
    st = torch.movedim(st, -1, q + 1)
    return st.reshape(B, -1)


def _cnot(state, c, t, n):
    B = state.shape[0]
    st = state.reshape(B, *([2] * n)).clone()
    idx1 = [slice(None)] * (n + 1); idx1[c + 1] = 1
    sub = st[tuple(idx1)]  # control=1 slice
    tt = t if t < c else t - 1
    sub = torch.flip(sub, dims=[tt + 1])
    st[tuple(idx1)] = sub
    return st.reshape(B, -1)


def _rot(phi, theta, omega):
    # Rot = RZ(omega) RY(theta) RZ(phi)
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    e = lambda a: torch.exp(1j * a)
    return torch.stack([torch.stack([e(-(phi + omega) / 2) * c, -e((phi - omega) / 2) * s]),
                        torch.stack([e(-(phi - omega) / 2) * s, e((phi + omega) / 2) * c])])


def strongly_entangling(state, W, n):
    # W (layers, n, 3)
    for l in range(W.shape[0]):
        for q in range(n):
            state = _apply_1q(state, _rot(W[l, q, 0], W[l, q, 1], W[l, q, 2]).to(state.dtype), q, n)
        r = (l % (n - 1)) + 1
        for q in range(n):
            state = _cnot(state, q, (q + r) % n, n)
    return state


def z_expect(state, n):
    p = (state.abs() ** 2).reshape(state.shape[0], *([2] * n))
    out = []
    for q in range(n):
        pq = torch.movedim(p, q + 1, -1).reshape(state.shape[0], -1, 2).sum(1)
        out.append(pq[:, 0] - pq[:, 1])
    return torch.stack(out, 1)


class DressedVQC(nn.Module):
    """Dressed variational quantum classifier: linear -> angle encoding (RY) -> strongly entangling -> <Z> -> linear."""

    def __init__(self, n_in=21, n_q=4, layers=3):
        super().__init__(); self.n = n_q
        self.pre = nn.Linear(n_in, n_q)
        self.W = nn.Parameter(torch.rand(layers, n_q, 3) * 2 * PI)
        self.post = nn.Linear(n_q, 1)

    def forward(self, x):
        a = torch.tanh(self.pre(x)) * PI / 2
        B = x.shape[0]
        st = torch.zeros(B, 2 ** self.n, dtype=torch.complex64); st[:, 0] = 1
        for q in range(self.n):
            c, s = torch.cos(a[:, q] / 2), torch.sin(a[:, q] / 2)
            U = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)
            st = _apply_1q(st, U, q, self.n)
        st = strongly_entangling(st, self.W, self.n)
        return self.post(z_expect(st, self.n)).squeeze(-1)


class EffKAN(nn.Module):
    """tfkan-style DenseKAN: spline (grid_size, order 3) on [-1,1] plus SiLU base, per input-output pair."""

    def __init__(self, n_in, n_out, grid=3):
        super().__init__(); self.G = grid
        self.coef = nn.Parameter(torch.randn(n_out, n_in, grid + 3) * 0.05)
        self.base = nn.Parameter(torch.randn(n_out, n_in) / math.sqrt(n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        Bs = bspline_basis((torch.tanh(x) + 1) / 2, self.G)
        return torch.einsum("bik,oik->bo", Bs, self.coef) + Fn.silu(x) @ self.base.t() + self.bias


class AmpQLayer(nn.Module):
    """Amplitude embedding of 16 features into 4 qubits, one strongly entangling layer, <Z_i> readout."""

    def __init__(self, n=4, layers=1):
        super().__init__(); self.n = n; self.W = nn.Parameter(torch.rand(layers, n, 3) * 2 * PI)

    def forward(self, x):
        st = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        st = st.to(torch.complex64)
        st = strongly_entangling(st, self.W, self.n)
        return z_expect(st, self.n)


class KACQDCNN(nn.Module):
    """Re-implementation of KACQ-DCNN (Jahin et al.) after the authors' released notebook: input = features + stacked
    CatBoost probability as a sequence; channel 1 = 2 x BiLSTM(64) -> Dense(320) -> KAN(256) -> KAN(32);
    channel 2 = KAN(112) per step -> KAN(176) -> KAN(48) -> 3 x [4-qubit amplitude-embedded SEL] -> KAN(112) -> KAN(32)."""

    def __init__(self, n_in=22, p_drop=0.2):
        super().__init__(); self.n_in = n_in
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.d320 = nn.Linear(n_in * 128, 320)
        self.k256 = EffKAN(320, 256); self.k32a = EffKAN(256, 32)
        self.k112s = EffKAN(1, 112)
        self.k176 = EffKAN(n_in * 112, 176)
        self.k48 = EffKAN(176, 48, grid=5)
        self.q = nn.ModuleList([AmpQLayer() for _ in range(3)])
        self.k112 = EffKAN(12, 112); self.k32b = EffKAN(112, 32)
        self.out = nn.Linear(64, 1); self.dp = nn.Dropout(p_drop)

    def forward(self, x):
        B = x.shape[0]
        s = x.unsqueeze(-1)
        h, _ = self.lstm(s); h = torch.relu(h)
        m1 = self.dp(torch.relu(self.d320(self.dp(h.reshape(B, -1)))))
        m1 = self.dp(self.k32a(self.dp(self.k256(m1))))
        m2 = self.k112s(s.reshape(-1, 1)).reshape(B, -1)
        m2 = self.dp(self.k176(self.dp(m2)))
        m2 = self.k48(self.dp(m2))
        qs = [self.q[i](m2[:, 16 * i:16 * (i + 1)]) for i in range(3)]
        m2 = self.dp(self.k32b(self.dp(self.k112(torch.cat(qs, 1)))))
        return self.out(torch.cat([m1, m2], 1)).squeeze(-1)


def n_params(m): return sum(p.numel() for p in m.parameters())


def build(name, n_in=21, **kw):
    if name.startswith("VCKAN:"):  # VCKAN:h<width>:G<grid>:L<layers>:r<residual 0/1>
        p = {t[0]: int(t[1:]) for t in name.split(":")[1:]}
        return KAN(n_in, p["h"], G=p["G"], L=p["L"], path="vc", residual=bool(p["r"]))
    return {"LR": lambda: LR(n_in), "MLP": lambda: MLP(n_in), "VQC": lambda: DressedVQC(n_in), "Spline-KAN": lambda: SplineKAN(n_in), "Fed-KAN-pub": lambda: FedKANPub(n_in),
            "VC-KAN": lambda: VCKAN(n_in, **kw), "VC-KAN-L1": lambda: VCKAN(n_in, L=1), "VC-KAN-L3": lambda: VCKAN(n_in, L=3),
            "Circuit-KAN": lambda: CircuitKAN(n_in), "Fourier-KAN": lambda: FourierKAN(n_in),
            "VC-KAN-nores": lambda: KAN(n_in, 8, G=5, L=2, path="vc", residual=False),
            "VC-KAN-fixgate": lambda: KAN(n_in, 8, G=5, L=2, path="vc", gate="fixed"),
            "KACQ-DCNN": lambda: KACQDCNN(n_in + 1)}[name]()
