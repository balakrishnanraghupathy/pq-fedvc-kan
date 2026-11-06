# PQ-FedVC-KAN

Code for the paper **"Post-Quantum Secure Federated Variational-Circuit KANs for Privacy-Preserving Heart Disease Screening"**
by Bala Krishnan Raghupathy, Subramaniyaswamy Vairavasundaram, and Rajesh Kumar Namachivayam (submitted to IEEE Access).

PQ-FedVC-KAN trains a compact heart disease risk model across hospitals. It combines:

- **VC-KAN edges.** Each edge is `λ·φ_Q(x) + (1 − λ)·φ_S(x) + w·SiLU(x)`, where:
  - `φ_S` is a cubic B-spline (G = 5);
  - `φ_Q` is a single-qubit data re-uploading circuit (L = 2), simulated exactly as a two-amplitude state vector;
  - `λ` is a learnable gate.

  The 21–8–1 BRFSS network has 3177 parameters.
- **Median norm-bounded FedAvg.** Each update is clipped to the median update norm of the round before weighted averaging.
- **Client-side DP-SGD** with a Rényi differential privacy accountant (Opacus).
- **A post-quantum secured and authenticated update channel.** It uses ML-KEM-768 encapsulation, HKDF-SHA256, AES-256-GCM and an ML-DSA-65 signature, with round binding and a nonce registry. One upload is 17,141 bytes.
- **Threshold calibration from aggregated counts.** The Youden threshold is computed from aggregated true-positive and true-negative counts only.

## Repository layout

```
pqfedvckan/          library
  models.py          VC-KAN, ablation variants, LR, MLP, dressed VQC, Fed-KAN (published architecture), KACQ-DCNN re-implementation
  train.py           federated training, DP-SGD, norm-bounded aggregation, threshold calibration, metrics
  pqc.py             post-quantum channel (Algorithm 2)
  data.py            BRFSS and UCI loading, codebook scaling, federated partitions
  stack.py           stacked CatBoost input of KACQ-DCNN, computed inside each hospital
scripts/
  run_experiments.py all federated experiments (Tables 5–7, Figs. 4–7, 9(b))
  attacks.py         membership inference, gradient inversion, channel tests, measurement noise (Tables 8–9, Fig. 9(a))
  stats.py           corrected resampled t-test with Holm adjustment
  roc_calibration.py ROC curves and reliability diagram (Fig. 4)
  summarize.py       mean and SD of every run group
  make_tables.py     prints Tables 5–9 from results/
examples/quickstart.py  three simulated hospitals, about one minute on a CPU
tests/test_core.py      parameter counts, B-spline basis, Proposition 1, channel attacks, threshold calibration
data/                   UCI four-site files (included); BRFSS file to be downloaded, see data/README.md
results/                JSON output of every run reported in the paper, see results/README.md
```

## Installation

```bash
git clone https://github.com/<user>/PQ-FedVC-KAN.git
cd PQ-FedVC-KAN
python -m pip install -r requirements.txt
python -m pytest -q tests
python examples/quickstart.py            # add --dp 4 for DP-SGD with epsilon = 4
```

Python 3.13 and the package versions in `requirements.txt` were used for the paper. No GPU or quantum hardware is required.

## Minimal use

```python
from pqfedvckan import build, fed_run
# hosp: list of (X_train, y_train, X_val, y_val) per hospital, features scaled to [0, 1]; test: (X_test, y_test)
model, res, p = fed_run(lambda: build("VC-KAN", n_in=21), hosp, test,
                        R=40, lr=1e-3, agg="norm", channel=True)   # dp=dict(steps, q, C, sigma) enables DP-SGD
print(res["auc"], res["bacc"], res["tau"], res["msg_bytes"])
```

## Reproducing the paper

1. Place the BRFSS file at `data/brfss.csv`, as described in `data/README.md`.
2. Run the phases in order. Each job writes one JSON file, and finished jobs are skipped on a rerun.

```bash
python scripts/run_experiments.py tune
python scripts/run_experiments.py fkp_tune
python scripts/run_experiments.py main
python scripts/run_experiments.py fkp
python scripts/run_experiments.py extra
python scripts/attacks.py mia
python scripts/attacks.py inv VC-KAN,MLP
python scripts/attacks.py inv Fed-KAN-pub
python scripts/attacks.py channel 500
python scripts/attacks.py meas
python scripts/stats.py
python scripts/roc_calibration.py
python scripts/make_tables.py
```

Pass `--results <folder>` to `run_experiments.py`, or set `PQFEDVCKAN_RESULTS`, to keep new runs separate from the shipped results.

**Seeds and determinism.** Every split and training run is seeded by its repetition index, so the runs are deterministic on one machine. The federated and centralized UCI runs of the first repetition were rerun before release and gave identical ROC-AUC values.

**Run time.** Measured on a two-core CPU node:
- one BRFSS repetition of the proposed model takes about 5 minutes;
- KACQ-DCNN is evaluated on UCI only, because one federated round on BRFSS took about 7 minutes.

## Notes

- The circuit path is simulated exactly. On hardware, the expectation `⟨Z⟩` is estimated from finite shots, which `KANLayer.shots` and `KANLayer.depol` emulate.
- KACQ-DCNN is a PyTorch re-implementation written from the architecture in the authors' released code (MIT License, © 2024 Md Abrar Jahin).
- The ML-KEM and ML-DSA implementations used, `kyber-py` and `dilithium-py`, are pure-Python reference implementations. They suit research and timing comparisons, not production. A deployment should use a constant-time library such as liboqs.

## Citation

```bibtex
@article{raghupathy2026pqfedvckan,
  title   = {Post-Quantum Secure Federated Variational-Circuit KANs for Privacy-Preserving Heart Disease Screening},
  author  = {Raghupathy, Bala Krishnan and Vairavasundaram, Subramaniyaswamy and Namachivayam, Rajesh Kumar},
  journal = {IEEE Access},
  year    = {2026},
  note    = {submitted}
}
```

## License

MIT, see `LICENSE`. The UCI data files are distributed under CC BY 4.0, see `data/README.md`.
