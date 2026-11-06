"""Data loading, codebook scaling and federated partitioning (BRFSS 2015 and UCI four-site)."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

import os
D = os.environ.get("PQFEDVCKAN_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
FEATS = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes", "PhysActivity", "Fruits", "Veggies",
         "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
         "Education", "Income"]
# fixed codebook / plausibility ranges (no statistic is estimated from the data)
RANGES = {"BMI": (12, 60), "Diabetes": (0, 2), "GenHlth": (1, 5), "MentHlth": (0, 30), "PhysHlth": (0, 30), "Age": (1, 13),
          "Education": (1, 6), "Income": (1, 8)}
# ten age-band hospitals from the 13 BRFSS age codes
AGE2H = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8, 13: 9}

_cache = {}


def load_brfss():
    if "b" in _cache: return _cache["b"]
    df = pd.read_csv(f"{D}/brfss.csv")
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    X = np.zeros((len(df), len(FEATS)), np.float32)
    for j, f in enumerate(FEATS):
        a, b = RANGES.get(f, (0, 1))
        X[:, j] = (np.clip(df[f].values, a, b) - a) / (b - a)
    y = df["HeartDiseaseorAttack"].values.astype(np.float32)
    age = df["Age"].values.astype(int)
    _cache["b"] = (X, y, age, n_raw)
    return _cache["b"]


def brfss_split(rep, scheme="age", K=10, alpha=None, val_frac=0.1):
    """Return list of hospitals [(Xtr,ytr,Xva,yva)], test (Xte,yte), hospital ids of test rows."""
    X, y, age, _ = load_brfss()
    rng = np.random.RandomState(1000 + rep)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=1000 + rep)
    if scheme == "age":
        hid = np.array([AGE2H[a] for a in age])
        K = 10
    elif scheme == "iid":
        hid = rng.randint(0, K, len(y))
    elif scheme == "dir":
        hid = np.zeros(len(y), int)
        for c in (0, 1):
            ic = np.where(y == c)[0]
            p = rng.dirichlet([alpha] * K)
            hid[ic] = rng.choice(K, len(ic), p=p)
    hosp = []
    for k in range(K):
        ik = tr[hid[tr] == k]
        if len(ik) < 50: continue
        a, b = train_test_split(ik, test_size=val_frac, stratify=y[ik], random_state=rep * 31 + k)
        hosp.append((X[a], y[a], X[b], y[b]))
    return hosp, (X[te], y[te]), hid[te]


# ------------------------------------------------------------------ UCI four sites
SITES = [("Cleveland", "cleveland14.csv"), ("Hungary", "hungarian14r.csv"), ("Switzerland", "switzerland14.csv"),
         ("Long Beach VA", "long_beach_va14.csv")]
UCI_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_uci():
    if "u" in _cache: return _cache["u"]
    out = []
    for name, f in SITES:
        df = pd.read_csv(f"{D}/uci/{f}", header=None, names=UCI_COLS, na_values=["?", "-9", -9, "-9.0"])
        df = df.apply(pd.to_numeric, errors="coerce")
        df.loc[df["chol"] == 0, "chol"] = np.nan
        df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
        out.append((name, df))
    _cache["u"] = out
    return out


# fixed clinical ranges and reference (fill) values
U_RNG = {"age": (25, 80, 55), "trestbps": (80, 200, 130), "chol": (100, 600, 240), "thalach": (60, 210, 140),
         "oldpeak": (-1, 6, 0.0), "restecg": (0, 2, 0), "slope": (1, 3, 2), "fbs": (0, 1, 0), "exang": (0, 1, 0), "sex": (0, 1, 1)}
MISS_IND = ["trestbps", "chol", "fbs", "thalach", "slope"]


def uci_encode(df):
    cols = []
    for c in ["age", "sex", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]:
        a, b, ref = U_RNG[c]
        v = df[c].fillna(ref).values
        cols.append((np.clip(v, a, b) - a) / (b - a))
    cp = df["cp"].fillna(4).values.astype(int)
    for k in (1, 2, 3, 4): cols.append((cp == k).astype(float))
    for c in MISS_IND: cols.append(df[c].isna().values.astype(float))
    X = np.stack(cols, 1).astype(np.float32)
    y = (df["num"].fillna(0).values > 0).astype(np.float32)
    return X, y


def uci_split(rep):
    hosp, tes = [], []
    for name, df in load_uci():
        X, y = uci_encode(df)
        a, b = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=2000 + rep)
        a, v = train_test_split(a, test_size=0.15, stratify=y[a], random_state=3000 + rep)
        hosp.append((X[a], y[a], X[v], y[v]))
        tes.append((X[b], y[b]))
    return hosp, tes
