"""Stacked CatBoost probability input of KACQ-DCNN, computed per hospital (out-of-fold for training rows) so that no record leaves its hospital."""
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold


def _cb(iters, seed):
    return CatBoostClassifier(iterations=iters, depth=6, learning_rate=0.1, verbose=0, thread_count=1, random_seed=seed)


def stack_catboost(hosp, test, hid, rep, iters=500):
    out = []; models = []
    for k, (Xtr, ytr, Xva, yva) in enumerate(hosp):
        oof = np.zeros(len(ytr), np.float32)
        nf = min(5, int(min(ytr.sum(), len(ytr) - ytr.sum())))
        if nf >= 2:
            for a, b in StratifiedKFold(nf, shuffle=True, random_state=rep).split(Xtr, ytr):
                m = _cb(iters, rep).fit(Xtr[a], ytr[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]
        else:
            oof[:] = ytr.mean()
        m = _cb(iters, rep).fit(Xtr, ytr); models.append(m)
        out.append((np.c_[Xtr, oof].astype(np.float32), ytr, np.c_[Xva, m.predict_proba(Xva)[:, 1]].astype(np.float32), yva))
    Xt, yt = test
    pt = np.zeros(len(yt), np.float32)
    for k, m in enumerate(models):
        sel = hid == k
        if sel.any(): pt[sel] = m.predict_proba(Xt[sel])[:, 1]
    return out, (np.c_[Xt, pt].astype(np.float32), yt)
