import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import polars as pl

def _auc_per_session(y, p, s):
    """
    Bir session içindeki veriler için AUC hesaplar.
    y: gerçek etiketler (0/1)
    p: tahminler
    s: session_id
    """
    df = pd.DataFrame({'y': y, 'p': p, 's': s})
    aucs = []
    for _, g in df.groupby('s', sort=False):
        if g['y'].nunique() == 2:  # oturumda hem 0 hem 1 varsa
            aucs.append(roc_auc_score(g['y'], g['p']))
    return float(np.mean(aucs)) if len(aucs) else np.nan

def trendyol_score(y_click, y_order, p, session_id, w_click=0.3, w_order=0.7):
    """
    Trendyol hackathon metriğini hesaplar.
    Click ve Order AUC’larını alıp ağırlıklı ortalama döner.
    """
    auc_c = _auc_per_session(y_click, p, session_id)
    auc_o = _auc_per_session(y_order, p, session_id)
    if np.isnan(auc_c):
        auc_c = 0.0
    if np.isnan(auc_o):
        auc_o = 0.0
    return w_click * auc_c + w_order * auc_o, auc_c, auc_o

def mm(c):
    """
    Polars için min-max normalizasyon.
    """
    return (pl.col(c) - pl.min(c)) / (pl.max(c) - pl.min(c) + 1e-12)
