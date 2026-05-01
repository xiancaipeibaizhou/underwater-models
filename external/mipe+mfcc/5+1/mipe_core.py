import numpy as np
from scipy.stats import norm as sp_norm

def _fix(x):
    return np.where(x >= 0, np.floor(x), np.ceil(x)).astype(int)

def calculate_ipe(data, m=4, tau=1, c=4):
    """
    改进排列熵 (单尺度)
    """
    data = np.array(data, dtype=float).ravel()
    mu, sigma = np.mean(data), np.std(data)
    if sigma < 1e-10:
        return 0.0
    y = sp_norm.cdf(data, loc=mu, scale=sigma)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    n = len(y)
    n_vectors = n - tau * (m - 1)
    if n_vectors <= 0:
        return 0.0
    delta_dist = 1.0 / c
    col0 = _fix(y[:n_vectors] / delta_dist) + 1
    col0 = np.clip(col0, 1, c)
    powers = np.array([c ** (m - 1 - i) for i in range(m)], dtype=np.int64)
    embd = (col0 - 1).astype(np.int64) * powers[0]
    for k in range(1, m):
        diff = y[k * tau: n_vectors + k * tau] - y[:n_vectors]
        delta_k = _fix(diff / delta_dist)
        embd += (col0 + delta_k - 1).astype(np.int64) * powers[k]
    counts = np.bincount(embd[embd < c ** m], minlength=c ** m)
    p = counts[counts > 0] / n_vectors
    return -np.sum(p * np.log(p)) / np.log(c ** m)

def multi_scale_mipe(signal, scales=10):
    """
    多尺度改进排列熵
    """
    feats = []
    signal = signal.copy().ravel()
    L = len(signal)
    for s in range(1, scales + 1):
        new_len = L // s
        if new_len <= 0:
            feats.append(0.0)
            continue
        resampled = signal[:new_len * s].reshape(new_len, s).mean(axis=1)
        mipe = calculate_ipe(resampled)
        feats.append(mipe)
    return np.array(feats, dtype=np.float32)