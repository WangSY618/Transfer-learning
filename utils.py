import numpy as np
def MAPE(v, v_, thr = 1):
    idx = v >= thr
    return np.mean(np.abs(v_[idx] - v[idx]) / (v[idx] + 1e-6))
def MAE(v, v_):
    return np.mean(np.abs(v_ - v))

