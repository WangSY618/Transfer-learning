import numpy as np

def MAPE(v, v_, thr = 1):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param thr: 
    :return: int, MAPE averages on all elements of input.
    '''

    idx = v >= thr

    return np.mean(np.abs(v_[idx] - v[idx]) / (v[idx] + 1e-6))

def SMAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''

    # idx = ((v > 0) & (v_ > 0))
    idx = (v > 0)

    return np.mean(np.abs(v_[idx] - v[idx]) / ((v[idx] + v_[idx]) / 2  + 1e-6))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v)**2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))

