"""Forecast error metrics: RMSE, MAE, MAPE, SMAPE, R2."""
import numpy as np


def _arrays(y, yhat):
    return np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)


def rmse(y, yhat):
    y, yhat = _arrays(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    y, yhat = _arrays(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    y, yhat = _arrays(y, yhat)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y, yhat):
    y, yhat = _arrays(y, yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    ratio = np.divide(np.abs(y - yhat), denom, out=np.zeros_like(y), where=denom != 0)
    return float(np.mean(ratio) * 100)


def r2(y, yhat):
    y, yhat = _arrays(y, yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def all_metrics(y, yhat):
    return {
        "rmse": rmse(y, yhat),
        "mae": mae(y, yhat),
        "mape": mape(y, yhat),
        "smape": smape(y, yhat),
        "r2": r2(y, yhat),
    }