"""Metrics computation for Remaining Useful Life (RUL) prediction.

Includes:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 (Coefficient of Determination)
- NASA Asymmetric Scoring Function for C-MAPSS
"""

from __future__ import annotations

import numpy as np


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the official NASA asymmetric scoring function for C-MAPSS.

    Penalty formula:
        d = y_pred - y_true
        s_i = exp(-d / 13) - 1   if d < 0 (early prediction, lower penalty)
        s_i = exp(d / 10) - 1    if d >= 0 (late prediction, higher penalty)

    Parameters
    ----------
    y_true : np.ndarray
        True remaining useful life values.
    y_pred : np.ndarray
        Predicted remaining useful life values.

    Returns
    -------
    float
        Total NASA asymmetric score (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Compute standard regression metrics and NASA score for RUL predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth RUL.
    y_pred : np.ndarray
        Model predicted RUL.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, R2, and NASA_Score.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # R-squared calculation
    y_mean = np.mean(y_true)
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))

    if ss_tot > 1e-12:
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = None

    nasa_score = compute_nasa_score(y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4) if r2 is not None else None,
        "NASA_Score": round(nasa_score, 2),
    }
