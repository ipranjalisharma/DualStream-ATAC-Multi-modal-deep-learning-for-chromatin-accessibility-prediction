"""Regression evaluation metrics for ATAC signal prediction."""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_mse(y_true, y_pred):
    """Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_r2(y_true, y_pred):
    """R-squared (coefficient of determination)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_pearson(y_true, y_pred):
    """Pearson correlation coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def compute_spearman(y_true, y_pred):
    """Spearman rank correlation coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def compute_all_metrics(y_true, y_pred):
    """
    Compute all regression metrics.

    Args:
        y_true: Ground truth continuous values (numpy array).
        y_pred: Predicted continuous values (numpy array).

    Returns:
        dict: Dictionary with all computed metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return {
        'mse': compute_mse(y_true, y_pred),
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred),
        'pearson_r': compute_pearson(y_true, y_pred),
        'spearman_rho': compute_spearman(y_true, y_pred),
    }
