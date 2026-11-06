"""Regression evaluation metrics.

Note the distinction from training: LinearRegression/RandomForestRegressor
internally minimize a *loss* (squared error) during fit(). These functions
compute *metrics* -- chosen for human interpretability, not for being
optimizable -- to judge the result afterward. See docs/glossary.md
("Loss function vs. evaluation metric").
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}
