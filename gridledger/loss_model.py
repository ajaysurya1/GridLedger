"""
GridLedger loss model — fit and residuals for one node.

Detection code only — never imports oracle or touches ground truth.
All functions are pure (no side effects, no global state).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def fit_loss_model(
    e_in: np.ndarray,
    metered: np.ndarray,
    tf: np.ndarray,
) -> tuple[float, float]:
    """
    Fit L = (a + b * E_in²) * tf by non-negative least squares.

    Parameters
    ----------
    e_in   : [N] hourly energy input (kWh)
    metered: [N] hourly metered sum (kWh), same length as e_in
    tf     : [N] temperature factor 1 + kappa*(temp - t_ref)

    Returns
    -------
    (a, b) — both >= 0
    """
    y = (e_in - metered) / np.where(tf > 0, tf, 1.0)
    X = np.column_stack([np.ones_like(e_in), e_in ** 2])
    scale = np.array([1.0, max(float(np.max(e_in)) ** 2, 1e-9)])
    coef, _ = nnls(X / scale, y)
    a, b = coef / scale
    return float(a), float(b)


def residual(
    e_in: np.ndarray,
    metered: np.ndarray,
    a: float,
    b: float,
    tf: np.ndarray,
) -> np.ndarray:
    """
    Unexplained kWh per hour.

    residual > 0  → energy unaccounted (possible NTL)
    residual < 0  → over-credited (possible mapping error)
    """
    return e_in - metered - (a + b * e_in ** 2) * tf


def temp_factor(temp: np.ndarray, kappa: float, t_ref: float) -> np.ndarray:
    """Return tf = 1 + kappa*(temp - t_ref)."""
    return 1.0 + kappa * (temp - t_ref)
