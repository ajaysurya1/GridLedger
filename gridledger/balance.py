"""
GridLedger balance engine — metered aggregation and CUSUM machinery.

No oracle imports. Receives only Observed data.
All functions are pure and vectorised.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from gridledger.config import Config, CFG


# ─────────────────────────────────────────────────────────────────────────────
# Metered sum aggregation
# ─────────────────────────────────────────────────────────────────────────────

def metered_sum_by_tx(
    meter_kwh: np.ndarray,          # [n_c, T]  may contain NaN
    mapping: pd.DataFrame,          # meter_idx, tx_idx, valid_from_day
    n_tx: int,
    n_days: int,
) -> np.ndarray:
    """
    Sum metered kWh into transformer buckets, honouring valid_from_day.

    Returns [n_tx, T] array with NaN propagated only where ALL customers NaN.
    Missing meters are treated as 0 contribution (not added to sum).
    """
    T = n_days * 24
    result = np.zeros((n_tx, T), dtype=np.float64)

    # Pre-sort: for each meter, find its tx assignment per day
    meter_indices = mapping["meter_idx"].values
    tx_indices = mapping["tx_idx"].values
    from_days = mapping["valid_from_day"].values

    # Build per-meter time-series of tx assignment
    # Group by meter_idx, take the last valid record before each day
    grouped = mapping.sort_values("valid_from_day").groupby("meter_idx")

    days = np.arange(n_days)
    for meter_idx, grp in grouped:
        grp = grp.sort_values("valid_from_day")
        from_arr = grp["valid_from_day"].values
        tx_arr = grp["tx_idx"].values

        # For each day, find the applicable tx
        # searchsorted: last record with valid_from_day <= day
        day_tx = tx_arr[np.searchsorted(from_arr, days, side="right") - 1]  # [n_days]

        for d, ji in enumerate(day_tx):
            if 0 <= ji < n_tx:
                h_start = d * 24
                h_end = h_start + 24
                row = meter_kwh[meter_idx, h_start:h_end]
                # NaN = missing; treat as 0 in the sum
                result[ji, h_start:h_end] += np.where(np.isnan(row), 0.0, row)

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Noise calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_blocks(
    r_cal: np.ndarray,      # residuals over calibration window [N_cal_hours]
    hours_cal: np.ndarray,  # hour index for each element [N_cal_hours]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit 6-hour block (μ, σ) on calibration residuals.

    Returns mu [4] and sig [4] (robust: MAD-based).
    """
    mu = np.zeros(4)
    sig = np.zeros(4)
    for k in range(4):
        x = r_cal[(hours_cal // 6) == k]
        if len(x) == 0:
            sig[k] = 1e-3
            continue
        mu[k] = np.median(x)
        sig[k] = max(1.4826 * np.median(np.abs(x - mu[k])), 1e-3)
    return mu, sig


def daily_sigma(
    r_cal_full: np.ndarray,     # residuals for the calibration window [cal_days*24]
    cal_days: int,
    sig_block: np.ndarray,      # [4] from calibrate_blocks
    inflate: float,
) -> float:
    """
    Compute robust daily sigma incorporating both empirical daily variation
    and an independent-noise floor from hourly block sigmas.
    """
    R = r_cal_full[: cal_days * 24].reshape(cal_days, 24).sum(axis=1)  # [cal_days]
    mad = 1.4826 * np.median(np.abs(R - np.median(R)))
    # Independent-noise floor: each 6-hour block contributes 6 hours of variance
    indep = float(np.sqrt((np.repeat(sig_block, 6) ** 2).sum()))
    return max(mad, indep, 1e-3) * inflate


# ─────────────────────────────────────────────────────────────────────────────
# CUSUM
# ─────────────────────────────────────────────────────────────────────────────

def cusum_runs(
    z: np.ndarray,
    k: float = 0.5,
    h: float = 6.0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    One-sided upper CUSUM on standardised daily scores z.

    Returns
    -------
    S     : CUSUM statistic array [len(z)]
    runs  : list of (start_day, end_day) pairs where an alarm fired.
            start_day = estimated change point (last zero crossing + 1).
    """
    n = len(z)
    S = np.zeros(n, dtype=float)
    s = 0.0
    last_zero = -1
    runs: List[Tuple[int, int]] = []
    run_start: int | None = None

    for i in range(n):
        s = max(0.0, s + z[i] - k)
        S[i] = s
        if s == 0.0:
            last_zero = i
        if s > h and run_start is None:
            run_start = last_zero + 1
        if run_start is not None and s == 0.0:
            runs.append((run_start, i - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, n - 1))

    return S, runs


# ─────────────────────────────────────────────────────────────────────────────
# Feeder balance (disjoint from transformer residuals)
# ─────────────────────────────────────────────────────────────────────────────

def feeder_residual(
    feeder_in: np.ndarray,      # [T]
    tx_in_sum: np.ndarray,      # [T] sum of all tx_in under this feeder
    tf: np.ndarray,             # [T] temperature factor
    a_f: float,
    b_f: float,
) -> np.ndarray:
    """
    r_f = F_in - sum(tx_in) - L_f(F_in)

    Disjoint from transformer residuals: never add to tx residuals.
    """
    L_f = (a_f + b_f * feeder_in ** 2) * tf
    return feeder_in - tx_in_sum - L_f
