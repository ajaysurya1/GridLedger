"""
GridLedger fusion module (Phase 3).

Fuses evidence from:
  1. Balance anomaly   (e_balance: CUSUM run probability proxy in [0,1])
  2. Customer suspects (e_suspect: best reconcile coverage in [0,1])
  3. Voltage physics   (e_volt: R² from LV digital-twin deviation in [0,1])

Uses a simple Dempster-Shafer-style product-of-likelihoods fusion.
All evidence values come from detection code only — NO oracle imports.
"""
from __future__ import annotations

import numpy as np


def fuse_evidence(
    e_balance: float,
    e_suspect: float,
    e_volt: float,
) -> dict:
    """
    Fuse three independent evidence channels into a unified case score.

    Parameters
    ----------
    e_balance : float  in [0,1]
        Strength of balance anomaly. Derived from CUSUM run metrics:
        S / h clipped to [0,1].  0 = no anomaly, 1 = strong alarm.
    e_suspect : float  in [0,1]
        Best reconcile coverage from customer-attribution layer.
        0 = no suspect identified, 1 = gap fully attributed to suspects.
    e_volt : float  in [0,1]
        Voltage cross-check R² score (× 1[alpha >= min]).
        0 = voltage is normal, 1 = strong voltage physics confirmation.

    Returns
    -------
    dict with keys:
        fused_score  float in [0,1]  — overall case confidence
        channel      str             — primary evidence driver
        note         str             — human-readable summary
    """
    # Weights (ASSUMPTION): balance is most reliable, voltage second, suspect third
    # Using log-odds fusion: log-odds_fused = sum of log-odds per channel
    # Prior odds = 0.1 (10% prior NTL probability per transformer-window)
    PRIOR_P = 0.10  # ASSUMPTION

    def _to_log_odds(p: float, strength: float, w: float) -> float:
        """Convert evidence strength to weighted log-odds update."""
        # LR = 1 + strength * (max_LR - 1) where max_LR ≈ 20 for strong channel
        max_lr = 20.0
        lr = 1.0 + float(np.clip(strength, 0.0, 1.0)) * (max_lr - 1.0)
        return w * np.log(lr)

    # Per-channel weights (sum to 1.0) — ASSUMPTION
    w_balance = 0.50   # ASSUMPTION
    w_volt    = 0.30   # ASSUMPTION
    w_suspect = 0.20   # ASSUMPTION

    prior_lo = np.log(PRIOR_P / (1.0 - PRIOR_P))
    update = (
        _to_log_odds(PRIOR_P, e_balance, w_balance)
        + _to_log_odds(PRIOR_P, e_volt,    w_volt)
        + _to_log_odds(PRIOR_P, e_suspect, w_suspect)
    )
    posterior_lo = prior_lo + update
    posterior_p = float(1.0 / (1.0 + np.exp(-posterior_lo)))
    fused_score = float(np.clip(posterior_p, 0.0, 1.0))

    # Identify primary driver
    channels = {
        "Balance (CUSUM)": e_balance,
        "Voltage physics": e_volt,
        "Customer suspects": e_suspect,
    }
    primary = max(channels, key=lambda k: channels[k])

    if fused_score >= 0.75:
        confidence = "HIGH"
    elif fused_score >= 0.45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    note = (
        f"Fused score {fused_score:.2f} ({confidence} confidence). "
        f"Primary driver: {primary} (e={channels[primary]:.2f}). "
        f"Balance={e_balance:.2f}, Voltage={e_volt:.2f}, Suspects={e_suspect:.2f}."
    )

    return {
        "fused_score": fused_score,
        "confidence": confidence,
        "channel": primary,
        "e_balance": float(e_balance),
        "e_volt": float(e_volt),
        "e_suspect": float(e_suspect),
        "note": note,
    }
