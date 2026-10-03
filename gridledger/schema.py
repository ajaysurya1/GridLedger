"""
GridLedger data schema.

Observed  — what a utility really has; the ONLY input to detection code.
Truth     — ground truth; accessible only to oracle.py, evaluate.py, verify.py, savings.py, tests.
World     — container returned by the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Observed:
    """Everything the utility can see. Detection code MUST only receive this."""

    # ── Timestamps ────────────────────────────────────────────────────────
    t0: pd.Timestamp                    # simulation start
    temp: np.ndarray                    # [T] hourly °C (public weather)

    # ── Reference DataFrames ──────────────────────────────────────────────
    customers: pd.DataFrame             # meter_idx, meter_id, cls, tariff_cls
    transformers: pd.DataFrame          # tx_idx, tx_id, feeder_idx, rated_kw
    feeders: pd.DataFrame               # feeder_idx, feeder_id

    # ── Interval data (kWh per hour) ──────────────────────────────────────
    meter_kwh: np.ndarray               # [n_customers, T]  NaN = missing
    tx_in_kwh: np.ndarray               # [n_tx, T]
    feeder_in_kwh: np.ndarray           # [n_feeders, T]

    # ── Mapping (piecewise-constant, utility records) ─────────────────────
    mapping: pd.DataFrame               # meter_idx, tx_idx, valid_from_day

    # ── Extras ────────────────────────────────────────────────────────────
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Truth:
    """Ground truth — only oracle.py / evaluate.py / savings.py / tests may use this."""

    true_kwh: np.ndarray                # [n_customers, T]  actual consumption
    tap_kwh: np.ndarray                 # [n_tx, T]  illegal tap load (0 if none)
    seg_kwh: np.ndarray                 # [n_feeders, T]  segment loss theft (0 if none)
    tx_true: np.ndarray                 # [n_customers]  int, physical tx assignment
    scenarios: List[Any]                # list of Scenario objects
    stolen_kwh: Dict[str, np.ndarray]   # scenario_id -> [T] kWh stolen per hour


@dataclass
class World:
    """Full simulation output. Callers must use world.observed for detection."""

    observed: Observed
    truth: Truth
    cfg: Any          # Config
    seed: int
