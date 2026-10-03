"""
GridLedger oracle — ground-truth wrapper.

ONLY the following modules may import this:
  evaluate.py, verify.py, savings.py, tests/

Detection code (loss_model, balance, persistence, suspects, reconcile, triage,
voltage, fusion, pipeline) must NEVER import this module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gridledger.schema import World


def stolen_kwh_by_node(world: "World", today_day: int) -> pd.DataFrame:
    """
    Return a DataFrame of (scenario_id, kind, node, stolen_kwh) up to today_day.
    For oracle / evaluation use only.
    """
    T_now = today_day * 24
    rows = []
    for sc in world.truth.scenarios:
        s_kwh = world.truth.stolen_kwh.get(sc.id, np.zeros(T_now))
        total = float(s_kwh[:T_now].sum())
        rows.append({
            "scenario_id": sc.id,
            "kind": sc.kind,
            "target": sc.target,
            "innocent": sc.innocent,
            "stolen_kwh": total,
        })
    return pd.DataFrame(rows)


def true_tx_assignment(world: "World") -> np.ndarray:
    """Return the physical transformer index per customer. Oracle only."""
    return world.truth.tx_true.copy()


def tap_profile(world: "World", tx_idx: int) -> np.ndarray:
    """Return illegal tap kWh[T] for a transformer. Oracle only."""
    return world.truth.tap_kwh[tx_idx].copy()


def seg_profile(world: "World", feeder_idx: int) -> np.ndarray:
    """Return feeder segment theft kWh[T]. Oracle only."""
    return world.truth.seg_kwh[feeder_idx].copy()
