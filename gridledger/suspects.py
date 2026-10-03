"""
GridLedger suspects module (Phase 2).

Ranks candidate causes within an already-localised transformer zone.
Stub for Phase 1 — implements the interface, logic added in Phase 2.
"""
from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd


def rank_suspects(
    tx_id: str,
    finding: Any,
    obs: Any,
    today_day: int,
) -> pd.DataFrame:
    """
    Rank customers within a transformer zone by suspicion score.
    Returns DataFrame[meter_id, score, reason].
    Phase 2 implementation.
    """
    return pd.DataFrame(columns=["meter_id", "score", "reason"])
