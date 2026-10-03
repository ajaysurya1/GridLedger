"""
GridLedger triage module (Phase 2).

Prioritises confirmed cases by urgency, financial impact and visit cost.
Stub for Phase 1.
"""
from __future__ import annotations
from typing import Any
import pandas as pd


def triage(results: Any, cfg: Any) -> pd.DataFrame:
    """
    Return prioritised case list with recommended action type.
    Phase 2 implementation.
    """
    return pd.DataFrame(columns=["node_id", "priority", "action", "est_inr_saved"])
