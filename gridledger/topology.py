"""
GridLedger topology utilities.
Build the grid tree and query adjacency without touching oracle data.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import pandas as pd


def build_grid_graph(
    feeders: pd.DataFrame,
    transformers: pd.DataFrame,
) -> nx.DiGraph:
    """
    Build a directed graph: substation -> feeder -> transformer.

    Node attributes:
        kind: 'substation' | 'feeder' | 'transformer'
        label: human-readable id
    """
    G = nx.DiGraph()
    G.add_node("SUB", kind="substation", label="Substation")

    for _, f in feeders.iterrows():
        fid = f["feeder_id"]
        G.add_node(fid, kind="feeder", label=fid)
        G.add_edge("SUB", fid)

    for _, t in transformers.iterrows():
        tid = t["tx_id"]
        fi = feeders[feeders["feeder_idx"] == t["feeder_idx"]]["feeder_id"].iloc[0]
        G.add_node(tid, kind="transformer", label=tid)
        G.add_edge(fi, tid)

    return G


def get_tx_of_feeder(transformers: pd.DataFrame, feeder_idx: int) -> List[str]:
    """Return list of tx_id strings belonging to a feeder."""
    return list(transformers[transformers["feeder_idx"] == feeder_idx]["tx_id"])


def get_feeder_of_tx(transformers: pd.DataFrame, tx_id: str) -> Optional[str]:
    """Return feeder_idx for a transformer id."""
    row = transformers[transformers["tx_id"] == tx_id]
    if row.empty:
        return None
    return int(row["feeder_idx"].iloc[0])
