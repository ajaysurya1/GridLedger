"""
GridLedger reconcile module — bounded least squares + backward elimination.

Attributes unexplained grid boundary loss (gap) to customer shortfalls.
Uses scipy.optimize.lsq_linear with bounds (0, 1) and sparse backward elimination.

Rule: Never imports gridledger.oracle or touches ground-truth objects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from gridledger.config import Config, CFG


def reconcile(gap: np.ndarray, A: np.ndarray) -> Dict[str, Any]:
    """
    Fit gap ~= A @ theta, with 0 <= theta <= 1.

    Parameters
    ----------
    gap : 1D array of shape [n_blocks], unexplained energy per block
    A   : 2D array of shape [n_blocks, n_candidates], candidate shortfalls per block

    Returns
    -------
    dict with theta, contrib, explained, gap, coverage, r2, ss
    """
    A = np.maximum(A, 0.0)
    gap = np.asarray(gap, float)
    G = float(gap.sum())

    if A.shape[1] == 0 or G <= 0:
        return dict(
            theta=np.zeros(A.shape[1]),
            contrib=np.zeros(A.shape[1]),
            explained=0.0,
            gap=G,
            coverage=0.0,
            r2=0.0,
            ss=float((gap ** 2).sum()),
        )

    th = lsq_linear(A, gap, bounds=(0.0, 1.0), method="bvls").x
    fit = A @ th
    ss = float(((gap - fit) ** 2).sum())
    sst = float(((gap - gap.mean()) ** 2).sum()) + 1e-12
    contrib = th * A.sum(axis=0)

    explained = float(contrib.sum())
    coverage = float(np.clip(explained / G, 0.0, 1.5))
    r2 = 1.0 - ss / sst

    return dict(
        theta=th,
        contrib=contrib,
        explained=explained,
        gap=G,
        coverage=coverage,
        r2=r2,
        ss=ss,
    )


def sparse_reconcile(gap: np.ndarray, A: np.ndarray, min_gain: float = 0.02) -> Dict[str, Any]:
    """
    Backward elimination: drop a candidate if removing it raises residual SS by < min_gain * SS_total.

    Returns dict with out from reconcile plus theta_full [n_candidates] and active [list of indices].
    """
    gap = np.asarray(gap, float)
    A = np.asarray(A, float)
    n_cands = A.shape[1]

    if n_cands == 0:
        out = reconcile(gap, A)
        out["theta_full"] = np.zeros(0)
        out["active"] = []
        return out

    if n_cands == 1:
        out = reconcile(gap, A)
        out["theta_full"] = out["theta"]
        out["active"] = [0] if out["theta"][0] > 0 else []
        return out

    active = list(range(n_cands))
    sst = float(((gap - np.mean(gap)) ** 2).sum()) + 1e-12

    while len(active) > 1:
        base = reconcile(gap, A[:, active])["ss"]
        trials = [
            (reconcile(gap, A[:, [a for a in active if a != k]])["ss"] - base, k)
            for k in active
        ]
        inc, k = min(trials)
        if inc < min_gain * sst:
            active.remove(k)
        else:
            break

    out = reconcile(gap, A[:, active])
    full = np.zeros(n_cands)
    full[active] = out["theta"]
    out["theta_full"] = full
    out["active"] = active
    return out


def classify_coverage(coverage: float, cfg: Config = CFG) -> str:
    """
    Classify energy attribution into diagnosis categories.
    coverage >= 0.7  -> CUSTOMER-ATTRIBUTABLE
    0.3 <= cov < 0.7 -> MIXED
    coverage < 0.3   -> UNMETERED / LINE
    """
    if coverage >= cfg.coverage_attributable:
        return "CUSTOMER-ATTRIBUTABLE"
    elif coverage >= cfg.coverage_mixed:
        return "MIXED"
    else:
        return "UNMETERED/LINE"


def reconcile_tree(obs: Any, results: Any, today_day: int) -> pd.DataFrame:
    """
    Full tree reconciliation table (Phase 1 legacy stub preserved).
    """
    rows = []
    for nid, nf in results.nodes.items():
        e_in = float(nf.e_in_daily.sum())
        loss = float(nf.loss_daily.sum())
        metered = float(nf.metered_daily.sum())
        unaccounted = e_in - loss - metered
        rows.append({
            "node_id": nid,
            "e_in": e_in,
            "e_out": metered,
            "loss": loss,
            "unaccounted": unaccounted,
        })
    return pd.DataFrame(rows)
