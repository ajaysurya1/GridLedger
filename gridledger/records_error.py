"""
GridLedger records error module — mapping swap detection and what-if simulation.

Detects when a customer meter is erroneously listed under one transformer (causing a negative run)
while its physical energy is being delivered by another transformer (causing a positive run).

Rule: Never imports gridledger.oracle or touches ground-truth objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import pandas as pd

from gridledger.config import Config, CFG
from gridledger.schema import Observed
from gridledger.reconcile import reconcile
from gridledger.loss_model import fit_loss_model, residual, temp_factor
from gridledger.balance import (
    metered_sum_by_tx,
    calibrate_blocks,
    daily_sigma,
    cusum_runs,
)
from gridledger.persistence import classify_runs, Run


@dataclass
class RecordsErrorCandidate:
    meter_idx: int
    meter_id: str
    from_tx_idx: int
    from_tx_id: str
    to_tx_idx: int
    to_tx_id: str
    theta: float
    coverage: float
    confirmed: bool
    explanation: str
    before_after: Dict[str, Any]


def whatif_reassign(
    obs: Observed,
    meter_idx: int,
    to_tx: int,
    today_day: int,
    cfg: Config = CFG,
    from_tx: Optional[int] = None,
    cleaned_meter: Optional[np.ndarray] = None,
    imputed_kwh: Optional[np.ndarray] = None,
    met_orig: Optional[np.ndarray] = None,
    imp_orig: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Simulate reassigning meter_idx to to_tx.
    Fast evaluation: adjusts metered sums of from_tx and to_tx directly.
    """
    T = today_day * 24
    n_days = today_day
    mapping_now = obs.mapping[obs.mapping["valid_from_day"] < today_day]

    # Determine original tx
    if from_tx is None:
        curr = mapping_now[mapping_now["meter_idx"] == meter_idx]["tx_idx"]
        if len(curr) == 0:
            return {"confirmed": False, "error": "meter not in mapping"}
        from_tx = int(curr.iloc[-1])

    if from_tx == to_tx:
        return {"confirmed": False, "error": "target tx is same as source tx"}

    # Clean meter data if not provided
    if cleaned_meter is None or imputed_kwh is None:
        from gridledger.pipeline import _clean_meter_data
        cleaned_meter, _, imputed_kwh = _clean_meter_data(
            obs.meter_kwh[:, :T], obs.customers, cfg
        )

    n_tx = len(obs.transformers)
    if met_orig is None:
        met_orig = metered_sum_by_tx(cleaned_meter, mapping_now, n_tx, n_days)
    if imp_orig is None:
        imp_orig = metered_sum_by_tx(imputed_kwh, mapping_now, n_tx, n_days)

    # Adjust only the hours where meter_idx was mapped to from_tx
    c_m = cleaned_meter[meter_idx, :T]
    c_imp = imputed_kwh[meter_idx, :T]

    meter_mapping = mapping_now[mapping_now["meter_idx"] == meter_idx].sort_values("valid_from_day")
    mapped_tx_hourly = np.zeros(T, dtype=np.int32)
    for _, row in meter_mapping.iterrows():
        start_h = int(row["valid_from_day"]) * 24
        mapped_tx_hourly[start_h:] = int(row["tx_idx"])

    mask_from = (mapped_tx_hourly == from_tx)

    met_new_from = met_orig[from_tx].copy()
    met_new_from[mask_from] -= c_m[mask_from]

    met_new_to = met_orig[to_tx].copy()
    met_new_to[mask_from] += c_m[mask_from]

    imp_new_from = imp_orig[from_tx].copy()
    imp_new_from[mask_from] -= c_imp[mask_from]

    imp_new_to = imp_orig[to_tx].copy()
    imp_new_to[mask_from] += c_imp[mask_from]

    tf = temp_factor(obs.temp[:T], cfg.kappa, cfg.t_ref)

    def _eval_tx_fast(ji: int, metered_arr: np.ndarray, imp_arr: np.ndarray):
        tx_row = obs.transformers[obs.transformers["tx_idx"] == ji].iloc[0]
        tx_id = str(tx_row["tx_id"])
        e_in = obs.tx_in_kwh[ji, :T].astype(np.float64)
        metered = metered_arr.astype(np.float64)
        imp = imp_arr.astype(np.float64)

        from gridledger.pipeline import _analyse_node
        finding = _analyse_node(
            tx_id, "transformer", e_in, metered, imp, tf, n_days, cfg.cal_days, cfg, [],
            tx_idx=ji,
        )
        return finding

    finding_from_before = _eval_tx_fast(from_tx, met_orig[from_tx], imp_orig[from_tx])
    finding_from_after  = _eval_tx_fast(from_tx, met_new_from, imp_new_from)

    finding_to_before   = _eval_tx_fast(to_tx, met_orig[to_tx], imp_orig[to_tx])
    finding_to_after    = _eval_tx_fast(to_tx, met_new_to, imp_new_to)

    pos_runs_to_before = [r for r in finding_to_before.runs if r.active and r.direction == "positive"]
    pos_runs_to_after  = [r for r in finding_to_after.runs if r.active and r.direction == "positive"]

    neg_runs_from_before = [r for r in finding_from_before.runs if r.active and r.direction == "negative"]
    neg_runs_from_after  = [r for r in finding_from_after.runs if r.active and r.direction == "negative"]

    # Genuine records swap has excess on from_tx and deficit on to_tx
    had_imbalance_from = (len(neg_runs_from_before) > 0) or (float(np.mean(finding_from_before.R_daily[-7:])) <= -1.5)
    had_imbalance_to   = (len(pos_runs_to_before) > 0) or (float(np.mean(finding_to_before.R_daily[-7:])) >= 1.5)
    if not (had_imbalance_from and had_imbalance_to):
        return {"confirmed": False, "reason": "Both transformers must have an active imbalance"}

    # Confirmed when negative run on from_tx and positive run on to_tx clear or drop by >= 50%
    cleared_from = (len(neg_runs_from_after) == 0) or (abs(float(np.mean(finding_from_after.R_daily[-7:]))) < 0.5 * max(1.0, abs(float(np.mean(finding_from_before.R_daily[-7:])))))
    cleared_to   = (len(pos_runs_to_after) == 0) or (float(np.mean(finding_to_after.R_daily[-7:])) < 0.5 * max(1.0, float(np.mean(finding_to_before.R_daily[-7:]))))
    confirmed = cleared_from and cleared_to

    return {
        "confirmed": confirmed,
        "from_tx": from_tx,
        "to_tx": to_tx,
        "before_runs_from": neg_runs_from_before,
        "after_runs_from": neg_runs_from_after,
        "before_runs_to": pos_runs_to_before,
        "after_runs_to": pos_runs_to_after,
        "before_finding_from": finding_from_before,
        "after_finding_from": finding_from_after,
        "before_finding_to": finding_to_before,
        "after_finding_to": finding_to_after,
    }


def detect_records_errors(
    nodes: Dict[str, Any],
    obs: Observed,
    cleaned_meter: np.ndarray,
    today_day: int,
    cfg: Config = CFG,
) -> List[RecordsErrorCandidate]:
    """
    Search for records errors (mapping swaps) across transformers.
    Identifies candidate customers on over-credited transformers (negative runs)
    whose consumption profile matches the deficit on an under-credited transformer (positive run).
    """
    tx_meta = obs.transformers.set_index("tx_id")
    mapping_now = obs.mapping[obs.mapping["valid_from_day"] < today_day]
    cust_meta = obs.customers.set_index("meter_idx")

    # Find transformers with negative active runs or persistent negative drift (metered > physical)
    neg_tx_nodes = []
    for nid, nf in nodes.items():
        if nf.kind != "transformer":
            continue
        neg_runs = [r for r in nf.runs if r.direction == "negative"]
        if neg_runs:
            for r in neg_runs:
                neg_tx_nodes.append((nid, nf, r.start, r.end))
        elif np.sum(nf.R_daily[cfg.cal_days:today_day] < -2.0) >= 3:
            start_d = max(cfg.cal_days, today_day - 20)
            end_d = today_day - 1
            neg_tx_nodes.append((nid, nf, start_d, end_d))

    # Find transformers with positive active runs or notable positive CUSUM
    pos_tx_nodes = [
        (nid, nf, r)
        for nid, nf in nodes.items()
        if nf.kind == "transformer"
        for r in nf.runs
        if r.direction == "positive"
    ]

    n_tx = len(obs.transformers)
    met_orig = metered_sum_by_tx(cleaned_meter, mapping_now, n_tx, today_day)
    imp_orig = np.zeros_like(met_orig)

    candidates: List[RecordsErrorCandidate] = []
    seen_meters = set()

    for neg_id, neg_nf, start_day, end_day in neg_tx_nodes:
        cust_neg = mapping_now[mapping_now["tx_idx"] == neg_nf.tx_idx]["meter_idx"].unique()
        if len(cust_neg) == 0:
            continue

        w_start = start_day * 24
        w_end = (end_day + 1) * 24
        n_blocks = max(1, (w_end - w_start) // 6)
        b_lim = n_blocks * 6

        # Negative deficit (excess metered energy)
        neg_gap_blocks = np.maximum(0.0, -neg_nf.r_hourly[w_start: w_start + b_lim]).reshape(n_blocks, 6).sum(axis=1)
        if neg_gap_blocks.sum() <= 0:
            continue

        # Build consumption matrix for negative transformer customers
        A_cons = np.zeros((n_blocks, len(cust_neg)), dtype=np.float64)
        for i, ci in enumerate(cust_neg):
            A_cons[:, i] = cleaned_meter[ci, w_start: w_start + b_lim].reshape(n_blocks, 6).sum(axis=1)

        out_neg = reconcile(neg_gap_blocks, A_cons)
        cand_indices = [
            i for i, th in enumerate(out_neg["theta"])
            if th >= 0.40 and out_neg["contrib"][i] >= 0.25 * neg_gap_blocks.sum()
        ]

        # If no strict threshold match, test top candidate by contribution
        if not cand_indices and len(out_neg["contrib"]) > 0:
            cand_indices = [int(np.argmax(out_neg["contrib"]))]

        for i in cand_indices:
            ci = int(cust_neg[i])
            if ci in seen_meters:
                continue

            c_row = cust_meta.loc[ci] if ci in cust_meta.index else None
            meter_id = str(c_row["meter_id"]) if c_row is not None else f"M{ci:04d}"
            c_blocks = A_cons[:, i]

            # Candidate destination transformers: prioritize positive run transformers, then all others
            cand_dest_nodes = [
                (p_id, p_nf.tx_idx) for p_id, p_nf, _ in pos_tx_nodes if p_nf.tx_idx != neg_nf.tx_idx
            ]
            other_nodes = [
                (str(r["tx_id"]), int(r["tx_idx"]))
                for _, r in obs.transformers.iterrows()
                if int(r["tx_idx"]) != neg_nf.tx_idx and int(r["tx_idx"]) not in [ji for _, ji in cand_dest_nodes]
            ]
            all_cand_dests = cand_dest_nodes + other_nodes

            for cand_id, cand_ji in all_cand_dests:
                swap_eval = whatif_reassign(
                    obs=obs,
                    meter_idx=ci,
                    to_tx=cand_ji,
                    today_day=today_day,
                    cfg=cfg,
                    from_tx=neg_nf.tx_idx,
                    cleaned_meter=cleaned_meter,
                    met_orig=met_orig,
                    imp_orig=imp_orig,
                )

                if swap_eval.get("confirmed", False):
                    seen_meters.add(ci)
                    candidates.append(RecordsErrorCandidate(
                        meter_idx=ci,
                        meter_id=meter_id,
                        from_tx_idx=neg_nf.tx_idx,
                        from_tx_id=neg_id,
                        to_tx_idx=cand_ji,
                        to_tx_id=cand_id,
                        theta=float(out_neg["theta"][i]),
                        coverage=1.0,
                        confirmed=True,
                        explanation=(
                            f"Meter {meter_id} registered on {neg_id} physically belongs to {cand_id}; "
                            f"what-if swap confirms both transformer balances normalize."
                        ),
                        before_after=swap_eval,
                    ))
                    break

    return candidates
