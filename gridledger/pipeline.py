"""
GridLedger detection pipeline.

pipeline.run(obs, today_day, cfg) -> Results

Rules:
- Receives ONLY Observed data; never imports oracle.
- No peeking: all data sliced to T = today_day * 24 before processing.
- Pure function — no side effects; caller is responsible for caching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gridledger.config import Config, CFG
from gridledger.schema import Observed
from gridledger.loss_model import fit_loss_model, residual, temp_factor
from gridledger.balance import (
    metered_sum_by_tx,
    calibrate_blocks,
    daily_sigma,
    cusum_runs,
    feeder_residual,
)
from gridledger.persistence import classify_runs, Run
from gridledger.suspects import (
    compute_baselines,
    analyse_transformer_customers,
    CustomerDiagnostic,
)
from gridledger.reconcile import sparse_reconcile, reconcile
from gridledger.records_error import detect_records_errors, whatif_reassign
from gridledger.triage import Case, build_case_for_node
from gridledger.voltage import analyse_tx_voltage
from gridledger.fusion import fuse_evidence


# ─────────────────────────────────────────────────────────────────────────────
# Output types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeFinding:
    node_id: str
    kind: str                   # 'transformer' | 'feeder'
    status: str                 # 'RED' | 'AMBER' | 'GREEN'
    runs: List[Run]             # validated anomaly runs
    S: np.ndarray               # CUSUM statistic [n_days]
    R_daily: np.ndarray         # daily residuals [n_days]
    r_hourly: np.ndarray        # hourly residuals [T]
    e_in_daily: np.ndarray      # daily input energy [n_days]
    metered_daily: np.ndarray   # daily metered sum [n_days]
    loss_daily: np.ndarray      # daily estimated technical loss [n_days]
    sigma: float                # calibration sigma
    a: float                    # fitted loss param a
    b: float                    # fitted loss param b
    n_imputed: np.ndarray       # [n_days] imputed kWh per day
    dq_warnings: List[str]      # data quality warnings
    zone_label: str             # plain-English zone description
    feeder_idx: Optional[int] = None
    tx_idx: Optional[int] = None


@dataclass
class Results:
    nodes: Dict[str, NodeFinding]       # node_id -> NodeFinding
    quarantine: pd.DataFrame            # rows: meter_idx, hour, reason
    status_summary: Dict[str, int]      # RED/AMBER/GREEN counts
    kpis: Dict[str, float]
    today_day: int
    cases: List[Case] = field(default_factory=list)
    cases_by_node: Dict[str, Case] = field(default_factory=dict)
    records_errors: List[Any] = field(default_factory=list)
    diagnostics_by_tx: Dict[str, List[CustomerDiagnostic]] = field(default_factory=dict)
    reconcile_by_tx: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    voltage_by_tx: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fusion_by_tx: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    baselines: Optional[np.ndarray] = None
    shortfalls: Optional[np.ndarray] = None
    cleaned_meter: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Data cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean_meter_data(
    meter_kwh: np.ndarray,      # [n_c, T]
    customers: pd.DataFrame,
    cfg: Config,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    1. Flag duplicates / negatives (except solar) / absurd spikes.
    2. Impute NaN with calibration-window hour/day-type mean.
    Returns cleaned array, quarantine DataFrame, imputed_kwh [n_c, T].
    """
    n_c, T = meter_kwh.shape
    cleaned = meter_kwh.copy().astype(np.float64)
    imputed = np.zeros_like(cleaned)
    quarantine_rows: List[Dict] = []

    for ci in range(n_c):
        row = cleaned[ci]

        # Flag absurd spikes (> spike_factor * p99.9)
        valid = row[~np.isnan(row)]
        if len(valid) > 10:
            p999 = np.percentile(valid, 99.9)
            spike_mask = row > cfg.spike_factor * p999
            if spike_mask.any():
                for t in np.where(spike_mask)[0]:
                    quarantine_rows.append({"meter_idx": ci, "hour": int(t), "reason": "spike"})
                row[spike_mask] = np.nan  # quarantine → treat as missing

        # Flag negatives (not solar — we can't distinguish here without oracle)
        neg_mask = row < -0.01
        if neg_mask.any():
            for t in np.where(neg_mask)[0]:
                quarantine_rows.append({"meter_idx": ci, "hour": int(t), "reason": "negative"})
            # Only zero out if clearly negative (not slight solar reverse)
            row[row < -1.0] = np.nan

        cleaned[ci] = row

    # Calibration-only hour-of-day x weekend means, computed across customers
    # and time in batches. Records-error what-if checks invoke this cleaner
    # repeatedly, so avoid thousands of small nanmean calls per invocation.
    cal_T = min(cfg.cal_days * 24, T)
    hours = np.arange(T) % 24
    days = np.arange(T) // 24
    is_weekend = ((days + 5) % 7 >= 5).astype(int)
    groups = hours + 24 * is_weekend
    cal_groups = groups[:cal_T]
    cal_data = cleaned[:, :cal_T]
    valid = np.isfinite(cal_data)
    counts = np.zeros((n_c, 48), dtype=np.int32)
    totals = np.zeros((n_c, 48), dtype=np.float64)
    for group in range(48):
        group_mask = cal_groups == group
        if group_mask.any():
            values = cal_data[:, group_mask]
            group_valid = valid[:, group_mask]
            counts[:, group] = group_valid.sum(axis=1)
            totals[:, group] = np.where(group_valid, values, 0.0).sum(axis=1)

    global_count = valid.sum(axis=1)
    global_total = np.where(valid, cal_data, 0.0).sum(axis=1)
    fallback = np.divide(global_total, global_count, out=np.zeros(n_c), where=global_count > 0)
    means = np.divide(totals, counts, out=np.broadcast_to(fallback[:, None], totals.shape).copy(),
                      where=counts > 0)

    missing = np.isnan(cleaned)
    for group in range(48):
        positions = groups == group
        if not positions.any():
            continue
        group_missing = missing[:, positions]
        if group_missing.any():
            fill_values = means[:, group, None]
            cleaned[:, positions] = np.where(group_missing, fill_values, cleaned[:, positions])
            imputed[:, positions] = np.where(group_missing, fill_values, imputed[:, positions])

    quarantine_df = pd.DataFrame(quarantine_rows) if quarantine_rows else pd.DataFrame(
        columns=["meter_idx", "hour", "reason"]
    )
    return cleaned.astype(np.float32), quarantine_df, imputed.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-node analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_node(
    node_id: str,
    kind: str,
    e_in: np.ndarray,           # [T] hourly
    metered: np.ndarray,        # [T] hourly
    imputed_kwh: np.ndarray,    # [T] hourly imputed amount
    tf: np.ndarray,             # [T] temp factor
    n_days: int,
    cal_days: int,
    cfg: Config,
    dq_warnings: List[str],
    feeder_idx: Optional[int] = None,
    tx_idx: Optional[int] = None,
) -> NodeFinding:
    """Core analysis for one transformer or feeder node."""
    T = n_days * 24
    hours = np.arange(T) % 24

    # ── Calibration window slice ──────────────────────────────────────────
    cal_T = cal_days * 24
    e_in_cal = e_in[:cal_T]
    met_cal = metered[:cal_T]
    tf_cal = tf[:cal_T]

    # ── Fit loss model (calibration window only) ──────────────────────────
    a, b = fit_loss_model(e_in_cal, met_cal, tf_cal)

    # ── Hourly residuals (full window) ────────────────────────────────────
    r_hourly = residual(e_in, metered, a, b, tf)  # [T]

    # ── Block calibration ─────────────────────────────────────────────────
    r_cal = r_hourly[:cal_T]
    hours_cal = hours[:cal_T]
    mu_block, sig_block = calibrate_blocks(r_cal, hours_cal)

    # ── Sigma calibration ─────────────────────────────────────────────────
    sigma = daily_sigma(r_cal, cal_days, sig_block, cfg.sigma_inflate)

    # ── De-bias and daily aggregation ────────────────────────────────────
    r_debiased = r_hourly - mu_block[hours // 6]
    R_daily = r_debiased[:n_days * 24].reshape(n_days, 24).sum(axis=1)  # [n_days]

    # ── Imputed variance inflator ─────────────────────────────────────────
    imp_daily = imputed_kwh[:n_days * 24].reshape(n_days, 24).sum(axis=1)  # [n_days]
    # Variance from imputation uncertainty: assume ±50% of imputed value
    imp_var_daily = (0.5 * imp_daily) ** 2

    # ── Standardised scores ───────────────────────────────────────────────
    denom = np.sqrt(sigma ** 2 + imp_var_daily + 1e-6)
    z = R_daily / denom  # [n_days]

    # Use only post-calibration window for CUSUM
    z_post = z.copy()
    z_post[:cal_days] = 0.0  # suppress calibration period

    # ── CUSUM (positive runs: energy missing) ─────────────────────────────
    S_pos, raw_pos = cusum_runs(z_post, cfg.cusum_k, cfg.cusum_h)
    runs_pos = classify_runs(R_daily, S_pos, raw_pos, r_debiased, n_days, cfg)
    for run in runs_pos:
        run.direction = "positive"

    # ── CUSUM (negative runs: over-credited) ──────────────────────────────
    S_neg, raw_neg = cusum_runs(-z_post, cfg.cusum_k, cfg.cusum_h)
    runs_neg = classify_runs(-R_daily, S_neg, raw_neg, -r_debiased, n_days, cfg)
    for run in runs_neg:
        run.excess_kwh = -abs(run.excess_kwh)
        run.kwh_per_day = -abs(run.kwh_per_day)
        run.direction = "negative"

    all_runs = runs_pos + runs_neg
    S = S_pos  # primary CUSUM stat for display

    # ── Status ────────────────────────────────────────────────────────────
    has_active_red = any(r.active for r in all_runs)
    has_recent_run = any(n_days - 1 - r.end <= cfg.amber_recent_days for r in all_runs)
    amber_s = S_pos[-1] > cfg.amber_s_fraction * cfg.cusum_h if len(S_pos) > 0 else False

    if has_active_red:
        status = "RED"
    elif has_recent_run or amber_s or bool(dq_warnings):
        status = "AMBER"
    else:
        status = "GREEN"

    # ── Daily energy breakdown ────────────────────────────────────────────
    e_in_daily = e_in[:n_days * 24].reshape(n_days, 24).sum(axis=1)
    metered_daily = metered[:n_days * 24].reshape(n_days, 24).sum(axis=1)
    loss_daily = ((a + b * e_in ** 2) * tf)[:n_days * 24].reshape(n_days, 24).sum(axis=1)

    # ── Zone label ────────────────────────────────────────────────────────
    if kind == "transformer":
        zone_label = f"Transformer {node_id} secondary (customers / records / unmetered tap)"
    else:
        zone_label = f"Feeder {node_id} segment (line between feeder meter and transformer meters)"

    return NodeFinding(
        node_id=node_id,
        kind=kind,
        status=status,
        runs=all_runs,
        S=S,
        R_daily=R_daily,
        r_hourly=r_debiased,
        e_in_daily=e_in_daily,
        metered_daily=metered_daily,
        loss_daily=loss_daily,
        sigma=sigma,
        a=a,
        b=b,
        n_imputed=imp_daily,
        dq_warnings=dq_warnings,
        zone_label=zone_label,
        feeder_idx=feeder_idx,
        tx_idx=tx_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    obs: Observed,
    today_day: int,
    cfg: Config = CFG,
) -> Results:
    """
    Full detection pipeline.

    Parameters
    ----------
    obs       : Observed — only utility-visible data
    today_day : int — pipeline uses ONLY data from days < today_day
    cfg       : Config

    Returns
    -------
    Results with per-node findings, quarantine list, and KPIs.
    """
    # ── No-peeking slice ──────────────────────────────────────────────────
    T_now = today_day * 24
    n_days = today_day  # only analyse days 0..today_day-1

    meter_kwh_now = obs.meter_kwh[:, :T_now].copy()
    tx_in_now = obs.tx_in_kwh[:, :T_now].copy()
    feeder_in_now = obs.feeder_in_kwh[:, :T_now].copy()
    temp_now = obs.temp[:T_now]
    tf = temp_factor(temp_now, cfg.kappa, cfg.t_ref)

    n_tx = len(obs.transformers)
    n_f = len(obs.feeders)

    # ── Data cleaning ─────────────────────────────────────────────────────
    cleaned_meter, quarantine_df, imputed_kwh = _clean_meter_data(
        meter_kwh_now, obs.customers, cfg
    )

    # ── Metered sum by transformer (honours mapping) ──────────────────────
    mapping_now = obs.mapping[obs.mapping["valid_from_day"] < today_day].copy()
    metered_by_tx = metered_sum_by_tx(cleaned_meter, mapping_now, n_tx, n_days)  # [n_tx, T_now]

    # ── Imputed kWh by transformer ────────────────────────────────────────
    imp_by_tx = metered_sum_by_tx(imputed_kwh, mapping_now, n_tx, n_days)  # [n_tx, T_now]

    # ── Transformer analysis ──────────────────────────────────────────────
    nodes: Dict[str, NodeFinding] = {}
    tx_findings: Dict[int, NodeFinding] = {}

    for _, tx_row in obs.transformers.iterrows():
        ji = int(tx_row["tx_idx"])
        tx_id = str(tx_row["tx_id"])
        fi = int(tx_row["feeder_idx"])

        e_in = tx_in_now[ji].astype(np.float64)
        metered = metered_by_tx[ji].astype(np.float64)
        imp = imp_by_tx[ji].astype(np.float64)

        # Data quality: count NaN in raw meter data for this tx
        tx_nan_frac = np.isnan(meter_kwh_now[
            mapping_now[mapping_now["tx_idx"] == ji]["meter_idx"].values, :
        ]).mean() if len(mapping_now[mapping_now["tx_idx"] == ji]) > 0 else 0.0

        dq = []
        if tx_nan_frac > 0.05:
            dq.append(f"High NaN rate ({tx_nan_frac:.1%}) in meter data")

        finding = _analyse_node(
            tx_id, "transformer", e_in, metered, imp, tf, n_days, cfg.cal_days, cfg, dq,
            feeder_idx=fi, tx_idx=ji,
        )
        nodes[tx_id] = finding
        tx_findings[ji] = finding

    # ── Feeder analysis (DISJOINT residuals — never adds tx residuals) ────
    for _, f_row in obs.feeders.iterrows():
        fi = int(f_row["feeder_idx"])
        fid = str(f_row["feeder_id"])

        e_in_f = feeder_in_now[fi].astype(np.float64)
        # Sum of tx_in under this feeder
        tx_mask = obs.transformers["feeder_idx"].values == fi
        tx_sum = tx_in_now[tx_mask].sum(axis=0).astype(np.float64)

        # Fit feeder loss model
        a_f, b_f = fit_loss_model(e_in_f[:cfg.cal_days * 24], tx_sum[:cfg.cal_days * 24], tf[:cfg.cal_days * 24])
        # Feeder residuals (disjoint — not adding tx residuals)
        r_f = feeder_residual(e_in_f, tx_sum, tf, a_f, b_f)

        # Imputed: aggregate imputed under feeder
        tx_ji_list = obs.transformers[tx_mask]["tx_idx"].values
        imp_f = imp_by_tx[tx_ji_list].sum(axis=0).astype(np.float64)

        # Check if any tx under this feeder has active RED run
        under_tx_ids = list(obs.transformers[tx_mask]["tx_id"])
        tx_under_red = any(
            nodes[tid].status == "RED" for tid in under_tx_ids if tid in nodes
        )

        dq_f: List[str] = []

        finding_f = _analyse_node(
            fid, "feeder", e_in_f, tx_sum, imp_f, tf, n_days, cfg.cal_days, cfg, dq_f,
            feeder_idx=fi,
        )

        # Localisation rule: feeder RED with no tx RED beneath it → segment/CT error
        if finding_f.status == "RED" and not tx_under_red:
            finding_f.zone_label = (
                f"Feeder {fid} segment between feeder meter and transformer meters "
                f"(or feeder CT/meter error)"
            )
        nodes[fid] = finding_f

    # ── Phase 2 Intelligence Layer: Baselines & Attribution ───────────────
    baselines, shortfalls, _ = compute_baselines(cleaned_meter, temp_now, cfg.cal_days, cfg)

    # Records error detection across all transformer pairs
    records_errors = detect_records_errors(nodes, obs, cleaned_meter, today_day, cfg)

    cases: List[Case] = []
    cases_by_node: Dict[str, Case] = {}
    diagnostics_by_tx: Dict[str, List[CustomerDiagnostic]] = {}
    reconcile_by_tx: Dict[str, Dict[str, Any]] = {}
    voltage_by_tx: Dict[str, Dict[str, Any]] = {}
    fusion_by_tx: Dict[str, Dict[str, Any]] = {}


    for _, tx_row in obs.transformers.iterrows():
        ji = int(tx_row["tx_idx"])
        tx_id = str(tx_row["tx_id"])
        finding = nodes[tx_id]

        active_runs = [r for r in finding.runs if r.active and r.direction == "positive"]
        if active_runs:
            run = active_runs[0]
            w_start = run.start * 24
            w_end = (run.end + 1) * 24
        else:
            w_start = cfg.cal_days * 24
            w_end = T_now

        diagnostics = analyse_transformer_customers(
            ji, obs, cleaned_meter, imputed_kwh, baselines, shortfalls,
            w_start, w_end, today_day, cfg
        )
        diagnostics_by_tx[tx_id] = diagnostics

        # Prepare candidate shortfall matrix for sparse reconciliation
        cands = [d for d in diagnostics if d.is_candidate]
        n_blocks = max(1, (w_end - w_start) // 6)
        b_lim = n_blocks * 6
        gap_blocks = np.maximum(0.0, finding.r_hourly[w_start: w_start + b_lim]).reshape(n_blocks, 6).sum(axis=1)

        if cands:
            A = np.column_stack([d.shortfall_blocks for d in cands])
        else:
            A = np.zeros((n_blocks, 0), dtype=np.float64)

        reconcile_out = sparse_reconcile(gap_blocks, A, min_gain=cfg.min_reconcile_gain)
        reconcile_by_tx[tx_id] = reconcile_out

        rated_kw = float(tx_row["rated_kw"]) if "rated_kw" in tx_row else 100.0
        case = build_case_for_node(
            tx_id, finding, records_errors, diagnostics, reconcile_out, today_day, cfg, rated_kw=rated_kw
        )
        if case is not None:
            cases.append(case)
            cases_by_node[tx_id] = case

        # ── Phase 3: Voltage cross-check ────────────────────────────────
        lv_nets = obs.extras.get("lv", {})
        v_meas_all = obs.extras.get("v_meas", {})
        v0_all = obs.extras.get("v0", {})

        if tx_id in lv_nets and tx_id in v_meas_all and tx_id in v0_all:
            lv = lv_nets[tx_id]
            v_meas = v_meas_all[tx_id]
            v0_t = v0_all[tx_id]

            # Metered consumption for this tx's customers (days < today_day)
            tx_mapping = mapping_now[mapping_now["tx_idx"] == ji]
            cust_indices = tx_mapping["meter_idx"].values
            n_cust_lv = len(lv["cust_nodes"])
            if len(cust_indices) >= n_cust_lv:
                cust_indices_lv = cust_indices[:n_cust_lv]
            else:
                cust_indices_lv = cust_indices

            metered_kwh_lv = np.zeros((n_cust_lv, T_now), dtype=np.float64)
            for k, ci in enumerate(cust_indices_lv):
                if k < n_cust_lv:
                    metered_kwh_lv[k, :] = np.nan_to_num(cleaned_meter[ci, :], nan=0.0)

            volt_result = analyse_tx_voltage(
                tx_id, lv,
                v_meas[:, :T_now],
                v0_t[:T_now],
                metered_kwh_lv,
                today_day=today_day,
                n_days=n_days,
                cal_days=cfg.cal_days,
                cfg=cfg,
            )
            voltage_by_tx[tx_id] = volt_result

            # ── Evidence fusion ─────────────────────────────────────────
            from gridledger.config import CFG as _CFG
            last_S = float(finding.S[-1]) if len(finding.S) > 0 else 0.0
            e_balance = float(np.clip(last_S / max(_CFG.cusum_h, 1e-6), 0.0, 1.0))

            # Best reconcile coverage from this transformer's reconcile output
            rec_out = reconcile_by_tx.get(tx_id, {})
            e_suspect = float(np.clip(rec_out.get("coverage", 0.0), 0.0, 1.0))

            e_volt = float(volt_result.get("e_volt", 0.0))

            fused = fuse_evidence(e_balance, e_suspect, e_volt)
            fusion_by_tx[tx_id] = fused

    # Feeder cases
    for _, f_row in obs.feeders.iterrows():
        fid = str(f_row["feeder_id"])
        finding_f = nodes[fid]
        case_f = build_case_for_node(fid, finding_f, [], [], {}, today_day, cfg)
        if case_f is not None:
            cases.append(case_f)
            cases_by_node[fid] = case_f

    # Group cases by feeder into route order: feeder_idx, then level (feeder first), then node id
    cases.sort(key=lambda c: (c.feeder_idx, 0 if c.level == "feeder" else 1, c.node))

    # ── KPIs ──────────────────────────────────────────────────────────────
    red_nodes = [nid for nid, nf in nodes.items() if nf.status == "RED"]
    unaccounted_kwh = sum(
        abs(r.excess_kwh) for nf in nodes.values() if nf.status == "RED"
        for r in nf.runs if r.active and r.direction == "positive"
    )
    # Estimated INR/month from cases or fallback
    avg_tariff = cfg.tariff_inr["DOMESTIC"]
    kwh_per_day_at_stake = sum(
        abs(r.kwh_per_day) for nf in nodes.values() if nf.status == "RED"
        for r in nf.runs if r.active
    )
    inr_per_month_baseline = kwh_per_day_at_stake * 30 * avg_tariff  # ASSUMPTION
    real_inr_per_month = sum(max(0.0, c.priority_inr) for c in cases)
    inr_per_month = real_inr_per_month if cases else inr_per_month_baseline

    status_counts = {"RED": 0, "AMBER": 0, "GREEN": 0}
    for nf in nodes.values():
        status_counts[nf.status] += 1

    kpis = {
        "red_nodes": float(len(red_nodes)),
        "unaccounted_kwh": float(unaccounted_kwh),
        "inr_per_month": float(inr_per_month),
        "dq_alerts": float(len(quarantine_df)),
        "kwh_per_day_at_stake": float(kwh_per_day_at_stake),
        "cases_count": float(len(cases)),
    }

    return Results(
        nodes=nodes,
        quarantine=quarantine_df,
        status_summary=status_counts,
        kpis=kpis,
        today_day=today_day,
        cases=cases,
        cases_by_node=cases_by_node,
        records_errors=records_errors,
        diagnostics_by_tx=diagnostics_by_tx,
        reconcile_by_tx=reconcile_by_tx,
        voltage_by_tx=voltage_by_tx,
        fusion_by_tx=fusion_by_tx,
        baselines=baselines,
        shortfalls=shortfalls,
        cleaned_meter=cleaned_meter,
    )
