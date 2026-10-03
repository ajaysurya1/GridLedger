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
from typing import Dict, List, Optional, Tuple

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

    # Imputation: calibration-window hour/day-type mean (no peeking beyond cal_days)
    cal_hours = cfg.cal_days * 24
    cal_T = min(cal_hours, T)

    for ci in range(n_c):
        row = cleaned[ci]
        nan_mask = np.isnan(row)
        if not nan_mask.any():
            continue

        # Build lookup: hour-of-day x is_weekend
        hours = np.arange(T) % 24
        days = np.arange(T) // 24
        # Compute day-of-week using offset (day 0 = 2026-08-01 = Saturday, dow=5)
        dow = (days + 5) % 7
        is_wend = (dow >= 5).astype(int)

        # Calibration data only
        cal_row = row[:cal_T]
        cal_nan = nan_mask[:cal_T]
        cal_hours_arr = hours[:cal_T]
        cal_wend = is_wend[:cal_T]

        for h in range(24):
            for w in range(2):
                mask_hw = (cal_hours_arr == h) & (cal_wend == w) & ~cal_nan
                if mask_hw.any():
                    mean_val = np.nanmean(cal_row[mask_hw])
                else:
                    mean_val = np.nanmean(cal_row[~cal_nan]) if (~cal_nan).any() else 0.0

                # Apply to full window
                fill_mask = nan_mask & (hours == h) & (is_wend == w)
                imputed[ci, fill_mask] = mean_val
                cleaned[ci, fill_mask] = mean_val

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

    # ── KPIs ──────────────────────────────────────────────────────────────
    red_nodes = [nid for nid, nf in nodes.items() if nf.status == "RED"]
    unaccounted_kwh = sum(
        abs(r.excess_kwh) for nf in nodes.values() if nf.status == "RED"
        for r in nf.runs if r.active and r.direction == "positive"
    )
    # Estimated INR/month at stake
    # ASSUMPTION: use blended tariff from config
    avg_tariff = cfg.tariff_inr["DOMESTIC"]
    kwh_per_day_at_stake = sum(
        abs(r.kwh_per_day) for nf in nodes.values() if nf.status == "RED"
        for r in nf.runs if r.active
    )
    inr_per_month = kwh_per_day_at_stake * 30 * avg_tariff  # ASSUMPTION

    status_counts = {"RED": 0, "AMBER": 0, "GREEN": 0}
    for nf in nodes.values():
        status_counts[nf.status] += 1

    kpis = {
        "red_nodes": float(len(red_nodes)),
        "unaccounted_kwh": float(unaccounted_kwh),
        "inr_per_month": float(inr_per_month),
        "dq_alerts": float(len(quarantine_df)),
        "kwh_per_day_at_stake": float(kwh_per_day_at_stake),
    }

    return Results(
        nodes=nodes,
        quarantine=quarantine_df,
        status_summary=status_counts,
        kpis=kpis,
        today_day=today_day,
    )
