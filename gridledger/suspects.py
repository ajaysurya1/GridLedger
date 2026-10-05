"""
GridLedger suspects module (Phase 2).

Computes weather-adjusted personal baselines vectorised over all customers.
Evaluates customer shortfall, step changes, innocent explanations, and candidate sets
inside a flagged transformer zone.

Rule: Never imports gridledger.oracle or touches ground-truth objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gridledger.config import Config, CFG
from gridledger.schema import Observed


@dataclass
class CustomerDiagnostic:
    meter_idx: int
    meter_id: str
    tx_idx: int
    cls: str
    tariff_cls: str
    ratio: float
    shortfall_kwh: float
    shortfall_blocks: np.ndarray        # [n_blocks] in 6-hour blocks
    consumption_blocks: np.ndarray      # [n_blocks] in 6-hour blocks
    step_day: Optional[int]
    night_shortfall_share: float
    flatline: bool
    zero_streak: bool
    imputed_share: float
    label: str                          # SOLAR_LIKE | VACANCY_LIKE | FAULTY_METER | DATA_GAP | DEFICIT_SUSPECT | NORMAL
    reason: str
    cleared: bool                       # True -> excluded from reconciliation
    is_candidate: bool                  # True -> column in reconciliation A matrix


# ─────────────────────────────────────────────────────────────────────────────
# 1. Weather-adjusted personal baselines (fully vectorised across all customers)
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(
    cleaned_meter: np.ndarray,      # [n_c, T]
    temp: np.ndarray,               # [T]
    cal_days: int,
    cfg: Config = CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute weather-adjusted personal baseline for all customers.

    profile[c, daytype, hour] = mean of calibration-window kWh (daytype: weekday/weekend)
    beta_c = clip( sum(x*(ratio-1)) / sum(x^2), -0.02, 0.12 )
             with x = max(0, temp-28), ratio = E/profile (profile > eps)
    baseline[c,t] = profile * (1 + beta_c * x_t)
    shortfall[c,t] = max(0, baseline - E_imputed)

    Returns
    -------
    (baseline [n_c, T], shortfall [n_c, T], beta_c [n_c])
    """
    n_c, T = cleaned_meter.shape
    cal_T = min(cal_days * 24, T)

    # Day-of-week: day 0 is 2026-08-01 (Saturday, dow=5)
    days = np.arange(T) // 24
    dow = (days + 5) % 7
    is_wend = (dow >= 5).astype(int)   # 0 = weekday, 1 = weekend
    hours = np.arange(T) % 24

    # Calibration window profile [n_c, 2, 24]
    profile = np.zeros((n_c, 2, 24), dtype=np.float32)
    for w in (0, 1):
        for h in range(24):
            cal_mask = (is_wend[:cal_T] == w) & (hours[:cal_T] == h)
            if cal_mask.any():
                profile[:, w, h] = cleaned_meter[:, :cal_T][:, cal_mask].mean(axis=1)

    # Full window profile broadcast [n_c, T]
    prof_t = profile[:, is_wend, hours]

    # Temperature sensitivity beta_c from calibration window
    x = np.maximum(0.0, temp[:T] - 28.0).astype(np.float32)
    x_cal = x[:cal_T]
    prof_cal = prof_t[:, :cal_T]
    meter_cal = cleaned_meter[:, :cal_T]

    eps = 1e-4
    valid_p = prof_cal > eps
    ratio_cal = np.where(valid_p, meter_cal / np.maximum(prof_cal, eps), 1.0)

    sum_x2 = float(np.sum(x_cal ** 2)) + 1e-12
    # sum(x * (ratio - 1)) over calibration hours
    numer = np.sum(x_cal[None, :] * (ratio_cal - 1.0), axis=1)
    beta_c = np.clip(numer / sum_x2, -0.02, 0.12).astype(np.float32)

    # Baseline & shortfall over full window
    baseline = prof_t * (1.0 + beta_c[:, None] * x[None, :])
    baseline = np.maximum(baseline, 0.0).astype(np.float32)
    shortfall = np.maximum(0.0, baseline - cleaned_meter).astype(np.float32)

    return baseline, shortfall, beta_c


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-customer analysis inside a transformer zone & window W
# ─────────────────────────────────────────────────────────────────────────────

def analyse_transformer_customers(
    tx_idx: int,
    obs: Observed,
    cleaned_meter: np.ndarray,      # [n_c, T]
    imputed_kwh: np.ndarray,        # [n_c, T]
    baseline: np.ndarray,           # [n_c, T]
    shortfall: np.ndarray,          # [n_c, T]
    w_start: int,                   # start hour
    w_end: int,                     # end hour (exclusive)
    today_day: int,
    cfg: Config = CFG,
) -> List[CustomerDiagnostic]:
    """
    Analyse each customer connected to tx_idx in the alarm window [w_start, w_end].
    Computes ratio, block shortfalls, step changes, streaks, and innocent explanations.
    """
    mapping_now = obs.mapping[obs.mapping["valid_from_day"] < today_day]
    cust_on_tx = mapping_now[mapping_now["tx_idx"] == tx_idx]["meter_idx"].unique()

    diagnostics: List[CustomerDiagnostic] = []
    w_start = max(0, w_start)
    w_end = min(cleaned_meter.shape[1], w_end)
    w_len = max(1, w_end - w_start)
    n_blocks = max(1, w_len // 6)
    block_limit = n_blocks * 6

    hours_all = np.arange(cleaned_meter.shape[1]) % 24
    hours_w = hours_all[w_start: w_start + block_limit]

    # Reference data lookup
    cust_meta = obs.customers.set_index("meter_idx")

    for ci in cust_on_tx:
        ci = int(ci)
        c_row = cust_meta.loc[ci] if ci in cust_meta.index else None
        meter_id = str(c_row["meter_id"]) if c_row is not None else f"M{ci:04d}"
        cls_name = str(c_row["cls"]) if c_row is not None else "DOM_LOW"
        tariff = str(c_row["tariff_cls"]) if c_row is not None else "DOMESTIC"

        E_w = cleaned_meter[ci, w_start: w_start + block_limit]
        B_w = baseline[ci, w_start: w_start + block_limit]
        S_w = shortfall[ci, w_start: w_start + block_limit]
        imp_w = imputed_kwh[ci, w_start: w_start + block_limit]

        tot_E = float(E_w.sum())
        tot_B = float(B_w.sum())
        tot_S = float(S_w.sum())
        ratio = tot_E / max(tot_B, 1e-6)

        # 6-hour block shortfalls & consumption
        shortfall_b = S_w.reshape(n_blocks, 6).sum(axis=1)
        consumption_b = E_w.reshape(n_blocks, 6).sum(axis=1)

        # Imputed share in window
        imputed_share = float(np.sum(imp_w > 0.01)) / max(len(imp_w), 1)

        # Night shortfall share (hours 22 to 05)
        night_mask = (hours_w >= 22) | (hours_w < 5)
        night_s = float(S_w[night_mask].sum())
        night_share = night_s / tot_S if tot_S > 1e-4 else 0.29

        # Step-change day (CUSUM on daily ratio)
        n_days = cleaned_meter.shape[1] // 24
        daily_E = cleaned_meter[ci, :n_days * 24].reshape(n_days, 24).sum(axis=1)
        daily_B = baseline[ci, :n_days * 24].reshape(n_days, 24).sum(axis=1)
        daily_ratio = daily_E / np.maximum(daily_B, 1e-6)

        # Daily CUSUM on deficit
        s_step = 0.0
        max_s = 0.0
        step_day: Optional[int] = None
        for d in range(cfg.cal_days, n_days):
            s_step = max(0.0, s_step + (1.0 - daily_ratio[d] - 0.15))
            if s_step > max_s:
                max_s = s_step
            if s_step > 2.0 and step_day is None:
                step_day = d

        # Sustained deficit check (>= min_run_days with ratio <= 0.92)
        end_d = w_end // 24
        sustained_count = 0
        for d in range(end_d - 1, max(cfg.cal_days - 1, end_d - 1 - 14), -1):
            if daily_ratio[d] <= 0.92:
                sustained_count += 1
            else:
                break
        is_sustained = sustained_count >= cfg.min_run_days

        # Flatline check: >=12 identical consecutive readings or std < 1e-6 over 48h
        diffs = np.abs(np.diff(E_w))
        is_same = diffs < 1e-5
        max_consec = 0
        cur_consec = 0
        for v in is_same:
            if v:
                cur_consec += 1
                if cur_consec > max_consec:
                    max_consec = cur_consec
            else:
                cur_consec = 0
        has_12_identical = max_consec >= 11
        std_48h = float(np.std(E_w[-48:])) if len(E_w) >= 48 else 999.0
        flatline = has_12_identical or (std_48h < cfg.flatline_std_thresh)

        # Zero streak check: >= 12 consecutive near-zero hours
        is_zero = E_w < 1e-3
        z_consec = 0
        max_z_consec = 0
        for v in is_zero:
            if v:
                z_consec += 1
                if z_consec > max_z_consec:
                    max_z_consec = z_consec
            else:
                z_consec = 0
        zero_streak = max_z_consec >= 12

        # ── Innocent explanation filter ───────────────────────────────────────
        # 1. SOLAR_LIKE: >=75% of shortfall in daylight (08-17h) while evening (18-22h) ratio in [0.85, 1.15]
        mask_daylight = (hours_w >= 8) & (hours_w <= 17)
        mask_18_22 = (hours_w >= 18) & (hours_w <= 22)
        solar_hrs_s = float(S_w[mask_daylight].sum())
        solar_share = solar_hrs_s / max(tot_S, 1e-6)
        eve_E = float(E_w[mask_18_22].sum())
        eve_B = float(B_w[mask_18_22].sum())
        eve_ratio = eve_E / max(eve_B, 1e-6)

        is_solar = (solar_share >= 0.70) and (
            0.85 <= eve_ratio <= 1.15
        )

        # 2. VACANCY_LIKE: mean kWh < 0.15/h or ratio <= 0.10 with sustained drop for >=5 days
        is_vacancy = False
        mean_vac_kwh = 0.0
        if len(E_w) >= cfg.vacancy_min_days * 24:
            vac_window = E_w[-cfg.vacancy_min_days * 24:]
            mean_vac_kwh = float(vac_window.mean())
            std_vac = float(vac_window.std())
            if (mean_vac_kwh < 0.15 or ratio <= 0.12) and std_vac < 0.10 and is_sustained:
                is_vacancy = True

        # 3. Label and reason assignment
        if is_solar:
            label = "SOLAR_LIKE"
            reason = (
                f"Rooftop solar: {solar_share:.0%} of shortfall during daylight (08-17h); "
                f"evening ratio {eve_ratio:.2f} is normal (cleared)"
            )
            cleared = True
            is_cand = False
        elif flatline or zero_streak:
            label = "FAULTY_METER"
            reason = (
                "Hardware fault: meter flatlined with consecutive identical readings "
                "or zero streak (action: test/replace meter, no penalty assumed)"
            )
            cleared = False
            is_cand = (ratio <= cfg.candidate_ratio_threshold) and is_sustained
        elif is_vacancy:
            label = "VACANCY_LIKE"
            reason = (
                f"Vacant premises pattern: sustained average {mean_vac_kwh:.3f} kWh/h "
                f"(ratio {ratio:.1%}) for >=5 days (cleared)"
            )
            cleared = True
            is_cand = False
        elif imputed_share > cfg.data_gap_imputed_share:
            label = "DATA_GAP"
            reason = f"Data gap: {imputed_share:.0%} of readings imputed; low confidence"
            cleared = False
            is_cand = (ratio <= cfg.candidate_ratio_threshold) and is_sustained
        elif ratio <= cfg.candidate_ratio_threshold and is_sustained:
            label = "DEFICIT_SUSPECT"
            reason = (
                f"Sustained deficit: consumption at {ratio:.0%} of weather-adjusted baseline "
                f"({tot_S:.1f} kWh deficit across window)"
            )
            cleared = False
            is_cand = True
        else:
            label = "NORMAL"
            reason = f"Normal consumption: {ratio:.0%} of expected baseline"
            cleared = True
            is_cand = False

        diagnostics.append(CustomerDiagnostic(
            meter_idx=ci,
            meter_id=meter_id,
            tx_idx=tx_idx,
            cls=cls_name,
            tariff_cls=tariff,
            ratio=ratio,
            shortfall_kwh=tot_S,
            shortfall_blocks=shortfall_b,
            consumption_blocks=consumption_b,
            step_day=step_day,
            night_shortfall_share=night_share,
            flatline=flatline,
            zero_streak=zero_streak,
            imputed_share=imputed_share,
            label=label,
            reason=reason,
            cleared=cleared,
            is_candidate=is_cand,
        ))

    return diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compatibility ranking function
# ─────────────────────────────────────────────────────────────────────────────

def rank_suspects(
    tx_id: str,
    finding: Any,
    obs: Observed,
    today_day: int,
    cfg: Config = CFG,
) -> pd.DataFrame:
    """
    Rank customers within a transformer zone by suspicion score.
    Returns DataFrame[meter_idx, meter_id, score, ratio, shortfall_kwh, label, reason].
    """
    tx_row = obs.transformers[obs.transformers["tx_id"] == tx_id]
    if len(tx_row) == 0:
        return pd.DataFrame(columns=["meter_id", "score", "reason"])

    tx_idx = int(tx_row["tx_idx"].iloc[0])
    T = today_day * 24

    from gridledger.pipeline import _clean_meter_data
    cleaned_meter, _, imputed_kwh = _clean_meter_data(
        obs.meter_kwh[:, :T], obs.customers, cfg
    )

    baseline, shortfall, _ = compute_baselines(cleaned_meter, obs.temp[:T], cfg.cal_days, cfg)

    # Determine window W
    active_runs = [r for r in (finding.runs if finding else []) if r.active and r.direction == "positive"]
    if active_runs:
        run = active_runs[0]
        w_start = run.start * 24
        w_end = (run.end + 1) * 24
    else:
        w_start = cfg.cal_days * 24
        w_end = T

    diagnostics = analyse_transformer_customers(
        tx_idx, obs, cleaned_meter, imputed_kwh, baseline, shortfall,
        w_start, w_end, today_day, cfg
    )

    rows = []
    for d in diagnostics:
        # Score ~ deficit magnitude * (1 - ratio)
        score = d.shortfall_kwh * max(0.0, 1.0 - d.ratio)
        rows.append({
            "meter_idx": d.meter_idx,
            "meter_id": d.meter_id,
            "score": score,
            "ratio": d.ratio,
            "shortfall_kwh": d.shortfall_kwh,
            "label": d.label,
            "reason": d.reason,
            "cleared": d.cleared,
            "is_candidate": d.is_candidate,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
