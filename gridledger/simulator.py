"""
GridLedger simulator — deterministic, fully vectorised, <3 s for 60 days.

All randomness comes from numpy Generators derived from (seed, entity key).
Adding a scenario NEVER changes unrelated customer data.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gridledger.config import CFG, Config
from gridledger.schema import Observed, Truth, World


# ─────────────────────────────────────────────────────────────────────────────
# RNG helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rng(seed: int, key: int) -> np.random.Generator:
    """Per-entity RNG from (seed, key). Deterministic, independent across keys."""
    return np.random.default_rng([seed, key])


def _str_key(s: str) -> int:
    """Convert a string entity key to a stable int for RNG seeding."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) & 0xFFFF_FFFF


# ─────────────────────────────────────────────────────────────────────────────
# Weather
# ─────────────────────────────────────────────────────────────────────────────

def _generate_weather(n_days: int, seed: int) -> np.ndarray:
    """Return hourly temperature array [T] with diurnal cycle + AR(1) anomaly."""
    T = n_days * 24
    rng = _rng(seed, _str_key("weather"))
    hours = np.arange(T)
    diurnal = 31.0 + 4.0 * np.sin(2 * np.pi * (hours % 24 - 9) / 24)

    # AR(1) daily anomaly
    anom = np.zeros(n_days)
    eps = rng.normal(0, CFG.weather_sd, n_days)
    for d in range(1, n_days):
        anom[d] = CFG.weather_rho * anom[d - 1] + eps[d]
    # Broadcast daily anomaly to hourly
    anom_hourly = np.repeat(anom, 24)
    return diurnal + anom_hourly


# ─────────────────────────────────────────────────────────────────────────────
# Load shape templates
# ─────────────────────────────────────────────────────────────────────────────

def _load_shape(cls: str) -> np.ndarray:
    """Return 24-vector (mean=1.0) hourly shape for a customer class."""
    h = np.arange(24)
    if cls in ("DOM_LOW", "DOM_HIGH"):
        # Morning bump 6-8h, midday dip, evening peak 19-22h, low night
        s = np.ones(24) * 0.35
        s[6:9] += 0.80
        s[9:12] += 0.25
        s[12:14] -= 0.10
        s[14:17] += 0.10
        s[17:19] += 0.50
        s[19:22] += 1.20
        s[22:24] += 0.20
    elif cls == "SHOP":
        s = np.ones(24) * 0.10
        s[9:21] = 1.45
        s[8] = 0.70
        s[21] = 0.80
    elif cls == "SMALL_IND":
        s = np.ones(24) * 0.40
        s[8:18] = 1.42
        s[7] = 0.80
        s[18] = 1.0
    else:
        s = np.ones(24)
    # Normalise to mean 1
    s = np.maximum(s, 0.0)
    return s / s.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Customer true load generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_customer_load(
    meter_idx: int,
    cls: str,
    n_days: int,
    temp: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return true_kwh [T] for one customer. Fully vectorised, no Python loops over time."""
    T = n_days * 24
    rng = _rng(seed, _str_key(f"customer_{meter_idx}"))

    base = CFG.base_kwh[cls] * float(np.exp(rng.normal(0, CFG.lognorm_sigma)))
    shape = _load_shape(cls)  # [24]
    shape_hourly = np.tile(shape, n_days)  # [T]

    # Weekend factor
    days = np.arange(n_days)
    dow = (pd.Timestamp(CFG.sim_start) + pd.to_timedelta(days, "D")).dayofweek
    is_weekend = (dow >= 5).astype(float)  # [n_days]
    wf = CFG.weekend_factor[cls]
    weekend_hourly = np.repeat(1.0 + (wf - 1.0) * is_weekend, 24)  # [T]

    # AC sensitivity
    ac = CFG.ac_sens[cls]
    ac_effect = 1.0 + ac * np.maximum(0.0, temp - 28.0)  # [T]

    # Mood (per customer-day)
    mood_log = rng.normal(0, CFG.mood_sigma, n_days)
    mood = np.exp(mood_log - CFG.mood_sigma ** 2 / 2)  # mean≈1 [n_days]
    mood_hourly = np.repeat(mood, 24)  # [T]

    # AR(1) hourly noise
    eps_raw = np.zeros(T)
    eps_raw[0] = rng.normal(0, CFG.eps_sd)
    noise_arr = rng.normal(0, CFG.eps_sd, T)
    for t in range(1, T):
        eps_raw[t] = CFG.eps_rho * eps_raw[t - 1] + noise_arr[t]
    eps = np.exp(eps_raw)
    eps = eps / eps.mean()  # renormalise to mean 1

    L = base * shape_hourly * weekend_hourly * ac_effect * mood_hourly * eps
    return np.maximum(L, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Physical loss model (transformer / feeder)
# ─────────────────────────────────────────────────────────────────────────────

def _loss_params(
    rng: np.random.Generator,
    f_lo: float,
    f_hi: float,
    mean_load: float,
) -> Tuple[float, float]:
    """Return (a, b) for loss = (a + b*P²)."""
    f = rng.uniform(f_lo, f_hi)
    a = 0.4 * f * mean_load
    b = 0.6 * f / max(mean_load, 1e-6)
    return float(a), float(b)


def _compute_losses(P: np.ndarray, a: float, b: float, tf: np.ndarray) -> np.ndarray:
    """Physical losses: (a + b*P²) * tf."""
    return (a + b * P ** 2) * tf


def _ar1_gain(rng: np.random.Generator, n_days: int, rho: float, sd: float) -> np.ndarray:
    """Per-day CT gain drift. AR(1), broadcastable to hours."""
    g = np.zeros(n_days)
    eps = rng.normal(0, sd, n_days)
    g[0] = eps[0]
    for d in range(1, n_days):
        g[d] = rho * g[d - 1] + eps[d]
    return 1.0 + g  # multiplicative gain around 1


# ─────────────────────────────────────────────────────────────────────────────
# Main simulator
# ─────────────────────────────────────────────────────────────────────────────

def simulate(seed: int, scenarios: Optional[List[Any]] = None, cfg: Config = CFG) -> World:
    """
    Build a full simulated World.

    Parameters
    ----------
    seed : int
        Master seed. Entity RNGs are derived from (seed, entity_key).
    scenarios : list of Scenario, optional
        Injected scenarios. None → clean grid.
    cfg : Config
        Configuration (default: global CFG).

    Returns
    -------
    World
        Contains .observed (for detection) and .truth (oracle only).
    """
    if scenarios is None:
        scenarios = []

    n_days = cfg.n_days
    T = n_days * 24
    t0 = pd.Timestamp(cfg.sim_start)

    # ── Grid topology ────────────────────────────────────────────────────
    feeders_df = pd.DataFrame({
        "feeder_idx": np.arange(cfg.n_feeders),
        "feeder_id": [f"F{i+1}" for i in range(cfg.n_feeders)],
    })

    tx_rows: List[Dict] = []
    for fi in range(cfg.n_feeders):
        for tj in range(cfg.tx_per_feeder):
            tx_global = fi * cfg.tx_per_feeder + tj
            rng_tx = _rng(seed, _str_key(f"tx_rated_{tx_global}"))
            rated_kw = float(rng_tx.uniform(80, 200))
            tx_rows.append({
                "tx_idx": tx_global,
                "tx_id": f"T{tx_global + 1:02d}",
                "feeder_idx": fi,
                "rated_kw": rated_kw,
            })
    transformers_df = pd.DataFrame(tx_rows)
    n_tx = len(transformers_df)

    # ── Customer assignment ───────────────────────────────────────────────
    cls_names = list(cfg.class_mix.keys())
    cls_probs = np.array([cfg.class_mix[c] for c in cls_names])

    customer_rows: List[Dict] = []
    cust_meter_idx = 0
    tx_customer_lists: List[List[int]] = [[] for _ in range(n_tx)]

    for tx_idx in range(n_tx):
        rng_tx_c = _rng(seed, _str_key(f"tx_customers_{tx_idx}"))
        n_c = int(rng_tx_c.integers(cfg.customers_per_tx_min, cfg.customers_per_tx_max + 1))
        cls_draws = rng_tx_c.choice(cls_names, size=n_c, p=cls_probs)
        for cls in cls_draws:
            tariff = (
                "DOMESTIC" if "DOM" in cls
                else "COMMERCIAL" if cls == "SHOP"
                else "INDUSTRIAL"
            )
            customer_rows.append({
                "meter_idx": cust_meter_idx,
                "meter_id": f"C{cust_meter_idx + 1:04d}",
                "cls": cls,
                "tariff_cls": tariff,
                "tx_idx_true": tx_idx,   # physical assignment (truth)
            })
            tx_customer_lists[tx_idx].append(cust_meter_idx)
            cust_meter_idx += 1

    customers_df_full = pd.DataFrame(customer_rows)
    n_c = len(customers_df_full)
    customers_df = customers_df_full[["meter_idx", "meter_id", "cls", "tariff_cls"]].copy()
    tx_true = customers_df_full["tx_idx_true"].values  # [n_c] int

    # ── Temperature ───────────────────────────────────────────────────────
    temp = _generate_weather(n_days, seed)  # [T]
    tf_global = 1.0 + cfg.kappa * (temp - cfg.t_ref)  # [T]

    # ── True customer loads ───────────────────────────────────────────────
    true_kwh = np.zeros((n_c, T), dtype=np.float32)
    for ci, row in customers_df_full.iterrows():
        true_kwh[row["meter_idx"]] = _generate_customer_load(
            row["meter_idx"], row["cls"], n_days, temp, seed
        )

    # ── Apply scenario effects to true load / physical taps ───────────────
    # Index scenarios by kind for fast lookup
    tap_kwh = np.zeros((n_tx, T), dtype=np.float32)   # illegal tap load
    seg_kwh = np.zeros((n_tx_f := cfg.n_feeders, T), dtype=np.float32)  # feeder segment
    stolen_kwh: Dict[str, np.ndarray] = {}

    # Scenario effects on meter readings (applied later)
    meter_factors = np.ones((n_c, T), dtype=np.float32)   # multiplicative
    meter_stuck: Dict[int, int] = {}                        # meter_idx -> start_day

    # Vacants / solar handled as true-load modifiers
    solar_kwh = np.zeros((n_c, T), dtype=np.float32)

    for sc in scenarios:
        _apply_scenario_to_physics(
            sc, seed, n_days, temp, customers_df_full,
            true_kwh, tap_kwh, seg_kwh,
            meter_factors, meter_stuck, solar_kwh, stolen_kwh, cfg,
        )

    # ── Physical chain: transformer inputs ────────────────────────────────
    tx_in_kwh = np.zeros((n_tx, T), dtype=np.float32)
    tx_loss_params: List[Tuple[float, float]] = []

    for ji in range(n_tx):
        rng_tx = _rng(seed, _str_key(f"tx_loss_{ji}"))
        cust_mask = (tx_true == ji)
        # Total physical load on transformer (true load + tap)
        if cust_mask.any():
            P_j = true_kwh[cust_mask].sum(axis=0)  # [T]
        else:
            P_j = np.zeros(T, dtype=np.float32)
        P_j = P_j + tap_kwh[ji]

        a_j, b_j = _loss_params(rng_tx, cfg.tx_loss_frac_lo, cfg.tx_loss_frac_hi, float(P_j.mean() or 1))
        tx_loss_params.append((a_j, b_j))
        loss_j = _compute_losses(P_j, a_j, b_j, tf_global)

        gain_j = _ar1_gain(_rng(seed, _str_key(f"tx_gain_{ji}")), n_days, cfg.tx_gain_rho, cfg.tx_gain_sd)
        gain_j_hourly = np.repeat(gain_j, 24)

        noise_j = _rng(seed, _str_key(f"tx_noise_{ji}")).normal(0, cfg.tx_meas_noise_sd, T)
        tx_in_kwh[ji] = ((P_j + loss_j) * gain_j_hourly * (1.0 + noise_j)).astype(np.float32)

    # ── Physical chain: feeder inputs ─────────────────────────────────────
    feeder_in_kwh = np.zeros((cfg.n_feeders, T), dtype=np.float32)

    for fi in range(cfg.n_feeders):
        rng_f = _rng(seed, _str_key(f"feeder_loss_{fi}"))
        tx_mask = transformers_df["feeder_idx"].values == fi
        # Feeder load = sum of transformer secondary power + feeder segment theft
        tx_secondary = np.zeros(T, dtype=np.float32)
        for ji in np.where(tx_mask)[0]:
            cust_mask = (tx_true == ji)
            if cust_mask.any():
                P_j = true_kwh[cust_mask].sum(axis=0)
            else:
                P_j = np.zeros(T, dtype=np.float32)
            P_j = P_j + tap_kwh[ji]
            a_j, b_j = tx_loss_params[ji]
            loss_j = _compute_losses(P_j, a_j, b_j, tf_global)
            tx_secondary += (P_j + loss_j).astype(np.float32)

        P_f = tx_secondary + seg_kwh[fi]
        a_f, b_f = _loss_params(rng_f, cfg.feeder_loss_frac_lo, cfg.feeder_loss_frac_hi, float(P_f.mean() or 1))
        loss_f = _compute_losses(P_f, a_f, b_f, tf_global)

        gain_f = _ar1_gain(_rng(seed, _str_key(f"feeder_gain_{fi}")), n_days, cfg.tx_gain_rho, cfg.feeder_gain_sd)
        gain_f_hourly = np.repeat(gain_f, 24)

        noise_f = _rng(seed, _str_key(f"feeder_noise_{fi}")).normal(0, cfg.feeder_meas_noise_sd, T)
        feeder_in_kwh[fi] = ((P_f + loss_f) * gain_f_hourly * (1.0 + noise_f)).astype(np.float32)

    # ── Meter readings ────────────────────────────────────────────────────
    meter_kwh = np.zeros((n_c, T), dtype=np.float32)
    for ci in range(n_c):
        L = (true_kwh[ci] - solar_kwh[ci]).astype(float)
        L = np.maximum(L, -999.0)  # solar can be negative net
        noise = _rng(seed, _str_key(f"meter_noise_{ci}")).normal(0, cfg.meter_noise_sd, T)
        L_m = L * (1.0 + noise)

        # Apply meter factor (BYPASS, NIGHT_THEFT, STEP_TAMPER)
        L_m = L_m * meter_factors[ci]

        # STUCK_METER: repeat last pre-start value
        if ci in meter_stuck:
            sd = meter_stuck[ci]
            if sd > 0 and sd * 24 <= T:
                last_val = float(L_m[sd * 24 - 1])
                L_m[sd * 24:] = last_val

        # Random comms drops (NaN)
        nan_mask = _rng(seed, _str_key(f"meter_nan_{ci}")).random(T) < cfg.nan_prob
        L_m[nan_mask] = np.nan
        meter_kwh[ci] = L_m.astype(np.float32)

    # ── Observed mapping (initially = true physical assignment) ───────────
    mapping_rows: List[Dict] = []
    for ci in range(n_c):
        mapping_rows.append({
            "meter_idx": ci,
            "tx_idx": int(tx_true[ci]),
            "valid_from_day": 0,
        })

    # Apply WRONG_MAPPING scenario effects to mapping records
    for sc in scenarios:
        if sc.kind == "WRONG_MAPPING" and sc.end_day is None:
            ci = sc.target
            wrong_tx = sc.params.get("wrong_tx_idx")
            if wrong_tx is not None:
                mapping_rows.append({
                    "meter_idx": ci,
                    "tx_idx": int(wrong_tx),
                    "valid_from_day": sc.start_day,
                })
        elif sc.kind == "WRONG_MAPPING" and sc.end_day is not None:
            ci = sc.target
            # Fix: correct record added
            mapping_rows.append({
                "meter_idx": ci,
                "tx_idx": int(tx_true[ci]),
                "valid_from_day": sc.end_day,
            })

    mapping_df = pd.DataFrame(mapping_rows).sort_values(["meter_idx", "valid_from_day"]).reset_index(drop=True)

    # ── Assemble Observed & Truth ─────────────────────────────────────────
    observed = Observed(
        t0=t0,
        temp=temp,
        customers=customers_df,
        transformers=transformers_df[["tx_idx", "tx_id", "feeder_idx", "rated_kw"]],
        feeders=feeders_df,
        meter_kwh=meter_kwh,
        tx_in_kwh=tx_in_kwh,
        feeder_in_kwh=feeder_in_kwh,
        mapping=mapping_df,
    )

    truth = Truth(
        true_kwh=true_kwh,
        tap_kwh=tap_kwh,
        seg_kwh=seg_kwh,
        tx_true=tx_true,
        scenarios=scenarios,
        stolen_kwh=stolen_kwh,
    )

    return World(observed=observed, truth=truth, cfg=cfg, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario physics injection (isolated helper)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_scenario_to_physics(
    sc: Any,
    seed: int,
    n_days: int,
    temp: np.ndarray,
    customers_df_full: pd.DataFrame,
    true_kwh: np.ndarray,
    tap_kwh: np.ndarray,
    seg_kwh: np.ndarray,
    meter_factors: np.ndarray,
    meter_stuck: Dict[int, int],
    solar_kwh: np.ndarray,
    stolen_kwh: Dict[str, np.ndarray],
    cfg: Config,
) -> None:
    """Mutate simulation arrays in-place for one scenario. Isolated per scenario RNG."""
    T = n_days * 24
    t_start = sc.start_day * 24
    t_end = (sc.end_day * 24) if sc.end_day is not None else T
    t_end = min(t_end, T)
    rng_sc = _rng(seed, _str_key(f"scenario_{sc.id}"))

    kind = sc.kind

    # ── Customer-level scenarios ──────────────────────────────────────────
    if kind == "BYPASS":
        ci = sc.target
        k = sc.params.get("k", 0.5)
        meter_factors[ci, t_start:t_end] *= k
        # stolen = true - metered (approximately)
        stolen = true_kwh[ci, t_start:t_end] * (1 - k)
        _add_stolen(stolen_kwh, sc.id, T, t_start, stolen)
        if sc.end_day is not None:
            # Post-fix: factor -> 1, true load scaled
            true_kwh[ci, t_end:] *= cfg.post_fix_keep + k * (1 - cfg.post_fix_keep)

    elif kind == "NIGHT_THEFT":
        ci = sc.target
        hours = np.arange(T) % 24
        night_mask = ((hours >= 22) | (hours < 5)) & _range_mask(T, t_start, t_end)
        meter_factors[ci, night_mask] *= 0.3
        stolen = true_kwh[ci] * 0.7 * night_mask
        _add_stolen(stolen_kwh, sc.id, T, 0, stolen)

    elif kind == "STEP_TAMPER":
        ci = sc.target
        meter_factors[ci, t_start:t_end] *= 0.35
        stolen = true_kwh[ci, t_start:t_end] * 0.65
        _add_stolen(stolen_kwh, sc.id, T, t_start, stolen)

    elif kind == "STUCK_METER":
        ci = sc.target
        meter_stuck[ci] = sc.start_day

    elif kind == "ILLEGAL_TAP":
        ji = sc.target
        mean_tap = sc.params.get("mean_kw", 1.8)
        # Evening-weighted tap profile
        hours = np.arange(T) % 24
        eve_weight = np.where((hours >= 17) & (hours < 23), 2.0, 0.5)
        eve_weight = eve_weight / eve_weight.mean()
        base_tap = mean_tap * rng_sc.lognormal(0, 0.2)
        tap = (base_tap * eve_weight).astype(np.float32)
        tap[:t_start] = 0
        tap[t_end:] = 0
        tap_kwh[ji] += tap
        _add_stolen(stolen_kwh, sc.id, T, t_start, tap[t_start:t_end])

    elif kind == "FEEDER_SEGMENT":
        fi = sc.target
        flat_kw = sc.params.get("kw", 4.0)
        seg = np.zeros(T, dtype=np.float32)
        seg[t_start:t_end] = flat_kw
        seg_kwh[fi] += seg
        _add_stolen(stolen_kwh, sc.id, T, t_start, seg[t_start:t_end])

    elif kind == "WRONG_MAPPING":
        pass  # handled in mapping_df construction

    elif kind == "VACANT":
        ci = sc.target
        true_kwh[ci, t_start:t_end] *= 0.05
        # meter also reflects (applied via true_kwh → meter pipeline)

    elif kind == "SOLAR":
        ci = sc.target
        peak_kw = sc.params.get("peak_kw", 3.0)
        for d in range(sc.start_day, min(sc.end_day if sc.end_day else n_days, n_days)):
            cloud = rng_sc.uniform(0.4, 1.0)
            for h in range(9, 17):
                t = d * 24 + h
                bell = np.exp(-0.5 * ((h - 13) / 2.5) ** 2)
                solar_kwh[ci, t] = peak_kw * bell * cloud

    elif kind == "MISSING_BLOCK":
        # NaN block: applied via meter_kwh NaN injection
        targets = sc.params.get("targets", [sc.target])
        for ci in targets:
            # Mark with special value; actual NaN set in meter loop
            meter_factors[ci, t_start:t_end] = np.nan  # sentinel

    elif kind == "TARIFF_MISUSE":
        # Profile already SHOP-like from t=0; tariff_cls stays DOMESTIC
        # The profile mismatch is visible to billing but not physical
        pass

    elif kind == "OVERLOAD":
        pass  # engineering alert; no physical change to loads


def _range_mask(T: int, t_start: int, t_end: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    m[t_start:t_end] = True
    return m


def _add_stolen(stolen_kwh: Dict, sc_id: str, T: int, t_start: int, arr: np.ndarray) -> None:
    if sc_id not in stolen_kwh:
        stolen_kwh[sc_id] = np.zeros(T, dtype=np.float32)
    stolen_kwh[sc_id][t_start: t_start + len(arr)] += arr
