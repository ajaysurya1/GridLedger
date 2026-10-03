"""
GridLedger configuration — single source of truth.
Every field with a monetary, CO2 or behavioural constant is labelled ASSUMPTION.
Modify here; never hard-code values elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Config:
    # ── Simulation dimensions ──────────────────────────────────────────────
    n_days: int = 60
    cal_days: int = 14          # calibration / burn-in window
    default_today_day: int = 40
    n_feeders: int = 2
    tx_per_feeder: int = 6
    customers_per_tx_min: int = 18
    customers_per_tx_max: int = 32

    # ── CUSUM parameters ──────────────────────────────────────────────────
    cusum_k: float = 0.5        # slack / reference value
    cusum_h: float = 9.0        # alarm threshold (tuned on seeds 1-5, FAR≤0.4/100 node-days)
    min_run_days: int = 3
    min_excess_kwh: float = 15.0

    # ── Loss-model physics ────────────────────────────────────────────────
    kappa: float = 0.004        # temperature sensitivity of losses (1/°C)
    t_ref: float = 25.0         # reference temperature °C

    # ── Noise calibration ─────────────────────────────────────────────────
    sigma_inflate: float = 3.0   # inflate calibration sigma to absorb AR(1) gain drift

    # ── ASSUMPTION: tariffs (INR / kWh) ───────────────────────────────────
    tariff_inr: Dict[str, float] = field(default_factory=lambda: {
        "DOMESTIC": 6.5,        # ASSUMPTION
        "COMMERCIAL": 9.5,      # ASSUMPTION
        "INDUSTRIAL": 8.0,      # ASSUMPTION
    })

    # ── ASSUMPTION: inspection visit costs (INR) ──────────────────────────
    visit_cost_inr: Dict[str, float] = field(default_factory=lambda: {
        "FIELD": 1500,          # ASSUMPTION
        "PATROL": 800,          # ASSUMPTION
        "DESK": 300,            # ASSUMPTION
    })

    # ── ASSUMPTION: financial / environmental projection horizon ──────────
    horizon_months: float = 3.0         # ASSUMPTION
    co2_t_per_mwh: float = 0.710        # ASSUMPTION — India grid emission factor tCO2/MWh
    post_fix_keep: float = 0.8          # ASSUMPTION — fraction of legitimate load retained after fix

    # ── Simulation start ──────────────────────────────────────────────────
    sim_start: str = "2026-08-01"

    # ── Class mix fractions ───────────────────────────────────────────────
    class_mix: Dict[str, float] = field(default_factory=lambda: {
        "DOM_LOW":   0.45,
        "DOM_HIGH":  0.25,
        "SHOP":      0.20,
        "SMALL_IND": 0.10,
    })

    # ── ASSUMPTION: base loads (kWh/hour mean) ────────────────────────────
    base_kwh: Dict[str, float] = field(default_factory=lambda: {
        "DOM_LOW":   0.22,      # ASSUMPTION
        "DOM_HIGH":  0.55,      # ASSUMPTION
        "SHOP":      0.90,      # ASSUMPTION
        "SMALL_IND": 2.50,      # ASSUMPTION
    })

    # ── ASSUMPTION: AC sensitivity per °C above 28 ────────────────────────
    ac_sens: Dict[str, float] = field(default_factory=lambda: {
        "DOM_LOW":   0.01,      # ASSUMPTION
        "DOM_HIGH":  0.05,      # ASSUMPTION
        "SHOP":      0.03,      # ASSUMPTION
        "SMALL_IND": 0.00,      # ASSUMPTION
    })

    # ── ASSUMPTION: weekend multipliers ──────────────────────────────────
    weekend_factor: Dict[str, float] = field(default_factory=lambda: {
        "DOM_LOW":   1.12,      # ASSUMPTION
        "DOM_HIGH":  1.12,      # ASSUMPTION
        "SHOP":      1.05,      # ASSUMPTION
        "SMALL_IND": 0.60,      # ASSUMPTION
    })

    # ── Meter / comms noise ───────────────────────────────────────────────
    meter_noise_sd: float = 0.005
    nan_prob: float = 0.003     # random comms drop probability

    # ── Transformer feeder loss fractions ─────────────────────────────────
    tx_loss_frac_lo: float = 0.035
    tx_loss_frac_hi: float = 0.060
    feeder_loss_frac_lo: float = 0.015
    feeder_loss_frac_hi: float = 0.030

    # ── CT / metering drift (AR1) ─────────────────────────────────────────
    tx_gain_rho: float = 0.8
    tx_gain_sd: float = 0.003
    feeder_gain_rho: float = 0.8
    feeder_gain_sd: float = 0.002
    tx_meas_noise_sd: float = 0.003
    feeder_meas_noise_sd: float = 0.002

    # ── Weather AR(1) parameters ──────────────────────────────────────────
    weather_rho: float = 0.7
    weather_sd: float = 1.5

    # ── Customer noise parameters ─────────────────────────────────────────
    lognorm_sigma: float = 0.25   # per-customer base scale
    mood_sigma: float = 0.12      # per-customer-day
    eps_rho: float = 0.5          # AR1 rho for hourly noise
    eps_sd: float = 0.28          # AR1 sd for hourly noise

    # ── Anomaly threshold ─────────────────────────────────────────────────
    spike_factor: float = 8.0     # multiples of p99.9 considered absurd

    # ── Status thresholds ─────────────────────────────────────────────────
    amber_s_fraction: float = 0.5  # fraction of h above which AMBER triggers
    amber_recent_days: int = 3     # days since last run end for AMBER


# Module-level default instance
CFG = Config()
