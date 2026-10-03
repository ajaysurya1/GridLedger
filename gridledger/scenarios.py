"""
GridLedger scenarios — Scenario dataclass and preset builders.

Detection code must NEVER import this module (it exposes scenario structure
but no oracle ground truth). The Scenario objects are stored in Truth.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from gridledger.config import CFG, Config


@dataclass
class Scenario:
    """Describes a single injected event or condition."""
    id: str                             # unique identifier (e.g. "S1")
    kind: str                           # see kind table in spec
    target: Any                         # customer meter_idx, tx_idx, or feeder_idx
    start_day: int
    end_day: Optional[int] = None       # None = active through sim end; set = fixed
    params: Dict[str, Any] = field(default_factory=dict)
    innocent: bool = False              # True if scenario should NOT produce a RED alert


# ─────────────────────────────────────────────────────────────────────────────
# Showcase scenarios (seed=7, deterministic target selection)
# ─────────────────────────────────────────────────────────────────────────────

def showcase_scenarios(world_seed: int = 7, cfg: Config = CFG) -> List[Scenario]:
    """
    Return the fixed demo scenario list with deterministic customer targets.
    Targets are chosen from the simulated topology without running detection.
    """
    from gridledger.simulator import simulate  # local import avoids circular at module level

    # Build a clean world just to get the topology (no scenarios yet)
    clean = simulate(world_seed, scenarios=[], cfg=cfg)
    obs = clean.observed
    tx_true = clean.truth.tx_true  # [n_c] physical tx assignment
    n_c = len(obs.customers)

    def first_on_tx(tx_id: str, cls_filter: Optional[str] = None) -> int:
        tx_row = obs.transformers[obs.transformers["tx_id"] == tx_id].iloc[0]
        ji = int(tx_row["tx_idx"])
        cands = np.where(tx_true == ji)[0]
        if cls_filter:
            cls_arr = obs.customers["cls"].values
            cands = cands[cls_arr[cands] == cls_filter]
        return int(cands[0])

    def tx_idx_of(tx_id: str) -> int:
        return int(obs.transformers[obs.transformers["tx_id"] == tx_id]["tx_idx"].iloc[0])

    def feeder_idx_of(fid: str) -> int:
        return int(obs.feeders[obs.feeders["feeder_id"] == fid]["feeder_idx"].iloc[0])

    # T04 → global idx 3 (feeder 0, tx 3)
    bypass_target = first_on_tx("T04", cls_filter="DOM_HIGH")
    solar_target  = first_on_tx("T04")
    # Make sure solar_target != bypass_target
    t04_ji = tx_idx_of("T04")
    t04_cands = np.where(tx_true == t04_ji)[0]
    dom_high_mask = obs.customers["cls"].values[t04_cands] == "DOM_HIGH"
    solar_candidates = t04_cands[~dom_high_mask] if (~dom_high_mask).any() else t04_cands
    solar_target = int(solar_candidates[0]) if len(solar_candidates) > 0 else bypass_target + 1

    wrong_map_target = first_on_tx("T02")
    t07_ji = tx_idx_of("T07")
    vacant_target   = first_on_tx("T06")
    stuck_target    = first_on_tx("T08")
    tariff_target   = first_on_tx("T10")

    # MISSING_BLOCK: 3 customers on T11
    t11_ji = tx_idx_of("T11")
    t11_cands = list(np.where(tx_true == t11_ji)[0][:3])

    scenarios: List[Scenario] = [
        # S1: BYPASS on a DOM_HIGH customer in T04, start day 20
        Scenario("S1", "BYPASS", bypass_target, start_day=20, params={"k": 0.5}),
        # S1b: SOLAR on another T04 customer (innocent decoy)
        Scenario("S1b", "SOLAR", solar_target, start_day=1, innocent=True,
                 params={"peak_kw": 3.0}),
        # S2: ILLEGAL_TAP on T09, 1.8 kW, day 22
        Scenario("S2", "ILLEGAL_TAP", tx_idx_of("T09"), start_day=22, params={"mean_kw": 1.8}),
        # S3: FEEDER_SEGMENT on F1, 4 kW, day 25
        Scenario("S3", "FEEDER_SEGMENT", feeder_idx_of("F1"), start_day=25, params={"kw": 4.0}),
        # S4: WRONG_MAPPING: T02 customer listed under T07 from day 24
        Scenario("S4", "WRONG_MAPPING", wrong_map_target, start_day=24,
                 params={"wrong_tx_idx": t07_ji}),
        # S5: VACANT customer in T06, day 21 (innocent)
        Scenario("S5", "VACANT", vacant_target, start_day=21, innocent=True),
        # S6: MISSING_BLOCK 3 customers in T11 days 30-33
        Scenario("S6", "MISSING_BLOCK", t11_cands[0], start_day=30, end_day=33, innocent=True,
                 params={"targets": t11_cands}),
        # S7: STUCK_METER customer in T08, day 28
        Scenario("S7", "STUCK_METER", stuck_target, start_day=28),
        # S8: TARIFF_MISUSE customer in T10
        Scenario("S8", "TARIFF_MISUSE", tariff_target, start_day=0),
        # S9: OVERLOAD T05
        Scenario("S9", "OVERLOAD", tx_idx_of("T05"), start_day=0, innocent=True),
    ]
    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Random scenarios (seeded)
# ─────────────────────────────────────────────────────────────────────────────

def random_scenarios(world_seed: int, random_seed: int, cfg: Config = CFG) -> List[Scenario]:
    """
    Generate a random-but-reproducible set of scenarios for evaluation.
    Used by eval harness (phase 5). Detection code never calls this.
    """
    from gridledger.simulator import simulate

    clean = simulate(world_seed, scenarios=[], cfg=cfg)
    obs = clean.observed
    tx_true = clean.truth.tx_true
    n_c = len(obs.customers)
    n_tx = len(obs.transformers)
    n_f = len(obs.feeders)

    rng = np.random.default_rng([random_seed, _str_key("random_scenarios")])
    scenarios: List[Scenario] = []

    # Random BYPASS
    ci = int(rng.integers(0, n_c))
    sd = int(rng.integers(10, 30))
    k = float(rng.uniform(0.4, 0.6))
    scenarios.append(Scenario("R_BYPASS", "BYPASS", ci, start_day=sd, params={"k": k}))

    # Random ILLEGAL_TAP
    ji = int(rng.integers(0, n_tx))
    sd = int(rng.integers(10, 30))
    kw = float(rng.uniform(1.0, 2.5))
    scenarios.append(Scenario("R_TAP", "ILLEGAL_TAP", ji, start_day=sd, params={"mean_kw": kw}))

    # Random FEEDER_SEGMENT
    fi = int(rng.integers(0, n_f))
    sd = int(rng.integers(10, 30))
    kw = float(rng.uniform(3.0, 6.0))
    scenarios.append(Scenario("R_SEG", "FEEDER_SEGMENT", fi, start_day=sd, params={"kw": kw}))

    return scenarios


def _str_key(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) & 0xFFFF_FFFF
