"""
GridLedger persistence detector — run classification, night share, active flag.

No oracle imports. Operates on residual arrays only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from gridledger.config import Config, CFG


@dataclass
class Run:
    """A persistent anomaly run on one node."""
    start: int              # day index (change-point estimate)
    end: int                # day index (last alarm day)
    days: int               # run length
    excess_kwh: float       # sum of daily residuals over the run (signed)
    kwh_per_day: float      # excess_kwh / days
    peak_S: float           # max CUSUM statistic during run
    active: bool            # True if run ends at today_day - 1
    night_share: float      # fraction of excess in night hours (22-05)
    direction: str          # 'positive' | 'negative'


def classify_runs(
    R_daily: np.ndarray,        # [n_days] daily residual totals
    S: np.ndarray,              # [n_days] CUSUM statistic
    raw_runs: List[Tuple[int, int]],
    r_hourly: np.ndarray,       # [T] hourly residuals (for night_share)
    today_day: int,
    cfg: Config = CFG,
) -> List[Run]:
    """
    Filter and annotate raw CUSUM runs.

    Filters: days >= min_run_days AND |excess_kwh| >= min_excess_kwh
    Night share: fraction of excess in hours 22-05 relative to expected 29%.
    """
    valid: List[Run] = []
    hours_all = np.arange(len(r_hourly)) % 24
    night_hours = (hours_all >= 22) | (hours_all < 5)

    for start, end in raw_runs:
        n_days_run = end - start + 1
        excess = float(R_daily[start: end + 1].sum())

        if n_days_run < cfg.min_run_days:
            continue
        if abs(excess) < cfg.min_excess_kwh:
            continue

        peak_S = float(S[start: end + 1].max())
        active = (end == today_day - 1)

        # Night share over the run hours
        h_start = start * 24
        h_end = (end + 1) * 24
        r_run = r_hourly[h_start:h_end]
        night_run = night_hours[h_start:h_end]
        run_sum = float(r_run.sum())
        night_sum = float(r_run[night_run].sum())
        night_share = night_sum / run_sum if abs(run_sum) > 1e-6 else 0.29

        direction = "positive" if excess > 0 else "negative"

        valid.append(Run(
            start=start,
            end=end,
            days=n_days_run,
            excess_kwh=excess,
            kwh_per_day=excess / n_days_run,
            peak_S=peak_S,
            active=active,
            night_share=night_share,
            direction=direction,
        ))

    return valid
