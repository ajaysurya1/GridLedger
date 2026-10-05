"""
GridLedger triage module (Phase 2).

Rule-based triage and economic prioritisation of confirmed grid cases.
Maps diagnostic evidence to concrete operational actions with visit costs and expected INR value.

Rule: Never imports gridledger.oracle or touches ground-truth objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from gridledger.config import Config, CFG
from gridledger.persistence import Run


@dataclass
class Case:
    id: str                             # CASE-T04, CASE-F1, etc.
    level: str                          # 'transformer' | 'feeder'
    node: str                           # T04, F1
    run: Optional[Run]
    excess_kwh: float
    kwh_per_day: float
    night_share: float
    suspects: List[Dict[str, Any]]       # [{meter, theta, contrib_kwh, label, reason, ratio, step_day}]
    coverage: float
    timing_r2: float
    unexplained_kwh: float
    action: str
    cost: float
    priority_inr: float
    evidence: Dict[str, Any]
    explanation: List[str]              # 3-5 plain-English bullets
    data_quality: List[str]
    feeder_idx: int = 0
    tx_idx: Optional[int] = None
    cleared_suspects: List[Dict[str, Any]] = field(default_factory=list)


def build_case_for_node(
    node_id: str,
    finding: Any,
    records_errors: List[Any],
    tx_cust_diagnostics: List[Any],
    reconcile_out: Dict[str, Any],
    today_day: int,
    cfg: Config = CFG,
    rated_kw: float = 100.0,
) -> Optional[Case]:
    """
    Build a comprehensive Case object for a flagged transformer or feeder node.
    Follows rule table precedence.
    """
    # Only active runs generate active triage cases
    active_runs = [r for r in finding.runs if r.active and r.direction == "positive"]
    active_neg_runs = [r for r in finding.runs if r.active and r.direction == "negative"]

    # Check for records error on negative or positive runs
    matching_rec_err = None
    for re in records_errors:
        if re.confirmed and (re.from_tx_id == node_id or re.to_tx_id == node_id):
            matching_rec_err = re
            break

    if not active_runs and not matching_rec_err:
        return None

    run = active_runs[0] if active_runs else (active_neg_runs[0] if active_neg_runs else (finding.runs[0] if finding.runs else None))
    kwh_per_day = abs(run.kwh_per_day) if run is not None else float(np.mean(np.abs(finding.R_daily[-7:])))
    excess_kwh = abs(run.excess_kwh) if run is not None else (kwh_per_day * 14.0)
    night_share = run.night_share if run is not None else 0.29
    level = finding.kind
    fi = finding.feeder_idx if finding.feeder_idx is not None else 0

    tariff = cfg.tariff_inr["DOMESTIC"]
    horizon = cfg.horizon_months

    # Separate candidates and cleared
    cleared_list = []
    candidates_list = []
    for cd in tx_cust_diagnostics:
        d_info = {
            "meter": cd.meter_id,
            "meter_idx": cd.meter_idx,
            "ratio": cd.ratio,
            "shortfall_kwh": cd.shortfall_kwh,
            "label": cd.label,
            "reason": cd.reason,
            "step_day": cd.step_day,
            "flatline": cd.flatline,
            "zero_streak": cd.zero_streak,
            "imputed_share": cd.imputed_share,
            "night_shortfall_share": cd.night_shortfall_share,
        }
        if cd.cleared:
            cleared_list.append(d_info)
        else:
            candidates_list.append(d_info)

    # ─────────────────────────────────────────────────────────────────────────
    # Rule 1: RECORDS_ERROR confirmed
    # ─────────────────────────────────────────────────────────────────────────
    if matching_rec_err is not None:
        cost = cfg.visit_cost_inr["DESK"]
        cov = float(matching_rec_err.coverage)
        p_true = 1.0  # confirmed by what-if simulation
        priority = p_true * kwh_per_day * 30.0 * tariff * horizon - cost

        suspects = [{
            "meter": matching_rec_err.meter_id,
            "meter_idx": matching_rec_err.meter_idx,
            "theta": float(matching_rec_err.theta),
            "contrib_kwh": excess_kwh * cov,
            "label": "RECORDS_ERROR",
            "reason": matching_rec_err.explanation,
            "ratio": 1.0,
            "step_day": run.start if run is not None else cfg.cal_days,
        }]

        expl = [
            f"Confirmed utility records mapping swap involving meter {matching_rec_err.meter_id}.",
            f"Meter is physically connected to {matching_rec_err.to_tx_id} but registered under {matching_rec_err.from_tx_id}.",
            f"Energy balance what-if simulation confirms reassigning this meter completely clears alarms on both transformers.",
            f"Desk action: Correct customer mapping in CIS/GIS database. Zero field visit cost required.",
        ]

        return Case(
            id=f"CASE-{node_id}",
            level=level,
            node=node_id,
            run=run,
            excess_kwh=excess_kwh,
            kwh_per_day=kwh_per_day,
            night_share=night_share,
            suspects=suspects,
            coverage=cov,
            timing_r2=1.0,
            unexplained_kwh=max(0.0, excess_kwh * (1.0 - cov)),
            action="Correct the record (desk, no visit)",
            cost=cost,
            priority_inr=priority,
            evidence={"type": "RECORDS_ERROR", "candidate": matching_rec_err},
            explanation=expl,
            data_quality=finding.dq_warnings,
            feeder_idx=fi,
            tx_idx=finding.tx_idx,
            cleared_suspects=cleared_list,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Feeder-only run
    # ─────────────────────────────────────────────────────────────────────────
    if level == "feeder":
        cost = cfg.visit_cost_inr["PATROL"]
        p_true = 0.65
        priority = p_true * kwh_per_day * 30.0 * tariff * horizon - cost

        expl = [
            f"Feeder {node_id} trunk segment has an active {kwh_per_day:.1f} kWh/day deficit since Day {run.start}.",
            f"Night shortfall share is {night_share:.0%} (expected normal is ~29%).",
            "None of the distribution transformers beneath this feeder exhibit matching shortfalls.",
            "Recommended action: Patrol medium-voltage line between substation feeder breaker and transformers, and verify feeder CT metering calibration.",
        ]

        return Case(
            id=f"CASE-{node_id}",
            level=level,
            node=node_id,
            run=run,
            excess_kwh=excess_kwh,
            kwh_per_day=kwh_per_day,
            night_share=night_share,
            suspects=[],
            coverage=0.0,
            timing_r2=0.0,
            unexplained_kwh=excess_kwh,
            action="Patrol segment between feeder meter and transformers; verify feeder CT",
            cost=cost,
            priority_inr=priority,
            evidence={"type": "FEEDER_SEGMENT"},
            explanation=expl,
            data_quality=finding.dq_warnings,
            feeder_idx=fi,
            tx_idx=None,
            cleared_suspects=cleared_list,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Transformer attribution analysis
    # ─────────────────────────────────────────────────────────────────────────
    cov = float(reconcile_out.get("coverage", 0.0))
    r2 = float(reconcile_out.get("r2", 0.0))
    theta_full = reconcile_out.get("theta_full", np.zeros(0))
    contrib = reconcile_out.get("contrib", np.zeros(0))
    explained = float(reconcile_out.get("explained", 0.0))
    unexplained = max(0.0, excess_kwh - explained)

    # Populate suspects list from candidate customers
    ranked_suspects = []
    for idx, c_info in enumerate(candidates_list):
        th = float(theta_full[idx]) if idx < len(theta_full) else 0.0
        c_kwh = float(contrib[idx]) if idx < len(contrib) else 0.0
        
        # Pattern-matching multipliers
        ratio_drop = max(0.0, 1.0 - float(c_info["ratio"]))
        depth_factor = ratio_drop ** 1.2
        step_bonus = 2.0 if c_info.get("step_day") is not None else 1.0
        
        # If transformer shows night theft, boost suspects with matching high night share
        cust_night = float(c_info.get("night_shortfall_share", 0.29))
        if night_share > 0.40:
            night_bonus = 1.0 + 4.0 * max(0.0, cust_night - 0.25)
        else:
            night_bonus = 1.0

        score = (c_kwh * 4.0 + float(c_info["shortfall_kwh"]) * depth_factor) * step_bonus * night_bonus * (1.0 + 3.0 * th)

        ranked_suspects.append({
            "meter": c_info["meter"],
            "meter_idx": c_info["meter_idx"],
            "theta": th,
            "contrib_kwh": c_kwh,
            "label": c_info["label"],
            "reason": c_info["reason"],
            "ratio": c_info["ratio"],
            "step_day": c_info["step_day"],
            "shortfall_kwh": c_info["shortfall_kwh"],
            "score": score,
        })

    # Sort suspects by composite score descending
    ranked_suspects.sort(key=lambda s: s["score"], reverse=True)
    top_suspect = ranked_suspects[0] if ranked_suspects else None

    # ─────────────────────────────────────────────────────────────────────────
    # Rule 2: Top suspect FAULTY_METER
    # ─────────────────────────────────────────────────────────────────────────
    if top_suspect and top_suspect["label"] == "FAULTY_METER":
        cost = cfg.visit_cost_inr["FIELD"]
        p_true = 0.5 + 0.5 * min(1.0, cov)
        priority = p_true * kwh_per_day * 30.0 * tariff * horizon - cost

        expl = [
            f"Top suspect meter {top_suspect['meter']} flatlined or experienced an extended zero streak.",
            f"Transformer imbalance is {kwh_per_day:.1f} kWh/day since Day {run.start}.",
            "Signature indicates internal meter hardware fault (stuck counter/converter) rather than customer theft.",
            "Recommended action: Test / replace meter (no penalty assumed).",
        ]

        return Case(
            id=f"CASE-{node_id}",
            level=level,
            node=node_id,
            run=run,
            excess_kwh=excess_kwh,
            kwh_per_day=kwh_per_day,
            night_share=night_share,
            suspects=ranked_suspects,
            coverage=cov,
            timing_r2=r2,
            unexplained_kwh=unexplained,
            action="Test / replace meter (no penalty assumed)",
            cost=cost,
            priority_inr=priority,
            evidence={"type": "FAULTY_METER", "meter": top_suspect["meter"]},
            explanation=expl,
            data_quality=finding.dq_warnings,
            feeder_idx=fi,
            tx_idx=finding.tx_idx,
            cleared_suspects=cleared_list,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rule 3: CUSTOMER-ATTRIBUTABLE (coverage >= 0.7)
    # ─────────────────────────────────────────────────────────────────────────
    if cov >= cfg.coverage_attributable and top_suspect is not None and top_suspect["theta"] > 0.2:
        cost = cfg.visit_cost_inr["FIELD"]
        p_true = 0.5 + 0.5 * min(1.0, cov)
        priority = p_true * kwh_per_day * 30.0 * tariff * horizon - cost

        expl = [
            f"Customer-attributable loss: customer {top_suspect['meter']} explains {cov:.0%} of the transformer gap.",
            f"Personal shortfall timing matches transformer loss profile with timing R² = {r2:.2f}.",
            f"Customer consumption fell to {top_suspect['ratio']:.0%} of baseline around Day {top_suspect['step_day'] or run.start}.",
            f"Night shortfall share of {night_share:.0%} strongly suggests deliberate bypass or meter tampering.",
            f"Recommended action: Surprise field inspection of meter {top_suspect['meter']} and incoming service cable.",
        ]

        return Case(
            id=f"CASE-{node_id}",
            level=level,
            node=node_id,
            run=run,
            excess_kwh=excess_kwh,
            kwh_per_day=kwh_per_day,
            night_share=night_share,
            suspects=ranked_suspects,
            coverage=cov,
            timing_r2=r2,
            unexplained_kwh=unexplained,
            action=f"Surprise inspection: meter {top_suspect['meter']}",
            cost=cost,
            priority_inr=priority,
            evidence={"type": "CUSTOMER-ATTRIBUTABLE", "top_meter": top_suspect["meter"]},
            explanation=expl,
            data_quality=finding.dq_warnings,
            feeder_idx=fi,
            tx_idx=finding.tx_idx,
            cleared_suspects=cleared_list,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rule 4: MIXED (0.3 <= coverage < 0.7)
    # ─────────────────────────────────────────────────────────────────────────
    if cov >= cfg.coverage_mixed:
        cost = cfg.visit_cost_inr["FIELD+PATROL"]
        p_true = 0.5 + 0.5 * min(1.0, cov)
        priority = p_true * kwh_per_day * 30.0 * tariff * horizon - cost

        target_meter = top_suspect["meter"] if top_suspect else "candidate"
        expl = [
            f"Mixed attribution: customer deficit accounts for {cov:.0%} of the {kwh_per_day:.1f} kWh/day transformer gap.",
            f"Unexplained balance of {unexplained:.1f} kWh suggests combined meter deficit and secondary line leak.",
            f"Timing correlation R² = {r2:.2f} across the alarm window.",
            f"Recommended action: Inspect meter {target_meter} + patrol the line.",
        ]

        return Case(
            id=f"CASE-{node_id}",
            level=level,
            node=node_id,
            run=run,
            excess_kwh=excess_kwh,
            kwh_per_day=kwh_per_day,
            night_share=night_share,
            suspects=ranked_suspects,
            coverage=cov,
            timing_r2=r2,
            unexplained_kwh=unexplained,
            action=f"Inspect meter {target_meter} + patrol the line",
            cost=cost,
            priority_inr=priority,
            evidence={"type": "MIXED", "meter": target_meter},
            explanation=expl,
            data_quality=finding.dq_warnings,
            feeder_idx=fi,
            tx_idx=finding.tx_idx,
            cleared_suspects=cleared_list,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rule 5: UNMETERED / LINE (coverage < 0.3)
    # ─────────────────────────────────────────────────────────────────────────
    cost = cfg.visit_cost_inr["PATROL"]
    p_true = 0.5 + 0.5 * min(1.0, cov)
    priority = p_true * kwh_per_day * 30.0 * tariff * horizon - cost

    expl = [
        f"Unmetered line theft: transformer {node_id} has {kwh_per_day:.1f} kWh/day missing with low customer attribution ({cov:.0%}).",
        "Individual customer meter shortfalls do not explain the magnitude or timing of the imbalance.",
        f"Night share is {night_share:.0%}, pointing towards direct unmetered tapping on the low-voltage network.",
        "Recommended action: Line patrol for illegal tap along secondary distribution mains.",
    ]

    return Case(
        id=f"CASE-{node_id}",
        level=level,
        node=node_id,
        run=run,
        excess_kwh=excess_kwh,
        kwh_per_day=kwh_per_day,
        night_share=night_share,
        suspects=ranked_suspects,
        coverage=cov,
        timing_r2=r2,
        unexplained_kwh=unexplained,
        action="Line patrol for illegal tap",
        cost=cost,
        priority_inr=priority,
        evidence={"type": "UNMETERED/LINE"},
        explanation=expl,
        data_quality=finding.dq_warnings,
        feeder_idx=fi,
        tx_idx=finding.tx_idx,
        cleared_suspects=cleared_list,
    )


def triage(results: Any, cfg: Config = CFG) -> pd.DataFrame:
    """
    Return prioritised case list as a DataFrame.
    """
    if not hasattr(results, "cases") or not results.cases:
        return pd.DataFrame(columns=["node_id", "priority", "action", "est_inr_saved"])

    rows = []
    for c in results.cases:
        rows.append({
            "case_id": c.id,
            "node_id": c.node,
            "level": c.level,
            "priority": c.priority_inr,
            "action": c.action,
            "coverage": c.coverage,
            "kwh_per_day": c.kwh_per_day,
            "cost_inr": c.cost,
            "est_inr_saved": max(0.0, c.priority_inr),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("priority", ascending=False).reset_index(drop=True)
    return df
