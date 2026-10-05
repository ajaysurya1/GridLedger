"""Comparable baseline and GridLedger evaluation; oracle access stays here."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


NTL_KINDS = {
    "BYPASS", "NIGHT_THEFT", "STEP_TAMPER", "STUCK_METER",
    "ILLEGAL_TAP", "FEEDER_SEGMENT", "WRONG_MAPPING",
}
CUSTOMER_KINDS = {"BYPASS", "NIGHT_THEFT", "STEP_TAMPER", "STUCK_METER", "WRONG_MAPPING"}
INNOCENT_KINDS = {"VACANT", "SOLAR", "MISSING_BLOCK"}


def _mapping_at(world: Any, today_day: int) -> Dict[int, int]:
    mapping = world.observed.mapping
    mapping = mapping[mapping["valid_from_day"] < today_day].sort_values("valid_from_day")
    latest = mapping.drop_duplicates("meter_idx", keep="last")
    return dict(zip(latest["meter_idx"].astype(int), latest["tx_idx"].astype(int)))


def _customer_baseline(meter: np.ndarray, cal_days: int) -> np.ndarray:
    daily = np.nansum(meter[:, :cal_days * 24].reshape(len(meter), cal_days, 24), axis=2)
    baseline = np.nanmedian(daily, axis=1)
    return np.maximum(baseline, 1e-6)


def customer_only_predictions(world: Any, today_day: int, cal_days: int = 14) -> List[dict]:
    """Daily ratio baseline: ratio below 0.7 for three consecutive days."""
    obs = world.observed
    meter = obs.meter_kwh[:, :today_day * 24]
    n_days = min(today_day, meter.shape[1] // 24)
    daily = np.nansum(meter[:, :n_days * 24].reshape(len(meter), n_days, 24), axis=2)
    baseline = _customer_baseline(meter, min(cal_days, max(1, n_days)))
    ratio = daily / baseline[:, None]
    listed_tx = _mapping_at(world, today_day)
    shortfall = np.maximum(0.0, baseline[:, None] - daily)
    out: List[dict] = []
    for ci in range(len(obs.customers)):
        low = ratio[ci] < 0.7
        for day in range(max(2, cal_days - 1), n_days):
            if low[day - 2:day + 1].all() and ci in listed_tx:
                tx = obs.transformers.loc[obs.transformers["tx_idx"] == listed_tx[ci], "tx_id"]
                if not tx.empty:
                    zone = str(tx.iloc[0])
                    prior = [p for p in out if p["zone"] == zone and p["start_day"] == day]
                    if prior:
                        prior[0]["suspects"].append(ci)
                        prior[0]["rank"] += float(shortfall[ci, day - 2:day + 1].sum())
                    else:
                        out.append({"zone": zone, "start_day": day, "detection_day": day, "suspects": [ci],
                                    "rank": float(shortfall[ci, day - 2:day + 1].sum())})
                break
    for prediction in out:
        prediction["suspects"] = sorted(prediction["suspects"],
                                         key=lambda ci: float(shortfall[ci].sum()), reverse=True)[:3]
    return out


def fixed_percent_predictions(world: Any, today_day: int) -> List[dict]:
    """Weekly transformer gap rule with an 8% threshold and shortfall ranking."""
    obs = world.observed
    meter = obs.meter_kwh[:, :today_day * 24]
    predictions: List[dict] = []
    mapping = obs.mapping
    for _, tx in obs.transformers.iterrows():
        tx_idx = int(tx["tx_idx"])
        for start in range(0, today_day, 7):
            end = min(start + 7, today_day)
            if end <= start:
                continue
            mask = (mapping["tx_idx"] == tx_idx) & (mapping["valid_from_day"] <= start)
            customers = mapping.loc[mask, "meter_idx"].astype(int).to_numpy()
            if len(customers) == 0:
                continue
            lo, hi = start * 24, end * 24
            energy_in = float(np.nansum(obs.tx_in_kwh[tx_idx, lo:hi]))
            energy_out = float(np.nansum(meter[customers, lo:hi]))
            gap = energy_in - energy_out
            if energy_in <= 0 or gap / energy_in <= 0.08:
                continue
            baseline = _customer_baseline(meter[:, :max(24, min(lo, meter.shape[1]))], min(14, max(1, lo // 24)))
            consumed = np.nansum(meter[customers, lo:hi], axis=1)
            expected = baseline[customers] * (end - start)
            ranked = customers[np.argsort(np.maximum(0, expected - consumed))[::-1][:3]]
            predictions.append({"zone": str(tx["tx_id"]), "start_day": end - 1,
                               "detection_day": end - 1, "suspects": ranked.astype(int).tolist(), "rank": gap})
    return predictions


def gridledger_predictions(results: Any) -> List[dict]:
    """Convert GridLedger's active cases to the shared prediction shape."""
    predictions = []
    for case in results.cases:
        run = case.run
        if run is not None and (not run.active or run.direction != "positive"):
            continue
        suspects = [int(s["meter_idx"]) for s in case.suspects[:3] if s.get("meter_idx") is not None]
        predictions.append({"zone": case.node, "start_day": int(run.start if run else results.today_day),
                            "detection_day": int(run.end if run else results.today_day),
                            "suspects": suspects, "rank": float(case.priority_inr or case.excess_kwh)})
    return predictions


def score_predictions(world: Any, predictions: Sequence[dict], today_day: int) -> Dict[str, Any]:
    """Score raw predictions against event definitions; exported for unit tests."""
    obs = world.observed
    tx_name = dict(zip(obs.transformers["tx_idx"].astype(int), obs.transformers["tx_id"].astype(str)))
    feeder_name = dict(zip(obs.feeders["feeder_idx"].astype(int), obs.feeders["feeder_id"].astype(str)))
    events = [s for s in world.truth.scenarios if not s.innocent and s.kind in NTL_KINDS and s.start_day <= today_day]

    def zone_for(scenario: Any) -> str:
        if scenario.kind == "FEEDER_SEGMENT":
            return feeder_name[int(scenario.target)]
        if scenario.kind == "ILLEGAL_TAP":
            return tx_name[int(scenario.target)]
        return tx_name[int(world.truth.tx_true[int(scenario.target)])]

    matched: Dict[str, List[dict]] = {}
    matched_cases = set()
    for event in events:
        true_zone = zone_for(event)
        candidates = [(i, case) for i, case in enumerate(predictions)
                      if case["zone"] == true_zone and event.start_day - 1 <= int(case["start_day"]) <= today_day]
        if candidates:
            matched[event.id] = [case for _, case in candidates]
            matched_cases.update(i for i, _ in candidates)

    total = len(events)
    hits = len(matched)
    tx_days = max(1, len(obs.transformers) * max(1, today_day - world.cfg.cal_days))
    matched_case_count = sum(1 for i in range(len(predictions)) if i in matched_cases)
    customer_events = [event for event in events if event.kind in CUSTOMER_KINDS]
    customer_hits = 0
    for event in customer_events:
        candidates = matched.get(event.id, [])
        if any(int(event.target) in [int(ci) for ci in case["suspects"][:3]] for case in candidates):
            customer_hits += 1

    false_cases = [case for i, case in enumerate(predictions) if i not in matched_cases
                   and case["zone"] in set(tx_name.values())]
    negative_events = [s for s in world.truth.scenarios if s.innocent and s.kind in INNOCENT_KINDS and s.start_day <= today_day]
    false_lookalikes = 0
    for event in negative_events:
        target_indices = event.params.get("targets", [event.target])
        true_zone = tx_name[int(world.truth.tx_true[int(target_indices[0])])]
        if any(case["zone"] == true_zone and event.start_day - 1 <= int(case["start_day"]) <= today_day
               and any(int(target) in [int(ci) for ci in case["suspects"][:3]] for target in target_indices)
               for case in predictions):
            false_lookalikes += 1

    ranked = sorted(predictions, key=lambda case: float(case.get("rank", 0.0)), reverse=True)[:5]
    recovered_ids = set()
    for event in events:
        if any(case in ranked for case in matched.get(event.id, [])):
            recovered_ids.add(event.id)
    recovered = sum(float(np.asarray(world.truth.stolen_kwh.get(event.id, np.zeros(0)))[:today_day * 24].sum())
                    for event in events if event.id in recovered_ids)
    kinds = sorted(NTL_KINDS)
    per_kind = {kind: sum(1 for event in events if event.kind == kind and event.id in matched) /
                max(1, sum(1 for event in events if event.kind == kind)) for kind in kinds}
    precision = matched_case_count / max(1, len(predictions))
    return {
        "zone_localisation_accuracy": hits / max(1, total),
        "event_recall": hits / max(1, total),
        "case_precision": precision,
        "customer_top3_hit_rate": customer_hits / max(1, len(customer_events)),
        "median_detection_delay_days": float(np.median([
            min(int(case.get("detection_day", case["start_day"])) - event.start_day for case in matched[event.id])
            for event in events if event.id in matched
        ])) if matched else float("nan"),
        "false_inspections_per_100_transformer_days": len(false_cases) / tx_days * 100.0,
        "innocent_lookalike_false_positive_rate": false_lookalikes / max(1, len(negative_events)),
        "recovered_simulated_kwh_top5": recovered,
        "per_kind_recall": per_kind,
        "events": total,
        "cases": len(predictions),
    }


def evaluate(world: Any, results: Any, today_day: int | None = None) -> Dict[str, Dict[str, Any]]:
    """Run all three systems on identical observations and return raw metrics."""
    today = int(today_day if today_day is not None else results.today_day)
    systems = {
        "A": customer_only_predictions(world, today, world.cfg.cal_days),
        "B": fixed_percent_predictions(world, today),
        "C": gridledger_predictions(results),
    }
    return {name: score_predictions(world, predictions, today) for name, predictions in systems.items()}
