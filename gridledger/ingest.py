"""Bring-your-own-data CSV ingestion for the existing Observed schema."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from gridledger.schema import Observed


@dataclass
class IngestResult:
    observed: Observed
    report: Dict[str, Any]
    quarantine: pd.DataFrame


def template_csvs() -> Dict[str, str]:
    """Return small downloadable starter templates."""
    return {
        "meters.csv": "timestamp,meter_id,kwh_interval\n2026-01-01T00:00:00,M001,0.42\n",
        "nodes.csv": "timestamp,node_id,level,kwh_in\n2026-01-01T00:00:00,T01,transformer,12.4\n2026-01-01T00:00:00,F01,feeder,13.0\n",
        "mapping.csv": "meter_id,transformer_id,feeder_id,valid_from\nM001,T01,F01,2025-01-01\n",
    }


def _read_csv(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Path):
        return pd.read_csv(value)
    if hasattr(value, "read"):
        return pd.read_csv(value)
    return pd.read_csv(value)


def _timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"Invalid or missing timestamps in {column}")
    return parsed.dt.tz_convert(None)


def import_csvs(
    meters_csv: Any,
    nodes_csv: Any,
    mapping_csv: Any,
    *,
    spike_factor: float = 8.0,
) -> IngestResult:
    """Parse hourly or cumulative meter exports into an ``Observed`` object."""
    meters = _read_csv(meters_csv)
    nodes = _read_csv(nodes_csv)
    mappings = _read_csv(mapping_csv)
    required_meter = {"timestamp", "meter_id"}
    required_nodes = {"timestamp", "node_id", "level", "kwh_in"}
    required_mapping = {"meter_id", "transformer_id", "feeder_id", "valid_from"}
    for label, frame, required in (
        ("meters", meters, required_meter), ("nodes", nodes, required_nodes),
        ("mapping", mappings, required_mapping),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label}.csv is missing columns: {', '.join(sorted(missing))}")
    if ("kwh_interval" in meters.columns) == ("cumulative_kwh" in meters.columns):
        raise ValueError("meters.csv must contain exactly one of kwh_interval or cumulative_kwh")

    meters["timestamp"] = _timestamps(meters, "timestamp").dt.floor("h")
    nodes["timestamp"] = _timestamps(nodes, "timestamp").dt.floor("h")
    mappings["valid_from"] = _timestamps(mappings, "valid_from").dt.floor("D")
    meter_value_col = "kwh_interval" if "kwh_interval" in meters else "cumulative_kwh"
    for frame, column in ((meters, meter_value_col), (nodes, "kwh_in")):
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any():
            raise ValueError(f"{column} must contain numeric values or blanks")
        frame[column] = numeric

    quarantine_rows = []
    duplicate_meter_count = int(meters.duplicated(["meter_id", "timestamp"], keep="last").sum())
    duplicate_node_count = int(nodes.duplicated(["node_id", "timestamp"], keep="last").sum())
    for _, row in meters[meters.duplicated(["meter_id", "timestamp"], keep="first")].iterrows():
        quarantine_rows.append({"source": "meter", "id": row["meter_id"], "timestamp": row["timestamp"], "reason": "duplicate"})
    for _, row in nodes[nodes.duplicated(["node_id", "timestamp"], keep="first")].iterrows():
        quarantine_rows.append({"source": "node", "id": row["node_id"], "timestamp": row["timestamp"], "reason": "duplicate"})
    meters = meters.drop_duplicates(["meter_id", "timestamp"], keep="last").copy()
    nodes = nodes.drop_duplicates(["node_id", "timestamp"], keep="last").copy()

    reset_count = 0
    if meter_value_col == "cumulative_kwh":
        meters = meters.sort_values(["meter_id", "timestamp"])
        differences = meters.groupby("meter_id")["cumulative_kwh"].diff()
        reset_mask = differences < 0
        reset_count = int(reset_mask.sum())
        for idx in meters.index[reset_mask]:
            row = meters.loc[idx]
            quarantine_rows.append({"source": "meter", "id": row["meter_id"], "timestamp": row["timestamp"], "reason": "register_reset"})
        meters["kwh_interval"] = differences.mask(reset_mask)
    else:
        meters["kwh_interval"] = meters["kwh_interval"].astype(float)

    negative_meter = meters["kwh_interval"] < 0
    negative_node = nodes["kwh_in"] < 0
    for _, row in meters[negative_meter].iterrows():
        quarantine_rows.append({"source": "meter", "id": row["meter_id"], "timestamp": row["timestamp"], "reason": "negative"})
    for _, row in nodes[negative_node].iterrows():
        quarantine_rows.append({"source": "node", "id": row["node_id"], "timestamp": row["timestamp"], "reason": "negative"})
    meters.loc[negative_meter, "kwh_interval"] = np.nan
    nodes.loc[negative_node, "kwh_in"] = np.nan

    spike_count = 0
    for meter_id, indices in meters.groupby("meter_id").groups.items():
        values = meters.loc[indices, "kwh_interval"]
        valid = values.dropna()
        if len(valid) < 8:
            continue
        threshold = max(float(valid.quantile(0.999)) * spike_factor, float(valid.median()) * spike_factor)
        spike_mask = values > threshold
        spike_count += int(spike_mask.sum())
        for idx in values.index[spike_mask]:
            row = meters.loc[idx]
            quarantine_rows.append({"source": "meter", "id": meter_id, "timestamp": row["timestamp"], "reason": "spike"})
        meters.loc[spike_mask.index[spike_mask], "kwh_interval"] = np.nan

    mappings["meter_id"] = mappings["meter_id"].astype(str)
    meters["meter_id"] = meters["meter_id"].astype(str)
    nodes["node_id"] = nodes["node_id"].astype(str)
    for col in ("transformer_id", "feeder_id"):
        mappings[col] = mappings[col].astype(str)
    mappings = mappings.sort_values(["meter_id", "valid_from"]).drop_duplicates(["meter_id", "valid_from"], keep="last")

    mapping_by_meter = {key: group.sort_values("valid_from") for key, group in mappings.groupby("meter_id")}
    stale_mask = np.zeros(len(meters), dtype=bool)
    meter_tx = np.full(len(meters), -1, dtype=int)
    meter_fdr = np.full(len(meters), -1, dtype=int)
    tx_ids = sorted(mappings["transformer_id"].unique().tolist())
    feeder_ids = sorted(mappings["feeder_id"].unique().tolist())
    tx_index = {value: index for index, value in enumerate(tx_ids)}
    feeder_index = {value: index for index, value in enumerate(feeder_ids)}
    for meter_id, indices in meters.groupby("meter_id").groups.items():
        map_rows = mapping_by_meter.get(meter_id)
        if map_rows is None:
            stale_mask[meters.index.get_indexer(indices)] = True
            continue
        positions = meters.index.get_indexer(indices)
        timestamps = meters.loc[indices, "timestamp"].to_numpy(dtype="datetime64[ns]")
        valid_from = map_rows["valid_from"].to_numpy(dtype="datetime64[ns]")
        selected = np.searchsorted(valid_from, timestamps, side="right") - 1
        valid = selected >= 0
        stale_mask[positions[~valid]] = True
        if valid.any():
            selected_rows = map_rows.iloc[selected[valid]]
            meter_tx[positions[valid]] = selected_rows["transformer_id"].map(tx_index).to_numpy()
            meter_fdr[positions[valid]] = selected_rows["feeder_id"].map(feeder_index).to_numpy()
    stale_count = int(stale_mask.sum())
    for pos in np.flatnonzero(stale_mask):
        row = meters.iloc[pos]
        quarantine_rows.append({"source": "meter", "id": row["meter_id"], "timestamp": row["timestamp"], "reason": "stale_mapping"})
    meters["tx_idx"] = meter_tx
    meters["feeder_idx"] = meter_fdr
    meters.loc[stale_mask, "kwh_interval"] = np.nan

    nodes["level"] = nodes["level"].astype(str).str.lower()
    if not nodes["level"].isin(["feeder", "transformer"]).all():
        raise ValueError("nodes.csv level must be feeder or transformer")
    if set(tx_ids) - set(nodes.loc[nodes["level"] == "transformer", "node_id"]):
        raise ValueError("nodes.csv must include every transformer listed in mapping.csv")
    if set(feeder_ids) - set(nodes.loc[nodes["level"] == "feeder", "node_id"]):
        raise ValueError("nodes.csv must include every feeder listed in mapping.csv")

    start = min(meters["timestamp"].min(), nodes["timestamp"].min())
    stop = max(meters["timestamp"].max(), nodes["timestamp"].max())
    index = pd.date_range(start, stop, freq="h")
    customer_ids = sorted(meters["meter_id"].unique().tolist())
    customer_index = {value: i for i, value in enumerate(customer_ids)}
    meter_series = meters.groupby(["meter_id", "timestamp"])["kwh_interval"].sum(min_count=1).unstack(0)
    meter_series = meter_series.reindex(index=index, columns=customer_ids)
    meter_array = meter_series.to_numpy(dtype=np.float32).T

    node_series = nodes.pivot_table(index="timestamp", columns="node_id", values="kwh_in", aggfunc="sum").reindex(index=index)
    node_missing = {}
    node_values = {}
    for _, row in nodes[["node_id", "level"]].drop_duplicates().iterrows():
        node_id = str(row["node_id"])
        values = node_series[node_id].astype(float)
        missing = values.isna()
        node_missing[node_id] = int(missing.sum())
        if values.notna().sum() == 0:
            raise ValueError(f"No usable energy readings for node {node_id}")
        node_values[node_id] = values.interpolate(method="time", limit_direction="both").to_numpy(dtype=np.float32)

    tx_rows = []
    for tx_id in tx_ids:
        tx_maps = mappings[mappings["transformer_id"] == tx_id]
        feeder_id = str(tx_maps["feeder_id"].mode().iloc[0])
        tx_rows.append({"tx_idx": tx_index[tx_id], "tx_id": tx_id, "feeder_idx": feeder_index[feeder_id], "rated_kw": np.nan})
    transformer_frame = pd.DataFrame(tx_rows)
    feeder_frame = pd.DataFrame({"feeder_idx": range(len(feeder_ids)), "feeder_id": feeder_ids})
    customers_frame = pd.DataFrame({
        "meter_idx": range(len(customer_ids)), "meter_id": customer_ids,
        "cls": "UNKNOWN", "tariff_cls": "UNKNOWN",
    })
    mapping_rows = []
    t0 = pd.Timestamp(start)
    for _, row in mappings.iterrows():
        if row["meter_id"] not in customer_index:
            continue
        mapping_rows.append({
            "meter_idx": customer_index[row["meter_id"]],
            "tx_idx": tx_index[row["transformer_id"]],
            "valid_from_day": max(0, int((row["valid_from"] - t0).days)),
        })
    mapping_frame = pd.DataFrame(mapping_rows).sort_values(["meter_idx", "valid_from_day"]).reset_index(drop=True)
    tx_input = np.vstack([node_values[tx_id] for tx_id in tx_ids])
    feeder_input = np.vstack([node_values[feeder_id] for feeder_id in feeder_ids])

    meter_missing_by_tx = {}
    for tx_id in tx_ids:
        tx_idx_value = tx_index[tx_id]
        mapped_customers = mapping_frame.loc[mapping_frame["tx_idx"] == tx_idx_value, "meter_idx"].unique()
        if len(mapped_customers):
            selected = meter_array[mapped_customers]
            meter_missing_by_tx[tx_id] = float(np.isnan(selected).mean())
        else:
            meter_missing_by_tx[tx_id] = 1.0
    expected_meter_points = len(customer_ids) * len(index)
    report = {
        "rows": {"meters": int(len(meters)), "nodes": int(len(nodes)), "mapping": int(len(mappings))},
        "duplicates": {"meters": duplicate_meter_count, "nodes": duplicate_node_count},
        "negative_readings": int(negative_meter.sum() + negative_node.sum()),
        "register_resets": reset_count,
        "spikes": spike_count,
        "stale_mapping": stale_count,
        "missing_meter_points": int(np.isnan(meter_array).sum()),
        "missing_meter_percent": 100.0 * float(np.isnan(meter_array).sum()) / max(1, expected_meter_points),
        "coverage": 1.0 - float(np.isnan(meter_array).sum()) / max(1, expected_meter_points),
        "missing_per_transformer": meter_missing_by_tx,
        "interpolated_node_points": node_missing,
        "quarantined_points": len(quarantine_rows),
    }
    quarantine = pd.DataFrame(quarantine_rows, columns=["source", "id", "timestamp", "reason"])
    observed = Observed(
        t0=t0,
        temp=np.full(len(index), 25.0, dtype=np.float32),
        customers=customers_frame,
        transformers=transformer_frame,
        feeders=feeder_frame,
        meter_kwh=meter_array,
        tx_in_kwh=tx_input,
        feeder_in_kwh=feeder_input,
        mapping=mapping_frame,
        extras={"ingest_quality": report, "node_imputed_counts": node_missing},
    )
    return IngestResult(observed=observed, report=report, quarantine=quarantine)


def load_from_csv(data_dir: Path) -> IngestResult:
    """Load meters.csv, nodes.csv, and mapping.csv from a directory."""
    data_dir = Path(data_dir)
    return import_csvs(data_dir / "meters.csv", data_dir / "nodes.csv", data_dir / "mapping.csv")


def legacy_weekly_audit(observed: Observed) -> pd.DataFrame:
    """Return weekly transformer balances and top shortfall candidates only."""
    hours = observed.meter_kwh.shape[1]
    week_hours = 7 * 24
    rows = []
    for _, transformer in observed.transformers.iterrows():
        tx_idx = int(transformer["tx_idx"])
        for start in range(0, hours, week_hours):
            end = min(start + week_hours, hours)
            start_day = start // 24
            active_map = observed.mapping[observed.mapping["valid_from_day"] <= start_day]
            active_map = active_map.sort_values("valid_from_day").drop_duplicates("meter_idx", keep="last")
            customers = active_map.loc[active_map["tx_idx"] == tx_idx, "meter_idx"].astype(int).to_numpy()
            energy_in = float(np.nansum(observed.tx_in_kwh[tx_idx, start:end]))
            meter_values = observed.meter_kwh[customers, start:end] if len(customers) else np.empty((0, end - start))
            energy_out = float(np.nansum(meter_values))
            gap = energy_in - energy_out
            daily_by_customer = np.nansum(meter_values, axis=1) if len(customers) else np.empty(0)
            top = customers[np.argsort(daily_by_customer)[:3]].astype(int).tolist() if len(customers) else []
            coverage = float(np.isfinite(meter_values).mean()) if meter_values.size else 0.0
            rows.append({
                "transformer_id": str(transformer["tx_id"]),
                "week_start_day": start_day,
                "week_end_day": (end - 1) // 24,
                "energy_in_kwh": energy_in,
                "metered_kwh": energy_out,
                "gap_kwh": gap,
                "gap_percent": 100.0 * gap / energy_in if energy_in > 0 else np.nan,
                "coverage_percent": 100.0 * coverage,
                "top_shortfall_customers": top,
            })
    return pd.DataFrame(rows)


def export_observed_csvs(observed: Observed) -> Dict[str, str]:
    """Export observed arrays using the documented import templates."""
    timestamps = pd.date_range(observed.t0, periods=observed.meter_kwh.shape[1], freq="h")
    meter_rows = pd.DataFrame({
        "timestamp": np.tile(timestamps, len(observed.customers)),
        "meter_id": np.repeat(observed.customers["meter_id"].astype(str).to_numpy(), len(timestamps)),
        "kwh_interval": observed.meter_kwh.reshape(-1),
    })
    node_frames = []
    for _, row in observed.transformers.iterrows():
        node_frames.append(pd.DataFrame({"timestamp": timestamps, "node_id": row["tx_id"], "level": "transformer",
                                         "kwh_in": observed.tx_in_kwh[int(row["tx_idx"])]}))
    for _, row in observed.feeders.iterrows():
        node_frames.append(pd.DataFrame({"timestamp": timestamps, "node_id": row["feeder_id"], "level": "feeder",
                                         "kwh_in": observed.feeder_in_kwh[int(row["feeder_idx"])]}))
    mapping = observed.mapping[["meter_idx", "tx_idx", "valid_from_day"]].copy()
    mapping["meter_id"] = mapping["meter_idx"].map(dict(zip(observed.customers["meter_idx"], observed.customers["meter_id"])))
    tx_ids = dict(zip(observed.transformers["tx_idx"], observed.transformers["tx_id"]))
    feeder_id_by_idx = dict(zip(observed.feeders["feeder_idx"], observed.feeders["feeder_id"]))
    feeder_ids = dict(zip(observed.transformers["tx_idx"], observed.transformers["feeder_idx"].map(feeder_id_by_idx)))
    mapping["transformer_id"] = mapping["tx_idx"].map(tx_ids)
    mapping["feeder_id"] = mapping["tx_idx"].map(feeder_ids)
    mapping["valid_from"] = observed.t0 + pd.to_timedelta(mapping["valid_from_day"], unit="D")
    return {
        "meters.csv": meter_rows.to_csv(index=False),
        "nodes.csv": pd.concat(node_frames, ignore_index=True).to_csv(index=False),
        "mapping.csv": mapping[["meter_id", "transformer_id", "feeder_id", "valid_from"]].to_csv(index=False),
    }