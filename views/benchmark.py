"""Held-out benchmark view; reads the committed report without rerunning it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


METRICS = {
    "event_recall": ("Events found", "Share of simulated loss events matched to a case."),
    "case_precision": ("Cases that matched", "Share of raised cases that matched a simulated loss event."),
    "customer_top3_hit_rate": ("Customer in top 3", "For customer-level events, share where the true customer appeared among the first three suspects."),
    "innocent_lookalike_false_positive_rate": ("Innocent look-alikes flagged", "Share of innocent vacant, solar, and missing-data events that were incorrectly flagged."),
    "false_inspections_per_100_transformer_days": ("False inspections / 100 transformer-days", "Cases matching no loss event, scaled by evaluated transformer-days."),
    "median_detection_delay_days": ("Detection delay (days)", "Median days from event start to the day the system raised its alert."),
    "recovered_simulated_kwh_top5": ("Simulated kWh in top 5 cases", "Oracle-measured stolen energy up to day 40 for matched events represented in the top five ranked cases."),
}


def _report() -> dict:
    path = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_frame(summary: dict, keys: list[str]) -> pd.DataFrame:
    names = {"A": "Customer-only", "B": "Fixed 8% rule", "C": "GridLedger"}
    return pd.DataFrame([
        {"Metric": METRICS[key][0], "System": names[system],
         "Mean": summary[system][key]["mean"], "Std": summary[system][key]["std"]}
        for key in keys for system in ("A", "B", "C")
    ])


def _summary_copy(summary: dict) -> str:
    c = summary["C"]
    a = summary["A"]
    b = summary["B"]
    recall = c["event_recall"]["mean"]
    best_baseline_recall = max(a["event_recall"]["mean"], b["event_recall"]["mean"])
    false_c = c["false_inspections_per_100_transformer_days"]["mean"]
    false_best_baseline = min(a["false_inspections_per_100_transformer_days"]["mean"],
                              b["false_inspections_per_100_transformer_days"]["mean"])
    innocent_c = c["innocent_lookalike_false_positive_rate"]["mean"]
    innocent_best_baseline = min(a["innocent_lookalike_false_positive_rate"]["mean"],
                                 b["innocent_lookalike_false_positive_rate"]["mean"])
    kwh_c = c["recovered_simulated_kwh_top5"]["mean"]
    kwh_best_baseline = max(a["recovered_simulated_kwh_top5"]["mean"],
                            b["recovered_simulated_kwh_top5"]["mean"])
    recall_text = (f"GridLedger found {recall:.0%} of events, below the best baseline ({best_baseline_recall:.0%})."
                   if recall < best_baseline_recall else f"GridLedger found {recall:.0%} of events, at or above the best baseline ({best_baseline_recall:.0%}).")
    false_text = (f"It raised fewer false inspections ({false_c:.2f} vs {false_best_baseline:.2f} per 100 transformer-days)"
                  if false_c < false_best_baseline else f"It raised {false_c:.2f} false inspections per 100 transformer-days; the best baseline raised {false_best_baseline:.2f}")
    innocent_text = (f"and flagged fewer innocent look-alikes ({innocent_c:.0%} vs {innocent_best_baseline:.0%})."
                     if innocent_c < innocent_best_baseline else f"and flagged {innocent_c:.0%} innocent look-alikes; the best baseline flagged {innocent_best_baseline:.0%}.")
    kwh_text = (f"Its top five cases covered {kwh_c:,.0f} simulated kWh, compared with {kwh_best_baseline:,.0f} for the strongest baseline."
                if kwh_c > kwh_best_baseline else f"Its top five cases covered {kwh_c:,.0f} simulated kWh, below the strongest baseline ({kwh_best_baseline:,.0f}).")
    return f"{recall_text} {false_text} {innocent_text} {kwh_text} These are simulation results, not field performance."


def _run_fresh() -> dict:
    from gridledger.config import CFG
    from gridledger.evaluate import evaluate
    from gridledger.pipeline import run
    from gridledger.scenarios import random_scenarios
    from gridledger.simulator import simulate

    seeds = np.random.default_rng().choice(np.arange(1000, 2**31 - 1), size=3, replace=False).tolist()
    rows = []
    progress = st.progress(0, text="Starting three fresh simulated worlds")
    for index, seed in enumerate(seeds, start=1):
        topology = simulate(int(seed), scenarios=[])
        scenarios = random_scenarios(np.random.default_rng([int(seed), 0x47524944]), topology.observed, CFG.n_days)
        world = simulate(int(seed), scenarios=scenarios)
        result = run(world.observed, today_day=40)
        rows.append({"seed": int(seed), "systems": evaluate(world, result, today_day=40)})
        progress.progress(index / 3, text=f"Finished fresh world {index} of 3")
    progress.empty()

    keys = [key for key in rows[0]["systems"]["A"] if isinstance(rows[0]["systems"]["A"][key], (int, float))]
    summary = {}
    for system in ("A", "B", "C"):
        summary[system] = {}
        for key in keys:
            values = np.asarray([row["systems"][system][key] for row in rows], dtype=float)
            summary[system][key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values, ddof=1))}
        summary[system]["per_kind_recall"] = {
            kind: {"mean": float(np.mean([row["systems"][system]["per_kind_recall"][kind] for row in rows])),
                   "std": float(np.std([row["systems"][system]["per_kind_recall"][kind] for row in rows], ddof=1))}
            for kind in rows[0]["systems"][system]["per_kind_recall"]
        }
    return {"metadata": {"seeds": seeds, "seed_set": "fresh", "today_day": 40}, "summary": summary, "per_seed": rows}


def render() -> None:
    st.title("Does it outperform simpler checks?")
    st.caption("Three systems see the same simulated meters and events. The report is precomputed and loads immediately.")
    st.info("Simulated data with hidden ground truth; not field results.", icon="ℹ️")

    systems = st.columns(3)
    for col, title, explanation in zip(
        systems,
        ("A · Customer-only", "B · Fixed 8% rule", "C · GridLedger"),
        ("Flags a customer's daily meter reading when it stays 30% below that customer's own earlier pattern.",
         "Flags a transformer when its weekly input is more than 8% above the sum of customer meters.",
         "Checks where a gap begins, tests innocent explanations, compares meter and voltage evidence, then ranks a human review."),
    ):
        with col:
            st.markdown(f"**{title}**")
            st.caption(explanation)

    report = _report()
    summary = report["summary"]
    metadata = report["metadata"]
    st.markdown(_summary_copy(summary))
    st.caption(f"Held-out seeds: {', '.join(map(str, metadata['seeds']))} · "
               f"Run date: {metadata['date_utc'][:10]} · "
               f"Runtime: {metadata['runtime_seconds']:.1f} seconds")

    percent_keys = ["event_recall", "case_precision", "customer_top3_hit_rate", "innocent_lookalike_false_positive_rate"]
    counts = _comparison_frame(summary, percent_keys)
    fig = px.bar(counts, x="Metric", y="Mean", color="System", barmode="group",
                 error_y="Std", color_discrete_map={"Customer-only": "#38BDF8", "Fixed 8% rule": "#F59E0B", "GridLedger": "#2563EB"})
    fig.update_layout(height=390, yaxis=dict(range=[0, 1], tickformat=".0%", title="Mean rate"),
                      legend_title_text="System", margin=dict(l=10, r=10, t=20, b=30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    operational_keys = ["false_inspections_per_100_transformer_days", "median_detection_delay_days", "recovered_simulated_kwh_top5"]
    operational = _comparison_frame(summary, operational_keys)
    fig_ops = px.bar(operational, x="Metric", y="Mean", color="System", barmode="group",
                     error_y="Std", color_discrete_map={"Customer-only": "#38BDF8", "Fixed 8% rule": "#F59E0B", "GridLedger": "#2563EB"})
    fig_ops.update_layout(height=350, legend_title_text="System", margin=dict(l=10, r=10, t=20, b=30),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ops, use_container_width=True, config={"displayModeBar": False})

    display_rows = []
    for key, (label, description) in METRICS.items():
        row = {"Metric": label}
        for system, name in (("A", "Customer-only"), ("B", "Fixed 8% rule"), ("C", "GridLedger")):
            metric = summary[system][key]
            mean, std = metric["mean"], metric["std"]
            row[name] = "n/a" if mean is None else f"{mean:.0%} ± {std:.0%}" if key in percent_keys else f"{mean:,.2f} ± {std:,.2f}"
        row["Meaning"] = description
        display_rows.append(row)
    st.subheader("Results at a glance")
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    kinds = list(summary["C"]["per_kind_recall"])
    heat = pd.DataFrame({name: [summary[sys]["per_kind_recall"][kind]["mean"] for kind in kinds]
                         for sys, name in (("A", "Customer-only"), ("B", "Fixed 8% rule"), ("C", "GridLedger"))}, index=kinds)
    heatmap = go.Figure(go.Heatmap(z=heat.to_numpy(dtype=float), x=heat.columns, y=heat.index,
                                   zmin=0, zmax=1, colorscale=[[0, "#F1F5F9"], [0.5, "#60A5FA"], [1, "#1D4ED8"]],
                                   text=np.vectorize(lambda value: f"{value:.0%}")(heat.to_numpy(dtype=float)),
                                   texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{z:.0%}<extra></extra>"))
    heatmap.update_layout(title="Which event types were found?", height=350, margin=dict(l=10, r=10, t=50, b=20),
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Run three fresh simulated seeds"):
        st.caption("Fresh runs are separate from the held-out report and are not used for tuning.")
        if st.button("Run fresh comparison", type="primary", help="Run the same three systems on three newly generated simulated worlds."):
            try:
                st.session_state.fresh_benchmark = _run_fresh()
            except Exception as exc:
                st.error(f"The fresh run could not finish: {exc}")
        if "fresh_benchmark" in st.session_state:
            fresh = st.session_state.fresh_benchmark
            st.caption(f"Fresh seeds: {', '.join(map(str, fresh['metadata']['seeds']))}")
            fresh_table = _comparison_frame(fresh["summary"], ["event_recall", "case_precision", "customer_top3_hit_rate"])
            st.dataframe(fresh_table.pivot(index="Metric", columns="System", values="Mean").style.format("{:.0%}"), use_container_width=True)