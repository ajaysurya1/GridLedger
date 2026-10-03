"""
GridLedger — Command Center view.

KPI strip + Plotly grid-tree + node selection + summary panel.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Semantic colours
RED = "#E5484D"
AMBER = "#F5A524"
GREEN = "#30A46C"
STATUS_COLOR = {"RED": RED, "AMBER": AMBER, "GREEN": GREEN}


def _status_badge(status: str) -> str:
    cls = {"RED": "badge-red", "AMBER": "badge-amber", "GREEN": "badge-green"}[status]
    return f'<span class="{cls}">{status}</span>'


def _kpi_card(value: str, label: str, color_class: str = "kpi-blue", unit: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-value {color_class}">{value}<span style="font-size:0.9rem;font-weight:400;color:#71717A;margin-left:2px">{unit}</span></div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def _build_grid_tree(world, results) -> go.Figure:
    """Build interactive Plotly grid-tree figure."""
    obs = world.observed
    nodes = results.nodes

    # Positions: substation at top, feeders middle, transformers bottom
    # Use a tree layout
    n_feeders = len(obs.feeders)
    n_tx = len(obs.transformers)
    tx_per_f = n_tx // n_feeders

    node_x, node_y, node_text, node_color, node_size, node_ids = [], [], [], [], [], []
    edge_x, edge_y = [], []

    # Substation
    sub_x, sub_y = 0.5, 1.0
    node_x.append(sub_x); node_y.append(sub_y)
    node_text.append("Substation")
    node_color.append("#52525B")
    node_size.append(28)
    node_ids.append("SUB")

    feeder_xs = []
    for fi, f_row in obs.feeders.iterrows():
        fid = f_row["feeder_id"]
        fx = (fi + 1) / (n_feeders + 1)
        feeder_xs.append(fx)
        node_x.append(fx); node_y.append(0.65)
        status = nodes.get(fid, None)
        color = STATUS_COLOR.get(status.status if status else "GREEN", GREEN)
        finding = nodes.get(fid)
        active_runs = [r for r in (finding.runs if finding else []) if r.active]
        exc = sum(abs(r.excess_kwh) for r in active_runs)
        hover = f"<b>{fid}</b><br>Status: {status.status if status else 'N/A'}<br>Unaccounted: {exc:.1f} kWh"
        node_text.append(hover)
        node_color.append(color)
        node_size.append(22)
        node_ids.append(fid)

        # Edge: SUB -> feeder
        edge_x += [sub_x, fx, None]
        edge_y += [sub_y, 0.65, None]

    for _, tx_row in obs.transformers.iterrows():
        ji = int(tx_row["tx_idx"])
        tid = tx_row["tx_id"]
        fi = int(tx_row["feeder_idx"])
        local_j = ji % tx_per_f
        tx_x = feeder_xs[fi] + (local_j - (tx_per_f - 1) / 2) * 0.12
        feeder_x = feeder_xs[fi]

        finding = nodes.get(tid)
        status_str = finding.status if finding else "GREEN"
        color = STATUS_COLOR[status_str]

        # Size ~ mean daily load
        mean_load = float(finding.e_in_daily.mean()) if finding is not None and len(finding.e_in_daily) > 0 else 10.0
        size = max(14, min(26, 12 + mean_load / 20))

        active_runs = [r for r in (finding.runs if finding else []) if r.active and r.direction == "positive"]
        exc = sum(abs(r.kwh_per_day) for r in active_runs)
        hover = (
            f"<b>{tid}</b><br>"
            f"Status: {status_str}<br>"
            f"Missing: {exc:.1f} kWh/day<br>"
            f"CUSUM S: {float(finding.S[-1]):.2f}" if finding is not None and len(finding.S) > 0 else f"<b>{tid}</b>"
        )

        node_x.append(tx_x); node_y.append(0.25)
        node_text.append(hover)
        node_color.append(color)
        node_size.append(size)
        node_ids.append(tid)

        # Edge: feeder -> tx
        edge_x += [feeder_x, tx_x, None]
        edge_y += [0.65, 0.25, None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#3F3F46", width=1.5),
        hoverinfo="none", showlegend=False,
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(color="#27272A", width=1.5),
            symbol="circle",
        ),
        text=[nid if nid != "SUB" else "" for nid in node_ids],
        textposition="bottom center",
        textfont=dict(color="#D4D4D8", size=10, family="Inter"),
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        customdata=node_ids,
        showlegend=False,
    ))

    # Highlight selected node
    sel = st.session_state.get("selected_node")
    if sel and sel in node_ids:
        idx = node_ids.index(sel)
        fig.add_trace(go.Scatter(
            x=[node_x[idx]], y=[node_y[idx]], mode="markers",
            marker=dict(
                color="rgba(0,0,0,0)",
                size=node_size[idx] + 10,
                line=dict(color="#FAFAFA", width=2),
            ),
            hoverinfo="none", showlegend=False,
        ))

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[0.0, 1.2]),
        hoverlabel=dict(
            bgcolor="#18181B",
            bordercolor="#3F3F46",
            font=dict(family="Inter", color="#FAFAFA", size=12),
        ),
        font=dict(family="Inter"),
        clickmode="event+select",
    )

    return fig, node_ids


def render(world, results) -> None:
    """Render the Command Center page."""
    obs = world.observed
    kpis = results.kpis
    nodes = results.nodes
    cfg = world.cfg

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.6rem;font-weight:700;letter-spacing:-0.03em;margin:0;color:#FAFAFA;">
            Command Center
        </h1>
        <p style="color:#71717A;font-size:0.85rem;margin:0.25rem 0 0 0;">
            Real-time grid energy reconciliation — every boundary, every boundary.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    red_count = int(kpis["red_nodes"])
    unacct = kpis["unaccounted_kwh"]
    inr = kpis["inr_per_month"]
    dq = int(kpis["dq_alerts"])

    with k1:
        color = "kpi-red" if red_count > 0 else "kpi-green"
        st.markdown(_kpi_card(str(red_count), "Red Nodes Active", color), unsafe_allow_html=True)
    with k2:
        st.markdown(_kpi_card(f"{unacct:,.0f}", "Unaccounted kWh (active runs)", "kpi-amber", "kWh"), unsafe_allow_html=True)
    with k3:
        st.markdown(
            _kpi_card(f"₹{inr:,.0f}", "Est. Revenue at Stake / Month", "kpi-red") +
            '<span class="assumption-badge">ASSUMPTION</span>',
            unsafe_allow_html=True
        )
    with k4:
        color = "kpi-amber" if dq > 0 else "kpi-green"
        st.markdown(_kpi_card(str(dq), "Data-Quality Alerts", color), unsafe_allow_html=True)

    st.markdown("---")

    # ── Grid tree + node panel ────────────────────────────────────────────
    left, right = st.columns([3, 1.4])

    with left:
        st.markdown('<div class="section-header">Grid Tree</div>', unsafe_allow_html=True)
        st.caption("Click a node to inspect · Size ∝ mean load · Colour = status")

        fig, node_ids = _build_grid_tree(world, results)

        # Plotly click event
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="grid_tree",
        )

        # Handle click
        if event and event.get("selection") and event["selection"].get("points"):
            pt = event["selection"]["points"][0]
            cd = pt.get("customdata")
            if cd and cd != "SUB" and cd != st.session_state.get("selected_node"):
                st.session_state.selected_node = cd
                st.rerun()

        # Fallback selectbox
        all_node_ids = [nid for nid in node_ids if nid != "SUB"]
        sel_idx = 0
        if st.session_state.get("selected_node") in all_node_ids:
            sel_idx = all_node_ids.index(st.session_state.selected_node)

        sel_fallback = st.selectbox(
            "Or select node",
            all_node_ids,
            index=sel_idx,
            key="node_fallback",
        )
        if sel_fallback != st.session_state.get("selected_node"):
            st.session_state.selected_node = sel_fallback
            st.rerun()

    with right:
        st.markdown('<div class="section-header">Node Summary</div>', unsafe_allow_html=True)

        sel = st.session_state.get("selected_node")
        if sel and sel in nodes:
            finding = nodes[sel]
            status = finding.status
            color = STATUS_COLOR[status]

            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="font-size:1.4rem;font-weight:700;color:#FAFAFA;margin-bottom:0.25rem;">{sel}</div>
                {_status_badge(status)}
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            last_S = float(finding.S[-1]) if len(finding.S) > 0 else 0.0
            active_runs = [r for r in finding.runs if r.active]
            exc_day = sum(abs(r.kwh_per_day) for r in active_runs)
            total_exc = sum(abs(r.excess_kwh) for r in active_runs)

            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:0.4rem;">
                <div class="kpi-value" style="font-size:1.4rem;color:{color};">{exc_day:.1f}</div>
                <div class="kpi-label">kWh/day missing</div>
            </div>
            <div class="kpi-card" style="margin-bottom:0.4rem;">
                <div class="kpi-value" style="font-size:1.4rem;color:{color};">{total_exc:.0f}</div>
                <div class="kpi-label">Total unaccounted kWh</div>
            </div>
            <div class="kpi-card" style="margin-bottom:0.4rem;">
                <div class="kpi-value" style="font-size:1.4rem;color:#A1A1AA;">{last_S:.2f}</div>
                <div class="kpi-label">CUSUM S (current)</div>
            </div>
            """, unsafe_allow_html=True)

            if active_runs:
                run = active_runs[0]
                st.markdown(f"""
                <div class="insight-box">
                    <strong>⚠ Active anomaly since day {run.start}</strong><br>
                    {run.days} days · {run.excess_kwh:.1f} kWh total<br>
                    Night share: {run.night_share:.0%}
                    {'· <strong>Night-heavy (theft signal)</strong>' if run.night_share > 0.45 else ''}
                </div>
                """, unsafe_allow_html=True)

            if finding.dq_warnings:
                for w in finding.dq_warnings:
                    st.markdown(f'<div class="warn-box">⚠ {w}</div>', unsafe_allow_html=True)

            st.markdown("")
            if st.button("📋 Open Case File", use_container_width=True, type="primary"):
                st.session_state.active_page = "Case File"
                st.rerun()

        else:
            st.markdown("""
            <div style="color:#52525B;font-size:0.9rem;padding:2rem 0;text-align:center;">
                Click a node in the<br>grid tree to inspect it
            </div>
            """, unsafe_allow_html=True)

    # ── Status table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">All Nodes at a Glance</div>', unsafe_allow_html=True)

    rows = []
    for nid, nf in results.nodes.items():
        active_runs = [r for r in nf.runs if r.active]
        exc = sum(abs(r.kwh_per_day) for r in active_runs)
        last_s = float(nf.S[-1]) if len(nf.S) > 0 else 0.0
        rows.append({
            "Node": nid,
            "Type": nf.kind.title(),
            "Status": nf.status,
            "kWh/day missing": f"{exc:.1f}" if exc > 0 else "—",
            "CUSUM S": f"{last_s:.2f}",
            "Active runs": len(active_runs),
            "DQ warnings": len(nf.dq_warnings),
        })

    df = pd.DataFrame(rows)

    def _color_status(val: str) -> str:
        return {
            "RED": "color: #E5484D; font-weight: 600",
            "AMBER": "color: #F5A524; font-weight: 600",
            "GREEN": "color: #30A46C; font-weight: 600",
        }.get(val, "")

    st.dataframe(
        df.style.map(_color_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
        height=400,
    )
