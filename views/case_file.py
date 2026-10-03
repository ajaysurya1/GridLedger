"""
GridLedger — Case File view.

Per-node deep-dive: CUSUM chart, waterfall, data quality, plain-English narrative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

RED = "#E5484D"
AMBER = "#F5A524"
GREEN = "#30A46C"
STATUS_COLOR = {"RED": RED, "AMBER": AMBER, "GREEN": GREEN}


def _cusum_chart(finding, today_day: int) -> go.Figure:
    """Daily residuals bar chart + CUSUM line + threshold + alarm shading."""
    n = len(finding.R_daily)
    days = np.arange(n)
    R = finding.R_daily
    S = finding.S

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.04,
    )

    # Shade alarm windows
    for run in finding.runs:
        if run.direction == "positive":
            shade_color = "rgba(229,72,77,0.1)"
        else:
            shade_color = "rgba(245,165,36,0.1)"
        fig.add_vrect(
            x0=run.start, x1=run.end + 1,
            fillcolor=shade_color, layer="below", line_width=0,
            row=1, col=1,
        )
        fig.add_vrect(
            x0=run.start, x1=run.end + 1,
            fillcolor=shade_color, layer="below", line_width=0,
            row=2, col=1,
        )

    # Daily residuals bar
    colors = [
        RED if v > finding.sigma else (AMBER if v > finding.sigma * 0.4 else "#3F3F46")
        for v in R
    ]
    fig.add_trace(go.Bar(
        x=days, y=R,
        marker_color=colors,
        name="Daily residual",
        hovertemplate="Day %{x}<br>Residual: %{y:.2f} kWh<extra></extra>",
    ), row=1, col=1)

    # ±sigma band
    fig.add_hline(y=finding.sigma, line_dash="dash", line_color="#71717A", line_width=1, row=1, col=1)
    fig.add_hline(y=-finding.sigma, line_dash="dash", line_color="#71717A", line_width=1, row=1, col=1)

    # CUSUM line
    fig.add_trace(go.Scatter(
        x=days, y=S,
        line=dict(color=AMBER, width=2),
        name="CUSUM S",
        hovertemplate="Day %{x}<br>S: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # Threshold
    from gridledger.config import CFG
    h = CFG.cusum_h
    fig.add_hline(y=h, line_dash="dot", line_color=RED, line_width=1.5, row=2, col=1,
                  annotation_text=f"h={h}", annotation_font_color=RED, annotation_font_size=10)

    # Today line
    fig.add_vline(x=today_day - 1, line_dash="solid", line_color="#52525B", line_width=1, row=1, col=1)
    fig.add_vline(x=today_day - 1, line_dash="solid", line_color="#52525B", line_width=1, row=2, col=1)

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            font=dict(color="#A1A1AA", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="Inter", color="#A1A1AA"),
        hoverlabel=dict(bgcolor="#18181B", bordercolor="#3F3F46", font=dict(family="Inter", color="#FAFAFA")),
        showlegend=True,
    )
    fig.update_xaxes(
        gridcolor="#27272A", zerolinecolor="#3F3F46",
        tickfont=dict(color="#71717A", size=10),
        title_text="Day", row=2, col=1,
    )
    fig.update_yaxes(
        gridcolor="#27272A", zerolinecolor="#3F3F46",
        tickfont=dict(color="#71717A", size=10),
    )
    fig.update_yaxes(title_text="kWh/day", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="CUSUM S", row=2, col=1, title_font=dict(size=10))

    return fig


def _waterfall_chart(finding, today_day: int) -> go.Figure:
    """7-day waterfall: E_in → Technical loss → Billed → Unaccounted."""
    last7 = min(7, today_day - 1)
    e_in = float(finding.e_in_daily[-last7:].sum())
    loss = float(finding.loss_daily[-last7:].sum())
    billed = float(finding.metered_daily[-last7:].sum())
    unaccounted = e_in - loss - billed

    labels = ["Energy In", "Tech. Loss", "Billed", "Unaccounted"]
    values = [e_in, -loss, -billed, unaccounted]
    measures = ["absolute", "relative", "relative", "total"]

    colors = [
        "#60A5FA",     # Energy In
        "#71717A",     # Tech loss
        GREEN,         # Billed
        RED if unaccounted > 5 else AMBER if unaccounted > 0 else GREEN,  # Unaccounted
    ]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector=dict(line=dict(color="#3F3F46", width=1, dash="dot")),
        decreasing=dict(marker=dict(color="#71717A")),
        increasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(
            color=RED if unaccounted > 5 else (AMBER if unaccounted > 0 else GREEN)
        )),
        text=[f"{v:+,.1f}" for v in values],
        textfont=dict(color="#FAFAFA", size=11),
        hovertemplate="%{x}: %{y:,.1f} kWh<extra></extra>",
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter", color="#A1A1AA"),
        title=dict(
            text=f"Energy Balance — Last {last7} Days",
            font=dict(color="#D4D4D8", size=13),
            x=0,
        ),
        yaxis=dict(gridcolor="#27272A", tickfont=dict(color="#71717A", size=10), title="kWh"),
        xaxis=dict(tickfont=dict(color="#D4D4D8", size=11)),
        hoverlabel=dict(bgcolor="#18181B", bordercolor="#3F3F46", font=dict(family="Inter", color="#FAFAFA")),
        showlegend=False,
    )
    return fig


def _plain_english(node_id: str, finding, today_day: int, cfg) -> str:
    """Generate a plain-English diagnostic narrative."""
    last7 = min(7, today_day - 1)
    e_in = float(finding.e_in_daily[-last7:].sum())
    loss = float(finding.loss_daily[-last7:].sum())
    billed = float(finding.metered_daily[-last7:].sum())
    unaccounted = e_in - loss - billed
    active = [r for r in finding.runs if r.active]

    kind = "Transformer" if finding.kind == "transformer" else "Feeder"

    if not active:
        return (
            f"<strong>{kind} {node_id}</strong> is currently <strong>balanced</strong>. "
            f"Over the last {last7} days it received <strong>{e_in:,.0f} kWh</strong>, "
            f"of which <strong>{loss:,.0f} kWh</strong> is explained by technical losses "
            f"and <strong>{billed:,.0f} kWh</strong> was billed to customers. "
            f"No active anomaly run detected."
        )

    run = active[0]
    start_day = run.start
    days_active = run.days
    exc_day = abs(run.kwh_per_day)

    night_note = ""
    if run.night_share > 0.45:
        night_note = (
            f" The anomaly is <strong>night-heavy</strong> ({run.night_share:.0%} of excess in hours 22–05 vs. "
            f"the expected 29%), a strong indicator of deliberate bypass or illegal tap."
        )

    return (
        f"<strong>{kind} {node_id}</strong> received <strong>{e_in:,.0f} kWh</strong> over the last {last7} days. "
        f"Technical losses account for <strong>{loss:,.0f} kWh</strong> and customers were billed "
        f"<strong>{billed:,.0f} kWh</strong>. "
        f"That leaves <strong>{max(0, unaccounted):,.0f} kWh unaccounted</strong> in this window.<br><br>"
        f"A persistent anomaly run has been active since <strong>Day {start_day}</strong> ({days_active} days). "
        f"The shortfall averages <strong>{exc_day:.1f} kWh/day</strong>. "
        f"The CUSUM statistic has exceeded the alarm threshold (h={cfg.cusum_h}), "
        f"confirming this is not random variation.{night_note}<br><br>"
        f"<strong>Zone:</strong> {finding.zone_label}. "
        f"<strong>Recommended action:</strong> Field inspection of the {kind.lower()} and associated meters."
    )


def render(world, results) -> None:
    """Render the Case File page."""
    obs = world.observed
    nodes = results.nodes
    cfg = world.cfg

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.6rem;font-weight:700;letter-spacing:-0.03em;margin:0;color:#FAFAFA;">
            Case File
        </h1>
        <p style="color:#71717A;font-size:0.85rem;margin:0.25rem 0 0 0;">
            Deep-dive diagnostics for any node in the grid tree.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Node selector
    all_ids = list(nodes.keys())
    # Default to selected node from command center
    default_sel = st.session_state.get("selected_node")
    default_idx = all_ids.index(default_sel) if default_sel in all_ids else 0

    selected = st.selectbox("Select node", all_ids, index=default_idx, key="case_file_selector")
    if selected != st.session_state.get("selected_node"):
        st.session_state.selected_node = selected

    finding = nodes[selected]
    status = finding.status
    color = STATUS_COLOR[status]

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin:1rem 0;">
        <div style="font-size:1.3rem;font-weight:700;color:#FAFAFA;">{selected}</div>
        <span class="badge-{status.lower()}">{status}</span>
        <div style="color:#71717A;font-size:0.8rem;">{finding.kind.title()} · {finding.zone_label[:60]}…</div>
    </div>
    """, unsafe_allow_html=True)

    today_day = results.today_day

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">CUSUM Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(_cusum_chart(finding, today_day), use_container_width=True, key=f"cusum_{selected}")

    st.markdown('<div class="section-header">Energy Balance Waterfall</div>', unsafe_allow_html=True)
    st.plotly_chart(_waterfall_chart(finding, today_day), use_container_width=True, key=f"waterfall_{selected}")

    # ── Data quality warnings ─────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Data Quality</div>', unsafe_allow_html=True)
        if finding.dq_warnings:
            for w in finding.dq_warnings:
                st.markdown(f'<div class="warn-box">⚠ {w}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#30A46C;font-size:0.85rem;">✓ No data quality issues detected</div>', unsafe_allow_html=True)

        # Imputation stats
        imp_total = float(finding.n_imputed.sum())
        if imp_total > 0:
            st.markdown(f'<div class="warn-box">ℹ {imp_total:.1f} kWh imputed from calibration-window means (NaN fill)</div>', unsafe_allow_html=True)

        # Quarantine
        qdf = results.quarantine
        if len(qdf) > 0:
            st.markdown(f'<div class="warn-box">⚠ {len(qdf)} meter readings quarantined (spikes/negatives)</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Active Runs</div>', unsafe_allow_html=True)
        if finding.runs:
            for run in finding.runs:
                active_str = "🔴 Active" if run.active else "⬜ Closed"
                direction = "↑ Positive" if run.direction == "positive" else "↓ Negative"
                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                        <span style="font-size:0.8rem;font-weight:600;color:#FAFAFA;">Day {run.start} → Day {run.end}</span>
                        <span style="font-size:0.75rem;color:#71717A;">{active_str} · {direction}</span>
                    </div>
                    <div style="font-size:0.8rem;color:#A1A1AA;">
                        {run.days} days · {abs(run.excess_kwh):.1f} kWh · {abs(run.kwh_per_day):.1f} kWh/day ·
                        Night share: {run.night_share:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#71717A;font-size:0.85rem;">No alarm runs detected</div>', unsafe_allow_html=True)

    # ── Plain-English narrative ───────────────────────────────────────────
    st.markdown('<div class="section-header">Diagnostic Summary</div>', unsafe_allow_html=True)
    narrative = _plain_english(selected, finding, today_day, cfg)
    st.markdown(f'<div class="insight-box">{narrative}</div>', unsafe_allow_html=True)

    # ── Raw daily table ───────────────────────────────────────────────────
    with st.expander("📊 Raw daily data table"):
        n = len(finding.e_in_daily)
        df = pd.DataFrame({
            "Day": np.arange(n),
            "E_in (kWh)": np.round(finding.e_in_daily, 2),
            "Billed (kWh)": np.round(finding.metered_daily, 2),
            "Tech Loss (kWh)": np.round(finding.loss_daily, 2),
            "Residual (kWh)": np.round(finding.R_daily, 2),
            "CUSUM S": np.round(finding.S, 3),
            "Imputed (kWh)": np.round(finding.n_imputed, 2),
        })
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
