"""
GridLedger — Case File view.

Per-node deep-dive: CUSUM chart, waterfall, suspect list, voltage cross-check,
evidence fusion score, and plain-English narrative.
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
BLUE = "#60A5FA"
STATUS_COLOR = {"RED": RED, "AMBER": AMBER, "GREEN": GREEN}


def _status_badge(status: str) -> str:
    cls = {"RED": "badge-red", "AMBER": "badge-amber", "GREEN": "badge-green"}[status]
    symbol = {"RED": "▲", "AMBER": "●", "GREEN": "■"}[status]
    return f'<span class="{cls}">{symbol} {status}</span>'


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
        shade_color = "rgba(229,72,77,0.1)" if run.direction == "positive" else "rgba(245,165,36,0.1)"
        fig.add_vrect(x0=run.start, x1=run.end + 1, fillcolor=shade_color, layer="below", line_width=0, row=1, col=1)
        fig.add_vrect(x0=run.start, x1=run.end + 1, fillcolor=shade_color, layer="below", line_width=0, row=2, col=1)

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
        name="Persistence score",
        hovertemplate="Day %{x}<br>S: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    from gridledger.config import CFG
    h = CFG.cusum_h
    fig.add_hline(y=h, line_dash="dot", line_color=RED, line_width=1.5, row=2, col=1,
                  annotation_text="Alarm threshold", annotation_font_color=RED, annotation_font_size=10)

    fig.add_vline(x=today_day - 1, line_dash="solid", line_color="#52525B", line_width=1, row=1, col=1)
    fig.add_vline(x=today_day - 1, line_dash="solid", line_color="#52525B", line_width=1, row=2, col=1)

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(color="#A1A1AA", size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter", color="#A1A1AA"),
        hoverlabel=dict(bgcolor="#18181B", bordercolor="#3F3F46", font=dict(family="Inter", color="#FAFAFA")),
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="#27272A", zerolinecolor="#3F3F46", tickfont=dict(color="#71717A", size=10), title_text="Day", row=2, col=1)
    fig.update_yaxes(gridcolor="#27272A", zerolinecolor="#3F3F46", tickfont=dict(color="#71717A", size=10))
    fig.update_yaxes(title_text="kWh/day", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Persistence score", row=2, col=1, title_font=dict(size=10))
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

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector=dict(line=dict(color="#3F3F46", width=1, dash="dot")),
        decreasing=dict(marker=dict(color="#71717A")),
        increasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=RED if unaccounted > 5 else (AMBER if unaccounted > 0 else GREEN))),
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
        title=dict(text=f"Energy Balance — Last {last7} Days", font=dict(color="#D4D4D8", size=13), x=0),
        yaxis=dict(gridcolor="#27272A", tickfont=dict(color="#71717A", size=10), title="kWh"),
        xaxis=dict(tickfont=dict(color="#D4D4D8", size=11)),
        hoverlabel=dict(bgcolor="#18181B", bordercolor="#3F3F46", font=dict(family="Inter", color="#FAFAFA")),
        showlegend=False,
    )
    return fig


def _voltage_chart(volt: dict, today_day: int) -> go.Figure:
    """Voltage cross-check: measured vs simulated daily minimum per customer."""
    daily_meas = volt.get("daily_meas_min")  # (n_cust, n_days)
    daily_sim = volt.get("daily_sim_min")
    z_scores = volt.get("z_scores")

    if daily_meas is None or daily_sim is None:
        fig = go.Figure()
        fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    n_cust, n_days = daily_meas.shape
    days = np.arange(n_days)

    # Mean across customers (ignoring NaN)
    mean_meas = np.nanmean(daily_meas, axis=0)
    mean_sim = np.nanmean(daily_sim, axis=0)
    mean_z = np.nanmean(z_scores, axis=0) if z_scores is not None else np.zeros(n_days)

    from gridledger.config import CFG
    alarm_days = volt.get("alarm_days", [])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.05,
        subplot_titles=["Voltage (V) — Daily Minimum", "Anomaly Z-Score"],
    )

    # Shade alarm days
    for d in alarm_days:
        fig.add_vrect(x0=d, x1=d + 1, fillcolor="rgba(229,72,77,0.12)", layer="below", line_width=0, row=1, col=1)
        fig.add_vrect(x0=d, x1=d + 1, fillcolor="rgba(229,72,77,0.12)", layer="below", line_width=0, row=2, col=1)

    # Simulated (digital twin)
    fig.add_trace(go.Scatter(
        x=days, y=mean_sim,
        mode="lines", name="Digital twin (expected)",
        line=dict(color=BLUE, width=2, dash="dash"),
        hovertemplate="Day %{x}<br>Simulated: %{y:.2f} V<extra></extra>",
    ), row=1, col=1)

    # Measured
    fig.add_trace(go.Scatter(
        x=days, y=mean_meas,
        mode="lines", name="Measured (smart meters)",
        line=dict(color=GREEN, width=2),
        hovertemplate="Day %{x}<br>Measured: %{y:.2f} V<extra></extra>",
    ), row=1, col=1)

    # Z-score
    z_colors = [RED if z >= CFG.volt_z_alarm else (AMBER if z >= CFG.volt_z_alarm * 0.6 else "#3F3F46") for z in mean_z]
    fig.add_trace(go.Bar(
        x=days, y=mean_z,
        marker_color=z_colors,
        name="Z-score",
        hovertemplate="Day %{x}<br>Z: %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=CFG.volt_z_alarm, line_dash="dot", line_color=RED, line_width=1.5, row=2, col=1,
                  annotation_text=f"z={CFG.volt_z_alarm}", annotation_font_color=RED, annotation_font_size=9)

    # Today line
    fig.add_vline(x=today_day - 1, line_dash="solid", line_color="#52525B", line_width=1, row=1, col=1)
    fig.add_vline(x=today_day - 1, line_dash="solid", line_color="#52525B", line_width=1, row=2, col=1)

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
        legend=dict(font=dict(color="#A1A1AA", size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter", color="#A1A1AA"),
        hoverlabel=dict(bgcolor="#18181B", bordercolor="#3F3F46", font=dict(family="Inter", color="#FAFAFA")),
    )
    fig.update_xaxes(gridcolor="#27272A", zerolinecolor="#3F3F46", tickfont=dict(color="#71717A", size=10), title_text="Day", row=2, col=1)
    fig.update_yaxes(gridcolor="#27272A", zerolinecolor="#3F3F46", tickfont=dict(color="#71717A", size=10))
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Z-score", row=2, col=1, title_font=dict(size=10))
    return fig


def _fusion_gauge(fused_score: float) -> go.Figure:
    """A simple gauge chart for the fused evidence confidence score."""
    if fused_score >= 0.75:
        bar_color = RED
    elif fused_score >= 0.45:
        bar_color = AMBER
    else:
        bar_color = GREEN

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(fused_score * 100, 1),
        number={"suffix": "%", "font": {"color": "#FAFAFA", "size": 28, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#52525B", "tickfont": {"color": "#71717A", "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#18181B",
            "borderwidth": 1,
            "bordercolor": "#27272A",
            "steps": [
                {"range": [0, 45], "color": "rgba(48,164,108,0.1)"},
                {"range": [45, 75], "color": "rgba(245,165,36,0.1)"},
                {"range": [75, 100], "color": "rgba(229,72,77,0.1)"},
            ],
            "threshold": {"line": {"color": "#FAFAFA", "width": 2}, "thickness": 0.6, "value": fused_score * 100},
        },
        title={"text": "Combined evidence", "font": {"color": "#A1A1AA", "size": 12, "family": "Inter"}},
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        font=dict(family="Inter"),
    )
    return fig


def _plain_english(node_id: str, finding, today_day: int, cfg, volt: dict = None, fused: dict = None) -> str:
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
            f" More of the gap occurs overnight ({run.night_share:.0%} between 10 PM and 5 AM, "
            f"compared with an assumed typical share of 29%). This timing can help prioritize "
            f"a meter and connection check; it does not prove the cause."
        )

    volt_note = ""
    if volt and volt.get("alarm"):
        max_z = volt.get("max_z", 0.0)
        alpha_kw = volt.get("alpha_kw", 0.0)
        volt_note = (
            f"<br><br><strong>⚡ Voltage physics cross-check ALARM:</strong> "
            f"Smart meter voltages are systematically lower than the digital twin predicts "
            f"(peak z-score: {max_z:.1f}). This suggests approximately "
            f"<strong>{alpha_kw:.1f} kW</strong> of unmetered load is drawing current "
            f"from this transformer's LV network."
        )

    fusion_note = ""
    if fused:
        score_pct = fused.get("fused_score", 0.0) * 100
        conf = fused.get("confidence", "LOW")
        channel = fused.get("channel", "Balance")
        conf_colors = {"HIGH": "#E5484D", "MEDIUM": "#F5A524", "LOW": "#30A46C"}
        conf_color = conf_colors.get(conf, "#71717A")
        fusion_note = (
            f"<br><br><strong>Evidence Fusion:</strong> All detection channels combined give "
            f"a case confidence of <strong style='color:{conf_color}'>{score_pct:.0f}% "
            f"({conf})</strong>. Primary evidence driver: {channel}."
        )

    return (
        f"<strong>{kind} {node_id}</strong> received <strong>{e_in:,.0f} kWh</strong> over the last {last7} days. "
        f"Technical losses account for <strong>{loss:,.0f} kWh</strong> and customers were billed "
        f"<strong>{billed:,.0f} kWh</strong>. "
        f"That leaves <strong>{max(0, unaccounted):,.0f} kWh unaccounted</strong> in this window.<br><br>"
        f"A persistent energy gap has been running since <strong>Day {start_day}</strong> ({days_active} days). "
        f"It averages <strong>{exc_day:.1f} kWh/day</strong> — roughly the daily consumption "
        f"of {max(1, int(exc_day / 8))} typical households. "
            f"The persistence score crossed its alarm threshold. This makes the gap less likely to be random variation, but does not prove its cause.{night_note}{volt_note}{fusion_note}"
    )


def render(world, results) -> None:
    """Render the Case File page."""
    obs = world.observed
    nodes = results.nodes
    cfg = world.cfg

    st.title("Leak found" if any(node.status == "RED" for node in nodes.values()) else "No active leak found")
    st.caption("Review the energy gap and supporting signals. A finding is a lead for human review, not an accusation.")

    # Node selector
    all_ids = list(nodes.keys())
    default_sel = st.session_state.get("selected_node")
    default_idx = all_ids.index(default_sel) if default_sel in all_ids else 0

    selected = st.selectbox("Select node", all_ids, index=default_idx, key="case_file_selector")
    if selected != st.session_state.get("selected_node"):
        st.session_state.selected_node = selected

    finding = nodes[selected]
    status = finding.status
    color = STATUS_COLOR[status]

    # Voltage + fusion data (transformers only)
    volt = results.voltage_by_tx.get(selected, {}) if hasattr(results, "voltage_by_tx") else {}
    fused = results.fusion_by_tx.get(selected, {}) if hasattr(results, "fusion_by_tx") else {}

    # Header row
    col_hdr, col_fused = st.columns([2, 1])
    with col_hdr:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin:1rem 0;">
            <div style="font-size:1.3rem;font-weight:700;color:#FAFAFA;">{selected}</div>
            <span class="badge-{status.lower()}">{status}</span>
            <div style="color:#71717A;font-size:0.8rem;">{finding.kind.title()} · {finding.zone_label[:60]}…</div>
        </div>
        """, unsafe_allow_html=True)

    with col_fused:
        if fused:
            st.plotly_chart(_fusion_gauge(fused.get("fused_score", 0.1)), use_container_width=True, key=f"gauge_{selected}")

    today_day = results.today_day

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Energy gap over time</div>', unsafe_allow_html=True)
    st.caption("Bars show daily differences; the line shows whether a gap persists across days.")
    st.plotly_chart(_cusum_chart(finding, today_day), use_container_width=True, key=f"cusum_{selected}")

    st.markdown('<div class="section-header">Where the energy went</div>', unsafe_allow_html=True)
    st.caption("Where did the energy go over the last 7 days?")
    st.plotly_chart(_waterfall_chart(finding, today_day), use_container_width=True, key=f"waterfall_{selected}")

    # ── Voltage physics (transformer-only) ────────────────────────────────
    if volt:
        st.markdown('<div class="section-header">Voltage cross-check</div>', unsafe_allow_html=True)

        # Alarm banner
        if volt.get("alarm"):
            alpha_kw = volt.get("alpha_kw", 0.0)
            cand_pole = volt.get("cand_pole")
            cand_meter = volt.get("cand_meter")
            loc_str = f"pole node {cand_pole}" if cand_pole else (f"meter C{cand_meter:04d}" if cand_meter else "unknown location")
            st.markdown(f"""
            <div class="warn-box" style="border-left-color:#E5484D;background:rgba(229,72,77,0.08);">
                ⚡ <strong>Voltage Alarm</strong> — Smart meter voltages are lower than the digital twin expects.
                Estimated unmetered load: <strong>~{alpha_kw:.1f} kW</strong> localised to <strong>{loc_str}</strong>.
                This corroborates an illegal tap or unregistered connection on this transformer's LV network.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#30A46C;font-size:0.85rem;padding:0.5rem 0;">
                ✓ Voltage physics cross-check: No anomaly detected — measured voltages match the digital twin.
            </div>
            """, unsafe_allow_html=True)

        st.caption("Blue dashed = what the utility's digital twin predicts · Green = actual smart meter readings · Red bars = alarm days")
        st.plotly_chart(_voltage_chart(volt, today_day), use_container_width=True, key=f"volt_{selected}")

        # Voltage detail metrics
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            max_z = volt.get("max_z", 0.0)
            st.metric("Peak voltage shift", f"{max_z:.2f}", help="How far measured voltage differs from the expected model, in standard deviations.")
        with v_col2:
            alpha = volt.get("alpha_kw", 0.0)
            st.metric("Estimated extra load", f"{alpha:.2f} kW", help="Approximate load inferred from a simplified single-phase voltage model; not a direct measurement.")
        with v_col3:
            r2 = volt.get("r2", 0.0)
            st.metric("Location fit", f"{r2:.2f}", help="How closely the voltage model fit matches observed readings; higher is a closer fit.")
        with v_col4:
            n_alarm = len(volt.get("alarm_days", []))
            st.metric("Days outside expected range", n_alarm, help="Days where measured voltage exceeded the configured anomaly threshold.")

    # ── Evidence fusion ────────────────────────────────────────────────────
    if fused:
        st.markdown('<div class="section-header">Signals considered together</div>', unsafe_allow_html=True)
        e_bal = fused.get("e_balance", 0.0)
        e_v = fused.get("e_volt", 0.0)
        e_s = fused.get("e_suspect", 0.0)

        fu_c1, fu_c2, fu_c3 = st.columns(3)
        for col, label, val, help_text in [
            (fu_c1, "Energy pattern", e_bal, "Evidence from the size and persistence of the energy gap."),
            (fu_c2, "Voltage readings", e_v, "Evidence from voltage differences compared with the simplified network model."),
            (fu_c3, "Meter patterns", e_s, "Evidence from customer readings that could explain the energy gap."),
        ]:
            with col:
                st.metric(label, f"{val:.0%}", help=help_text)

        st.caption(f"💡 {fused.get('note', '')}")

    # ── Suspects & diagnostics (transformer only) ──────────────────────────
    case = results.cases_by_node.get(selected)
    if case is not None:
        why_col, action_col = st.columns([1.4, 1])
        with why_col:
            st.subheader("Why we think so")
            for explanation in case.explanation[:4]:
                st.markdown(f"- {explanation}")
        with action_col:
            st.subheader("What to do")
            st.info(case.action)
            st.caption("Recommendation for review only; verify locally before taking action.")
    elif status == "GREEN":
        st.success("No active energy gap is being flagged at this location.")
    if case and case.suspects:
        st.markdown('<div class="section-header">Customers to check first</div>', unsafe_allow_html=True)
        st.caption("Customers whose consumption patterns best explain the energy gap")

        for i, s in enumerate(case.suspects[:5]):
            rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#A1A1AA", "#71717A"]
            rank_color = rank_colors[min(i, 4)]
            meter = s.get("meter", "?")
            theta = s.get("theta", 0.0)
            contrib = s.get("contrib_kwh", 0.0)
            label = s.get("label", "")
            reason = s.get("reason", "")
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:0.4rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="color:{rank_color};font-weight:700;font-size:0.9rem;">#{i+1}</span>
                        <span style="color:#FAFAFA;font-weight:600;margin-left:0.5rem;">{meter}</span>
                        {f'<span style="color:#71717A;font-size:0.78rem;margin-left:0.5rem;">— {label}</span>' if label else ''}
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.85rem;color:#F5A524;font-family:monospace;">θ={theta:.3f}</div>
                        <div style="font-size:0.75rem;color:#71717A;">{contrib:.1f} kWh shortfall</div>
                    </div>
                </div>
                {f'<div style="font-size:0.75rem;color:#71717A;margin-top:0.2rem;">{reason}</div>' if reason else ''}
            </div>""", unsafe_allow_html=True)

    # ── Data quality + runs ────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Data quality</div>', unsafe_allow_html=True)
        if finding.dq_warnings:
            for w in finding.dq_warnings:
                st.markdown(f'<div class="warn-box">⚠ {w}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#30A46C;font-size:0.85rem;">✓ No data quality issues detected</div>', unsafe_allow_html=True)

        imp_total = float(finding.n_imputed.sum())
        if imp_total > 0:
            st.markdown(f'<div class="warn-box">ℹ {imp_total:.1f} kWh imputed (NaN fill from calibration window means)</div>', unsafe_allow_html=True)

        qdf = results.quarantine
        if len(qdf) > 0:
            st.markdown(f'<div class="warn-box">⚠ {len(qdf)} meter readings quarantined (spikes/negatives)</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Ongoing energy gaps</div>', unsafe_allow_html=True)
        if finding.runs:
            for run in finding.runs:
                active_str = "🔴 Active" if run.active else "⬜ Closed"
                direction = "↑ Energy missing" if run.direction == "positive" else "↓ Over-credited"
                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                        <span style="font-size:0.8rem;font-weight:600;color:#FAFAFA;">Day {run.start} → Day {run.end}</span>
                        <span style="font-size:0.75rem;color:#71717A;">{active_str} · {direction}</span>
                    </div>
                    <div style="font-size:0.8rem;color:#A1A1AA;">
                        {run.days} days · {abs(run.excess_kwh):.1f} kWh total · {abs(run.kwh_per_day):.1f} kWh/day ·
                        Night share: {run.night_share:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#71717A;font-size:0.85rem;">No alarm runs detected</div>', unsafe_allow_html=True)

    # ── Plain-English narrative ───────────────────────────────────────────
    st.markdown('<div class="section-header">Why this location was flagged</div>', unsafe_allow_html=True)
    narrative = _plain_english(selected, finding, today_day, cfg, volt=volt, fused=fused)
    st.markdown(f'<div class="insight-box">{narrative}</div>', unsafe_allow_html=True)

    # ── Raw daily table ───────────────────────────────────────────────────
    with st.expander("Daily readings and calculations"):
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
