"""
GridLedger — Scenario Lab view.

Shows active scenarios from the current preset.
Does NOT leak ground truth into the detection pipeline.
The Lab is an operator view; detection never sees this.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


# Scenario kind descriptions (no oracle data)
KIND_DESC = {
    "BYPASS": "Meter bypass — meter registers a fraction of true consumption",
    "NIGHT_THEFT": "Night-time theft — meter reads 30% during hours 22–05",
    "STEP_TAMPER": "Step tamper — meter factor set to 0.35 continuously",
    "STUCK_METER": "Stuck meter — meter repeats the last pre-event reading",
    "ILLEGAL_TAP": "Illegal tap — extra load on transformer secondary (not metered)",
    "FEEDER_SEGMENT": "Feeder segment theft — additional load between feeder & transformer meters",
    "WRONG_MAPPING": "Wrong mapping — meter physically on Tx A but billed under Tx B (records error)",
    "VACANT": "Vacant premise — very low legitimate consumption (innocent)",
    "SOLAR": "Rooftop solar — net metering; consumption may appear lower (innocent)",
    "MISSING_BLOCK": "Missing data block — NaN readings for several days (innocent)",
    "TARIFF_MISUSE": "Tariff misuse — SHOP-profile customer billed as DOMESTIC (revenue loss)",
    "OVERLOAD": "Overload — transformer operating near rated capacity (engineering alert)",
}

INNOCENT_LABEL = {True: "✓ Innocent", False: "✗ NTL / Loss"}
INNOCENT_COLOR = {True: "#30A46C", False: "#E5484D"}


def _normalize_end_day(value):
    """Convert mixed integer/placeholder values into a consistent string column."""
    if value is None or pd.isna(value):
        return "—"
    return str(value)


def render(world, results) -> None:
    """Render the Scenario Lab page."""
    st.title("What can happen")
    st.caption("Examples in the selected simulated world. These details are not available to the detection pipeline.")

    obs = world.observed
    scenarios = world.truth.scenarios
    today_day = results.today_day

    if not scenarios:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#52525B;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">✓</div>
            <div style="font-size:1rem;font-weight:600;color:#71717A;">Clean Grid — No scenarios active</div>
            <div style="font-size:0.85rem;color:#52525B;margin-top:0.25rem;">Switch to the Showcase or Random preset to inject events.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Build readable table
    rows = []
    for sc in scenarios:
        # Resolve target name
        target_name = str(sc.target)
        if sc.kind in ("BYPASS", "NIGHT_THEFT", "STEP_TAMPER", "STUCK_METER", "WRONG_MAPPING",
                        "VACANT", "SOLAR", "TARIFF_MISUSE"):
            row = obs.customers[obs.customers["meter_idx"] == sc.target]
            if not row.empty:
                target_name = f"{row.iloc[0]['meter_id']} ({row.iloc[0]['cls']})"
        elif sc.kind in ("ILLEGAL_TAP", "OVERLOAD"):
            row = obs.transformers[obs.transformers["tx_idx"] == sc.target]
            if not row.empty:
                target_name = row.iloc[0]["tx_id"]
        elif sc.kind == "FEEDER_SEGMENT":
            row = obs.feeders[obs.feeders["feeder_idx"] == sc.target]
            if not row.empty:
                target_name = row.iloc[0]["feeder_id"]
        elif sc.kind == "MISSING_BLOCK":
            tgts = sc.params.get("targets", [sc.target])
            names = []
            for ti in tgts:
                row = obs.customers[obs.customers["meter_idx"] == ti]
                if not row.empty:
                    names.append(row.iloc[0]["meter_id"])
            target_name = ", ".join(names) if names else str(tgts)

        active = (sc.end_day is None or sc.end_day > today_day) and sc.start_day < today_day
        status_str = "Active" if active else ("Future" if sc.start_day >= today_day else "Fixed")

        rows.append({
            "ID": sc.id,
            "Kind": sc.kind,
            "Target": target_name,
            "Start Day": sc.start_day,
            "End Day": _normalize_end_day(sc.end_day),
            "Active?": status_str,
            "Innocent?": INNOCENT_LABEL[sc.innocent],
            "Description": KIND_DESC.get(sc.kind, sc.kind),
        })

    df = pd.DataFrame(rows)

    def _color_innocent(val: str) -> str:
        return f"color: {INNOCENT_COLOR[val == '✓ Innocent']}; font-weight: 600"

    def _color_active(val: str) -> str:
        return {
            "Active": "color: #E5484D; font-weight: 600",
            "Future": "color: #71717A",
            "Fixed": "color: #30A46C",
        }.get(val, "")

    styled = df.style.map(_color_innocent, subset=["Innocent?"]).map(_color_active, subset=["Active?"])
    st.dataframe(styled, width="stretch", hide_index=True)

    # Summary cards
    st.markdown("---")
    n_active = sum(1 for r in rows if r["Active?"] == "Active")
    n_ntl = sum(1 for r in rows if r["Innocent?"] == "✗ NTL / Loss" and r["Active?"] == "Active")
    n_innocent = n_active - n_ntl

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value kpi-amber">{n_active}</div>
            <div class="kpi-label">Active scenarios</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value kpi-red">{n_ntl}</div>
            <div class="kpi-label">NTL / Loss events</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value kpi-green">{n_innocent}</div>
            <div class="kpi-label">Innocent events (should NOT alert)</div>
        </div>""", unsafe_allow_html=True)

    # Kind legend
    st.markdown('<div class="section-header">Scenario Kind Reference</div>', unsafe_allow_html=True)
    for kind, desc in KIND_DESC.items():
        st.markdown(f"""
        <div style="display:flex;gap:0.75rem;padding:0.4rem 0;border-bottom:1px solid #27272A;">
            <div style="min-width:140px;font-size:0.78rem;font-weight:600;color:#D4D4D8;font-family:'JetBrains Mono',monospace;">{kind}</div>
            <div style="font-size:0.82rem;color:#A1A1AA;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
