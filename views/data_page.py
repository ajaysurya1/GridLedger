"""Bring-your-own-data workflow."""
from __future__ import annotations

import streamlit as st

from gridledger.ingest import import_csvs, legacy_weekly_audit, template_csvs


def _template_downloads() -> None:
    templates = template_csvs()
    cols = st.columns(3)
    for col, (name, contents) in zip(cols, templates.items()):
        with col:
            st.download_button(f"Download {name}", contents, file_name=name, mime="text/csv",
                               use_container_width=True, help=f"Download the {name} column template.")


def render() -> None:
    st.title("Bring your utility data")
    st.caption("Upload interval readings, grid inputs, and customer-to-transformer records. Files stay in this session.")
    st.warning("Real-data mode has no hidden ground truth. Benchmark comparisons and oracle-based dispatch estimates are unavailable here.", icon="ℹ️")

    if st.session_state.get("data_mode_active"):
        if st.button("Return to simulated demo", help="Leave this upload session and return to the prepared Showcase world."):
            st.session_state.data_mode_active = False
            st.session_state.data_result = None
            st.session_state.active_page = "Command Center"
            st.rerun()

    st.subheader("CSV templates")
    _template_downloads()
    with st.expander("Required columns and rules"):
        st.markdown("""
        - `meters.csv`: `timestamp`, `meter_id`, and exactly one of `kwh_interval` or `cumulative_kwh`.
        - `nodes.csv`: `timestamp`, `node_id`, `level` (`feeder` or `transformer`), `kwh_in`.
        - `mapping.csv`: `meter_id`, `transformer_id`, `feeder_id`, `valid_from`.

        Timestamps are aligned to the hour. Duplicate rows keep the latest value and are reported. A falling cumulative register is quarantined as a reset. Meter gaps remain missing; node input gaps are interpolated and counted.
        """)

    legacy = st.toggle("Audit mode for utilities without hourly data", key="legacy_audit_mode",
                       help="Aggregate imported readings into weekly balance checks. This mode excludes time-of-day signals and voltage analysis.")
    meter_file = st.file_uploader("Meters", type="csv", key="upload_meters")
    node_file = st.file_uploader("Grid input readings", type="csv", key="upload_nodes")
    mapping_file = st.file_uploader("Customer mapping", type="csv", key="upload_mapping")

    if st.button("Import and review", type="primary", disabled=not all((meter_file, node_file, mapping_file)),
                  help="Validate all three files, align readings, and run the selected analysis."):
        try:
            with st.spinner("Checking files and aligning readings…"):
                result = import_csvs(meter_file, node_file, mapping_file)
                st.session_state.data_result = result
                st.session_state.data_mode_active = True
                if legacy:
                    st.session_state.legacy_weekly = legacy_weekly_audit(result.observed)
                    st.session_state.imported_detection = None
                else:
                    from gridledger.config import CFG
                    from gridledger.pipeline import run
                    days = result.observed.meter_kwh.shape[1] // 24
                    if days <= CFG.cal_days:
                        raise ValueError(f"At least {CFG.cal_days + 1} full days are needed for the hourly analysis; this upload contains {days}.")
                    today = min(days, CFG.default_today_day)
                    st.session_state.imported_detection = run(result.observed, today_day=today)
                    st.session_state.legacy_weekly = None
            st.success("Files imported. Review the data checks and findings below.")
        except Exception as exc:
            st.error(f"We could not import these files: {exc}")

    result = st.session_state.get("data_result")
    if result is None:
        st.info("Add the three CSV files to begin. Nothing is sent to a server outside this app session.")
        return

    st.subheader("Data quality")
    report = result.report
    a, b, c, d = st.columns(4)
    a.metric("Readings kept", f"{report['rows']['meters']:,}", help="Meter rows after duplicate removal.")
    b.metric("Coverage", f"{report['coverage']:.1%}", help="Observed meter intervals divided by expected meter intervals. Missing readings are not treated as zero.")
    c.metric("Quarantined", f"{report['quarantined_points']:,}", help="Duplicates, negative values, register resets, spikes, or stale mappings.")
    d.metric("Reset registers", f"{report['register_resets']:,}", help="Cumulative registers that decreased; the affected interval was quarantined.")
    with st.expander("Coverage by transformer"):
        st.dataframe([{"Transformer": node, "Missing meter readings": f"{share:.1%}"}
                      for node, share in report["missing_per_transformer"].items()], use_container_width=True, hide_index=True)
    if not result.quarantine.empty:
        with st.expander("Quarantined rows"):
            st.dataframe(result.quarantine, use_container_width=True, hide_index=True)

    if st.session_state.get("legacy_weekly") is not None:
        st.subheader("Weekly energy checks")
        st.caption("Audit mode for utilities without hourly data. No time-of-day patterns, voltage checks, or automated actions are included.")
        st.dataframe(st.session_state.legacy_weekly, use_container_width=True, hide_index=True)
        return

    results = st.session_state.get("imported_detection")
    if results is None:
        st.caption("The hourly analysis needs at least 15 complete days so it can learn a baseline before checking for changes.")
        return
    st.subheader("Findings to review")
    findings = []
    for node_id, finding in results.nodes.items():
        findings.append({
            "Grid location": node_id,
            "Status": {"RED": "▲ Review", "AMBER": "● Watch", "GREEN": "■ Balanced"}.get(finding.status, finding.status),
            "Unaccounted energy (kWh/day)": round(sum(abs(run.kwh_per_day) for run in finding.runs if run.active), 1),
            "Data warnings": len(finding.dq_warnings),
        })
    st.dataframe(findings, use_container_width=True, hide_index=True)
    st.caption("These results support prioritisation only. They are not proof of theft or grounds for customer action.")