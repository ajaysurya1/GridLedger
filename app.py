"""
GridLedger — main Streamlit entry point.

Navigation + global sidebar (demo clock, scenario preset).
All heavy computation lives behind st.cache_resource / st.cache_data.
"""
from __future__ import annotations

import json
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="GridLedger — NTL Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0D0D0F;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111113 0%, #0D0D0F 100%);
    border-right: 1px solid #27272A;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FAFAFA;
}

/* Logo */
.gl-logo {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #30A46C 0%, #4ade80 50%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.gl-tagline {
    font-size: 0.7rem;
    color: #71717A;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* KPI cards */
.kpi-card {
    background: #18181B;
    border: 1px solid #27272A;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s ease;
}
.kpi-card:hover { border-color: #3F3F46; }
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-label {
    font-size: 0.72rem;
    color: #A1A1AA;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.kpi-red { color: #E5484D; }
.kpi-amber { color: #F5A524; }
.kpi-green { color: #30A46C; }
.kpi-blue { color: #60A5FA; }

/* Status badges */
.badge-red {
    display: inline-block;
    background: rgba(229,72,77,0.15);
    color: #E5484D;
    border: 1px solid rgba(229,72,77,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-amber {
    display: inline-block;
    background: rgba(245,165,36,0.15);
    color: #F5A524;
    border: 1px solid rgba(245,165,36,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-green {
    display: inline-block;
    background: rgba(48,164,108,0.15);
    color: #30A46C;
    border: 1px solid rgba(48,164,108,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #71717A;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #27272A;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, rgba(48,164,108,0.08) 0%, rgba(34,211,238,0.05) 100%);
    border: 1px solid rgba(48,164,108,0.25);
    border-left: 3px solid #30A46C;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #D4D4D8;
}
.insight-box strong { color: #FAFAFA; }

/* Warning box */
.warn-box {
    background: rgba(229,72,77,0.08);
    border: 1px solid rgba(229,72,77,0.2);
    border-left: 3px solid #E5484D;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #FCA5A5;
}

/* Assumption badge */
.assumption-badge {
    display: inline-block;
    background: rgba(245,165,36,0.1);
    color: #F5A524;
    border-radius: 4px;
    padding: 0 6px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 4px;
    vertical-align: middle;
}

/* Demo badge */
.demo-badge {
    display: inline-block;
    background: rgba(96,165,250,0.1);
    color: #60A5FA;
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Clock display */
.clock-display {
    background: #18181B;
    border: 1px solid #27272A;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
    margin: 0.75rem 0;
}
.clock-day {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #FAFAFA;
    letter-spacing: -0.04em;
}
.clock-label {
    font-size: 0.65rem;
    color: #71717A;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Streamlit overrides */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
    border-radius: 8px;
    border: 1px solid #3F3F46;
    background: #18181B;
    color: #FAFAFA;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #27272A;
    border-color: #52525B;
    color: #FAFAFA;
}
.stSelectbox > div > div {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

/* Dividers */
hr { border-color: #27272A; margin: 1rem 0; }

/* Metric delta */
[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #27272A;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    color: #71717A;
    padding: 0.5rem 1rem;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #FAFAFA;
    border-bottom: 2px solid #30A46C;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #18181B; }
::-webkit-scrollbar-thumb { background: #3F3F46; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #52525B; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
def _init_state():
    if "today_day" not in st.session_state:
        from gridledger.config import CFG
        st.session_state.today_day = CFG.default_today_day
    if "scenario_preset" not in st.session_state:
        st.session_state.scenario_preset = "Showcase"
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Command Center"


_init_state()


# ── Cached simulation ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Simulating grid…")
def _get_world(preset: str, random_seed: int):
    from gridledger.simulator import simulate
    from gridledger.scenarios import showcase_scenarios, random_scenarios

    if preset == "Showcase":
        scenarios = showcase_scenarios(world_seed=7)
        return simulate(seed=7, scenarios=scenarios)
    elif preset == "Clean grid":
        return simulate(seed=7, scenarios=[])
    else:  # Random
        scenarios = random_scenarios(world_seed=7, random_seed=random_seed)
        return simulate(seed=7, scenarios=scenarios)


@st.cache_data(show_spinner="Running detection…", ttl=None)
def _get_results(preset: str, random_seed: int, today_day: int):
    from gridledger import pipeline
    world = _get_world(preset, random_seed)
    return pipeline.run(world.observed, today_day)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar():
    from gridledger.config import CFG

    with st.sidebar:
        st.markdown('<div class="gl-logo">⚡ GridLedger</div>', unsafe_allow_html=True)
        st.markdown('<div class="gl-tagline">Non-Technical Loss Detection</div>', unsafe_allow_html=True)
        st.markdown('<span class="demo-badge">Simulated Data</span>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Demo Clock</div>', unsafe_allow_html=True)

        today = st.session_state.today_day
        st.markdown(f"""
        <div class="clock-display">
            <div class="clock-day">Day {today}</div>
            <div class="clock-label">of {CFG.n_days} simulated days</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("+1", use_container_width=True):
                st.session_state.today_day = min(st.session_state.today_day + 1, CFG.n_days)
                st.rerun()
        with col2:
            if st.button("+5", use_container_width=True):
                st.session_state.today_day = min(st.session_state.today_day + 5, CFG.n_days)
                st.rerun()
        with col3:
            if st.button("↺", use_container_width=True):
                st.session_state.today_day = CFG.default_today_day
                st.rerun()

        st.markdown('<div class="section-header">Scenario Preset</div>', unsafe_allow_html=True)

        preset = st.selectbox(
            "Preset",
            ["Showcase", "Clean grid", "Random"],
            index=["Showcase", "Clean grid", "Random"].index(st.session_state.scenario_preset),
            label_visibility="collapsed",
        )
        if preset != st.session_state.scenario_preset:
            st.session_state.scenario_preset = preset
            st.session_state.selected_node = None
            st.rerun()

        if preset == "Random":
            rseed = st.number_input("Random seed", value=st.session_state.random_seed, min_value=0, step=1)
            if rseed != st.session_state.random_seed:
                st.session_state.random_seed = int(rseed)
                st.session_state.selected_node = None
                st.rerun()

        # Navigation
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)

        pages = {
            "Command Center": "🗺️",
            "Case File": "📋",
            "Scenario Lab": "🧪",
        }
        for page, icon in pages.items():
            is_active = st.session_state.active_page == page
            btn_style = "primary" if is_active else "secondary"
            if st.button(f"{icon} {page}", use_container_width=True, type=btn_style if is_active else "secondary"):
                st.session_state.active_page = page
                st.rerun()

        # Status legend
        st.markdown('<div class="section-header">Legend</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem; line-height:2;">
            <span class="badge-red">RED</span> Active anomaly run<br>
            <span class="badge-amber">AMBER</span> Watch — elevated risk<br>
            <span class="badge-green">GREEN</span> Normal balance
        </div>
        """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
_render_sidebar()

# Preload data
world = _get_world(st.session_state.scenario_preset, st.session_state.random_seed)
results = _get_results(st.session_state.scenario_preset, st.session_state.random_seed, st.session_state.today_day)

# Route to active page
page = st.session_state.active_page

if page == "Command Center":
    from views import command_center
    command_center.render(world, results)
elif page == "Case File":
    from views import case_file
    case_file.render(world, results)
elif page == "Scenario Lab":
    from views import scenario_lab
    scenario_lab.render(world, results)
