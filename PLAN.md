# GridLedger — Phase 1 Build Plan

## Mission
Detect non-technical losses by energy reconciliation at every grid boundary.
Escalate ONLY when a transformer/feeder imbalance is confirmed.

## Milestones (this session)
1. **Repo scaffold** — folders, stubs, requirements, config.toml
2. **Schema + Config** — frozen dataclasses; all constants in one place
3. **Simulator** — deterministic, vectorised, <3 s; full physics chain
4. **Scenarios** — 9 scenario kinds + showcase + clean + random presets
5. **Oracle** — ground-truth wrapper; only tests/evaluate.py may import it
6. **Balance engine** — metered_sum_by_tx, loss model, CUSUM, localisation
7. **Persistence detector** — run tracking, night_share, active flag
8. **Pipeline** — pure function; cached by caller; no peeking enforced
9. **App shell** — app.py navigation; sidebar; demo clock state
10. **Command Center** — KPI strip, Plotly grid-tree, node selection
11. **Case File** — waterfall + CUSUM chart + data-quality + plain-English
12. **Scenario Lab** — stub readable table
13. **Tests** — determinism, conservation, detection, no-peek, AppTest smoke
14. **Tune CUSUM** — h/k on seeds 1-5; ≤1 false RED per 100 tx-days
15. **Final check** — pytest -q; target: T04 RED, T09 RED, F1 RED, T06 GREEN, T11 GREEN
16. **Commit + push** — runnable at every commit

## Key Design Decisions
- Detection receives ONLY `Observed`; never imports oracle
- RNG: `default_rng([seed, entity_key])` per entity — adding scenarios changes nothing else
- Loss model fitted on calibration window only (days 0..cal_days)
- CUSUM on daily standardised residuals; disjoint tx vs feeder residuals
- All money/CO2/behaviour constants labelled ASSUMPTION in config.py
- Streamlit heavy work behind st.cache_resource / st.cache_data
- Wide layout; semantic colours RED #E5484D AMBER #F5A524 GREEN #30A46C
