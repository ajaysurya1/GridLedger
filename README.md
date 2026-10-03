# GridLedger ⚡

> **Detecting Non-Technical Losses in Electrical Distribution** — a hackathon-ready, demo-deployable product.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.37+-red)](https://streamlit.io)

## What It Does

GridLedger reconciles energy at every boundary of the grid tree (Feeder → Transformer → Customer meter).
It finds the exact branch where **energy-in stops matching energy-billed**, then ranks probable causes inside that branch.

**Thesis:** A meter that looks odd while its transformer still balances is **never escalated**.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens on the **Showcase world** (seed 7, day 40) with:
- 🔴 **T04** — BYPASS on DOM_HIGH customer (since day 20)  
- 🔴 **T09** — Illegal tap 1.8 kW (since day 22)  
- 🔴 **F1**  — Feeder segment theft 4 kW (since day 25)  
- 🟢 **T06** — Vacant premise (innocent — no false alert)  
- 🟢 **T11** — Missing data block (innocent — no false alert)

## Architecture

```
Feeder meter → Transformer meter → Customer meters
     F1, F2         T01–T12           ~300 meters
```

**Detection pipeline** (no oracle, no peeking):
1. Clean meter data (spikes, NaN imputation)
2. Aggregate metered sum by transformer (honouring mapping records)
3. Fit loss model on calibration window (NNLS: `L = (a + b·E²)·tf`)
4. Compute daily residuals → standardise → CUSUM
5. Localise: feeder run with no TX run beneath → **segment theft**; TX run → **customer zone**
6. Status: RED/AMBER/GREEN per node

## Repo Layout

```
app.py                    # Streamlit navigation + sidebar
views/
  command_center.py       # KPI strip + grid tree + node summary
  case_file.py            # CUSUM chart + waterfall + narrative
  scenario_lab.py         # Active scenario table (no oracle leak)
gridledger/
  config.py               # All constants (labelled ASSUMPTION)
  schema.py               # Observed / Truth / World dataclasses
  simulator.py            # Deterministic vectorised simulator
  scenarios.py            # Scenario dataclass + showcase/random presets
  topology.py             # NetworkX grid graph
  loss_model.py           # NNLS loss fitting + residuals
  balance.py              # Metered aggregation + CUSUM + feeder residuals
  persistence.py          # Run classification (night share, active flag)
  pipeline.py             # Main detection pipeline (oracle-free)
  oracle.py               # Ground truth (tests / evaluate only)
  suspects.py             # Phase 2 stub
  reconcile.py            # Phase 2 stub
  triage.py               # Phase 2 stub
  voltage.py              # Phase 3 stub
  fusion.py               # Phase 3 stub
  verify.py               # Phase 4 stub (oracle access allowed)
  savings.py              # Phase 5 stub (oracle access allowed)
  evaluate.py             # Phase 5 stub (oracle access allowed)
  ingest.py               # Phase 6 stub
tests/
  test_core.py            # Determinism, detection, conservation, no-peek
  test_apptest.py         # Streamlit AppTest smoke tests
```

## Running Tests

```bash
pytest -q
```

## Design Rules

- **No oracle in detection**: `pipeline.py`, `balance.py`, `loss_model.py`, etc. never import `oracle.py`
- **Deterministic**: All RNGs are `np.random.default_rng([seed, entity_key])`; adding a scenario never changes unrelated customers
- **No peeking**: Pipeline slices all data to `T = today_day × 24` as first step
- **All constants labelled**: Every money/CO2/behaviour value in `config.py` is marked `ASSUMPTION`
- **Vectorised**: No Python loops over time steps; numpy throughout

## Configuration

All constants in [`gridledger/config.py`](gridledger/config.py). Key fields:

| Field | Default | Label |
|-------|---------|-------|
| `tariff_inr` | DOMESTIC: ₹6.5, COMMERCIAL: ₹9.5, INDUSTRIAL: ₹8.0 | ASSUMPTION |
| `co2_t_per_mwh` | 0.710 | ASSUMPTION |
| `cusum_h` | 5.5 | Tuned on seeds 1-5 |
| `cal_days` | 14 | Calibration window |

## Phases Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ **Done** | Simulator + balance engine + CUSUM + Command Center + Case File |
| 2 | 🔜 Planned | Suspects ranking + full tree reconciliation + triage |
| 3 | 🔜 Planned | Voltage cross-check + evidence fusion |
| 4 | 🔜 Planned | Demo loop + inspection outcome simulation |
| 5 | 🔜 Planned | Benchmark + evaluate + savings counterfactual |
| 6 | 🔜 Planned | Real data ingest + polish |
