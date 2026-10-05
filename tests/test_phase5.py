from __future__ import annotations

from io import StringIO
from pathlib import Path
import time

import numpy as np

from gridledger.evaluate import score_predictions
from gridledger.ingest import export_observed_csvs, import_csvs
from gridledger.pipeline import run
from gridledger.scenarios import Scenario, random_scenarios, showcase_scenarios
from gridledger.simulator import simulate
from scripts.run_eval import DEV_SEEDS, TEST_SEEDS, run as run_eval


def test_random_scenario_mix_and_solar_co_location():
    topology = simulate(101, scenarios=[])
    scenarios = random_scenarios(np.random.default_rng(101), topology.observed, 60)
    kinds = [scenario.kind for scenario in scenarios]
    for kind in ("BYPASS", "NIGHT_THEFT", "STEP_TAMPER", "ILLEGAL_TAP", "WRONG_MAPPING", "STUCK_METER",
                 "VACANT", "SOLAR", "MISSING_BLOCK", "TARIFF_MISUSE"):
        expected = 2 if kind in ("VACANT", "SOLAR") else 1
        assert kinds.count(kind) == expected
    assert kinds.count("FEEDER_SEGMENT") in (0, 1)
    assert len([s for s in scenarios if s.kind == "FEEDER_SEGMENT"]) <= 1
    assert all(18 <= s.start_day <= 34 for s in scenarios if s.kind in {
        "BYPASS", "NIGHT_THEFT", "STEP_TAMPER", "ILLEGAL_TAP", "WRONG_MAPPING", "STUCK_METER",
        "FEEDER_SEGMENT", "VACANT", "MISSING_BLOCK",
    })
    assert scenarios == random_scenarios(np.random.default_rng(101), topology.observed, 60)
    bypass = next(s for s in scenarios if s.kind == "BYPASS")
    solar = next(s for s in scenarios if s.kind == "SOLAR")
    assert topology.truth.tx_true[bypass.target] == topology.truth.tx_true[solar.target]


def test_raw_prediction_metrics_with_hand_made_case():
    scenarios = [
        Scenario("E1", "BYPASS", 0, 20, params={"k": 0.5}),
        Scenario("N1", "VACANT", 1, 20, innocent=True),
    ]
    world = simulate(13, scenarios=scenarios)
    tx_idx = int(world.truth.tx_true[0])
    zone = str(world.observed.transformers.loc[world.observed.transformers.tx_idx == tx_idx, "tx_id"].iloc[0])
    result = score_predictions(world, [{"zone": zone, "start_day": 22, "suspects": [0], "rank": 10}], 40)
    assert result["zone_localisation_accuracy"] == 1.0
    assert result["event_recall"] == 1.0
    assert result["case_precision"] == 1.0
    assert result["customer_top3_hit_rate"] == 1.0
    assert result["median_detection_delay_days"] == 2.0
    assert result["false_inspections_per_100_transformer_days"] == 0.0
    assert result["innocent_lookalike_false_positive_rate"] == 0.0
    assert result["recovered_simulated_kwh_top5"] == float(world.truth.stolen_kwh["E1"][:40 * 24].sum())


def test_tuning_and_test_seed_sets_are_disjoint_and_separated():
    assert DEV_SEEDS == (1, 2, 3, 4, 5)
    assert TEST_SEEDS == tuple(range(101, 111))
    assert set(DEV_SEEDS).isdisjoint(TEST_SEEDS)
    runner_source = Path("scripts/run_eval.py").read_text(encoding="utf-8")
    assert 'args.seeds == "dev"' in runner_source
    assert 'args.seeds == "test"' in runner_source
    assert '"eval_dev_results.json"' in runner_source
    assert '"eval_results.json"' in runner_source


def test_two_dev_seed_evaluation_under_30_seconds():
    started = time.perf_counter()
    rows = run_eval([1, 2])
    elapsed = time.perf_counter() - started
    assert len(rows) == 2
    assert all(set(row["systems"]) == {"A", "B", "C"} for row in rows)
    assert elapsed < 30.0


def test_showcase_csv_round_trip_preserves_red_nodes():
    world = simulate(7, scenarios=showcase_scenarios(7))
    csvs = export_observed_csvs(world.observed)
    imported = import_csvs(*(StringIO(csvs[name]) for name in ("meters.csv", "nodes.csv", "mapping.csv")))
    original_red = {node for node, finding in run(world.observed, 40).nodes.items() if finding.status == "RED"}
    imported_red = {node for node, finding in run(imported.observed, 40).nodes.items() if finding.status == "RED"}
    assert imported_red == original_red
    assert imported.report["missing_meter_percent"] > 0.0
    assert np.isnan(imported.observed.meter_kwh).any()