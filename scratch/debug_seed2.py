from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger import pipeline

seed = 2
scenarios = showcase_scenarios(world_seed=seed)
world = simulate(seed=seed, scenarios=scenarios)
obs = world.observed
truth = world.truth
res = pipeline.run(obs, today_day=40)

# 1. T04 BYPASS
bypass_sc = next(s for s in truth.scenarios if s.kind == "BYPASS")
bypass_ci = bypass_sc.target
bypass_meter_id = obs.customers.loc[obs.customers["meter_idx"] == bypass_ci, "meter_id"].iloc[0]
case_t04 = res.cases_by_node.get("T04")
top1_meter = case_t04.suspects[0]["meter"] if case_t04 and case_t04.suspects else "None"
print(f"Seed 2 T04 BYPASS: expected {bypass_meter_id}, got {top1_meter} -> {'PASS' if top1_meter == bypass_meter_id else 'FAIL'}")

# Solar decoy
solar_sc = next(s for s in truth.scenarios if s.kind == "SOLAR")
solar_ci = solar_sc.target
solar_meter_id = obs.customers.loc[obs.customers["meter_idx"] == solar_ci, "meter_id"].iloc[0]
top3_meters = [s["meter"] for s in case_t04.suspects[:3]] if case_t04 and case_t04.suspects else []
print(f"Seed 2 Solar decoy {solar_meter_id} not in top3: {top3_meters} -> {'PASS' if solar_meter_id not in top3_meters else 'FAIL'}")

# 2. T09 ILLEGAL_TAP
case_t09 = res.cases_by_node.get("T09")
cov_t09 = case_t09.coverage if case_t09 else -1
print(f"Seed 2 T09 cov={cov_t09:.2f} (<0.3) -> {'PASS' if cov_t09 < 0.3 else 'FAIL'}")

# 3. T08 STUCK_METER
stuck_sc = next(s for s in truth.scenarios if s.kind == "STUCK_METER")
stuck_ci = stuck_sc.target
stuck_meter_id = obs.customers.loc[obs.customers["meter_idx"] == stuck_ci, "meter_id"].iloc[0]
t08_diags = res.diagnostics_by_tx.get("T08", [])
stuck_diag = next((d for d in t08_diags if d.meter_idx == stuck_ci), None)
print(f"Seed 2 T08 STUCK_METER label={stuck_diag.label if stuck_diag else 'None'} -> {'PASS' if stuck_diag and stuck_diag.label == 'FAULTY_METER' else 'FAIL'}")

# 4. Records error
wm_sc = next(s for s in truth.scenarios if s.kind == "WRONG_MAPPING")
wm_ci = wm_sc.target
wm_meter_id = obs.customers.loc[obs.customers["meter_idx"] == wm_ci, "meter_id"].iloc[0]
found_rec_err = any(re.meter_idx == wm_ci and re.confirmed for re in res.records_errors)
print(f"Seed 2 Records error for {wm_meter_id} (ci={wm_ci}): found={found_rec_err} -> {'PASS' if found_rec_err else 'FAIL'}")
for re in res.records_errors:
    print(f"  Existing records error: {re.meter_id} (ci={re.meter_idx}) {re.from_tx_id} -> {re.to_tx_id} confirmed={re.confirmed}")

print(f"\nT02 node runs: {[(r.direction, r.active, r.start, r.end, round(r.excess_kwh, 1)) for r in res.nodes['T02'].runs]}")
print(f"T07 node runs: {[(r.direction, r.active, r.start, r.end, round(r.excess_kwh, 1)) for r in res.nodes['T07'].runs]}")
