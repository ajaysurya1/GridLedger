from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger import pipeline
from gridledger.records_error import detect_records_errors, whatif_reassign

seed = 1
scenarios = showcase_scenarios(world_seed=seed)
world = simulate(seed=seed, scenarios=scenarios)
obs = world.observed
truth = world.truth

wm_sc = next(s for s in truth.scenarios if s.kind == "WRONG_MAPPING")
wm_ci = wm_sc.target
wm_meter_id = obs.customers.loc[obs.customers["meter_idx"] == wm_ci, "meter_id"].iloc[0]
print(f"Target WRONG_MAPPING meter: {wm_meter_id} (ci={wm_ci}), wrong_tx_idx={wm_sc.params.get('wrong_tx_idx')}")

res = pipeline.run(obs, today_day=40)
print("Records errors found in pipeline:", len(res.records_errors))
for re in res.records_errors:
    print("  Found:", re.meter_id, re.from_tx_id, "->", re.to_tx_id, "confirmed:", re.confirmed)

# Check nodes
print("\nT02 (physical) runs:", [(r.direction, r.active, r.start, r.end, round(r.excess_kwh, 1)) for r in res.nodes["T02"].runs])
print("T02 S:", res.nodes["T02"].S[-5:])
print("T07 (wrong map) runs:", [(r.direction, r.active, r.start, r.end, round(r.excess_kwh, 1)) for r in res.nodes["T07"].runs])
print("T07 S:", res.nodes["T07"].S[-5:])

# Directly test whatif_reassign
swap = whatif_reassign(obs, wm_ci, to_tx=int(obs.transformers[obs.transformers["tx_id"] == "T02"]["tx_idx"].iloc[0]), today_day=40, from_tx=int(obs.transformers[obs.transformers["tx_id"] == "T07"]["tx_idx"].iloc[0]))
print("\nDirect whatif_reassign swap result:", swap)
