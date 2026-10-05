from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger import pipeline

for seed in range(1, 6):
    world = simulate(seed=seed, scenarios=showcase_scenarios(world_seed=seed))
    obs = world.observed
    truth = world.truth
    res = pipeline.run(obs, today_day=40)

    wm_sc = next(s for s in truth.scenarios if s.kind == "WRONG_MAPPING")
    wm_ci = wm_sc.target
    wm_meter_id = obs.customers.loc[obs.customers["meter_idx"] == wm_ci, "meter_id"].iloc[0]
    found_rec_err = any(re.meter_idx == wm_ci and re.confirmed for re in res.records_errors)

    bypass_sc = next(s for s in truth.scenarios if s.kind == "BYPASS")
    bypass_ci = bypass_sc.target
    bypass_meter_id = obs.customers.loc[obs.customers["meter_idx"] == bypass_ci, "meter_id"].iloc[0]
    case_t04 = res.cases_by_node.get("T04")
    top1_meter = case_t04.suspects[0]["meter"] if case_t04 and case_t04.suspects else "None"
    
    print(f"Seed {seed}: RecErr={found_rec_err} (meter={wm_meter_id}, found={[r.meter_id for r in res.records_errors]}), T04_top1={top1_meter==bypass_meter_id}")
