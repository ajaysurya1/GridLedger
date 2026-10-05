import numpy as np
from gridledger.simulator import simulate
from gridledger.scenarios import Scenario
from gridledger import pipeline

kinds = ["BYPASS", "STEP_TAMPER", "NIGHT_THEFT"]
seed = 1

clean = simulate(seed=seed, scenarios=[])
obs = clean.observed
tx_true = clean.truth.tx_true

tx_list = ["T01", "T03", "T05"]
scenarios = []
target_info = {}

for kind, tx_name in zip(kinds, tx_list):
    tx_row = obs.transformers[obs.transformers["tx_id"] == tx_name].iloc[0]
    ji = int(tx_row["tx_idx"])
    cands = np.where(tx_true == ji)[0]
    target_ci = int(cands[0])
    target_meter_id = obs.customers.loc[obs.customers["meter_idx"] == target_ci, "meter_id"].iloc[0]

    sc = Scenario(f"TEST_{kind}", kind, target_ci, start_day=18, params={"k": 0.5})
    scenarios.append(sc)
    target_info[tx_name] = (target_ci, target_meter_id, kind)

world = simulate(seed=seed, scenarios=scenarios)
res = pipeline.run(world.observed, today_day=40)

for tx_name, (target_ci, target_meter, kind) in target_info.items():
    case = res.cases_by_node.get(tx_name)
    rec = res.reconcile_by_tx.get(tx_name, {})
    diags = res.diagnostics_by_tx.get(tx_name, [])
    cand_diags = [d for d in diags if d.is_candidate]
    
    print(f"\n=================== {tx_name} ({kind}) ===================")
    print(f"Target meter: {target_meter} (ci={target_ci})")
    
    if case:
        print("Case coverage:", case.coverage, "kwh/day:", case.kwh_per_day)
        print("Top 5 suspects in Case:")
        for s in case.suspects[:5]:
            is_tgt = " *** TARGET ***" if s["meter"] == target_meter else ""
            print(f"  {s['meter']}: theta={s.get('theta',0):.3f}, score={s.get('score',0):.2f}, contrib={s.get('contrib_kwh',0):.1f}, ratio={s.get('ratio',0):.3f}, step_day={s.get('step_day')}{is_tgt}")
    else:
        print("NO CASE FOR", tx_name)
    
    print("Candidates in reconcile:", len(cand_diags))
    for d in cand_diags:
        is_tgt = " *** TARGET ***" if d.meter_idx == target_ci else ""
        print(f"  cand {d.meter_id} (ci={d.meter_idx}): shortfall={d.shortfall_kwh:.1f}, ratio={d.ratio:.3f}, step={d.step_day}, night_share={d.night_shortfall_share:.2f}{is_tgt}")
