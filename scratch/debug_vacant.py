from gridledger.simulator import simulate
from gridledger.scenarios import Scenario
from gridledger import pipeline
import numpy as np

seed = 7
clean = simulate(seed=seed, scenarios=[])
obs = clean.observed
tx_true = clean.truth.tx_true

t04_ji = int(obs.transformers[obs.transformers["tx_id"] == "T04"]["tx_idx"].iloc[0])
cands = list(np.where(tx_true == t04_ji)[0])
thief_ci = cands[0]
vacant_ci = cands[1]

scenarios = [
    Scenario("THEFT", "BYPASS", thief_ci, start_day=20, params={"k": 0.5}),
    Scenario("VACANT", "VACANT", vacant_ci, start_day=20, innocent=True),
]

world = simulate(seed=seed, scenarios=scenarios)
res = pipeline.run(world.observed, today_day=40)

diags = res.diagnostics_by_tx["T04"]
cand_diags = [d for d in diags if d.is_candidate]
print("Candidate meters:", [d.meter_id for d in cand_diags])
print("Thief meter:", obs.customers.loc[obs.customers["meter_idx"] == thief_ci, "meter_id"].iloc[0])
print("Vacant meter:", obs.customers.loc[obs.customers["meter_idx"] == vacant_ci, "meter_id"].iloc[0])

for d in cand_diags:
    print(d.meter_id, "label:", d.label, "ratio:", d.ratio, "shortfall:", d.shortfall_kwh)

rec = res.reconcile_by_tx["T04"]
print("Reconcile output:")
for k, v in rec.items():
    print(k, v)
