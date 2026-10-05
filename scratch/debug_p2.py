from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger import pipeline
import numpy as np

w = simulate(seed=1, scenarios=showcase_scenarios(1))
obs = w.observed
res = pipeline.run(obs, today_day=40)
diags = res.diagnostics_by_tx['T04']

solar_sc = next(s for s in w.truth.scenarios if s.kind == 'SOLAR')
solar_ci = solar_sc.target
print("solar target ci:", solar_ci)
for d in diags:
    if d.meter_idx == solar_ci:
        print("Solar diag:", d.meter_id, d.label, "ratio:", d.ratio, "shortfall:", d.shortfall_kwh, "cleared:", d.cleared, "is_cand:", d.is_candidate)

case_t04 = res.cases_by_node.get('T04')
print("T04 suspects:")
for s in case_t04.suspects:
    print(s)
