from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger import pipeline

seed = 1
world = simulate(seed=seed, scenarios=showcase_scenarios(world_seed=seed))
res = pipeline.run(world.observed, today_day=40)
nf7 = res.nodes["T07"]
print("T07 R_daily[14:40]:", nf7.R_daily[14:40])
print("T07 R_daily < -2.0 count:", (nf7.R_daily[14:40] < -2.0).sum())
print("T07 runs:", [(r.direction, r.active, r.start, r.end, round(r.excess_kwh, 1)) for r in nf7.runs])

nf2 = res.nodes["T02"]
print("\nT02 R_daily[14:40]:", nf2.R_daily[14:40])
print("T02 runs:", [(r.direction, r.active, r.start, r.end, round(r.excess_kwh, 1)) for r in nf2.runs])
