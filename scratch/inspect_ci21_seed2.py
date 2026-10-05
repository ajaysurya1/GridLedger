from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger.pipeline import _clean_meter_data
import numpy as np

seed = 2
world = simulate(seed=seed, scenarios=showcase_scenarios(world_seed=seed))
obs = world.observed
cfg = world.cfg
today_day = 40
T = today_day * 24

cleaned_meter, _, _ = _clean_meter_data(obs.meter_kwh[:, :T], obs.customers, cfg)

ci = 21
raw_kwh = obs.meter_kwh[ci, :T]
clean_kwh = cleaned_meter[ci, :T]
true_kwh = world.truth.true_kwh[ci, :T]

print("ci 21 mean raw:", np.nanmean(raw_kwh))
print("ci 21 mean true:", true_kwh.mean())
print("ci 21 daily sum (days 0..13 cal):", [round(float(np.nansum(raw_kwh[d*24:(d+1)*24])), 1) for d in range(14)])
print("ci 21 daily sum (days 14..23 pre-swap):", [round(float(np.nansum(raw_kwh[d*24:(d+1)*24])), 1) for d in range(14, 24)])
print("ci 21 daily sum (days 24..39 post-swap):", [round(float(np.nansum(raw_kwh[d*24:(d+1)*24])), 1) for d in range(24, 40)])

# Now let's check T02 physical load and metered load
tx2_in = obs.tx_in_kwh[1, :T]
print("\nT02 in daily:", [round(float(tx2_in[d*24:(d+1)*24].sum()), 1) for d in range(24, 40)])
