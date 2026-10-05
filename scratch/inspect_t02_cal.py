from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger.records_error import whatif_reassign
import numpy as np

seed = 2
world = simulate(seed=seed, scenarios=showcase_scenarios(world_seed=seed))
obs = world.observed
cfg = world.cfg
today_day = 40

swap = whatif_reassign(obs, 21, to_tx=1, today_day=40, from_tx=6)
aft_to = swap['after_finding_to']

print("T02 fitted a, b:", aft_to.a, aft_to.b)
print("T02 sigma:", aft_to.sigma)
print("T02 R_daily days 0..13 (cal):", [round(x, 2) for x in aft_to.R_daily[:14]])
print("T02 R_daily days 14..23 (post-cal, pre-swap):", [round(x, 2) for x in aft_to.R_daily[14:24]])
print("T02 R_daily days 24..39 (post-swap):", [round(x, 2) for x in aft_to.R_daily[24:]])
print("T02 mean R_daily days 14..23:", np.mean(aft_to.R_daily[14:24]))
print("T02 mean R_daily days 24..39:", np.mean(aft_to.R_daily[24:]))
