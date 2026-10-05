from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger.records_error import whatif_reassign
from gridledger.pipeline import _clean_meter_data
import numpy as np

seed = 2
world = simulate(seed=seed, scenarios=showcase_scenarios(world_seed=seed))
obs = world.observed
cfg = world.cfg
today_day = 40

swap = whatif_reassign(obs, 21, to_tx=1, today_day=40, from_tx=6)
bef_to = swap['before_finding_to']
aft_to = swap['after_finding_to']

print("=== BEFORE SWAP T02 ===")
print("R_daily (last 16 days):", [round(x, 1) for x in bef_to.R_daily[24:]])
print("S pos (last 16 days):", [round(x, 2) for x in bef_to.S[24:]])
print("Runs:", bef_to.runs)

print("\n=== AFTER SWAP T02 ===")
print("R_daily (last 16 days):", [round(x, 1) for x in aft_to.R_daily[24:]])
print("Runs:", aft_to.runs)

print("\n=== BEFORE SWAP T07 ===")
bef_from = swap['before_finding_from']
aft_from = swap['after_finding_from']
print("T07 before runs:", bef_from.runs)
print("T07 after runs:", aft_from.runs)
