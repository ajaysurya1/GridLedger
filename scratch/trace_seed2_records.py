from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger import pipeline
from gridledger.records_error import whatif_reassign
from gridledger.reconcile import reconcile
import numpy as np

seed = 2
scenarios = showcase_scenarios(world_seed=seed)
world = simulate(seed=seed, scenarios=scenarios)
obs = world.observed
cfg = world.cfg
today_day = 40
T = today_day * 24

res = pipeline.run(obs, today_day=40)
nodes = res.nodes
cleaned_meter = res.cleaned_meter
mapping_now = obs.mapping[obs.mapping["valid_from_day"] < today_day]

neg_nf = nodes["T07"]
cust_neg = mapping_now[mapping_now["tx_idx"] == neg_nf.tx_idx]["meter_idx"].unique()
print("cust_neg on T07:", cust_neg)
print("Is C0022 (ci=21) in cust_neg?", 21 in cust_neg)

start_day = 24
end_day = 39
w_start = start_day * 24
w_end = (end_day + 1) * 24
n_blocks = max(1, (w_end - w_start) // 6)
b_lim = n_blocks * 6

neg_gap_blocks = np.maximum(0.0, -neg_nf.r_hourly[w_start: w_start + b_lim]).reshape(n_blocks, 6).sum(axis=1)
print("neg_gap_blocks sum:", neg_gap_blocks.sum())

A_cons = np.zeros((n_blocks, len(cust_neg)), dtype=np.float64)
for i, ci in enumerate(cust_neg):
    A_cons[:, i] = cleaned_meter[ci, w_start: w_start + b_lim].reshape(n_blocks, 6).sum(axis=1)

out_neg = reconcile(neg_gap_blocks, A_cons)
print("theta:", out_neg["theta"])
print("contrib:", out_neg["contrib"])

cand_indices = [
    i for i, th in enumerate(out_neg["theta"])
    if th >= 0.40 and out_neg["contrib"][i] >= 0.25 * neg_gap_blocks.sum()
]
print("cand_indices:", cand_indices)

# Check whatif_reassign for C0022 to T02 (tx_idx=1)
swap = whatif_reassign(obs, 21, to_tx=1, today_day=40, from_tx=neg_nf.tx_idx, cleaned_meter=cleaned_meter)
print("\nswap C0022 to T02:", swap)
