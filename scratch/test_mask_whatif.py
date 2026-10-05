import numpy as np
from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios
from gridledger.balance import metered_sum_by_tx
from gridledger.records_error import whatif_reassign
from gridledger.pipeline import _clean_meter_data

seed = 1
world = simulate(seed=seed, scenarios=showcase_scenarios(world_seed=seed))
obs = world.observed
cfg = world.cfg
today_day = 40
T = today_day * 24

cleaned_meter, _, imputed_kwh = _clean_meter_data(obs.meter_kwh[:, :T], obs.customers, cfg)
mapping_now = obs.mapping[obs.mapping["valid_from_day"] < today_day]
n_tx = len(obs.transformers)
met_orig = metered_sum_by_tx(cleaned_meter, mapping_now, n_tx, today_day)
imp_orig = metered_sum_by_tx(imputed_kwh, mapping_now, n_tx, today_day)

wm_ci = 19  # C0020
from_tx = 6  # T07
to_tx = 1    # T02

meter_mapping = mapping_now[mapping_now["meter_idx"] == wm_ci].sort_values("valid_from_day")
mapped_tx_hourly = np.zeros(T, dtype=np.int32)
for _, row in meter_mapping.iterrows():
    start_h = int(row["valid_from_day"]) * 24
    mapped_tx_hourly[start_h:] = int(row["tx_idx"])

mask_from = (mapped_tx_hourly == from_tx)
print("Hours mapped to T07:", mask_from.sum())

c_m = cleaned_meter[wm_ci, :T]
c_imp = imputed_kwh[wm_ci, :T]

met_new_from = met_orig[from_tx].copy()
met_new_from[mask_from] -= c_m[mask_from]

met_new_to = met_orig[to_tx].copy()
met_new_to[mask_from] += c_m[mask_from]

imp_new_from = imp_orig[from_tx].copy()
imp_new_from[mask_from] -= c_imp[mask_from]

imp_new_to = imp_orig[to_tx].copy()
imp_new_to[mask_from] += c_imp[mask_from]

from gridledger.loss_model import temp_factor
from gridledger.pipeline import _analyse_node

tf = temp_factor(obs.temp[:T], cfg.kappa, cfg.t_ref)

finding_from_after = _analyse_node(
    "T07", "transformer", obs.tx_in_kwh[from_tx, :T].astype(float), met_new_from.astype(float),
    imp_new_from.astype(float), tf, today_day, cfg.cal_days, cfg, [], tx_idx=from_tx
)

finding_to_after = _analyse_node(
    "T02", "transformer", obs.tx_in_kwh[to_tx, :T].astype(float), met_new_to.astype(float),
    imp_new_to.astype(float), tf, today_day, cfg.cal_days, cfg, [], tx_idx=to_tx
)

print("T07 after runs:", finding_from_after.runs)
print("T07 after S:", finding_from_after.S[-5:])
print("T02 after runs:", finding_to_after.runs)
print("T02 after S:", finding_to_after.S[-5:])
