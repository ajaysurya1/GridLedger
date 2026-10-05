from gridledger.simulator import simulate
from gridledger.scenarios import showcase_scenarios

seed = 2
scs = showcase_scenarios(world_seed=seed)
world = simulate(seed=seed, scenarios=scs)
obs = world.observed
tx_true = world.truth.tx_true

for sc in scs:
    if isinstance(sc.target, int) and sc.target < len(tx_true):
        tgt_tx = tx_true[sc.target]
        tx_id = obs.transformers.loc[obs.transformers['tx_idx'] == tgt_tx, 'tx_id'].iloc[0]
        print(f"Scenario {sc.id} ({sc.kind}): target ci={sc.target} physically on {tx_id} (tx_idx={tgt_tx}), start_day={sc.start_day}")
    else:
        print(f"Scenario {sc.id} ({sc.kind}): target={sc.target}, start_day={sc.start_day}")
