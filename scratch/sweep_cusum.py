import numpy as np
from gridledger.simulator import simulate
from gridledger.scenarios import Scenario, showcase_scenarios
from gridledger.config import Config, CFG
from gridledger import pipeline

def test_config(k, h, sigma_inflate, cand_ratio):
    cfg = Config(cusum_k=k, cusum_h=h, sigma_inflate=sigma_inflate, candidate_ratio_threshold=cand_ratio)
    
    # 1. Test Clean Grid False Alarm Rate
    total_reds = 0
    total_node_days = 0
    for seed in range(1, 6):
        world = simulate(seed=seed, scenarios=[], cfg=cfg)
        res = pipeline.run(world.observed, today_day=40, cfg=cfg)
        n_red = sum(1 for nf in res.nodes.values() if nf.status == "RED")
        total_reds += n_red
        total_node_days += len(res.nodes) * (40 - cfg.cal_days)
    far = (total_reds / total_node_days) * 100
    
    # 2. Test Theft Attribution Rate
    kinds = ["BYPASS", "STEP_TAMPER", "NIGHT_THEFT"]
    total_events = 0
    top1_correct = 0
    top3_correct = 0
    
    for seed in range(1, 6):
        clean = simulate(seed=seed, scenarios=[], cfg=cfg)
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
            target_info[tx_name] = target_meter_id
            
        world = simulate(seed=seed, scenarios=scenarios, cfg=cfg)
        res = pipeline.run(world.observed, today_day=40, cfg=cfg)
        for tx_name, target_meter in target_info.items():
            case = res.cases_by_node.get(tx_name)
            total_events += 1
            if case and case.suspects:
                top_meter = case.suspects[0]["meter"]
                top3 = [s["meter"] for s in case.suspects[:3]]
                if top_meter == target_meter:
                    top1_correct += 1
                if target_meter in top3:
                    top3_correct += 1
                    
    top1_rate = top1_correct / total_events
    top3_rate = top3_correct / total_events
    
    print(f"k={k}, h={h}, sig_inf={sigma_inflate}, cand_ratio={cand_ratio} => FAR: {far:.2f}/100 node-days (reds={total_reds}), Top1: {top1_rate:.1%} ({top1_correct}/{total_events}), Top3: {top3_rate:.1%} ({top3_correct}/{total_events})")

for k in [0.25, 0.30, 0.35]:
    for h in [4.5, 5.0, 5.5]:
        for sig in [1.2, 1.5, 1.8]:
            test_config(k, h, sig, 0.95)
