"""
Diagnostic script: check theft attribution rate across seeds 1-5 with full verbose output.
"""
import numpy as np
from gridledger.simulator import simulate
from gridledger.scenarios import Scenario
from gridledger import pipeline

kinds = ["BYPASS", "STEP_TAMPER", "NIGHT_THEFT"]

total_events = 0
top1_correct = 0
top3_correct = 0

for seed in range(1, 6):
    clean = simulate(seed=seed, scenarios=[])
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
        target_info[tx_name] = (target_ci, target_meter_id)

    world = simulate(seed=seed, scenarios=scenarios)
    res = pipeline.run(world.observed, today_day=40)

    for tx_name, (target_ci, target_meter) in target_info.items():
        case = res.cases_by_node.get(tx_name)
        total_events += 1
        finding = res.nodes.get(tx_name)
        status = finding.status if finding else "N/A"
        
        if case is None:
            print(f"Seed {seed} {tx_name} ({kinds[tx_list.index(tx_name)]}): NO CASE (status={status})")
            continue
        
        suspects = case.suspects
        if not suspects:
            print(f"Seed {seed} {tx_name} ({kinds[tx_list.index(tx_name)]}): case exists but NO SUSPECTS. cov={case.coverage:.2f}")
            continue

        top_meter = suspects[0]["meter"]
        top3 = [s["meter"] for s in suspects[:3]]
        hit1 = top_meter == target_meter
        hit3 = target_meter in top3
        if hit1:
            top1_correct += 1
        if hit3:
            top3_correct += 1

        kind_name = kinds[tx_list.index(tx_name)]
        
        # Print target's diagnostics
        tx_diags = res.diagnostics_by_tx.get(tx_name, [])
        target_diag = next((d for d in tx_diags if d.meter_idx == target_ci), None)
        
        print(f"\nSeed {seed} {tx_name} ({kind_name}): top1={'YES' if hit1 else 'NO'} top3={'YES' if hit3 else 'NO'}")
        print(f"  Target: {target_meter} (ci={target_ci})")
        print(f"  Top3: {top3}")
        if target_diag:
            print(f"  Target diag: label={target_diag.label}, ratio={target_diag.ratio:.3f}, shortfall={target_diag.shortfall_kwh:.1f}, "
                  f"step_day={target_diag.step_day}, night_share={target_diag.night_shortfall_share:.2f}, "
                  f"is_cand={target_diag.is_candidate}, cleared={target_diag.cleared}")
        else:
            print(f"  Target diag: NOT FOUND in diagnostics")
        
        # Print all candidate suspects with scores
        if suspects:
            print(f"  All suspects (top 5):")
            for s in suspects[:5]:
                flag = " <-- TARGET" if s["meter"] == target_meter else ""
                print(f"    {s['meter']}: theta={s.get('theta',0):.3f}, score={s.get('score',0):.2f}, "
                      f"contrib={s.get('contrib_kwh',0):.1f}, label={s.get('label','?')}, "
                      f"ratio={s.get('ratio',0):.3f}{flag}")
        
        print(f"  Case: cov={case.coverage:.2f}, kwh/day={case.kwh_per_day:.1f}, action={case.action[:50]}")

print(f"\n=== SUMMARY ===")
print(f"Top-1: {top1_correct}/{total_events} = {top1_correct/total_events:.1%}")
print(f"Top-3: {top3_correct}/{total_events} = {top3_correct/total_events:.1%}")
