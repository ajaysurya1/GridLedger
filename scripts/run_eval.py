"""Run the held-out three-system benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gridledger.config import CFG
from gridledger.evaluate import evaluate
from gridledger.pipeline import run as run_pipeline
from gridledger.scenarios import random_scenarios
from gridledger.simulator import simulate


DEV_SEEDS = tuple(range(1, 6))
TEST_SEEDS = tuple(range(101, 111))


def _summarize(rows: list[dict]) -> dict:
    systems = {}
    metrics = [key for key, value in rows[0]["systems"]["A"].items() if isinstance(value, (int, float))]
    for name in ("A", "B", "C"):
        summary = {}
        for metric in metrics:
            values = np.asarray([row["systems"][name][metric] for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            summary[metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        all_kinds = sorted(rows[0]["systems"][name]["per_kind_recall"])
        summary["per_kind_recall"] = {
            kind: {
                "mean": float(np.mean([row["systems"][name]["per_kind_recall"][kind] for row in rows])),
                "std": float(np.std([row["systems"][name]["per_kind_recall"][kind] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for kind in all_kinds
        }
        systems[name] = summary
    return systems


def run(seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        topology = simulate(seed=seed, scenarios=[])
        rng = np.random.default_rng([seed, 0x47524944])
        scenarios = random_scenarios(rng, topology.observed, CFG.n_days)
        world = simulate(seed=seed, scenarios=scenarios)
        result = run_pipeline(world.observed, today_day=40)
        rows.append({"seed": seed, "systems": evaluate(world, result, today_day=40)})
        print(f"completed seed {seed}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", choices=("dev", "test", "fresh"), required=True)
    parser.add_argument("--count", type=int, help="Number of seeds for DEV or fresh runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.seeds == "dev":
        count = args.count if args.count is not None else len(DEV_SEEDS)
        if count < 1 or count > len(DEV_SEEDS):
            parser.error(f"DEV count must be between 1 and {len(DEV_SEEDS)}")
        seeds = list(DEV_SEEDS[:count])
        default_output = ROOT / "data" / "eval_dev_results.json"
    elif args.seeds == "test":
        seeds = list(TEST_SEEDS)
        default_output = ROOT / "data" / "eval_results.json"
    else:
        count = args.count if args.count is not None else 3
        if count < 1:
            parser.error("--count must be positive")
        seeds = np.random.default_rng().choice(np.arange(1000, 2**31 - 1), size=count, replace=False).tolist()
        default_output = None

    started = time.perf_counter()
    rows = run(seeds)
    elapsed = time.perf_counter() - started
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    payload = {
        "metadata": {
            "seeds": seeds,
            "seed_set": args.seeds,
            "git_sha": git_sha,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": elapsed,
            "today_day": 40,
            "systems": {"A": "Customer-only", "B": "Fixed 8% weekly gap", "C": "GridLedger"},
        },
        "summary": _summarize(rows),
        "per_seed": rows,
    }
    output = args.output or default_output
    if output is not None:
        output = output if output.is_absolute() else ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    else:
        print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()