"""
Multi-seed benchmark runner.

Runs training across multiple seeds, collects test metrics, and reports
mean +/- std — the standard for PDE operator learning papers.

Usage:
  python run_benchmark.py --config configs/burgers1d.yaml --seeds 3
  python run_benchmark.py --config configs/darcy2d.yaml --seeds 5
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def run_single_seed(config_path, seed, results_dir):
    """Run a single training + eval with a given seed via subprocess."""
    env = os.environ.copy()
    env["PNO_SEED"] = str(seed)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_pde.py"),
        "--config", config_path,
    ]

    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Multi-seed PDE benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to run")
    parser.add_argument("--start-seed", type=int, default=42)
    args = parser.parse_args()

    seeds = [args.start_seed + i for i in range(args.seeds)]
    results_dir = Path("benchmark_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    return_codes = []
    for seed in seeds:
        rc = run_single_seed(args.config, seed, results_dir)
        return_codes.append(rc)
        if rc != 0:
            print(f"WARNING: Seed {seed} exited with code {rc}")

    # Summary
    summary = {
        "config": args.config,
        "seeds": seeds,
        "return_codes": return_codes,
        "n_successful": sum(1 for rc in return_codes if rc == 0),
    }

    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete: {summary['n_successful']}/{len(seeds)} successful")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
