"""
Compare the trained REINFORCE policies against the HIF baseline.

Reads the RL results stored in results/rl_sweep/sweep_results.json and re-runs the
HIF baseline over several tie-breaking seeds, so that the comparison is not
anchored to a single (possibly lucky or unlucky) baseline run.

Metric: mean of MaxIdle over the last 1,000 steps of a 5,000-step episode,
averaged over the five test starting positions (and over seeds for HIF).

Usage:
    python scripts/compare_to_baseline.py --baseline-seeds 5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rl_core import (TEST_STARTS, build_graph_with_removal, build_grid_graph,  # noqa: E402
                     detect_collapse, run_baseline_hif)

GRAPHS = {"0%": 0.0, "10%": 0.10, "20%": 0.20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "rl_sweep", "sweep_results.json"))
    ap.add_argument("--baseline-seeds", type=int, default=5)
    args = ap.parse_args()

    sweep = json.load(open(args.results))
    seeds = range(args.baseline_seeds)

    print(f"{'graph':>6} {'HIF':>8} {'RL':>8} {'improvement':>12} {'RL wins':>8}")
    for name, frac in GRAPHS.items():
        nb = build_grid_graph() if frac == 0 else build_graph_with_removal(frac, 123)
        base = {}
        for r1, r2 in TEST_STARTS:
            vals = [detect_collapse(run_baseline_hif(nb, seed=s, r1_start=r1, r2_start=r2))[1]
                    for s in seeds]
            base[(r1, r2)] = float(np.mean(vals))
        rows = [x for run in sweep if run["graph"] == name for x in run["rows"]]
        rl = float(np.mean([x["rl_max"] for x in rows]))
        hif = float(np.mean(list(base.values())))
        wins = sum((not x["collapsed"]) and x["rl_max"] < base[tuple(x["start"])] for x in rows)
        print(f"{name:>6} {hif:8.1f} {rl:8.1f} {100 * (hif - rl) / hif:11.1f}% {wins:>4}/{len(rows)}")


if __name__ == "__main__":
    main()
