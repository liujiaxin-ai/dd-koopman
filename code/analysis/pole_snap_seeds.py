"""Seed robustness of the zero-training grid-snap intervention.

The intervention (force the pole sub-bin offset to 0.00 bin = the classical
16-point DFT grid, same weights, no training) is measured on all three reported
checkpoints, each paired against its own stock run on the official 162-setting
grid. Regenerates the table in evidence-report section 16.3.

Output: results/analysis/pole_snap_seeds.csv
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"

SEEDS = [
    (42, "SNAPPOLE_stock_full162.csv", "SNAPPOLE_off000_full162.csv"),
    (43, "CAPNOAUXS43_full162.csv", "SNAP0_s43_full162.csv"),
    (44, "HEADNOAUXS44_full162.csv", "SNAP0_s44_full162.csv"),
]
KEY = ["cm", "ds", "ms", "noise_degree"]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    frame = frame[frame["noise_type"] == "vanilla"].copy()
    frame["nmse"] = frame["nmse_mean"].map(
        lambda text: float(np.mean([
            float(x) for x in
            re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))])))
    frame["ds"] = frame["ds"].astype(float).round(10)
    frame["ms"] = frame["ms"].astype(float)
    frame["noise_degree"] = frame["noise_degree"].astype(float)
    return frame[KEY + ["nmse"]]


def main() -> int:
    rows = []
    for seed, stock_name, snap_name in SEEDS:
        stock_path, snap_path = RUNS / stock_name, RUNS / snap_name
        if not snap_path.exists():
            print(f"[skip] seed {seed}: {snap_name} missing")
            continue
        stock, snap = load(stock_path), load(snap_path)
        merged = stock.merge(snap, on=KEY, suffixes=("_stock", "_snap"))
        diff = (merged["nmse_snap"] - merged["nmse_stock"]).to_numpy()
        base = float(merged["nmse_stock"].mean())
        forced = float(merged["nmse_snap"].mean())
        rng = np.random.default_rng(20260913)
        idx = rng.integers(0, len(diff), size=(5000, len(diff)))
        draws = diff[idx].mean(axis=1)
        worse = int((diff > 0).sum())
        k = max(worse, len(diff) - worse)
        tail = sum(math.comb(len(diff), i) for i in range(k, len(diff) + 1)) \
            / 2 ** len(diff)
        rows.append({"seed": seed,
                     "nmse_stock": round(base, 4),
                     "nmse_snap0": round(forced, 4),
                     "pct": round((forced / base - 1) * 100, 1),
                     "mean_diff": round(float(diff.mean()), 4),
                     "ci_low": round(float(np.quantile(draws, 0.025)), 4),
                     "ci_high": round(float(np.quantile(draws, 0.975)), 4),
                     "worse_settings": worse,
                     "n": int(len(diff)),
                     "sign_p": float(min(1.0, 2 * tail))})
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "pole_snap_seeds.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'pole_snap_seeds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
