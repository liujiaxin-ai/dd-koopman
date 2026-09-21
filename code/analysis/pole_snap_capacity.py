"""Mechanism x capacity: does the 58 k model still need the off-grid poles?

Capacity sweep: 57,894-parameter TINY matches the reported 173,190-parameter
model on the official 162 grid (0.1400 vs 0.1385). This leg repeats the
zero-training pole intervention (delta forced to 0.00 bin = the classical
16-point DFT grid) on the TINY checkpoint, so the capacity story and the
mechanism story are tied together: if the small model shows the same reliance on
the learned off-grid placement, the gain is structural rather than capacity.

Sources: results/run_csv/TINY_{stock,snap0}_full162.csv and the reported-model
pair SNAPPOLE_{stock,off000}_full162.csv.
Output: results/analysis/pole_snap_capacity.csv
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
KEY = ["cm", "ds", "ms", "noise_degree"]

PAIRS = [
    ("reported-173k", 173190,
     RUNS / "SNAPPOLE_stock_full162.csv", RUNS / "SNAPPOLE_off000_full162.csv"),
    ("capacity-XS-77k", 77110,
     RUNS / "CAPLONG_CAPACITY_XS_full162.csv", RUNS / "XS_snap0_full162.csv"),
    ("capacity-TINY-58k", 57894,
     RUNS / "TINY169_stock_full162.csv", RUNS / "TINY169_snap0_full162.csv"),
]


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


def cell_mean(frame: pd.DataFrame) -> float:
    cells = frame.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
    return float(cells["nmse"].mean())


def main() -> int:
    rows = []
    for label, params, stock_path, snap_path in PAIRS:
        if not snap_path.exists():
            print(f"[skip] {label}: {snap_path.name} missing")
            continue
        stock, snap = load(stock_path), load(snap_path)
        merged = stock.merge(snap, on=KEY, suffixes=("_stock", "_snap"))
        diff = (merged["nmse_snap"] - merged["nmse_stock"]).to_numpy()
        rng = np.random.default_rng(20260913)
        idx = rng.integers(0, len(diff), size=(5000, len(diff)))
        draws = diff[idx].mean(axis=1)
        worse = int((diff > 0).sum())
        k = max(worse, len(diff) - worse)
        tail = sum(math.comb(len(diff), i)
                   for i in range(k, len(diff) + 1)) / 2 ** len(diff)
        base, forced = cell_mean(stock), cell_mean(snap)
        rows.append({"label": label, "params": params,
                     "nmse_stock": round(base, 4),
                     "nmse_snap0": round(forced, 4),
                     "pct": round((forced / base - 1) * 100, 1),
                     "mean_diff": round(float(diff.mean()), 4),
                     "ci_low": round(float(np.quantile(draws, 0.025)), 4),
                     "ci_high": round(float(np.quantile(draws, 0.975)), 4),
                     "worse_settings": worse, "n": int(len(diff)),
                     "sign_p": float(min(1.0, 2 * tail))})
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "pole_snap_capacity.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'pole_snap_capacity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
