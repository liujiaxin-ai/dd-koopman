"""Data-efficiency ladder of the reported operator (official 162 grid).

The reported model trains on 24,300 packed training windows. This ladder repeats
the identical loss-off protocol (240 epochs, patience 25, seed 42) while keeping
a deterministic 25 / 50 / 75 % subset of those windows, then scores each
checkpoint on the official 162-setting grid and compares it with the full-data
reference (CAPNOAUX600, 0.1385).

Output: results/analysis/data_frac_curve.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"

# Each point is scored with its best-validation checkpoint, matching the
# protocol used for the capacity curve. For the 75 % run the queue's automatic
# scorer fell back to `last.ckpt`, so the best-checkpoint run is preferred
# explicitly (0.1433 vs 0.1436 on the 162 grid - reported for transparency).
POINTS = [(1.00, RUNS / "CAPNOAUX600_full162.csv"),
          (0.25, RUNS / "CAPLONG_DATAFRAC25_full162.csv"),
          (0.50, RUNS / "CAPLONG_DATAFRAC50_full162.csv"),
          (0.75, RUNS / "CAPLONG_DATAFRAC75BEST_full162.csv")]


def cell_mean(path: Path) -> float:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    frame = frame[frame["noise_type"] == "vanilla"].copy()
    frame["nmse"] = frame["nmse_mean"].map(
        lambda text: float(np.mean([
            float(x) for x in
            re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))])))
    cells = frame.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
    return float(cells["nmse"].mean())


def main() -> int:
    rows = []
    for frac, path in POINTS:
        if not path.exists():
            print(f"[skip] {frac:.0%}: {path.name} missing")
            continue
        rows.append({"train_frac": frac,
                     "windows": int(round(24300 * frac)),
                     "nmse": round(cell_mean(path), 4)})
    if not rows:
        return 1
    out = pd.DataFrame(rows).sort_values("train_frac").reset_index(drop=True)
    base = float(out.loc[out.train_frac == 1.0, "nmse"].iloc[0])
    out["pct_vs_full"] = ((out.nmse / base - 1) * 100).round(1)
    out.to_csv(ANA / "data_frac_curve.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'data_frac_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
