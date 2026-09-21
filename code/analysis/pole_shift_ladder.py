"""Resolution ladder of the learned pole offsets on the interpolation-shift slice.

Mirrors `fig_pole_gain` panel b, but on the 648-setting A/C/D slice: the learned
sub-bin offset is rounded to r = 0.125 / 0.25 / 0.50 bin (keeping its per-mode,
per-sample structure) and paired per setting against the stock run of the same
checkpoint. Together with `shift_snap.csv` (delta forced to a constant) this
answers whether the precision requirement is slice-independent.

Output: results/analysis/pole_shift_ladder.csv
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
STOCK = RUNS / "CAPNOAUX600_gen_acd.csv"
LADDER = [(0.125, RUNS / "POLEQ0125_gen_ACD.csv"),
          (0.250, RUNS / "POLEQ0250_gen_ACD.csv"),
          (0.500, RUNS / "POLEQ0500_gen_ACD.csv")]
KEY = ["cm", "ds", "ms", "noise_degree"]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    if "noise_type" in frame:
        frame = frame[frame["noise_type"] == "vanilla"]
    frame = frame.copy()
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
    stock = load(STOCK)
    base = cell_mean(stock)
    rows = [{"resolution": "free fit", "nmse": round(base, 4), "pct": 0.0,
             "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
             "worse_settings": 0, "n": len(stock), "sign_p": None}]
    for resolution, path in LADDER:
        if not path.exists():
            print(f"[skip] r={resolution}: {path.name} missing")
            continue
        other = load(path)
        merged = stock.merge(other, on=KEY, suffixes=("_stock", "_r"))
        diff = (merged["nmse_r"] - merged["nmse_stock"]).to_numpy()
        rng = np.random.default_rng(20260913)
        idx = rng.integers(0, len(diff), size=(5000, len(diff)))
        draws = diff[idx].mean(axis=1)
        worse = int((diff > 0).sum())
        k = max(worse, len(diff) - worse)
        tail = sum(math.comb(len(diff), i)
                   for i in range(k, len(diff) + 1)) / 2 ** len(diff)
        nmse = cell_mean(other)
        rows.append({"resolution": resolution, "nmse": round(nmse, 4),
                     "pct": round((nmse / base - 1) * 100, 1),
                     "mean_diff": round(float(diff.mean()), 4),
                     "ci_low": round(float(np.quantile(draws, 0.025)), 4),
                     "ci_high": round(float(np.quantile(draws, 0.975)), 4),
                     "worse_settings": worse, "n": int(len(diff)),
                     "sign_p": float(min(1.0, 2 * tail))})
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "pole_shift_ladder.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'pole_shift_ladder.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
