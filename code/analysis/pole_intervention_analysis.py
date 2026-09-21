"""Unified table for the zero-training pole interventions.

Three intervention families all reuse the *same* trained checkpoint and only
change how the learned sub-bin phase offset delta is exposed to the model:

* ``offset``  - delta forced to a single constant for every mode / sample
                (0.00 / 0.15 / 0.25 / 0.50 bin; 0.00 == the classical 16-point
                DFT grid);
* ``quant``   - delta kept per mode / per sample but rounded to a grid of
                resolution r (0.500 / 0.250 / 0.125 bin);
* ``topk``    - only the k most important modes keep their learned delta, the
                remaining 16-k are snapped to the DFT grid.

Convention (shared with the other analyses): ``nmse_mean`` is the per-horizon
vector -> plain mean over the four horizons, then averaged over the six SNR
levels inside each (cm, ds, ms) cell and finally over the 27 cells.

Output: results/analysis/pole_intervention_curve.csv
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

STOCK = RUNS / "SNAPPOLE_stock_full162.csv"

# (family, tag, x-value, run file)
ENTRIES = [
    ("offset", "off000", 0.00, RUNS / "SNAPPOLE_off000_full162.csv"),
    ("offset", "off015", 0.15, RUNS / "SNAPPOLE_off015_full162.csv"),
    ("offset", "off025", 0.25, RUNS / "SNAPPOLE_off025_full162.csv"),
    ("offset", "off050", 0.50, RUNS / "SNAPPOLE_off050_full162.csv"),
    ("quant", "quant0500", 0.500, RUNS / "POLEQ_quant0500_full162.csv"),
    ("quant", "quant0250", 0.250, RUNS / "POLEQ_quant0250_full162.csv"),
    ("quant", "quant0125", 0.125, RUNS / "POLEQ_quant0125_full162.csv"),
    ("topk", "top8", 8, RUNS / "POLETOPtop8_full162.csv"),
    ("topk", "top4", 4, RUNS / "POLETOPtop4_full162.csv"),
]


def _numbers(text: str) -> float:
    return float(np.mean([
        float(x) for x in
        re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]))


def per_setting(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    frame = frame[frame["noise_type"] == "vanilla"].copy()
    frame["nmse"] = frame["nmse_mean"].map(_numbers)
    return frame[["cm", "ds", "ms", "noise_degree", "nmse"]]


def cell_mean(path: Path) -> float:
    frame = per_setting(path)
    cells = frame.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
    return float(cells["nmse"].mean())


def paired_stats(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    merged = base.merge(other, on=["cm", "ds", "ms", "noise_degree"],
                        suffixes=("_a", "_b"))
    diff = (merged["nmse_b"] - merged["nmse_a"]).to_numpy()
    rng = np.random.default_rng(20260913)
    idx = rng.integers(0, len(diff), size=(5000, len(diff)))
    means = diff[idx].mean(axis=1)
    wins = int((diff > 0).sum())        # b worse than a
    n = len(diff)
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n": n,
            "mean_diff": round(float(diff.mean()), 4),
            "ci_low": round(float(np.quantile(means, 0.025)), 4),
            "ci_high": round(float(np.quantile(means, 0.975)), 4),
            "worse_settings": wins,
            "sign_p": round(float(min(1.0, 2 * tail)), 8)}


def main() -> int:
    base = per_setting(STOCK)
    stock_nmse = cell_mean(STOCK)
    rows = [{"family": "stock", "tag": "stock", "x": None,
             "nmse": round(stock_nmse, 4), "pct_vs_stock": 0.0,
             "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
             "worse_settings": 0, "n": len(base), "sign_p": None}]
    for family, tag, x, path in ENTRIES:
        if not path.exists():
            print(f"[skip] {path.name} missing")
            continue
        other = per_setting(path)
        stats = paired_stats(base, other)
        nmse = cell_mean(path)
        rows.append({"family": family, "tag": tag, "x": x,
                     "nmse": round(nmse, 4),
                     "pct_vs_stock": round((nmse / stock_nmse - 1) * 100, 1),
                     **stats})
    out = pd.DataFrame(rows)
    path = ANA / "pole_intervention_curve.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
