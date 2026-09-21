"""Turn the pole-offset intervention runs into a dose-response curve.

Inputs are the five harness tables produced by code/analysis/pole_offset_sweep.py
(stock model, then the pole sub-bin offset forced to 0.00 / 0.15 / 0.25 / 0.50
bin). The metric convention is the shared one: `nmse_mean` is the per-horizon
vector, so it is reduced with a plain mean over the four horizons, then averaged
over the six SNR levels inside each cell and over the 27 cells.

Output: results/analysis/pole_offset_curve.csv  (+ console table)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"
TAGS = [("stock", None), ("off000", 0.00), ("off015", 0.15),
        ("off025", 0.25), ("off050", 0.50)]


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


def per_setting(path: Path) -> pd.DataFrame:
    """Per (cm, ds, ms, snr) NMSE so two runs can be paired exactly."""
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    frame = frame[frame["noise_type"] == "vanilla"].copy()
    frame["nmse"] = frame["nmse_mean"].map(
        lambda text: float(np.mean([
            float(x) for x in
            re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))])))
    return frame[["cm", "ds", "ms", "noise_degree", "nmse"]]


def paired_stats(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    merged = a.merge(b, on=["cm", "ds", "ms", "noise_degree"],
                     suffixes=("_a", "_b"))
    diff = (merged["nmse_b"] - merged["nmse_a"]).to_numpy()
    rng = np.random.default_rng(20260913)
    idx = rng.integers(0, len(diff), size=(5000, len(diff)))
    means = diff[idx].mean(axis=1)
    wins = int((diff > 0).sum())          # b worse than a
    n = len(diff)
    import math
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n": n, "mean_diff": round(float(diff.mean()), 4),
            "ci_low": round(float(np.quantile(means, 0.025)), 4),
            "ci_high": round(float(np.quantile(means, 0.975)), 4),
            "worse_cells": wins,
            "sign_p": round(float(min(1.0, 2 * tail)), 6)}


def main() -> int:
    rows = []
    for tag, offset in TAGS:
        path = RUNS / f"SNAPPOLE_{tag}_full162.csv"
        if not path.exists():
            print(f"[skip] {path.name} not present yet")
            continue
        rows.append({"tag": tag, "forced_offset_bin": offset,
                     "nmse": round(cell_mean(path), 4)})
    if not rows:
        return 1
    out = pd.DataFrame(rows)
    stock = float(out.loc[out.tag == "stock", "nmse"].iloc[0])
    out["delta_vs_stock"] = (out["nmse"] - stock).round(4)
    out["pct_vs_stock"] = ((out["nmse"] / stock - 1) * 100).round(1)
    out.to_csv(ANA / "pole_offset_curve.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'pole_offset_curve.csv'}")

    stock_csv = RUNS / "SNAPPOLE_stock_full162.csv"
    if stock_csv.exists():
        base = per_setting(stock_csv)
        print("\npaired comparison against the stock (free-fit) model, per setting:")
        for tag, _ in TAGS[1:]:
            path = RUNS / f"SNAPPOLE_{tag}_full162.csv"
            if not path.exists():
                continue
            stats = paired_stats(base, per_setting(path))
            print(f"  {tag}: n={stats['n']} mean +{stats['mean_diff']:.4f} "
                  f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] "
                  f"worse on {stats['worse_cells']}/{stats['n']} settings "
                  f"(sign p={stats['sign_p']:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
