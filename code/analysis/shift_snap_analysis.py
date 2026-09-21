"""Does the grid-snap intervention cost the same under distribution shift?

The mechanism claim (`fig_pole_gain`) is measured on the regular 162-setting
grid. The generalisation claim lives on two other slices. This script ties the
two lines together with one object: the same frozen checkpoint, the same
zero-training intervention (delta := 0, i.e. the classical 16-point DFT grid),
evaluated on

  * the unseen-channel-model slice  cm B/E   (432 settings),
  * the interpolation slice         cm A/C/D at unseen delay spread / speed
    (648 settings),

and paired against the *stock* run of the same checkpoint on the same slice.

Sources (all TDD, vanilla noise):
  results/run_csv/CAPNOAUX600_gen_gen.csv   (stock, cm B/E)
  results/run_csv/SNAP0_gen_BE.csv          (delta = 0, cm B/E)
  results/run_csv/CAPNOAUX600_gen_acd.csv   (stock, cm A/C/D shift)
  results/run_csv/SNAP0_gen_ACD.csv         (delta = 0, cm A/C/D shift)

Output: results/analysis/shift_snap.csv
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

SLICES = [
    ("unseen cm B/E (432)", RUNS / "CAPNOAUX600_gen_gen.csv",
     RUNS / "SNAP0_gen_BE.csv"),
    ("interp. A/C/D shift (648)", RUNS / "CAPNOAUX600_gen_acd.csv",
     RUNS / "SNAP0_gen_ACD.csv"),
    ("robustness-486", RUNS / "CAPNOAUX600_robust486.csv",
     RUNS / "SNAP0_robust486.csv"),
]

# the robustness split *is* the corruption axis, everything else is vanilla
NOISY = {"robustness-486"}


def _numbers(text: str) -> float:
    return float(np.mean([
        float(x) for x in
        re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]))


def per_setting(path: Path, slice_name: str = "") -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    if "noise_type" in frame and slice_name not in NOISY:
        frame = frame[frame["noise_type"] == "vanilla"]
    frame["nmse"] = frame["nmse_mean"].map(_numbers)
    # normalise every join key to float: the harness writes `ms` and
    # `noise_degree` as int in some tables and float in others, and a
    # mixed int/float merge silently keeps only the rows that happen to match
    frame["ds"] = frame["ds"].astype(float).round(10)
    frame["ms"] = frame["ms"].astype(float)
    frame["noise_degree"] = frame["noise_degree"].astype(float)
    keys = ["cm", "ds", "ms", "noise_degree"]
    if slice_name in NOISY:
        keys.insert(3, "noise_type")
    return frame[keys + ["nmse"]]


def _key(slice_name: str) -> list[str]:
    keys = ["cm", "ds", "ms", "noise_degree"]
    if slice_name in NOISY:
        keys.insert(3, "noise_type")
    return keys


def cell_mean(frame: pd.DataFrame) -> float:
    cells = frame.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
    return float(cells["nmse"].mean())


def paired(base: pd.DataFrame, other: pd.DataFrame, keys: list[str]) -> dict:
    merged = base.merge(other, on=keys, suffixes=("_a", "_b"))
    diff = (merged["nmse_b"] - merged["nmse_a"]).to_numpy()
    rng = np.random.default_rng(20260913)
    idx = rng.integers(0, len(diff), size=(5000, len(diff)))
    draws = diff[idx].mean(axis=1)
    n = len(diff)
    worse = int((diff > 0).sum())
    k = max(worse, n - worse)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n_paired": n, "mean_diff": round(float(diff.mean()), 4),
            "ci_low": round(float(np.quantile(draws, 0.025)), 4),
            "ci_high": round(float(np.quantile(draws, 0.975)), 4),
            "worse_settings": worse,
            "sign_p": round(float(min(1.0, 2 * tail)), 8)}


def by_key(base: pd.DataFrame, other: pd.DataFrame, keys: list[str],
           key: str) -> dict:
    merged = base.merge(other, on=keys, suffixes=("_a", "_b"))
    merged["delta"] = merged["nmse_b"] - merged["nmse_a"]
    out = {}
    for value, part in merged.groupby(key):
        out[str(value)] = round(float(part["delta"].mean()), 4)
    return out


def main() -> int:
    rows = []
    for name, stock_path, snap_path in SLICES:
        if not snap_path.exists():
            print(f"[skip] {snap_path.name} not present yet")
            continue
        stock = per_setting(stock_path, name)
        snap = per_setting(snap_path, name)
        keys = _key(name)
        stats = paired(stock, snap, keys)
        rows.append({"slice": name,
                     "nmse_stock": round(cell_mean(stock), 4),
                     "nmse_snap0": round(cell_mean(snap), 4),
                     "pct": round((cell_mean(snap) / cell_mean(stock) - 1) * 100, 1),
                     **stats})
        print(f"{name}: stock {cell_mean(stock):.4f} -> snap {cell_mean(snap):.4f} "
              f"({rows[-1]['pct']:+.1f}%) worse on "
              f"{stats['worse_settings']}/{stats['n_paired']} "
              f"(sign p={stats['sign_p']:.1e})")
        for key in ("cm", "ds"):
            print(f"   by {key}: {by_key(stock, snap, keys, key)}")
    if not rows:
        return 1
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "shift_snap.csv", index=False)
    print(f"\nwrote {ANA / 'shift_snap.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
