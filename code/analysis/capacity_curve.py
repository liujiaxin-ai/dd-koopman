"""Capacity scaling of the reported operator (same protocol, only widths change).

The reported model is 173,190 parameters. The capacity sweep trains two smaller
and two larger variants of the *same* architecture under the identical loss-off
protocol (240 epochs, patience 25, seed 42, 27 subsets, official 162 grid), so
the resulting curve answers whether the reported accuracy is a capacity artefact
or a property of the mechanism.

Sources:
  results/analysis/capacity_profiles.csv        (params / MACs, CPU profile)
  results/run_csv/CAPNOAUX600_full162.csv       (173k reference)
  results/run_csv/CAPLONG_CAPACITY_{S,M,L}_full162.csv

Output: results/analysis/capacity_curve.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"

POINTS = [
    ("capacity-TINY", RUNS / "CAPLONG_CAPACITY_TINY_full162.csv"),
    ("capacity-XS", RUNS / "CAPLONG_CAPACITY_XS_full162.csv"),
    ("capacity-S", RUNS / "CAPLONG_CAPACITY_S_full162.csv"),
    ("reported-173k", RUNS / "CAPNOAUX600_full162.csv"),
    ("capacity-M", RUNS / "CAPLONG_CAPACITY_M_full162.csv"),
    ("capacity-L", RUNS / "CAPLONG_CAPACITY_L_full162.csv"),
]


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
    profiles = pd.read_csv(ANA / "capacity_profiles.csv")
    key = {"capacity-TINY": "CAPACITY_TINY", "capacity-XS": "CAPACITY_XS",
           "capacity-S": "CAPACITY_S", "capacity-M": "CAPACITY_M",
           "capacity-L": "CAPACITY_L"}
    reported = {"params": 173190, "macs_per_sample": 6514464}
    rows = []
    for label, path in POINTS:
        if not path.exists():
            print(f"[skip] {label}: {path.name} missing")
            continue
        if label in key:
            match = profiles[profiles.label == key[label]].iloc[0]
            params, macs = int(match.params), int(match.macs_per_sample)
        else:
            params, macs = reported["params"], reported["macs_per_sample"]
        rows.append({"label": label, "params": params,
                     "macs_per_sample": macs,
                     "nmse": round(cell_mean(path), 4)})
    if not rows:
        return 1
    out = pd.DataFrame(rows).sort_values("params").reset_index(drop=True)
    ref = out.loc[out.label == "reported-173k"]
    if not ref.empty:
        base = float(ref.nmse.iloc[0])
        out["pct_vs_reported"] = ((out.nmse / base - 1) * 100).round(1)
        out["params_vs_reported"] = (out.params / float(ref.params.iloc[0])) \
            .round(2)
    out.to_csv(ANA / "capacity_curve.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'capacity_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
