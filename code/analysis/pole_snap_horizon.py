"""Where in the prediction horizon does the off-grid pole freedom pay off?

Same pair of runs as `pole_intervention_analysis.py` (seed-42 checkpoint,
regular 162 grid, delta forced to 0.00 bin vs the free fit) but resolved per
horizon and per SNR band instead of averaged. Regenerates the numbers quoted in
the evidence report section 16.2.

Output: results/analysis/pole_snap_horizon.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"
PAIR = [("stock", RUNS / "SNAPPOLE_stock_full162.csv"),
        ("snap0", RUNS / "SNAPPOLE_off000_full162.csv")]
KEY = ["cm", "ds", "ms", "noise_degree"]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    frame = frame[frame["noise_type"] == "vanilla"].copy()
    vectors = frame["nmse_mean"].map(
        lambda text: [float(x) for x in
                      re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))])
    for index in range(4):
        frame[f"h{index + 1}"] = vectors.map(
            lambda values, i=index: values[i] if len(values) > i else np.nan)
    return frame[KEY + ["h1", "h2", "h3", "h4"]]


def main() -> int:
    frames = {tag: load(path) for tag, path in PAIR}
    merged = frames["stock"].merge(
        frames["snap0"], on=KEY, suffixes=("_stock", "_snap0"))
    rows = []
    for horizon in ("h1", "h2", "h3", "h4"):
        base = float(merged[f"{horizon}_stock"].mean())
        snap = float(merged[f"{horizon}_snap0"].mean())
        rows.append({"group": "horizon", "value": horizon,
                     "nmse_stock": round(base, 4),
                     "nmse_snap0": round(snap, 4),
                     "pct": round((snap / base - 1) * 100, 1),
                     "n": int(len(merged))})
    bands = pd.cut(merged["noise_degree"], [-1, 4.9, 14.9, 100],
                   labels=["0 dB", "5-10 dB", "15-25 dB"])
    for band, part in merged.groupby(bands, observed=True):
        base = float(part["h4_stock"].mean())
        snap = float(part["h4_snap0"].mean())
        rows.append({"group": "snr-band (h4)", "value": str(band),
                     "nmse_stock": round(base, 4),
                     "nmse_snap0": round(snap, 4),
                     "pct": round((snap / base - 1) * 100, 1),
                     "n": int(len(part))})
    for cm, part in merged.groupby("cm"):
        base = float(part["h4_stock"].mean())
        snap = float(part["h4_snap0"].mean())
        rows.append({"group": "cm (h4)", "value": cm,
                     "nmse_stock": round(base, 4),
                     "nmse_snap0": round(snap, 4),
                     "pct": round((snap / base - 1) * 100, 1),
                     "n": int(len(part))})
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "pole_snap_horizon.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'pole_snap_horizon.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
