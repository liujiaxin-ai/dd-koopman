"""Summarise a harness result CSV: overall mean NMSE plus a breakdown.

The harness writes one row per setting with the four per-horizon values inside a
printed numpy array, so the mean over horizons has to be recovered by parsing.
Robustness runs vary both the noise type and the noise degree, and those are the
contrasts that matter for the paper's robustness claim - a model that is fine on
phase noise and broken on burst noise should not be reported as "robust".

Usage:
    python summarize_csv.py <result.csv> [more.csv ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def nmse(text: object) -> float:
    values = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]
    return float(np.mean(values)) if values else float("nan")

def main() -> int:
    for raw in sys.argv[1:]:
        path = Path(raw)
        frame = pd.read_csv(path)
        frame["nmse"] = frame["nmse_mean"].map(nmse)
        print(f"=== {path.name}")
        print(f"    rows={len(frame)}  overall mean NMSE={frame['nmse'].mean():.4f}")
        if "noise_type" in frame:
            for noise, group in frame.groupby("noise_type"):
                degrees = group["noise_degree"].nunique() if "noise_degree" in group else 0
                print(f"    {noise:>12}: n={len(group):>4}  mean={group['nmse'].mean():.4f}"
                      f"  degrees={degrees}")
        if "is_gen" in frame and frame["is_gen"].any():
            for flag, group in frame.groupby("is_gen"):
                print(f"    is_gen={flag}: n={len(group)} mean={group['nmse'].mean():.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
