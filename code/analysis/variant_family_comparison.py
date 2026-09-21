"""Three-seed comparison of the reported 173 k model against the 77 k variant.

Answers the question the capacity sweep raised: should the paper report the
smaller 77 k configuration instead? Both families have three seeds; every cell
is the mean over the six SNR levels and then averaged over the family's three
seeds, exactly like the main table. The paired comparison uses 27 (regular,
robustness) resp. 72 (cm B/E) cells with the repository bootstrap RNG.

Slices:
  regular       results/run_csv/{CAPNOAUX600,CAPNOAUXS43,HEADNOAUXS44}_full162.csv
                 vs {CAPLONG_CAPACITY_XS,CAPLONG_XS_S43,CAPLONG_XS_S44}_full162.csv
  B/E shift     {CAPNOAUX600,CAPNOAUXS43,CAPNOAUXS44}_gen_gen.csv
                 vs {XS42,XS_S43,XS_S44}_gen_BE.csv
  robustness    {CAPNOAUX600,CAPNOAUXS43,HEADNOAUXS44}_robust486.csv
                 vs {XS42,XS_S43,XS_S44}_robust486.csv

Output: results/analysis/variant_family_comparison.csv
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
KEY = ["cm", "ds", "ms"]

SLICES = [
    ("regular 162", True,
     ["CAPNOAUX600_full162.csv", "CAPNOAUXS43_full162.csv",
      "HEADNOAUXS44_full162.csv"],
     ["CAPLONG_CAPACITY_XS_full162.csv", "CAPLONG_XS_S43_full162.csv",
      "CAPLONG_XS_S44_full162.csv"]),
    ("B/E 432", True,
     ["CAPNOAUX600_gen_gen.csv", "CAPNOAUXS43_gen_gen.csv",
      "CAPNOAUXS44_gen_gen.csv"],
     ["XS42_gen_BE.csv", "XS_S43_gen_BE.csv", "XS_S44_gen_BE.csv"]),
    ("robustness 486", False,
     ["CAPNOAUX600_robust486.csv", "CAPNOAUXS43_robust486.csv",
      "HEADNOAUXS44_robust486.csv"],
     ["XS42_robust486.csv", "XS_S43_robust486.csv", "XS_S44_robust486.csv"]),
]


def family(names: list[str], vanilla: bool) -> pd.DataFrame:
    cells = []
    for name in names:
        frame = pd.read_csv(RUNS / name)
        if "scenario" in frame:
            frame = frame[frame["scenario"] == "TDD"]
        if vanilla and "noise_type" in frame:
            frame = frame[frame["noise_type"] == "vanilla"]
        frame = frame.copy()
        frame["nmse"] = frame["nmse_mean"].map(
            lambda text: float(np.mean([
                float(x) for x in
                re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))])))
        cells.append(frame.groupby(KEY, as_index=False)["nmse"].mean())
    stacked = pd.concat(cells)
    return stacked.groupby(KEY, as_index=False)["nmse"].mean()


def main() -> int:
    rows = []
    for name, vanilla, ours_names, xs_names in SLICES:
        ours = family(ours_names, vanilla).rename(columns={"nmse": "nmse_173k"})
        xs = family(xs_names, vanilla).rename(columns={"nmse": "nmse_77k"})
        merged = xs.merge(ours, on=KEY)
        diff = (merged["nmse_77k"] - merged["nmse_173k"]).to_numpy()
        rng = np.random.default_rng(20260913)
        idx = rng.integers(0, len(diff), size=(5000, len(diff)))
        draws = diff[idx].mean(axis=1)
        better = int((diff < 0).sum())
        k = max(better, len(diff) - better)
        tail = sum(math.comb(len(diff), i)
                   for i in range(k, len(diff) + 1)) / 2 ** len(diff)
        rows.append({"slice": name,
                     "nmse_173k_mean": round(float(merged["nmse_173k"].mean()), 4),
                     "nmse_77k_mean": round(float(merged["nmse_77k"].mean()), 4),
                     "paired_diff_77k_minus_173k": round(float(diff.mean()), 4),
                     "ci_low": round(float(np.quantile(draws, 0.025)), 4),
                     "ci_high": round(float(np.quantile(draws, 0.975)), 4),
                     "cells_77k_better": better,
                     "n_cells": int(len(diff)),
                     "sign_p": float(min(1.0, 2 * tail))})
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "variant_family_comparison.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'variant_family_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
