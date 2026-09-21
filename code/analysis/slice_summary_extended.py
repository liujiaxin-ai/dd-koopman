"""Extended per-model tables for the other two evaluation slices.

Companion to `generalization_432_extended.py` (cm B/E) and to the robustness
rows already in `robustness_table.csv`:

  * A/C/D 648 - the interpolation shift (seen channel models, unseen delay
    spread / speed combinations), paired against the published reference rows;
  * robustness 486 - the three corruption families, paired the same way. The
    pairing key here additionally contains the corruption name because the SNR
    grids differ between `burst`, `packagedrop` and `phase`.

Both tables use the repository convention: cell = (channel model, delay spread,
speed [, corruption]); NMSE averaged over the SNR levels inside a cell, then
over the cells; paired bootstrap (5000, RNG 20260913) plus an exact sign test.

Outputs: results/analysis/generalization_acd_extended.csv
         results/analysis/robustness_extended.csv
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
REF = ROOT / "results" / "reference" / "official_per_setting.csv"
SPEEDS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 42]
SPREADS = [5e-08, 2e-07, 4e-07]

ACD_MODELS = [
    ("ours seed 42 (reported)", 173190, "CAPNOAUX600_gen_acd.csv"),
    ("ours seed 43", 173190, "CAPNOAUXS43_gen_acd.csv"),
    ("ours seed 44", 173190, "HEADNOAUXS44_gen_acd.csv"),
    ("intervention: reported model, delta=0", 173190, "SNAP0_gen_ACD.csv"),
]

ROBUST_MODELS = [
    ("ours seed 42 (reported)", 173190, "CAPNOAUX600_robust486.csv"),
    ("ours seed 43", 173190, "CAPNOAUXS43_robust486.csv"),
    ("ours seed 44", 173190, "HEADNOAUXS44_robust486.csv"),
    ("MambaCSP trained to plateau", 881604, "MAMBA64CONV_C_robust486.csv"),
    ("intervention: reported model, delta=0", 173190, "SNAP0_robust486.csv"),
]


def load(path: Path, key: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    frame = frame.copy()
    frame["nmse"] = frame["nmse_mean"].map(
        lambda text: float(np.mean([
            float(x) for x in
            re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))])))
    frame["ds"] = frame["ds"].astype(float).round(10)
    frame["ms"] = frame["ms"].astype(float)
    frame["noise_degree"] = frame["noise_degree"].astype(float)
    return frame[key + ["nmse"]]


def published(split: str, cms: list[str], key: list[str]) -> pd.DataFrame:
    ref = pd.read_csv(REF)
    ref = ref[(ref.model == "MODEL") & (ref.duplex == "TDD")
              & (ref.split == split)]
    ref = ref[ref.cm.isin(cms)]
    if split == "generalization":
        ref = ref[ref.ds.round(10).isin(SPREADS) & ref.ms.isin(SPEEDS)]
    ref = ref.copy()
    ref["ds"] = ref["ds"].astype(float).round(10)
    ref["ms"] = ref["ms"].astype(float)
    ref["noise_degree"] = ref["snr"].astype(float)
    return ref[key + ["nmse_mean"]].rename(columns={"nmse_mean": "published"})


def build(models, ref, key, cell_key) -> pd.DataFrame:
    rows = []
    for label, params, name in models:
        path = RUNS / name
        if not path.exists():
            print(f"[skip] {label}: {name} missing")
            continue
        merged = load(path, key).merge(ref, on=key)
        cells = merged.groupby(cell_key, as_index=False)[["nmse", "published"]].mean()
        diff = (cells["nmse"] - cells["published"]).to_numpy()
        rng = np.random.default_rng(20260913)
        idx = rng.integers(0, len(diff), size=(5000, len(diff)))
        draws = diff[idx].mean(axis=1)
        enjoy = int((diff < 0).sum())
        k = max(enjoy, len(diff) - enjoy)
        tail = sum(math.comb(len(diff), i)
                   for i in range(k, len(diff) + 1)) / 2 ** len(diff)
        rows.append({"model": label, "params": params,
                     "nmse": round(float(cells["nmse"].mean()), 4),
                     "published": round(float(cells["published"].mean()), 4),
                     "paired_diff": round(float(diff.mean()), 4),
                     "ci_low": round(float(np.quantile(draws, 0.025)), 4),
                     "ci_high": round(float(np.quantile(draws, 0.975)), 4),
                     "cells_better": enjoy, "n_cells": int(len(diff)),
                     "sign_p": float(min(1.0, 2 * tail)),
                     "source": f"results/run_csv/{name}"})
    return pd.DataFrame(rows).sort_values("nmse").reset_index(drop=True)


def main() -> int:
    key_acd = ["cm", "ds", "ms", "noise_degree"]
    acd = build(ACD_MODELS,
                published("generalization", ["A", "C", "D"], key_acd),
                key_acd, ["cm", "ds", "ms"])
    acd.to_csv(ANA / "generalization_acd_extended.csv", index=False)
    print("=== A/C/D 648 (interpolation shift) ===")
    print(acd.to_string(index=False))

    # The paper's robustness numbers are plain means over the 486 settings,
    # which equals the cell mean over (cm, ds, ms) because every such cell
    # contains the same number of corruption x SNR rows. Adding `noise_type`
    # to the cell key would change the scale (0.142 vs 0.162) - do not do it.
    key_rob = ["cm", "ds", "ms", "noise_type", "noise_degree"]
    rob = build(ROBUST_MODELS,
                published("robustness", ["A", "C", "D"], key_rob),
                key_rob, ["cm", "ds", "ms"])
    rob.to_csv(ANA / "robustness_extended.csv", index=False)
    print("\n=== robustness 486 ===")
    print(rob.to_string(index=False))
    print(f"\nwrote {ANA / 'generalization_acd_extended.csv'} and "
          f"{ANA / 'robustness_extended.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
