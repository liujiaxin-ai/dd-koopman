"""Consolidated cm B/E shift-slice table for every model we evaluated.

`generalization_432_summary.py` covers the six configurations that existed when
the slice was first studied. Tonight's runs added a fairly-trained MambaCSP, the
77 k capacity variant (three seeds), the data-efficiency ladder and the two
zero-training interventions, so this script rebuilds one table from the raw
per-setting CSVs with a single, identical protocol:

  * cell = (channel model, delay spread, speed); 72 cells / 432 settings
  * NMSE averaged over the six SNR levels inside a cell, then over the cells
  * paired difference against the published reference rows on the same settings
  * 5000-resample bootstrap CI + exact sign test (RNG 20260913)

Output: results/analysis/generalization_432_extended.csv
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
KEY = ["cm", "ds", "ms", "noise_degree"]

# label -> (params, run csv, family)
MODELS = [
    ("ours seed 42 (reported)", 173190, "CAPNOAUX600_gen_gen.csv"),
    ("ours seed 43", 173190, "CAPNOAUXS43_gen_gen.csv"),
    ("ours seed 44", 173190, "CAPNOAUXS44_gen_gen.csv"),
    ("77k variant seed 42", 77110, "XS42_gen_BE.csv"),
    ("77k variant seed 43", 77110, "XS_S43_gen_BE.csv"),
    ("77k variant seed 44", 77110, "XS_S44_gen_BE.csv"),
    ("ours 25% data", 173190, "DATAFRAC25_gen_BE.csv"),
    ("ours 50% data", 173190, "DATAFRAC50_gen_BE.csv"),
    ("ours 75% data", 173190, "DATAFRAC75_gen_BE.csv"),
    # Grid-constrained retrain: the off-grid phase offset delta_m + eps_m(x) is
    # pinned to zero and the model is retrained from scratch, everything else
    # (radius conditioning, denoiser, delay decomposition, warm start) unchanged.
    # Answers whether the shift advantage survives without the off-grid degree of
    # freedom, i.e. whether that term is structurally necessary.
    ("grid-constrained retrain seed 42", 173190, "GRIDFIX_S42_gen_gen.csv"),
    ("grid-constrained retrain seed 43", 173190, "GRIDFIX_S43_gen_gen.csv"),
    ("grid-constrained retrain seed 44", 173190, "GRIDFIX_S44_gen_gen.csv"),
    ("MambaCSP trained to plateau", 881604, "MAMBA64CONV_C_gen.csv"),
    ("MambaCSP port (8 epochs)", 881604, "MAMBA64_432_gen.csv"),
    ("published-architecture 91k", 91410, "P3CONV_gen_gen.csv"),
    ("capacity-matched transformer", 149560, "TRANSMATCHED_432_gen.csv"),
    ("intervention: reported model, delta=0", 173190, "SNAP0_gen_BE.csv"),
    ("intervention: 77k variant, delta=0", 77110, "XS_snap0_gen_BE.csv"),
]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    frame = frame[frame["noise_type"] == "vanilla"].copy()
    frame["nmse"] = frame["nmse_mean"].map(
        lambda text: float(np.mean([
            float(x) for x in
            re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))])))
    frame["ds"] = frame["ds"].astype(float).round(10)
    frame["ms"] = frame["ms"].astype(float)
    frame["noise_degree"] = frame["noise_degree"].astype(float)
    return frame[KEY + ["nmse"]]


def published() -> pd.DataFrame:
    ref = pd.read_csv(REF)
    ref = ref[(ref.model == "MODEL") & (ref.duplex == "TDD")
              & (ref.split == "generalization")]
    ref = ref[ref.cm.isin(["B", "E"])]
    ref = ref[ref.ds.round(10).isin([5e-08, 2e-07, 4e-07])]
    ref = ref[ref.ms.isin([3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 42])]
    ref = ref.copy()
    ref["ds"] = ref["ds"].astype(float).round(10)
    ref["ms"] = ref["ms"].astype(float)
    ref["noise_degree"] = ref["snr"].astype(float)
    return ref[KEY + ["nmse_mean"]].rename(columns={"nmse_mean": "published"})


def main() -> int:
    ref = published()
    rows = []
    for label, params, name in MODELS:
        path = RUNS / name
        if not path.exists():
            print(f"[skip] {label}: {name} missing")
            continue
        run = load(path)
        merged = run.merge(ref, on=KEY)
        cells = merged.groupby(["cm", "ds", "ms"], as_index=False)[
            ["nmse", "published"]].mean()
        diff = (cells["nmse"] - cells["published"]).to_numpy()
        rng = np.random.default_rng(20260913)
        idx = rng.integers(0, len(diff), size=(5000, len(diff)))
        draws = diff[idx].mean(axis=1)
        enjoy = int((diff < 0).sum())
        k = max(enjoy, len(diff) - enjoy)
        tail = sum(math.comb(len(diff), i)
                   for i in range(k, len(diff) + 1)) / 2 ** len(diff)
        rows.append({"model": label, "params": params,
                     "nmse_BE": round(float(cells["nmse"].mean()), 4),
                     "published_BE": round(float(cells["published"].mean()), 4),
                     "paired_diff": round(float(diff.mean()), 4),
                     "ci_low": round(float(np.quantile(draws, 0.025)), 4),
                     "ci_high": round(float(np.quantile(draws, 0.975)), 4),
                     "cells_better": enjoy, "n_cells": int(len(diff)),
                     "sign_p": float(min(1.0, 2 * tail)),
                     "source": f"results/run_csv/{name}"})
    out = pd.DataFrame(rows).sort_values("nmse_BE").reset_index(drop=True)
    out.to_csv(ANA / "generalization_432_extended.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'generalization_432_extended.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
