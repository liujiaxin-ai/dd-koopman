"""Append the DFT-grid and LS-gain rows to the classical ladder.

The ladder's other rungs (NP / PAD / AR / WIENER) come from the shipped
reference table; the DFT-grid extrapolator and the ridge-LS gain predictor are
our own implementations, so their rows are (re)generated here from the run CSVs.

Two label facts are corrected in the same pass:
  * K = 1 is *not* a fixed grid. It keeps the single strongest bin per
    (sample, antenna, tap), so the retained bin moves with the input; the label
    now says `adaptive top-1`.
  * K = 16 keeps every bin, so it is the fixed-grid rung proper — the P0 of
    Proposition 1 under the full support (E0 in the ladder).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"

# (split, label, results-relative path). The K = 1 rows keep their existing
# values, the K = 16 and LS rows are new; all of them are recomputed from the
# CSVs so the table can always be regenerated from `results/`.
RUNS = [
    ("regular", "DFTGRID (K=1, adaptive top-1)", "results/run_csv/DFTGRID_K1_full162.csv"),
    ("robustness", "DFTGRID (K=1, adaptive top-1)", "results/run_csv/DFTGRID_K1_robust486.csv"),
    ("generalization-432", "DFTGRID (K=1, adaptive top-1)", "results/run_csv/DFTGRID_K1_gen432.csv"),
    ("regular", "DFTGRID (K=16, full grid)", "results/run_csv/DFTGRID_K16_full162.csv"),
    ("robustness", "DFTGRID (K=16, full grid)", "results/run_csv/DFTGRID_K16_robust486.csv"),
    ("generalization-432", "DFTGRID (K=16, full grid)", "results/run_csv/DFTGRID_K16_gen432.csv"),
    ("regular", "LS-fitted grid gains (K=16)", "results/run_csv/LSGAINS_K16_full162.csv"),
    ("robustness", "LS-fitted grid gains (K=16)", "results/run_csv/LSGAINS_K16_robust486.csv"),
    ("generalization-432", "LS-fitted grid gains (K=16)", "results/run_csv/LSGAINS_K16_gen432.csv"),
    ("regular", "LS-fitted grid gains (K=16, full-training fit)",
     "results/run_csv/LSGAINSFULL_K16_full162.csv"),
    ("robustness", "LS-fitted grid gains (K=16, full-training fit)",
     "results/run_csv/LSGAINSFULL_K16_robust486.csv"),
    ("generalization-432", "LS-fitted grid gains (K=16, full-training fit)",
     "results/run_csv/LSGAINSFULL_K16_gen432.csv"),
]

SUPERSEDED = {"DFTGRID (K=1, fixed grid)": "DFTGRID (K=1, adaptive top-1)"}


def cell_mean(path: Path) -> tuple[int, float]:
    df = pd.read_csv(path)
    df["ds"] = df["ds"].astype(float).round(10)
    df["nmse"] = df["nmse_mean"].map(
        lambda v: float(np.mean(np.fromstring(str(v).strip("[]"), sep=" ")))
    )
    cells = df.groupby(["cm", "ds", "ms"])["nmse"].mean()
    return len(df), float(cells.mean())


def main() -> int:
    ladder = pd.read_csv(OUT / "classical_ladder.csv")
    ladder["model"] = ladder["model"].replace(SUPERSEDED)
    rows = []
    missing = []
    for split, label, rel in RUNS:
        if not (ROOT / rel).exists():
            missing.append(rel)
            continue
        n, value = cell_mean(ROOT / rel)
        rows.append({
            "split": split,
            "model": label,
            "settings": n,
            "cell_mean_nmse": round(value, 4),
            "source": rel,
        })
    combined = pd.concat([ladder, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["split", "model"], keep="last")
    combined.to_csv(OUT / "classical_ladder.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if missing:
        print("\nMISSING (left as-is in the table): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
