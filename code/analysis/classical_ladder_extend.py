"""Append the fixed-grid extrapolator rows to the classical ladder.

The ladder's other rungs (NP / PAD / AR / WIENER) come from the shipped
reference table; the DFT-grid extrapolator is our own implementation, so its
three rows are appended here from the run CSVs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"

RUNS = [
    ("regular", "results/run_csv/DFTGRID_K1_full162.csv"),
    ("robustness", "results/run_csv/DFTGRID_K1_robust486.csv"),
    ("generalization-432", "results/run_csv/DFTGRID_K1_gen432.csv"),
]


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
    rows = []
    for split, rel in RUNS:
        n, value = cell_mean(ROOT / rel)
        rows.append({
            "split": split,
            "model": "DFTGRID (K=1, fixed grid)",
            "settings": n,
            "cell_mean_nmse": round(value, 4),
            "source": rel,
        })
    combined = pd.concat([ladder, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["split", "model"], keep="last")
    combined.to_csv(OUT / "classical_ladder.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
