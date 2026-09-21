"""Consolidated 432-setting generalization-slice table (S1 + S3 + S4).

Every row is one configuration evaluated on the same 72-cell / 432-setting
slice, with the paired difference against the published model and the
bootstrap interval produced by generalization_by_cm.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"

RUNS = [
    ("ours, seed 42 (reported)", 173190, "results/run_csv/CAPNOAUX600_gen_gen.csv", ""),
    ("ours, seed 43", 173190, "results/run_csv/CAPNOAUXS43_gen_gen.csv", "_CAPNOAUXS43"),
    ("ours, seed 44", 173190, "results/run_csv/CAPNOAUXS44_gen_gen.csv", "_CAPNOAUXS44"),
    ("CSI-4CAST shrunk (91k, converged)", 91410, "results/run_csv/P3CONV_gen_gen.csv", "_P3CONV"),
    ("MambaCSP port (8 ep)", 881604, "results/run_csv/MAMBA64_432_gen.csv", "_MAMBA64_432_gen"),
    ("capacity-matched Transformer", 149560, "results/run_csv/TRANSMATCHED_432_gen.csv", "_TRANSMATCHED_432_gen"),
]

rows = []
for label, params, source, suffix in RUNS:
    by_cm = pd.read_csv(OUT / f"generalization_by_cm{suffix}.csv")
    all_row = by_cm[by_cm["slice"] == "all"].iloc[0]
    rows.append({
        "model": label,
        "params": params,
        "nmse_432": all_row["ours"],
        "published_432": all_row["published"],
        "paired_diff": all_row["paired_mean_diff"],
        "ci_low": all_row["ci_low"],
        "ci_high": all_row["ci_high"],
        "ci_excludes_zero": bool(all_row["ci_low"] > 0 or all_row["ci_high"] < 0),
        "source": source,
    })

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "generalization_432_summary.csv", index=False)
print(frame.to_string(index=False))


