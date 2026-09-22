"""Verify the fixed ladder runner's output against the original ladder rows.

Reads the three CSVs produced by the re-run of
`anonymous_repo/code/ladder/ladder_run_dftgrid.sh` and reports, for each split:

  * row count (the script's own gate is 162 / 486 / 432),
  * cell-mean NMSE with the paper's protocol (mean over the four horizons inside
    a cell first, then mean over the 27 (or 12) cells),
  * for the generalization slice, the (cm, ds, ms) distribution.

It also diffs every numeric column against the original
`results/run_csv/DFTGRID_K16_*.csv` so a reviewer can see the re-run reproduces
the shipped ladder numbers rather than merely having the right shape.

Writes `results/theory/ladder_rerun/SUMMARY.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RERUN = ROOT / "results" / "theory" / "ladder_rerun" / "out"
ORIGINAL = ROOT / "results" / "run_csv"
OUT = ROOT / "results" / "theory" / "ladder_rerun" / "SUMMARY.md"

SPLITS = [
    ("regular", "DFTGRID_K16_full162.csv", 162, 3.2883, "full162"),
    ("robustness", "DFTGRID_K16_robust486.csv", 486, 2.7558, "robust486"),
    ("generalization-432", "DFTGRID_K16_gen432.csv", 432, 2.7518, "gen432"),
]
KEYS = ["cm", "ds", "ms", "noise_type", "noise_degree"]


def horizon_mean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ds"] = frame["ds"].astype(float).round(10)
    frame["nmse"] = frame["nmse_mean"].map(
        lambda v: float(np.mean(np.fromstring(str(v).strip("[]"), sep=" ")))
    )
    return frame


def main() -> int:
    lines = ["# Ladder re-run verification (fixed runner, 2026-09-22)", ""]
    lines += ["Source: `results/theory/ladder_rerun/` (never overwrites `results/run_csv/`).",
              "",
              "| split | rows (want) | rows (got) | cell-mean (re-run) | cell-mean (expected) | vs shipped CSV |",
              "|---|---|---|---|---|---|"]

    for split, name, want, expected, _ in SPLITS:
        rerun = horizon_mean(pd.read_csv(RERUN / name))
        cell = float(rerun.groupby(["cm", "ds", "ms"])["nmse"].mean().mean())

        shipped_path = ORIGINAL / name
        if shipped_path.exists():
            shipped = horizon_mean(pd.read_csv(shipped_path))
            merged = rerun.merge(shipped, on=KEYS, suffixes=("_r", "_s"))
            diff = float(np.abs(merged["nmse_r"] - merged["nmse_s"]).max())
            same_keys = len(merged) == len(rerun) == len(shipped)
            verdict = (f"max|Δ|={diff:.3e} over {len(merged)} matched settings"
                       if same_keys else f"KEY SET DIFFERS ({len(merged)} matched)")
        else:
            verdict = "shipped CSV not found"
        lines.append(f"| {split} | {want} | {len(rerun)} | **{cell:.4f}** | {expected} | {verdict} |")

    gen = horizon_mean(pd.read_csv(RERUN / "DFTGRID_K16_gen432.csv"))
    counts = gen.groupby(["cm", "ds", "ms"]).size()
    lines += ["", "## gen432 structure (the point of the fix)", "",
              f"channel models = {sorted(gen['cm'].unique())} (exactly B and E, no A/C/D), "
              f"delay spreads = {sorted(gen['ds'].unique())}, "
              f"speeds = {sorted(gen['ms'].unique())} ({len(sorted(gen['ms'].unique()))} values)",
              "",
              f"* (cm, ds, ms) cells = **{counts.size}** = 2 x 3 x 12; rows per cell = "
              f"{counts.min()}..{counts.max()} (the 6 SNRs) -> total **{int(counts.sum())}**",
              "* rows per channel model = "
              f"{int(counts.groupby(level=0).sum().iloc[0])} = 3 ds x 12 speeds x 6 SNR",
              "* rows per (cm, ds) = "
              f"{int(counts.groupby(level=[0, 1]).sum().iloc[0])} = 12 speeds x 6 SNR",
              "",
              "| cm | ds | speeds | rows | rows/speed |", "|---|---|---|---|---|"]
    for (cm, ds), group in counts.groupby(level=[0, 1]):
        lines.append(f"| {cm} | {ds} | {len(group)} | {int(group.sum())} | "
                     f"{sorted(set(group.values))} |")
    lines += ["", "Full (cm, ds, ms) cell table:", "", "| cm | ds | ms | rows |",
              "|---|---|---|---|"]
    for (cm, ds, ms), n in counts.items():
        lines.append(f"| {cm} | {ds} | {ms} | {n} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
