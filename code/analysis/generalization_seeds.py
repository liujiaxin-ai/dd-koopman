"""Aggregate the generalization slice across the reported configuration's seeds.

Reads one run CSV per seed, averages the six SNR settings inside each
(channel model, delay spread, speed) cell, and bootstraps the paired cell
differences against the published reference rows with the repository RNG, so
the intervals are directly comparable with results/analysis/generalization_by_cm.csv.

Run from the repository root:
    python code/analysis/generalization_seeds.py
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
RNG_SEED = 20260913
BOOTSTRAP = 5000

SEEDS = {
    42: "results/run_csv/CAPNOAUX600_gen_gen.csv",
    43: "results/run_csv/CAPNOAUXS43_gen_gen.csv",
    44: "results/run_csv/CAPNOAUXS44_gen_gen.csv",
}


def _read_run(path: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path)
    df["ds"] = df["ds"].astype(float).round(10)
    df["nmse"] = df["nmse_mean"].map(
        lambda v: float(np.mean(np.fromstring(str(v).strip("[]"), sep=" ")))
    )
    return df[["cm", "ds", "ms", "noise_degree", "nmse"]].rename(
        columns={"noise_degree": "snr"}
    )


def _published() -> pd.DataFrame:
    ref = pd.read_csv(ROOT / "results" / "reference" / "official_per_setting.csv")
    ref = ref[
        (ref.model == "MODEL") & (ref.duplex == "TDD") & (ref.split == "generalization")
    ].copy()
    ref["cm"] = ref["cm"].astype(str)
    ref["ds"] = ref["ds"].astype(float).round(10)
    ref["ms"] = ref["ms"].astype(float)
    ref["snr"] = ref["snr"].astype(float)
    return ref[["cm", "ds", "ms", "snr", "nmse_mean"]].rename(
        columns={"nmse_mean": "published"}
    )


def _sign_test(favour: int, n: int) -> float:
    start = max(favour, n - favour)
    tail = sum(comb(n, k) for k in range(start, n + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def cells_for(run: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    merged = run.merge(ref, on=["cm", "ds", "ms", "snr"], how="inner")
    merged["diff"] = merged["nmse"] - merged["published"]
    return merged.groupby(["cm", "ds", "ms"])[["diff", "nmse", "published"]].mean()


def summarise(name: str, cells: pd.DataFrame, rng: np.random.Generator) -> dict:
    diff = cells["diff"].to_numpy()
    draws = rng.choice(diff, size=(BOOTSTRAP, diff.size), replace=True).mean(axis=1)
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    # Ties carry no sign information, so they are removed from the sign test and
    # the effective sample size is reported next to the raw cell count.
    nonzero = diff[diff != 0.0]
    favour = int((nonzero < 0).sum())
    n_effective = int(nonzero.size)
    return {
        "slice": name,
        "cells": int(diff.size),
        "ours": round(float(cells["nmse"].mean()), 4),
        "published": round(float(cells["published"].mean()), 4),
        "rel_diff": round(float((cells["nmse"].mean() - cells["published"].mean())
                               / cells["published"].mean()), 3),
        "paired_mean_diff": round(float(diff.mean()), 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "favour_cells": f"{favour}/{n_effective}",
        "ties": int(diff.size - n_effective),
        "n_effective": n_effective,
        "sign_p": round(_sign_test(favour, n_effective), 4),
        # Two-sided semantics, matching acd_three_seeds / acd_shift_eval /
        # generalization_432_summary / stats_hardening: the interval is reported
        # as excluding zero whenever it lies entirely on one side, and a separate
        # flag says whether it lies on the side that favours this work.
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "ours_better": bool(high < 0),
    }


def main() -> int:
    ref = _published()
    per_seed: dict[int, pd.DataFrame] = {}
    rows = []
    for seed, path in SEEDS.items():
        per_seed[seed] = cells_for(_read_run(path), ref)

    for label, table in (("all", None), ("cmB", "B"), ("cmE", "E")):
        for seed, cells in per_seed.items():
            part = cells if table is None else cells[cells.index.get_level_values("cm") == table]
            row = summarise(f"{label}-seed{seed}", part, np.random.default_rng(RNG_SEED))
            row["seed"] = seed
            row["source"] = SEEDS[seed]
            rows.append(row)

    # seed-mean curve: average the per-cell differences across seeds
    stacked = pd.concat([c["diff"] for c in per_seed.values()], axis=1).mean(axis=1)
    mean_cells = pd.concat([c[["nmse", "published"]] for c in per_seed.values()]).groupby(
        level=[0, 1, 2]
    ).mean()
    mean_cells["diff"] = stacked
    for label, table in (("all", None), ("cmB", "B"), ("cmE", "E")):
        part = mean_cells if table is None else mean_cells[
            mean_cells.index.get_level_values("cm") == table
        ]
        row = summarise(f"{label}-seedmean", part, np.random.default_rng(RNG_SEED))
        row["seed"] = "mean(42,43,44)"
        row["source"] = "; ".join(SEEDS.values())
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "generalization_seeds.csv", index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
