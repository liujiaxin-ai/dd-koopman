"""DEPRECATED for interval reporting — kept for point estimates only.

Its bootstrap intervals differ from `generalization_seeds.csv` /
`stats_hardening.csv` in the 4th decimal (a different bootstrap RNG consumption
order). Every number that enters the paper must come from
`generalization_seeds.csv` (three-seed shift) or `stats_hardening.csv`
(sign test / Wilcoxon); see
`results/analysis/generalization_by_cm.DEPRECATED.md`.

Group the 432-setting generalization slice by unseen channel model.

Pairs results/run_csv/CAPNOAUX600_gen_gen.csv (ours, seed 42) with
results/reference/official_per_setting.csv (MODEL, TDD, generalization) on
(cm, ds, ms, snr), averages the paired difference inside each
(channel model, delay spread, speed) cell and bootstraps the cells with the
repository RNG (default_rng(20260913), 5000 resamples) in a fixed slice
order, so the confidence intervals stay methodologically identical to
results/analysis/generalization_table.csv.

Run from the repository root:
    python code/analysis/generalization_by_cm.py
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
RNG_SEED = 20260913
BOOTSTRAP = 5000
OURS_CSV = "results/run_csv/CAPNOAUX600_gen_gen.csv"
SOURCE = (OURS_CSV + " + results/reference/official_per_setting.csv "
          "(MODEL, TDD, generalization)")


def _set_ours(csv: str) -> None:
    """Point the analysis at another run CSV (one seed at a time)."""
    global OURS_CSV, SOURCE
    OURS_CSV = csv
    SOURCE = (csv + " + results/reference/official_per_setting.csv "
              "(MODEL, TDD, generalization)")


def _sign_test(favour: int, n: int) -> float:
    """Two-sided exact binomial sign test under the fair-coin null."""
    start = max(favour, n - favour)
    tail = sum(comb(n, k) for k in range(start, n + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def paired_cells(rng: np.random.Generator) -> pd.DataFrame:
    """Per-cell paired difference of ours minus published, per slice."""
    from build_tables import read_run

    ours = read_run(ROOT / OURS_CSV)
    ours["ds"] = ours["ds"].astype(float).round(10)
    official = pd.read_csv(ROOT / "results" / "reference" / "official_per_setting.csv")
    ref = official[
        (official["model"] == "MODEL") & (official["duplex"] == "TDD")
        & (official["split"] == "generalization")
    ].copy()
    ref["cm"] = ref["cm"].astype(str)
    ref["ds"] = ref["ds"].astype(float).round(10)
    ref["ms"] = ref["ms"].astype(float)
    ref["snr"] = ref["snr"].astype(float)
    key = ["cm", "ds", "ms", "snr"]
    merged = ours.merge(ref[key + ["nmse_mean"]], on=key, how="inner")
    merged = merged.rename(columns={"nmse_mean": "published_nmse"})
    merged["diff"] = merged["nmse"] - merged["published_nmse"]
    cells = merged.groupby(["cm", "ds", "ms"])[["diff", "nmse", "published_nmse"]].mean()
    return cells


def _slice_rows(name: str, cells: pd.DataFrame, rng: np.random.Generator) -> dict:
    diff = cells["diff"].to_numpy()
    draws = rng.choice(diff, size=(BOOTSTRAP, diff.size), replace=True).mean(axis=1)
    favour = int((diff < 0).sum())
    ours = round(float(cells["nmse"].mean()), 4)
    published = round(float(cells["published_nmse"].mean()), 4)
    return {
        "slice": name,
        "cells": int(diff.size),
        "ours": ours,
        "published": published,
        "rel_diff": round((ours - published) / published, 3),
        "paired_mean_diff": round(float(diff.mean()), 4),
        "ci_low": round(float(np.percentile(draws, 2.5)), 4),
        "ci_high": round(float(np.percentile(draws, 97.5)), 4),
        "favour_cells": f"{favour}/{diff.size}",
        "sign_p": round(_sign_test(favour, diff.size), 4),
        "source": SOURCE,
    }


def condition_table(rng: np.random.Generator, suffix: str = "") -> pd.DataFrame:
    """Per-(cm, ds, ms, snr) paired table behind the condition breakdown.

    ``results/analysis/generalization_by_condition.csv`` is the raw material
    for the "where does the gain concentrate" analysis: every row is one
    (unseen channel model, delay spread, speed, SNR) setting with ours, the
    published model, and their difference, plus a coarse speed band so the
    heatmap grouping does not need to be re-derived by hand.
    """
    from build_tables import read_run

    ours = read_run(ROOT / OURS_CSV)
    ours["ds"] = ours["ds"].astype(float).round(10)
    official = pd.read_csv(ROOT / "results" / "reference" / "official_per_setting.csv")
    ref = official[
        (official["model"] == "MODEL") & (official["duplex"] == "TDD")
        & (official["split"] == "generalization")
    ].copy()
    ref["cm"] = ref["cm"].astype(str)
    ref["ds"] = ref["ds"].astype(float).round(10)
    ref["ms"] = ref["ms"].astype(float)
    ref["snr"] = ref["snr"].astype(float)
    key = ["cm", "ds", "ms", "snr"]
    merged = ours.merge(ref[key + ["nmse_mean"]], on=key, how="inner")
    merged = merged.rename(columns={"nmse_mean": "published_nmse"})
    merged["diff"] = merged["nmse"] - merged["published_nmse"]
    merged["speed_band"] = pd.cut(
        merged["ms"], [0.0, 10.0, 30.0, np.inf], labels=["le10", "10to30", "gt30"]
    ).astype(str)
    merged["ours_favoured"] = merged["diff"] < 0
    cols = ["cm", "ds", "ms", "snr", "speed_band", "nmse", "published_nmse",
            "diff", "ours_favoured"]
    table = merged[cols].sort_values(["cm", "ds", "ms", "snr"]).reset_index(drop=True)
    table.to_csv(OUT / f"generalization_by_condition{suffix}.csv", index=False)
    return table


def run(rng: np.random.Generator) -> pd.DataFrame:
    """Write results/analysis/generalization_by_cm.csv and return the frame."""
    cells = paired_cells(rng)
    rows = [
        _slice_rows("all", cells, rng),
        _slice_rows("cmB", cells[cells.index.get_level_values("cm") == "B"], rng),
        _slice_rows("cmE", cells[cells.index.get_level_values("cm") == "E"], rng),
        _slice_rows("le10ms", cells[cells.index.get_level_values("ms") <= 10], rng),
        _slice_rows("gt30ms", cells[cells.index.get_level_values("ms") > 30], rng),
    ]
    # Equal weighting of the two unseen channel models in ratio terms: the
    # mean of the cmB and cmE relative differences.
    eqweight = (rows[1]["rel_diff"] + rows[2]["rel_diff"]) / 2
    rows.append({
        "slice": "eqweight-relative", "cells": int(len(cells)),
        "ours": np.nan, "published": np.nan, "rel_diff": round(eqweight, 3),
        "paired_mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "favour_cells": f"{int((cells['diff'] < 0).sum())}/{len(cells)}",
        "sign_p": np.nan,
        "source": SOURCE + "; mean of the cmB and cmE rel_diff values",
    })
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if OURS_CSV == "results/run_csv/CAPNOAUX600_gen_gen.csv" else \
        "_" + Path(OURS_CSV).stem.rsplit("_gen_gen", 1)[0]
    frame.to_csv(OUT / f"generalization_by_cm{suffix}.csv", index=False)
    condition_table(rng, suffix)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", default=OURS_CSV,
                    help="run CSV for the in-house model (one seed at a time)")
    args = ap.parse_args()
    _set_ours(args.ours)
    frame = run(np.random.default_rng(RNG_SEED))
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
