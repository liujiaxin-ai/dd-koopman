"""Three-seed aggregation of the A/C/D extended-shift slice.

Inputs are the three per-seed harness tables produced on the local 5060
(seed 42 = CAPNOAUX600, seed 43 = CAPNOAUXS43, seed 44 = HEADNOAUXS44), each
648 settings = 108 (cm, ds, speed) combinations x 6 SNR levels. The reduction
matches code/analysis/acd_shift_eval.py and build_tables.read_run exactly:
`nmse_mean` is stored as the per-horizon vector, so it is first reduced with a
plain mean over the four horizons, then averaged over the six SNRs inside a
cell, and finally averaged over the three seeds. Intervals are paired
bootstraps over the 108 cells (5,000 resamples, RNG 20260913).

Output: results/analysis/generalization_acd_seeds.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"
REFERENCE = ROOT / "results" / "reference" / "official_per_setting.csv"

SEEDS = {
    42: "CAPNOAUX600_gen_acd.csv",
    43: "CAPNOAUXS43_gen_acd.csv",
    44: "HEADNOAUXS44_gen_acd.csv",
}
CHANNEL_MODELS = ("A", "C", "D")
DELAY_SPREADS = (50e-9, 200e-9, 400e-9)
SPEEDS = (3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 42)
BOOTSTRAP, SEED = 5000, 20260913


def horizon_mean(text: str) -> float:
    return float(np.mean([float(x) for x in
                          re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]))


def seed_cells(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    level = "noise_degree" if "noise_degree" in df.columns else "snr"
    df = df[df["noise_type"] == "vanilla"].drop_duplicates(
        ["cm", "ds", "ms", level]).copy()
    df["nmse"] = df["nmse_mean"].map(horizon_mean)
    return (df.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean())


def published_cells() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE)
    # IMPORTANT: the reference table stores BOTH duplexes for the trained channel
    # models. Comparing our TDD model against a TDD+FDD average silently inflates
    # the published side (cm A/C/D FDD cell-mean 0.946 vs TDD 0.218), so the
    # duplex must be pinned to TDD here.
    ref = ref[(ref["split"] == "generalization")
              & (ref["duplex"] == "TDD")
              & (ref["cm"].isin(CHANNEL_MODELS))
              & (ref["ds"].isin(DELAY_SPREADS))
              & (ref["ms"].isin(SPEEDS))
              & (ref["noise_type"] == "vanilla")
              & (ref["model"] == "MODEL")]
    ref = ref.drop_duplicates(["cm", "ds", "ms", "snr"])
    return (ref.groupby(["cm", "ds", "ms"], as_index=False)["nmse_mean"].mean()
               .rename(columns={"nmse_mean": "published"}))


def paired_stats(diff: np.ndarray) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP, len(diff)))
    means = diff[idx].mean(axis=1)
    return (float(diff.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)), int((diff < 0).sum()))


def main() -> int:
    per_seed = {seed: seed_cells(RUNS / name) for seed, name in SEEDS.items()}
    for seed, frame in per_seed.items():
        print(f"[seed {seed}] cells={len(frame)} "
              f"mean={frame['nmse'].mean():.4f}")

    stacked = pd.concat(
        [frame.assign(seed=seed) for seed, frame in per_seed.items()],
        ignore_index=True)
    ours = (stacked.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
                   .rename(columns={"nmse": "ours"}))
    merged = ours.merge(published_cells(), on=["cm", "ds", "ms"], how="inner")
    merged["diff"] = merged["ours"] - merged["published"]
    merged["speed_band"] = np.where(merged["ms"] <= 9, "le10",
                                    np.where(merged["ms"] <= 27, "10to30",
                                             "gt30"))
    print(f"[three seeds] paired cells: {len(merged)}")

    slices: list[tuple[str, pd.DataFrame]] = [("all", merged)]
    slices += [(f"cm{cm}", merged[merged.cm == cm]) for cm in CHANNEL_MODELS]
    slices += [(f"ds{round(ds * 1e9)}", merged[merged.ds == ds])
               for ds in DELAY_SPREADS]
    slices += [(f"band{band}", merged[merged.speed_band == band])
               for band in ("le10", "10to30", "gt30")]

    rows = []
    for name, part in slices:
        if part.empty:
            continue
        mean, lo, hi, wins = paired_stats(part["diff"].to_numpy())
        rows.append({"slice": name, "cells": len(part),
                     "ours": round(float(part["ours"].mean()), 4),
                     "published": round(float(part["published"].mean()), 4),
                     "paired_mean_diff": round(mean, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "ci_excludes_zero": bool(lo > 0 or hi < 0),
                     "ours_favoured_cells": f"{wins}/{len(part)}"})
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "generalization_acd_seeds.csv", index=False)

    seed_rows = []
    for seed, frame in per_seed.items():
        part = frame.merge(published_cells(), on=["cm", "ds", "ms"], how="inner")
        diff = (part["nmse"] - part["published"]).to_numpy()
        mean, lo, hi, wins = paired_stats(diff)
        seed_rows.append({"seed": seed, "cells": len(part),
                          "ours": round(float(part["nmse"].mean()), 4),
                          "paired_mean_diff": round(mean, 4),
                          "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                          "ours_favoured_cells": f"{wins}/{len(part)}"})
    pd.DataFrame(seed_rows).to_csv(ANA / "generalization_acd_by_seed.csv",
                                   index=False)

    print(out.to_string(index=False))
    print()
    print(pd.DataFrame(seed_rows).to_string(index=False))
    print(f"\nwrote {ANA / 'generalization_acd_seeds.csv'}")
    print(f"wrote {ANA / 'generalization_acd_by_seed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
