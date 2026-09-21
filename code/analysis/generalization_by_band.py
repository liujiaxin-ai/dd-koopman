"""Persist the mobility-band three-seed numbers behind fig_condition / §IV-C.

The band-level values used to live only inside the plotting script, which made
them untraceable in an audit. This writes them to
results/analysis/generalization_by_band.csv with the paper's convention:

  cell = (channel model, delay spread, speed); NMSE is averaged over the six
  SNRs inside a cell and then over the three seeds (42/43/44); the interval is a
  paired bootstrap over cells (5,000 resamples, RNG 20260913); a two-sided sign
  test is reported alongside so the reader sees both the mean shift and how many
  cells actually favour the reported operator.

Scope: the 432-setting unseen-channel-model slice (cm B and E), which is the
slice fig_condition uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANA = ROOT / "results" / "analysis"
sys.path.insert(0, str(ROOT / "code" / "analysis"))
import stats_hardening as SH  # noqa: E402

SEED_FILES = {
    42: ANA / "generalization_by_condition.csv",
    43: ANA / "generalization_by_condition_CAPNOAUXS43.csv",
    44: ANA / "generalization_by_condition_CAPNOAUXS44.csv",
}
BANDS = ["le10", "10to30", "gt30"]
BAND_LABEL = {"le10": "<=10 m/s", "10to30": "10-30 m/s", "gt30": ">30 m/s"}


def cells_by_band(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["cm"].isin(["B", "E"])]
    cells = (frame.groupby(["cm", "ds", "ms", "speed_band"], as_index=False)
                  [["nmse", "published_nmse"]].mean())
    cells["diff"] = cells["nmse"] - cells["published_nmse"]
    return cells


def main() -> int:
    per_seed = {seed: cells_by_band(path) for seed, path in SEED_FILES.items()}
    rows = []
    for band in BANDS:
        per_seed_means = {}
        for seed, frame in per_seed.items():
            part = frame[frame["speed_band"] == band]
            if part.empty:
                continue
            stats = SH.summarise(f"seed{seed}", part["diff"].to_numpy())
            per_seed_means[seed] = part["diff"].mean()
            rows.append({"band": band, "band_label": BAND_LABEL[band],
                         "seed": seed, "cells": stats["cells"],
                         "mean_diff": stats["mean_diff"],
                         "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                         "sign_test_p": stats["sign_test_p"],
                         "cells_favoured": stats["cells_favoured"]})
        # the three-seed mean per band, paired over cells and seeds
        stacked = []
        for seed, frame in per_seed.items():
            part = frame[frame["speed_band"] == band].copy()
            stacked.append(part[["cm", "ds", "ms", "diff"]])
        merged = (pd.concat(stacked, ignore_index=True)
                    .groupby(["cm", "ds", "ms"], as_index=False)["diff"].mean())
        stats = SH.summarise("three-seeds", merged["diff"].to_numpy())
        rows.append({"band": band, "band_label": BAND_LABEL[band],
                     "seed": "mean", "cells": stats["cells"],
                     "mean_diff": stats["mean_diff"],
                     "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                     "sign_test_p": stats["sign_test_p"],
                     "cells_favoured": stats["cells_favoured"]})
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "generalization_by_band.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'generalization_by_band.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
