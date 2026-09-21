"""Non-parametric statistics for every paired comparison in the paper.

The tables already report paired bootstrap CIs over cells. Reviewers of a
signal-processing venue usually also expect a distribution-free test, so this
script adds, per comparison slice:

  * n paired cells,
  * the mean paired difference and its 95% bootstrap CI (same convention as the
    tables: cells = (cm, delay spread, speed), bootstrap over cells, RNG
    20260913),
  * a two-sided sign test (exact binomial on the number of favoured cells),
  * a two-sided Wilcoxon signed-rank test when scipy is available.

Output: results/analysis/stats_hardening.csv
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"
REFERENCE = ROOT / "results" / "reference" / "official_per_setting.csv"
BOOTSTRAP, SEED = 5000, 20260913

sys.path.insert(0, str(ROOT / "code" / "analysis"))
import acd_three_seeds as ATS  # noqa: E402


def horizon_mean(text: str) -> float:
    return float(np.mean([float(x) for x in
                          re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]))


def read_run(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    out = pd.DataFrame({
        "cm": frame["cm"].astype(str),
        "ds": frame["ds"].astype(float).round(10),
        "ms": frame["ms"].astype(float),
        "nmse": frame["nmse_mean"].map(horizon_mean),
    })
    if "noise_degree" in frame:
        out["snr"] = frame["noise_degree"].astype(float)
    if "noise_type" in frame:
        out["noise_type"] = frame["noise_type"].astype(str)
    return out


def bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_test_p(diff: np.ndarray) -> float:
    """Two-sided exact binomial test that the median paired difference is zero."""
    wins = int((diff < 0).sum())
    n = int((diff != 0).sum())
    if n == 0:
        return float("nan")
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def wilcoxon_p(diff: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon
    except Exception:                       # noqa: BLE001
        return float("nan")
    diff = diff[diff != 0]
    if len(diff) < 10:
        return float("nan")
    try:
        return float(wilcoxon(diff, alternative="two-sided").pvalue)
    except Exception:                       # noqa: BLE001
        return float("nan")


def summarise(name: str, diff: np.ndarray) -> dict:
    lo, hi = bootstrap_ci(diff)
    return {"comparison": name, "cells": len(diff),
            "mean_diff": round(float(diff.mean()), 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "cells_favoured": int((diff < 0).sum()),
            "sign_test_p": round(sign_test_p(diff), 5),
            "wilcoxon_p": round(wilcoxon_p(diff), 5)}


def cells_from_run(path: Path) -> pd.DataFrame:
    frame = read_run(path)
    frame = frame[frame["noise_type"] == "vanilla"]
    return (frame.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
                 .rename(columns={"nmse": "ours"}))


def published(split: str) -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE)
    ref = ref[(ref["split"] == split) & (ref["model"] == "MODEL")
              & (ref["duplex"] == "TDD")
              & (ref["noise_type"] == "vanilla")]
    ref = ref.drop_duplicates(["cm", "ds", "ms", "snr"])
    return (ref.groupby(["cm", "ds", "ms"], as_index=False)["nmse_mean"].mean()
               .rename(columns={"nmse_mean": "published"}))


def main() -> int:
    rows = []

    # ---- regular 162 grid, three seeds ------------------------------------
    pub_reg = published("regular")
    for tag, path in (("seed42", "CAPNOAUX600_full162.csv"),
                      ("seed43", "CAPNOAUXS43_full162.csv"),
                      ("seed44", "HEADNOAUXS44_full162.csv")):
        file = RUNS / path
        if not file.exists():
            continue
        merged = cells_from_run(file).merge(pub_reg, on=["cm", "ds", "ms"])
        rows.append(summarise(f"regular-162 {tag}",
                              (merged["ours"] - merged["published"]).to_numpy()))

    # ---- unseen channel models B/E, 432 settings, three seeds --------------
    pub_gen = published("generalization")
    for tag, path in (("seed42", "CAPNOAUX600_gen_gen.csv"),
                      ("seed43", "CAPNOAUXS43_gen_gen.csv"),
                      ("seed44", "CAPNOAUXS44_gen_gen.csv")):
        file = RUNS / path
        if not file.exists():
            continue
        merged = cells_from_run(file).merge(pub_gen, on=["cm", "ds", "ms"])
        merged = merged[merged.cm.isin(["B", "E"])]
        rows.append(summarise(f"unseen-cm 432 {tag}",
                              (merged["ours"] - merged["published"]).to_numpy()))

    # ---- interpolation shift A/C/D, 108 cells, three seeds -----------------
    per_seed = {seed: ATS.seed_cells(ATS.RUNS / name)
                for seed, name in ATS.SEEDS.items()}
    stacked = pd.concat([f.assign(seed=s) for s, f in per_seed.items()],
                        ignore_index=True)
    ours = (stacked.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
                   .rename(columns={"nmse": "ours"}))
    merged = ours.merge(pub_gen, on=["cm", "ds", "ms"])
    merged = merged[merged.cm.isin(["A", "C", "D"])]
    rows.append(summarise("interpolation 108 (3 seeds)",
                          (merged["ours"] - merged["published"]).to_numpy()))

    # ---- robustness 486 (per-setting pairing) ------------------------------
    ref_all = pd.read_csv(REFERENCE)
    pub_rob = ref_all[(ref_all["split"] == "robustness")
                      & (ref_all["model"] == "MODEL")
                      & (ref_all["duplex"] == "TDD")].copy()
    level = "snr" if "snr" in pub_rob.columns else "noise_degree"
    pub_rob = pub_rob.rename(columns={"nmse_mean": "published"})
    pub_rob = pub_rob[["cm", "ds", "ms", "noise_type", level, "published"]]
    pub_rob["cm"] = pub_rob["cm"].astype(str)
    pub_rob["ds"] = pub_rob["ds"].astype(float).round(10)
    pub_rob["ms"] = pub_rob["ms"].astype(float)
    for tag, path in (("seed42", "CAPNOAUX600_robust486.csv"),
                      ("seed43", "CAPNOAUXS43_robust486.csv"),
                      ("seed44", "HEADNOAUXS44_robust486.csv")):
        file = RUNS / path
        if not file.exists():
            continue
        ours = read_run(file)
        ours = ours.rename(columns={"nmse": "ours"})
        ours_key = "snr" if "snr" in ours.columns else level
        merged = ours.merge(pub_rob, left_on=["cm", "ds", "ms", "noise_type",
                                              ours_key],
                            right_on=["cm", "ds", "ms", "noise_type", level],
                            how="inner")
        if len(merged) >= 10:
            rows.append(summarise(f"robustness-486 {tag} (per setting)",
                                  (merged["ours"] - merged["published"]).to_numpy()))

    out = pd.DataFrame(rows)
    out.to_csv(ANA / "stats_hardening.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {ANA / 'stats_hardening.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
