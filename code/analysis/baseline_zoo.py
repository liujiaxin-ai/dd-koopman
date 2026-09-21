"""Zero-GPU comparison table: every reference baseline x every evaluation slice.

The shipped reference table carries per-setting NMSE for the whole benchmark
family (MODEL / CNN / RNN / LLM4CP / STEMGNN / PAD / WIENER / AR / NP) on the
regular, robustness and generalization splits. Our own runs cover the same
slices, so one table can show where the reported operator sits against all of
them - not only against the headline MODEL row.

Slice definitions (TDD only, exactly the grids used elsewhere in the project):
  regular       cm A/C/D x ds {3e-8, 1e-7, 3e-7} x ms {1,10,30}      162
  robustness    cm A/C/D x the three corruptions                      486
  shift B/E     cm B/E   x ds {5e-8, 2e-7, 4e-7} x 12 speeds          432
  shift A/C/D   cm A/C/D x the same 3 ds x 12 speeds                  648

Convention: average the six SNR levels inside each (cm, ds, ms[, noise]) cell,
then average the cells. Reference table -> mean of `nmse_mean`; our runs ->
mean of the four horizons then the same cell reduction.

Output: results/analysis/baseline_zoo.csv
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

DS_SHIFT = (5e-08, 2e-07, 4e-07)
MS_SHIFT = (3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 42)

OURS = {
    "regular": [RUNS / "CAPNOAUX600_full162.csv"],
    "robustness": [RUNS / "CAPNOAUX600_robust486.csv"],
    "shift B/E": [RUNS / "CAPNOAUX600_gen_gen.csv"],
    "shift A/C/D": [RUNS / "CAPNOAUX600_gen_acd.csv"],
}


def _numbers(text: str) -> float:
    return float(np.mean([
        float(x) for x in
        re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]))


def our_slice(path: Path, slice_name: str) -> float:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    # the robustness split *is* the corruption axis, so only the clean slices
    # are restricted to `vanilla` noise
    if "noise_type" in frame and slice_name != "robustness":
        frame = frame[frame["noise_type"] == "vanilla"]
    frame = frame.copy()
    frame["ds"] = frame["ds"].astype(float).round(10)
    frame["nmse"] = frame["nmse_mean"].map(_numbers)
    if slice_name == "shift B/E":
        frame = frame[frame["cm"].isin(["B", "E"])]
    if slice_name == "shift A/C/D":
        frame = frame[frame["cm"].isin(["A", "C", "D"])]
    cells = frame.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
    return float(cells["nmse"].mean())


def ref_slices() -> pd.DataFrame:
    ref = pd.read_csv(REF)
    ref = ref[ref.duplex == "TDD"].copy()
    ref["ds"] = ref["ds"].astype(float).round(10)
    ref["nmse"] = ref["nmse_mean"].astype(float)

    def cell_mean(part: pd.DataFrame) -> float:
        cells = part.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
        return float(cells["nmse"].mean())

    rows = []
    for model, part in ref.groupby("model"):
        rec = {"model": model}
        reg = part[part.split == "regular"]
        rob = part[part.split == "robustness"]
        gen = part[part.split == "generalization"]
        gen = gen[gen.ds.isin(DS_SHIFT) & gen.ms.isin(MS_SHIFT)]
        rec["regular"] = round(cell_mean(reg), 4)
        rec["robustness"] = round(cell_mean(rob), 4)
        rec["shift B/E"] = round(cell_mean(gen[gen.cm.isin(["B", "E"])]), 4)
        rec["shift A/C/D"] = round(
            cell_mean(gen[gen.cm.isin(["A", "C", "D"])]), 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def _ours_frame(path: Path, slice_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    if "noise_type" in frame and slice_name != "robustness":
        frame = frame[frame["noise_type"] == "vanilla"]
    frame = frame.copy()
    frame["ds"] = frame["ds"].astype(float).round(10)
    frame["nmse"] = frame["nmse_mean"].map(_numbers)
    key = ["cm", "ds", "ms"]
    if slice_name == "robustness":
        key.append("noise_type")
    else:
        frame = frame[frame["noise_type"] == "vanilla"] if "noise_type" in frame else frame
    return frame.groupby(key, as_index=False)["nmse"].mean()


def _ref_frame(slice_name: str, model: str) -> pd.DataFrame:
    ref = pd.read_csv(REF)
    ref = ref[(ref.duplex == "TDD") & (ref.model == model)].copy()
    ref["ds"] = ref["ds"].astype(float).round(10)
    ref["nmse"] = ref["nmse_mean"].astype(float)
    if slice_name == "regular":
        part = ref[ref.split == "regular"]
    elif slice_name == "robustness":
        part = ref[ref.split == "robustness"]
    else:
        part = ref[ref.split == "generalization"]
        part = part[part.ds.isin(DS_SHIFT) & part.ms.isin(MS_SHIFT)]
        part = part[part.cm.isin(["B", "E"] if slice_name == "shift B/E"
                                 else ["A", "C", "D"])]
    key = ["cm", "ds", "ms"]
    if slice_name == "robustness":
        key.append("noise_type")
    return part.groupby(key, as_index=False)["nmse"].mean()


def paired_against(path: Path, slice_name: str, rival: str) -> dict:
    ours = _ours_frame(path, slice_name)
    theirs = _ref_frame(slice_name, rival)
    key = ["cm", "ds", "ms"] + (["noise_type"] if slice_name == "robustness" else [])
    merged = ours.merge(theirs, on=key, suffixes=("_ours", "_ref"))
    diff = (merged["nmse_ref"] - merged["nmse_ours"]).to_numpy()
    rng = np.random.default_rng(20260913)
    idx = rng.integers(0, len(diff), size=(5000, len(diff)))
    draws = diff[idx].mean(axis=1)
    n = len(diff)
    better = int((diff > 0).sum())
    k = max(better, n - better)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n": n, "mean_diff": round(float(diff.mean()), 4),
            "ci_low": round(float(np.quantile(draws, 0.025)), 4),
            "ci_high": round(float(np.quantile(draws, 0.975)), 4),
            "ours_better": better,
            "sign_p": round(float(min(1.0, 2 * tail)), 8)}


def main() -> int:
    zoo = ref_slices()
    ours = {"model": "ours (DD-Koopman)"}
    for name, paths in OURS.items():
        values = [our_slice(p, name) for p in paths if p.exists()]
        ours[name] = round(float(np.mean(values)), 4) if values else np.nan
    zoo = pd.concat([pd.DataFrame([ours]), zoo], ignore_index=True)
    zoo = zoo.sort_values("regular").reset_index(drop=True)
    zoo.to_csv(ANA / "baseline_zoo.csv", index=False)
    print(zoo.to_string(index=False))
    print(f"\nwrote {ANA / 'baseline_zoo.csv'}")

    # paired, per-setting comparison against the two closest rivals
    for rival in ("MODEL", "LLM4CP"):
        print(f"\npaired vs {rival} (ours = seed-42 run on the same settings):")
        for name, paths in OURS.items():
            if not paths[0].exists():
                continue
            stats = paired_against(paths[0], name, rival)
            print(f"  {name:12s}: n={stats['n']:4d} "
                  f"diff {stats['mean_diff']:+.4f} "
                  f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] "
                  f"ours better {stats['ours_better']}/{stats['n']} "
                  f"(sign p={stats['sign_p']:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
