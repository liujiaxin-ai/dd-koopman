"""Score the reported model on the A/C/D generalization slice and compare it
with the published reference rows that already exist for those settings.

Why this exists: the shipped shift slice covers channel models B and E, the two
models absent from training. A reviewer's follow-up is what happens to the
*seen* channel models at unseen delay spreads and speeds - an interpolation-style
shift rather than an unseen-environment shift. The published model's numbers for
exactly those 648 settings are already in the benchmark's reference table, so
only our own model has to be evaluated.

Run from the harness root with the harness python:

    python code/analysis/acd_shift_eval.py --wait

`--wait` blocks until the A/C/D download has placed all combinations, then runs
the harness evaluation and writes the paired table.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(os.environ.get("CSI_HARNESS", "../CSI-4CAST-main"))
GEN_DIR = Path(os.environ.get("CSI_ACD_DATA", "../csi4cast_gen"))
RUNS = ROOT / "results" / "run_csv"
ANA = ROOT / "results" / "analysis"
REFERENCE = ROOT / "results" / "reference" / "official_per_setting.csv"

CHANNEL_MODELS = ("A", "C", "D")
DELAY_SPREADS = (50e-9, 200e-9, 400e-9)
SPEEDS = (3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 42)
BOOTSTRAP = 5000
SEED = 20260913


def expected_folders() -> list[str]:
    return [f"cm_{cm}_ds_{round(ds * 1e9):03d}_ms_{ms:03d}"
            for cm in CHANNEL_MODELS for ds in DELAY_SPREADS for ms in SPEEDS]


def wait_for_data(timeout_s: float = 6 * 3600) -> int:
    folders = expected_folders()
    started = time.time()
    while True:
        missing = [f for f in folders if not (GEN_DIR / f / "H_U_hist.pt").exists()]
        if not missing:
            print(f"[acd] all {len(folders)} combinations present")
            return 0
        if time.time() - started > timeout_s:
            print(f"[acd] timeout with {len(missing)} combinations missing")
            return 1
        print(f"[acd] waiting: {len(folders) - len(missing)}/{len(folders)} "
              f"downloaded (missing {len(missing)})", flush=True)
        time.sleep(120)


def run_harness_eval(out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    speeds = ",".join(str(s) for s in SPEEDS)
    cmd = [
        sys.executable, "eval_ours.py",
        "--model", "DD_KOOP_TDD", "--duplex", "TDD",
        "--test-type", "generalization", "--limit", "0",
        "--cm", ",".join(CHANNEL_MODELS),
        "--ds", ",".join(f"{ds:g}" for ds in DELAY_SPREADS),
        "--ms", speeds,
        "--out", str(out_csv),
    ]
    print(f"[acd] running harness eval: {' '.join(cmd[:6])} ...", flush=True)
    # torch>=2.6 defaults to weights_only=True, which refuses the Lightning
    # checkpoints in this project (OptimizerConfig is not an allowed global).
    # The benchmark's own evaluation scripts therefore require this escape
    # hatch; without it the harness dies with a WeightsUnpickler error.
    child_env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    proc = subprocess.run(cmd, cwd=str(HARNESS), env=child_env)
    if proc.returncode != 0:
        raise SystemExit(f"[acd] harness eval failed with exit {proc.returncode}")
    return out_csv


def cells_from_ours(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The harness writes the six SNR levels of the generalization split as
    # `noise_degree` under `noise_type == "vanilla"`; the reference table calls
    # the same axis `snr`. Accept either so the two sides merge on (cm, ds, ms).
    level = "noise_degree" if "noise_degree" in df.columns else "snr"
    df = df[df["noise_type"] == "vanilla"].drop_duplicates(
        ["cm", "ds", "ms", level])
    # `nmse_mean` is stored as the per-horizon vector; the benchmark's own
    # analysis (build_tables.read_run) reduces it with a plain mean over the
    # four horizons, so the identical reduction is used here.
    df = df.assign(nmse=df["nmse_mean"].map(
        lambda text: float(np.mean([
            float(x) for x in
            re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]))))
    out = (df.groupby(["cm", "ds", "ms"], as_index=False)["nmse"].mean()
             .rename(columns={"nmse": "ours"}))
    return out


def cells_from_reference() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE)
    ref = ref[(ref["split"] == "generalization")
              & (ref["duplex"] == "TDD")   # never mix the FDD rows in
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--ours", default=str(RUNS / "CAPNOAUX600_gen_acd.csv"))
    ap.add_argument("--no-eval", action="store_true",
                    help="reuse an existing ours CSV instead of running the harness")
    args = ap.parse_args()

    if args.wait and wait_for_data() != 0:
        return 1
    ours_csv = Path(args.ours)
    if not args.no_eval or not ours_csv.exists():
        # This host mirrors every python launch with a second process that shares
        # the same job object, so two identical evaluations can be in flight and
        # both would open the same output path. Give each run its own file and
        # publish it atomically at the end, so the final CSV is always one
        # complete table rather than two interleaved writers.
        scratch = ours_csv.with_suffix(f".{os.getpid()}.csv")
        run_harness_eval(scratch)
        scratch.replace(ours_csv)

    ours = cells_from_ours(ours_csv)
    ref = cells_from_reference()
    merged = ours.merge(ref, on=["cm", "ds", "ms"], how="inner")
    merged["diff"] = merged["ours"] - merged["published"]
    merged["speed_band"] = np.where(merged["ms"] <= 9, "le10",
                                    np.where(merged["ms"] <= 27, "10to30", "gt30"))
    print(f"[acd] paired cells: {len(merged)}")

    rows = []
    slices: list[tuple[str, pd.DataFrame]] = [("all", merged)]
    slices += [(f"cm{cm}", merged[merged.cm == cm]) for cm in CHANNEL_MODELS]
    slices += [(f"ds{round(ds * 1e9)}", merged[merged.ds == ds])
               for ds in DELAY_SPREADS]
    slices += [(f"band{band}", merged[merged.speed_band == band])
               for band in ("le10", "10to30", "gt30")]
    for name, part in slices:
        if part.empty:
            continue
        mean, lo, hi, wins = paired_stats(part["diff"].to_numpy())
        rows.append({
            "slice": name, "cells": len(part),
            "ours": round(float(part["ours"].mean()), 4),
            "published": round(float(part["published"].mean()), 4),
            "paired_mean_diff": round(mean, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "ours_favoured_cells": f"{wins}/{len(part)}",
        })
    out = pd.DataFrame(rows)
    out.to_csv(ANA / "generalization_acd.csv", index=False)
    merged.to_csv(ANA / "generalization_acd_by_cell.csv", index=False)
    print(out.to_string(index=False))
    print(f"[acd] wrote {ANA / 'generalization_acd.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
