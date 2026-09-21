"""Run the official CSI-4CAST evaluation harness on official checkpoints.

This deliberately calls the repository's own ``test_unit`` so that data loading,
normalization, noise injection, denormalization and metric computation are the
published code paths. It exists to answer one question before any modelling
work starts: does the local/remote setup reproduce the shipped numbers?

Run from the repository root, for example::

    python repro_official.py --model MODEL --duplex TDD --test-type regular --limit 6

The output CSV has the same columns as the official ``result.csv`` files, so it
can be diffed against the reference table field by field.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from src.cp.loss.loss import MSELoss, NMSELoss, SELoss
from src.noise.noise_testing import Noise
from src.testing.config import BATCH_SIZE, create_combinations_for_setting
from src.testing.get_models import get_eval_model, wrap_model_with_imputer
from src.testing.prediction_performance.test_unit import test_unit
from src.utils.dirs import DIR_DATA

def select_combinations(
    duplex: str,
    test_type: str,
    limit: int | None,
    cm: str | None,
    ds: float | None,
    ms: int | None,
    snr: float | None,
) -> list[tuple]:
    combos = create_combinations_for_setting(duplex, test_type)
    if cm is not None:
        combos = [c for c in combos if c[4] == cm]
    if ds is not None:
        combos = [c for c in combos if abs(c[5] - ds) < 1e-12]
    if ms is not None:
        combos = [c for c in combos if c[6] == ms]
    if snr is not None:
        combos = [c for c in combos if float(c[3]) == float(snr)]
    return combos[:limit] if limit else combos

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="MODEL")
    ap.add_argument("--duplex", default="TDD", choices=["TDD", "FDD"])
    ap.add_argument("--test-type", default="regular", choices=["regular", "robustness", "generalization"])
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--cm", default=None)
    ap.add_argument("--ds", type=float, default=None)
    ap.add_argument("--ms", type=int, default=None)
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="override the official BATCH_SIZE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("z_artifacts/outputs/repro/result.csv"))
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    batch_size = args.batch_size or BATCH_SIZE
    print(f"[repro] model={args.model} duplex={args.duplex} split={args.test_type}")
    print(f"[repro] device={device} batch_size={batch_size} cwd={Path.cwd()}")

    combos = select_combinations(args.duplex, args.test_type, args.limit, args.cm, args.ds, args.ms, args.snr)
    if not combos:
        print("[repro] no combinations matched")
        return 2
    print(f"[repro] evaluating {len(combos)} combinations")

    criterion_nmse = NMSELoss().to(device)
    criterion_mse = MSELoss().to(device)
    criterion_se = SELoss(SNR=10).to(device)
    noise = Noise()

    model = get_eval_model(model_name=args.model, device=device, scenario=args.duplex)
    model_imputer = wrap_model_with_imputer(model)
    print(f"[repro] model loaded: name={getattr(model, 'name', '?')} separate_antennas={getattr(model, 'is_separate_antennas', '?')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)

    started = time.time()
    for index, (_scenario, is_gen, noise_type, nd, cm, ds, ms) in enumerate(combos, start=1):
        current = model_imputer if noise_type == "packagedrop" else model
        current.to(device).eval()
        t0 = time.time()
        test_unit(
            scenario=args.duplex,
            is_gen=is_gen,
            is_U2D=args.duplex == "FDD",
            dir_data=Path(DIR_DATA),
            batch_size=batch_size,
            device=device,
            list_models=[current],
            criterion_nmse=criterion_nmse,
            criterion_mse=criterion_mse,
            criterion_se=criterion_se,
            cm=cm,
            ds=ds,
            ms=ms,
            noise_type=noise_type,
            noise_func=getattr(noise, noise_type),
            noise_degree=nd,
            df_path=args.out,
        )
        current.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[repro] {index}/{len(combos)} cm={cm} ds={ds:g} ms={ms:g} "
            f"noise={noise_type} deg={nd}  {time.time() - t0:.1f}s",
            flush=True,
        )

    df = pd.read_csv(args.out)
    print(f"[repro] wrote {len(df)} rows to {args.out} in {time.time() - started:.1f}s")
    print(df[["model", "cm", "ds", "ms", "noise_type", "noise_degree", "nmse_mean"]].to_string(index=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
