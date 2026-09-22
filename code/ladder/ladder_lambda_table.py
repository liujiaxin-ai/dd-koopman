"""Validation-set NMSE for each fitted gain matrix (delivery summary only).

Cheap relative comparison of the fits: every `.npz` produced by
`ladder_lsgains_fit.py` is applied to the cached validation fold (27 cells x the
last 100 samples of the training permutation) with the same coordinate
convention as the harness, and the per-cell NMSE is averaged.

The numbers are for the delivery summary, not for the paper: the split here is
the training split's validation fold, not the regular test split, and the
comparison is on normalised data.

Usage::

    python ladder_lambda_table.py --gains <dir> --cache $LADDER_CACHE \
        --out lambda_table.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import diag_lib as D


def predict(model_input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LSGAINS forward, matched to the harness coordinate convention."""
    delay = torch.fft.ifft(model_input, dim=-1)
    spectrum = torch.fft.fft(delay, dim=-2)
    future = torch.einsum("pm,baml->bapl", weight, spectrum)
    return torch.fft.fft(future, dim=-1)


def nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # squared magnitude, matching the harness NMSELoss on complex tensors
    error = (pred - target).abs().pow(2).flatten(1).sum(-1)
    power = target.abs().pow(2).flatten(1).sum(-1) + 1e-12
    return error / power


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gains", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    D.setup_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for path in sorted(args.gains.glob("W_*.npz")):
        payload = np.load(path)
        weight = torch.from_numpy(
            np.asarray(payload["W_fp64"], dtype=np.complex128)
        ).to(device).to(torch.complex64)
        per_cell = []
        for cm, ds, ms in D.combos():
            hist, pred = D.val_fold(cm, ds, ms)
            # `is_separate_antennas` is False for this baseline, so the model sees
            # the complex subcarrier-domain history directly: [n, A, T, K].
            with torch.no_grad():
                y = predict(hist.to(device), weight)
                per_cell.append(float(nmse(y, pred.to(device)).mean()))
            del y, hist, pred
        rows.append({
            "fit": path.stem.replace("W_", ""),
            "path": str(path),
            "ridge": float(payload["ridge"]) if "ridge" in payload else float("nan"),
            "val_cell_mean_nmse": float(np.mean(per_cell)),
            "val_cells": len(per_cell),
        })
        print(f"{rows[-1]['fit']:22s} ridge={rows[-1]['ridge']:g} "
              f"val_cell_mean_nmse={rows[-1]['val_cell_mean_nmse']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
