"""Fit the E1 predictor: fixed grid gains by ridge least squares (no training).

The object is exactly the one the model's own warm start fits:

    spectrum[m, l] = sum_n D[n, l] exp(-2 pi i m n / N)      (unnormalised)
    target[p, l]   = future delay profile at step p
    W              = argmin ||W spectrum - target||^2 + ridge * ||W||^2

so `W` is a [T, N] complex matrix shared across antennas and delay taps, with
no conditioning and no gating. Evaluating it isolates "the gains the exactness
constraint imposes" from "the gains the data prefers", which is the E0 -> E1
step of the ladder.

Three fits are produced:

  native   the recipe training actually ran: `H_hist[:256]` / `H_pred[:256]` of
           the packed training tensors, i.e. 256 samples WITH the training AWGN
           draw, ridge = 1e-6 (the default of `least_squares_spectrum_map`);
  full     all 24,300 packed training samples, same noise, same ridge;
  clean    all 24,300 raw training samples without the AWGN draw (diagnostic
           only: it shows how much of the difference is the noise draw).

The packing is reproduced bit for bit by seeding exactly as the training driver
does (`pl.seed_everything(config.seed, workers=True)` before `setup()`), so the
`native` gains are the ones the reported model was warm started with.

Usage::

    python ladder_lsgains_fit.py --config z_artifacts/config/ours/sweep/tdd_caplong.yaml \
        --out-dir <ladder-out>/gains
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import train_sweep
from src.utils.main_utils import make_config

SLOTS = 16
HORIZON = 4


def _chunks(tensor_hist: torch.Tensor, tensor_pred: torch.Tensor, size: int):
    for start in range(0, tensor_hist.shape[0], size):
        yield (tensor_hist[start:start + size], tensor_pred[start:start + size])


def fit_gains(hist: torch.Tensor, pred: torch.Tensor, taps: int, ridge: float,
              chunk: int = 256) -> np.ndarray:
    """Ridge least squares in the warm start's own coordinates, complex128."""
    gram = torch.zeros(SLOTS, SLOTS, dtype=torch.complex128)
    rhs = torch.zeros(SLOTS, HORIZON, dtype=torch.complex128)
    for h, p in _chunks(hist, pred, chunk):
        observed = torch.fft.fft(torch.fft.ifft(h, dim=-1)[..., :taps], dim=2)
        target = torch.fft.ifft(p, dim=-1)[..., :taps]
        design = observed.permute(0, 1, 3, 2).reshape(-1, observed.shape[-2])
        future = target.permute(0, 1, 3, 2).reshape(-1, target.shape[-2])
        design = design.to(torch.complex128)
        future = future.to(torch.complex128)
        gram += design.conj().T @ design
        rhs += design.conj().T @ future
    scale = torch.diagonal(gram).real.sum() / SLOTS
    gram = gram + (ridge * scale + 1e-12) * torch.eye(SLOTS, dtype=torch.complex128)
    weight = torch.linalg.solve(gram, rhs).T.contiguous()      # [HORIZON, SLOTS]
    return weight.numpy()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default="z_artifacts/config/ours/sweep/tdd_caplong.yaml")
    parser.add_argument("--out-dir", type=Path, default=Path("gains"))
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--native-samples", type=int, default=256)
    parser.add_argument("--lambdas", default="1e-07,1e-06,1e-05")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args.config)
    taps = int(config.model.params["num_delay_taps"])
    print(f"[lsgains] config={args.config} taps={taps} ridge={args.ridge}", flush=True)

    started = time.time()
    torch.manual_seed(config.seed)
    train_sweep.pl.seed_everything(config.seed, workers=True)
    torch.set_num_threads(8)
    module = train_sweep.PackOnceDataModule(config.data)
    module.setup()
    hist = module.train_dataset.H_hist
    pred = module.train_dataset.H_pred
    print(f"[lsgains] packed in {time.time() - started:.0f}s: "
          f"hist={tuple(hist.shape)} pred={tuple(pred.shape)}", flush=True)

    records = {}
    for name, sl in (("native", slice(0, args.native_samples)),
                     ("full", slice(None))):
        t0 = time.time()
        weight = fit_gains(hist[sl], pred[sl], taps, args.ridge)
        path = args.out_dir / f"W_{name}.npz"
        np.savez(path, W=weight.astype(np.complex64), W_fp64=weight,
                 ridge=args.ridge, samples=int(hist[sl].shape[0]), taps=taps)
        records[name] = {
            "path": str(path), "sha256": sha256(path),
            "samples": int(hist[sl].shape[0]), "ridge": args.ridge,
            "dtype": "fp64_fit_stored_fp64_and_fp32", "seconds": round(time.time() - t0, 1),
        }
        print(f"[lsgains] {name}: samples={records[name]['samples']} "
              f"{time.time() - t0:.0f}s -> {path}", flush=True)

    for lam in [float(v) for v in args.lambdas.split(",")]:
        weight = fit_gains(hist, pred, taps, lam)
        path = args.out_dir / f"W_full_lambda{lam:g}.npz"
        np.savez(path, W=weight.astype(np.complex64), W_fp64=weight, ridge=lam)
        records[f"full_lambda{lam:g}"] = {"path": str(path), "sha256": sha256(path),
                                          "ridge": lam, "samples": int(hist.shape[0])}
        print(f"[lsgains] lambda={lam:g} -> {path}", flush=True)

    payload = {
        "config": args.config,
        "seed": config.seed,
        "taps": taps,
        "slots": SLOTS,
        "horizon": HORIZON,
        "coordinates": "warm-start coordinates: delay = ifft_unnormalised(x), "
                       "spectrum = fft_unnormalised(delay, time), "
                       "target = ifft_unnormalised(future)",
        "fits": records,
        "total_seconds": round(time.time() - started, 1),
    }
    (args.out_dir / "fit_summary.json").write_text(json.dumps(payload, indent=2),
                                                   encoding="utf-8")
    print(f"[lsgains] wrote {args.out_dir / 'fit_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
