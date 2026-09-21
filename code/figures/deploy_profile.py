"""Reduced-precision / compiled inference profile for the reported model.

The efficiency claim is currently one FP32 number on one GPU. This script adds
the deployment axis honestly, in this order:

    1. FP32 CUDA latency (the reference protocol, 3 warm-ups + 12 repeats);
    2. BF16 and FP16 CUDA latency, with an NMSE drift check against FP32;
    3. torch.compile latency (if the installed torch supports it);
    4. INT8 dynamic quantization - attempted, and reported as unsupported if the
       complex-valued / FFT operators reject it (which is the expected outcome
       for this architecture; the script never fakes a quantized number).

Every latency is measured on one 32-antenna scene (batch 32), matching
results/analysis/profiles_*.txt.

Usage (repo root, GPU host):
    python deploy_profile.py --out /root/rivermind-data/csi4cast/deploy_3090.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# machine-local default; set CSI4CAST_ROOT (or pass --out) on another host
HARNESS = Path(os.environ.get("CSI4CAST_ROOT", "/root/rivermind-data/csi4cast/code/CSI-4CAST-main"))
sys.path.insert(0, str(HARNESS))
os.chdir(HARNESS)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import src.cp.models.ours.dd_koop  # noqa: E402,F401
from src.cp.models import PREDICTORS  # noqa: E402
from src.testing.get_models import get_eval_model  # noqa: E402
from src.utils.data_utils import load_data  # noqa: E402
from src.utils.dirs import DIR_DATA  # noqa: E402
from src.utils.norm_utils import normalize_input  # noqa: E402


def timed(model, x: torch.Tensor, warmup: int = 3, repeats: int = 12,
          sync: bool = True) -> tuple[float, float]:
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    if sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times = np.asarray(times)
    return float(np.median(times)), float(times.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/root/rivermind-data/csi4cast/deploy_profile.txt")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    model = get_eval_model(model_name="DD_KOOP_TDD", device=device, scenario="TDD").eval()
    params = sum(p.numel() for p in model.parameters())

    # Latency is a function of shapes, not values, so the input follows the
    # exact convention of `profile_models.py` (the source of the paper's
    # latency table): a float tensor [antennas, history, 2*subcarriers].
    # Route through `load_data` instead and the tensor stays complex, which the
    # denoiser's real-valued convs reject ("Input type (c10::complex<float>)
    # and bias type (float)") - that crash is why this profile runs on synthetic
    # input.
    x = torch.randn(args.batch, 16, 600, device=device)

    lines = [f"# reduced-precision deployment profile",
             f"# device: {gpu} | params: {params:,} | batch: {args.batch} antennas",
             f"# protocol: eval mode, 3 warm-up calls, 12 timed repeats, "
             f"forward only"]

    with torch.no_grad():
        ref = model(x).float().cpu()
    med, mean = timed(model, x)
    lines.append(f"fp32            median={med:6.2f} ms  mean={mean:6.2f} ms")
    print(lines[-1], flush=True)

    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        try:
            half = model.to(dtype=dtype)
            xh = x.to(dtype=dtype)
            with torch.no_grad():
                out = half(xh).float().cpu()
            drift = float((out - ref).abs().mean() / ref.abs().mean())
            med, mean = timed(half, xh)
            lines.append(f"{name:9s}       median={med:6.2f} ms  mean={mean:6.2f} ms"
                         f"  rel.drift={drift:.2e}")
            print(lines[-1], flush=True)
            model = model.float()
        except Exception as exc:                       # pragma: no cover
            lines.append(f"{name:9s}       unsupported: {type(exc).__name__}: {exc}")
            print(lines[-1], flush=True)
            model = model.float()

    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(x)
        med, mean = timed(compiled, x)
        lines.append(f"compiled      median={med:6.2f} ms  mean={mean:6.2f} ms")
    except Exception as exc:                           # pragma: no cover
        lines.append(f"compiled      unsupported: {type(exc).__name__}: {exc}")
    print(lines[-1], flush=True)

    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            model.cpu(), {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8)
        with torch.no_grad():
            quantized(x.cpu())
        med, mean = timed(quantized, x.cpu(), sync=False)
        lines.append(f"int8-dynamic  median={med:6.2f} ms  mean={mean:6.2f} ms (CPU)")
    except Exception as exc:
        lines.append(f"int8-dynamic  unsupported: {type(exc).__name__}: {exc}")
    print(lines[-1], flush=True)

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[deploy] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
