"""Efficiency profile for every model in the comparison set.

A lightweight paper is expected to report more than a parameter count, and this
project reports none of the rest. This measures, for each registered model on the
same batch shape the harness uses:

  * parameters
  * multiply-accumulates per sample (via `torch.utils.flop_counter`)
  * median and p90 forward latency per 32-antenna scene
  * peak GPU memory for a forward+backward step

Timing is deliberately reported as a distribution rather than a single number,
because the first call includes kernel autotuning and would flatter whichever
model happens to run first.

Usage (remote, repo root, venv python):
    python profile_models.py --repeats 30 \
        z_artifacts/config/ours/sweep/tdd_s49k300.yaml \
        z_artifacts/config/ours/sweep/tdd_base_gru.yaml ...
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

import src.cp.models.ours.dd_koop
import src.cp.models.baseline.baselines_suite
import src.cp.models.baseline.mambacsp
import src.cp.models.baseline.new_baselines

from src.utils.main_utils import make_config
from src.utils.model_utils import load_model

SCENE = 32
HISTORY = 16
FEATURES = 600

def profile(config_path: Path, reference_scenes: int = 1, repeats: int = 30) -> None:
    device = torch.device("cuda")

    config = make_config(str(config_path))
    try:
        wrapper = load_model(config, device=device)
    except Exception as error:
        print(f"{config_path.stem:16s} failed to build: {type(error).__name__}: {error}")
        return
    network = wrapper.model
    name = config.model_name or config_path.stem
    network = network.to(device).eval()

    parameters = sum(p.numel() for p in network.parameters())
    batch = reference_scenes * SCENE
    x = torch.randn(batch, HISTORY, FEATURES, device=device)

    with torch.no_grad():
        for _ in range(3):
            network(x)
        torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            network(x)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as counter:
            with torch.no_grad():
                network(x)
        flops = counter.get_total_flops()
        macs = flops / 2.0 / batch
    except Exception as error:
        macs = float("nan")
        print(f"    (flop counter unavailable: {type(error).__name__})")

    torch.cuda.reset_peak_memory_stats()

    network.train()
    y = network(x)
    loss = y.pow(2).mean()
    loss.backward()
    network.eval()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    torch.cuda.empty_cache()

    print(f"{name:16s} params={parameters:>10,d}  MACs/sample={macs:>12,.0f}  "
          f"median={statistics.median(timings):7.1f} ms  p90={sorted(timings)[int(0.9 * repeats)]:7.1f} ms  "
          f"peak_train={peak:6.0f} MiB", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--configs", nargs="+", default=[
        "z_artifacts/config/ours/sweep/tdd_s49k300.yaml",
        "z_artifacts/config/ours/sweep/tdd_base_dlinear.yaml",
        "z_artifacts/config/ours/sweep/tdd_base_mlp.yaml",
        "z_artifacts/config/ours/sweep/tdd_base_gru.yaml",
        "z_artifacts/config/ours/sweep/tdd_base_tcn.yaml",
        "z_artifacts/config/ours/sweep/tdd_base_transformer.yaml",
        "z_artifacts/config/ours/sweep/tdd_base_patchtst.yaml",
        "z_artifacts/config/ours/sweep/tdd_mamba64.yaml",
        "z_artifacts/config/ours/sweep/tdd_official.yaml",
    ])
    args = parser.parse_args()

    print(f"batch = 1 scene = {SCENE} antennas, history {HISTORY}, {FEATURES} features")
    for path in args.configs:
        if Path(path).exists():
            profile(Path(path), repeats=args.repeats)
        else:
            print(f"{Path(path).stem:16s} config not found; skipped")

if __name__ == "__main__":
    main()
