"""Local profile of every model in the comparison set (RTX 5060 Laptop).

Why this exists next to `profile_models.py`: the remote 4090 instance that
produced the original efficiency table has been destroyed, so the numbers in it
have no surviving artifact. This script re-measures parameters and MACs per
sample on whatever device is available, which is what the handover needs, and
labels the device in the output so nobody confuses a laptop-with-a-laptop
latency with a 4090 latency.

MACs per sample and parameter counts are device independent (defined by the
arithmetic, not the wall clock); latency is not, and is reported with the device
name so it can be replaced later.

Run from a repo root that has the project's model modules installed:
    python profile_local.py --out profile.txt <config.yaml> ...
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

import src.cp.models.ours.dd_koop_v6  # noqa: F401
import src.cp.models.ours.dd_koop  # noqa: F401
import src.cp.models.ours.dd_koop_v9  # noqa: F401
import src.cp.models.ours.dd_koop_v10  # noqa: F401
import src.cp.models.baseline.mambacsp  # noqa: F401
import src.cp.models.baseline.baselines_suite  # noqa: F401
import src.cp.models.baseline.new_baselines  # noqa: F401

from src.utils.main_utils import make_config
from src.utils.model_utils import load_model

SCENE = 32     # one scene is 32 antennas, flattened into the batch axis
HISTORY = 16
FEATURES = 600


def profile(config_path: Path, repeats: int) -> str:
    """Return one formatted result line for a configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = make_config(str(config_path))
    wrapper = load_model(config, device=device)
    network = wrapper.model.to(device).eval()
    parameters = sum(p.numel() for p in network.parameters())
    batch = SCENE
    x = torch.randn(batch, HISTORY, FEATURES, device=device)

    with torch.no_grad():
        for _ in range(3):
            network(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            network(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    macs = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with counter:
            with torch.no_grad():
                network(x)
        macs = counter.get_total_flops() / 2.0 / batch
    except Exception as error:  # noqa: BLE001
        macs = float("nan")
        print(f"    (flop counter unavailable: {type(error).__name__})")

    return (f"{config.model_name or config_path.stem:16s} "
            f"params={parameters:>10,d}  MACs/sample={macs:>14,.0f}  "
            f"median={statistics.median(timings):7.2f} ms  "
            f"p90={sorted(timings)[min(int(0.9 * repeats), len(timings) - 1)]:7.2f} ms  "
            f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")


def main() -> int:
    """Profile every configuration given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(f"batch = one scene = {SCENE} antennas, history {HISTORY}, {FEATURES} features")
    for raw in args.configs:
        path = Path(raw)
        if not path.exists():
            print(f"{path.stem:16s} config not found; skipped")
            continue
        try:
            print(profile(path, args.repeats), flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"{path.stem:16s} failed: {type(error).__name__}: {error}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
