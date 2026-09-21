"""Params and MACs for the three capacity-sweep points (CPU only).

The capacity sweep on the training host varies only two widths of the reported
operator (`hidden`, `arl_hidden`). Parameter and MAC counts are architecture
properties, so they can be computed locally on the CPU without touching the
GPU, which matters because the sweep is training the very models we profile.

Reads the reported architecture config from the local harness and writes
results/analysis/capacity_profiles.csv with the same two columns the paper's
main table expects (params, macs_per_sample).

Run from the harness root with the harness python:
    python ../icassp-paper/code/analysis/profile_capacity_cpu.py
"""

from __future__ import annotations

import csv
import re
import os
from pathlib import Path

import torch

import src.cp.models.ours.dd_koop  # noqa: F401  (registers DD_KOOP_TDD)
from src.utils.main_utils import make_config
from src.utils.model_utils import load_model

HARNESS = Path(os.environ.get("CSI_HARNESS", "../CSI-4CAST-main"))
BASE = HARNESS / "z_artifacts" / "config" / "ours" / "sweep" / "tdd_capfull.yaml"
TMP = HARNESS / "z_artifacts" / "config" / "ours" / "sweep" / "_profile_tmp"
OUT = ROOT / "results" / "analysis" / "capacity_profiles.csv"

POINTS = [
    ("CAPACITY_TINY", 16, 32),
    ("CAPACITY_XS", 24, 48),
    ("CAPACITY_S", 32, 64),
    ("CAPACITY_M", 96, 192),
    ("CAPACITY_L", 128, 256),
]
SCENE = 32
HISTORY = 16
FEATURES = 600


def variant(name: str, hidden: int, arl: int) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    text = BASE.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^(\s*hidden:\s*)\d+", rf"\g<1>{hidden}", text)
    text = re.sub(r"(?m)^(\s*arl_hidden:\s*)\d+", rf"\g<1>{arl}", text)
    text = re.sub(r"(?m)^(experiment_name:\s*).*$",
                  rf"\g<1>PROFILE_{name}", text)
    text = re.sub(r"(?m)^(output_dir:\s*).*$",
                  rf"\g<1>z_artifacts/outputs/profile_tmp/{name}", text)
    path = TMP / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    device = torch.device("cpu")
    rows = []
    for name, hidden, arl in POINTS:
        config = make_config(str(variant(name, hidden, arl)))
        wrapper = load_model(config, device=device)
        network = wrapper.model.to(device).eval()
        params = sum(p.numel() for p in network.parameters())
        batch = SCENE
        x = torch.randn(batch, HISTORY, FEATURES, device=device)
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as counter:
                with torch.no_grad():
                    network(x)
            macs = counter.get_total_flops() / 2.0 / batch
        except Exception as error:  # noqa: BLE001
            print(f"  {name}: flop counter failed ({type(error).__name__})")
            macs = float("nan")
        rows.append({"label": name, "hidden": hidden, "arl_hidden": arl,
                     "params": int(params), "macs_per_sample": int(macs)
                     if macs == macs else ""})
        print(f"{name:12s} hidden={hidden:3d} arl={arl:3d} "
              f"params={params:,}  MACs/sample={macs:,.0f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
