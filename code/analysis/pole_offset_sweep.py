"""Drive the pole-offset intervention over the official 162-setting grid.

Install the reported checkpoint (CAPNOAUX600, seed 42, loss-off) into the harness
weight slot once and evaluate it five times: stock, and with the pole sub-bin
offset forced to 0.00 / 0.15 / 0.25 / 0.50 bin. The first is the free-fit
reference (0.1385 on the 162 grid); the rest trace how the error responds when
the off-grid freedom is removed. Every other part of the network and the
checkpoint stay identical.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(os.environ.get("CSI_HARNESS", "../CSI-4CAST-main"))
SLOT = HARNESS / "z_artifacts" / "weights" / "tdd" / "dd_koop_tdd" / "model.ckpt"
BACKUP = ROOT / "results" / "run_csv" / "_slot_backup_pole_offset.ckpt"
RUNS = ROOT / "results" / "run_csv"
LOG = ROOT / "results" / "analysis" / "pole_offset_sweep.log"
CKPT_GLOB = ("remote_recovery/host1/z_artifacts/outputs/sweep/CAPLONG_CAPNOAUX600/"
             "*_*/ckpts/epoch=*.ckpt")
OFFSETS = [None, 0.0, 0.15, 0.25, 0.50]


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    candidates = sorted(ROOT.glob(CKPT_GLOB))
    if not candidates:
        log("CAPNOAUX600 checkpoint not found")
        return 1
    best = min(candidates, key=lambda p: float(
        p.name.split("val_loss=")[1].split(".ckpt")[0]))
    log(f"checkpoint {best.name}")
    shutil.copy2(SLOT, BACKUP)
    shutil.copy2(best, SLOT)
    failed = 0
    for offset in OFFSETS:
        tag = "stock" if offset is None else f"off{int(round(offset * 100)):03d}"
        out = RUNS / f"SNAPPOLE_{tag}_full162.csv"
        if out.exists() and out.stat().st_size > 10_000:
            log(f"{tag}: already evaluated")
            continue
        env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
        if offset is not None:
            env["POLE_OFFSET"] = str(offset)
        cmd = [sys.executable, "eval_pole_offset.py", "--model", "DD_KOOP_TDD",
               "--duplex", "TDD", "--test-type", "regular", "--limit", "0",
               "--out", str(out)]
        log(f"{tag}: evaluating")
        proc = subprocess.run(cmd, cwd=str(HARNESS), env=env)
        if proc.returncode == 0 and out.exists():
            log(f"{tag}: ok ({out.stat().st_size // 1024} KB)")
        else:
            failed += 1
            log(f"{tag}: FAILED exit {proc.returncode}")
    shutil.copy2(BACKUP, SLOT)
    log(f"done (failed={failed}); weight slot restored")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
