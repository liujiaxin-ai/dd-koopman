"""Complete the robustness-486 split for the third reported seed.

Seeds 42 and 43 already have robustness-486 numbers (0.1772 / 0.1735). Seed 44's
loss-off checkpoint lives in the recovered archive and is the same checkpoint
used for the A/C/D slice, so the third seed can be filled in on the local GPU
without training anything. The harness weight slot is backed up and restored.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(os.environ.get("CSI_HARNESS", "../CSI-4CAST-main"))
SLOT = HARNESS / "z_artifacts" / "weights" / "tdd" / "dd_koop_tdd" / "model.ckpt"
OUT = ROOT / "results" / "run_csv" / "HEADNOAUXS44_robust486.csv"
BACKUP = ROOT / "results" / "run_csv" / "_slot_backup_robust_seed44.ckpt"
LOG = ROOT / "results" / "analysis" / "robust_seed44.log"
CKPT_GLOB = ("remote_recovery/host2/checkpoints/z_artifacts/outputs/sweep/"
             "CAPLONG_HEADNOAUXS44/*/ckpts/epoch=*.ckpt")


def log(message: str) -> None:
    # No wall-clock stamp and a log truncated once per run: the file is then a
    # pure function of the inputs, so regenerate_all.py stays byte-idempotent.
    print(message, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    if OUT.exists() and OUT.stat().st_size > 10_000:
        log(f"already evaluated: {OUT.name}")
        return 0
    candidates = sorted(ROOT.glob(CKPT_GLOB))
    if not candidates:
        log("seed-44 checkpoint not found")
        return 1
    best = min(candidates, key=lambda p: float(
        p.name.split("val_loss=")[1].split(".ckpt")[0]))
    log(f"checkpoint {best.name}")
    shutil.copy2(SLOT, BACKUP)
    shutil.copy2(best, SLOT)
    scratch = OUT.with_suffix(f".{os.getpid()}.csv")
    cmd = [sys.executable, "eval_ours.py", "--model", "DD_KOOP_TDD",
           "--duplex", "TDD", "--test-type", "robustness", "--limit", "0",
           "--out", str(scratch)]
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    proc = subprocess.run(cmd, cwd=str(HARNESS), env=env)
    shutil.copy2(BACKUP, SLOT)
    if proc.returncode == 0 and scratch.exists():
        scratch.replace(OUT)
        log(f"ok -> {OUT.name} ({OUT.stat().st_size // 1024} KB)")
        return 0
    log(f"failed with exit {proc.returncode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
