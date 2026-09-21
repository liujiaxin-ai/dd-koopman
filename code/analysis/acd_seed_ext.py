"""Extend the A/C/D shift evaluation from one seed to the three reported seeds.

The first A/C/D result (`results/run_csv/CAPNOAUX600_gen_acd.csv`, seed 42)
already separates the two models by a wide margin (108 cells, 107 favoured).
The paper reports three seeds everywhere else, so this driver re-runs the very
same slice for the seed-43 and seed-44 checkpoints, which are both available in
the recovered archives. Each run installs its checkpoint into the harness
weight slot, evaluates, and restores the slot, so nothing else in the workspace
changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(os.environ.get("CSI_HARNESS", "../CSI-4CAST-main"))
SLOT = HARNESS / "z_artifacts" / "weights" / "tdd" / "dd_koop_tdd" / "model.ckpt"
RUNS = ROOT / "results" / "run_csv"
BACKUP = RUNS / "_slot_backup_acd_seedext.ckpt"
LOG = ROOT / "results" / "analysis" / "acd_seed_ext.log"

CM = "A,C,D"
DS = "5e-08,2e-07,4e-07"
MS = "3,6,9,12,15,18,21,24,27,33,36,42"

CANDIDATES = {
    "CAPNOAUXS43": ROOT / "remote_recovery/host1/z_artifacts/outputs/sweep",
    "HEADNOAUXS44": ROOT / "remote_recovery/host2/checkpoints/z_artifacts/outputs/sweep",
}


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def best_checkpoint(stem: str) -> Path | None:
    """Lowest-val_loss epoch checkpoint of that run, else its last checkpoint."""
    root = CANDIDATES[stem]
    if not root.exists():
        return None
    scored: list[tuple[float, Path]] = []
    for path in root.glob(f"*{stem}*/*/ckpts/*.ckpt"):
        # filenames look like `epoch=211-step=161120-val_loss=0.142389.ckpt`;
        # the number must stop before the extension dot
        match = re.search(r"val_loss=([0-9]+\.[0-9]+)", path.name)
        if match and path.name.startswith("epoch="):
            scored.append((float(match.group(1)), path))
    if scored:
        return min(scored, key=lambda item: item[0])[1]
    lasts = sorted(root.glob(f"*{stem}*/*/ckpts/last.ckpt"))
    return lasts[-1] if lasts else None


def main() -> int:
    log("start")
    if not SLOT.exists():
        log(f"weight slot missing: {SLOT}")
        return 1
    shutil.copy2(SLOT, BACKUP)
    failed = 0
    for stem in CANDIDATES:
        out = RUNS / f"{stem}_gen_acd.csv"
        if out.exists() and out.stat().st_size > 10_000:
            log(f"{stem}: already evaluated ({out.name})")
            continue
        ckpt = best_checkpoint(stem)
        if ckpt is None:
            log(f"{stem}: no checkpoint found")
            failed += 1
            continue
        log(f"{stem}: checkpoint {ckpt.name}")
        shutil.copy2(ckpt, SLOT)
        scratch = out.with_suffix(f".{os.getpid()}.csv")
        cmd = [sys.executable, "eval_ours.py", "--model", "DD_KOOP_TDD",
               "--duplex", "TDD", "--test-type", "generalization",
               "--limit", "0", "--cm", CM, "--ds", DS, "--ms", MS,
               "--out", str(scratch)]
        env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
        proc = subprocess.run(cmd, cwd=str(HARNESS), env=env)
        if proc.returncode == 0 and scratch.exists():
            scratch.replace(out)
            log(f"{stem}: ok -> {out.name}")
        else:
            failed += 1
            log(f"{stem}: eval failed with exit {proc.returncode}")
    shutil.copy2(BACKUP, SLOT)
    log(f"done (failed={failed}), weight slot restored")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
