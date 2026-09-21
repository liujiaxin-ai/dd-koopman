"""Queue the beamforming-rate evaluation for both models on the local 5060.

Waits for `local_gpu_queue.py` to finish (they share the harness weight slots),
then evaluates the system-level metric for

  1. the reported operator (CAPNOAUX600 installed in the dd_koop_tdd slot), and
  2. the published 21.9M model (the recovered registry checkpoint installed in
     the plain `model` slot it is registered under),

over cm A/C/D x ds 30/100/300 ns x ms 1/30 x SNR 0..25 (108 settings each),
and merges the two tables into results/analysis/rate_comparison.csv.
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
WEIGHTS = HARNESS / "z_artifacts" / "weights" / "tdd"
OURS_SLOT = WEIGHTS / "dd_koop_tdd" / "model.ckpt"
PUB_SLOT = WEIGHTS / "model" / "model.ckpt"
PUB_CKPT = ROOT / "remote_recovery/host1/registry/registry_backup_model_102238.ckpt"
ANA = ROOT / "results" / "analysis"
LOG = ANA / "rate_queue.log"
CKPT_GLOB = ("remote_recovery/host1/z_artifacts/outputs/sweep/"
             "CAPLONG_CAPNOAUX600/*_*/ckpts/epoch=*.ckpt")
ENV = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
GRID = ["--cm", "A,C,D", "--ds", "3e-08,1e-07,3e-07", "--ms", "1,30",
        "--snr", "0,5,10,15,20,25"]


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wait_for_queue() -> None:
    log("waiting for local_gpu_queue to finish")
    marker = ANA / "local_gpu_queue.log"
    while True:
        if marker.exists() and "queue done" in marker.read_text(encoding="utf-8"):
            log("predecessor finished")
            return
        time.sleep(30)


def run_rate(model: str, out: Path) -> bool:
    cmd = [sys.executable, "rate_eval.py", "--model", model, "--out", str(out)] + GRID
    log(f"{model}: evaluating rate on {out.name}")
    proc = subprocess.run(cmd, cwd=str(HARNESS), env=ENV)
    ok = proc.returncode == 0 and out.exists()
    log(f"{model}: {'ok' if ok else 'FAILED'}")
    return ok


def main() -> int:
    wait_for_queue()
    candidates = sorted(ROOT.glob(CKPT_GLOB))
    if not candidates:
        log("CAPNOAUX600 checkpoint not found")
        return 1
    ours_ckpt = min(candidates, key=lambda p: float(
        p.name.split("val_loss=")[1].split(".ckpt")[0]))
    ours_backup = ANA / "_slot_backup_rate_ours.ckpt"
    pub_backup = ANA / "_slot_backup_rate_pub.ckpt"
    failures = []

    shutil.copy2(OURS_SLOT, ours_backup)
    shutil.copy2(ours_ckpt, OURS_SLOT)
    if not run_rate("DD_KOOP_TDD", ANA / "rate_ours.csv"):
        failures.append("ours")
    shutil.copy2(ours_backup, OURS_SLOT)

    if not PUB_CKPT.exists():
        log(f"published checkpoint missing: {PUB_CKPT}")
        failures.append("published-checkpoint")
    else:
        shutil.copy2(PUB_SLOT, pub_backup)
        shutil.copy2(PUB_CKPT, PUB_SLOT)
        if not run_rate("MODEL", ANA / "rate_published.csv"):
            failures.append("published")
        shutil.copy2(pub_backup, PUB_SLOT)

    ours_csv = ANA / "rate_ours.csv"
    pub_csv = ANA / "rate_published.csv"
    if ours_csv.exists() and pub_csv.exists():
        import pandas as pd
        ours = pd.read_csv(ours_csv).rename(
            columns={"rate_model": "rate_ours", "loss_pct": "loss_ours_pct"})
        pub = pd.read_csv(pub_csv).rename(
            columns={"rate_model": "rate_published",
                     "loss_pct": "loss_published_pct"})
        merged = ours.merge(
            pub[["cm", "ds", "ms", "snr", "rate_published", "loss_published_pct"]],
            on=["cm", "ds", "ms", "snr"], how="inner")
        keep = ["cm", "ds", "ms", "snr", "rate_perfect", "rate_ours",
                "rate_published", "loss_ours_pct", "loss_published_pct", "n"]
        merged = merged[keep]
        merged.to_csv(ANA / "rate_comparison.csv", index=False)
        log("wrote rate_comparison.csv; mean losses: "
            f"ours {merged['loss_ours_pct'].mean():.3f}% "
            f"published {merged['loss_published_pct'].mean():.3f}%")
    log(f"done; failures={failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
