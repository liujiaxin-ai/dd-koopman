"""Build the validation-fold cache once (27 cells x 100 samples).

Each raw cell is ~1.75 GB on the shared volume; the diagnostics re-read all 27
cells for every (family, seed) pass. Writing the 100-sample validation slices to
one small file per cell cuts that to a single pass, after which every diagnostic
reads 122 MB instead of 1.75 GB.
"""
import os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "<theory-out>")
os.environ.setdefault("DIAG_REMOTE", "1")
import diag_lib as D
D.setup_env()

OUT = Path("<ladder-out>/cache")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for i, (cm, ds, ms) in enumerate(D.combos(), 1):
    blob = OUT / ("cm%s_ds%03d_ms%03d.pt" % (cm, round(ds * 1e9), ms))
    if blob.exists():
        print("[%2d/27] cm%s %03dns ms%03d cached" % (i, cm, round(ds * 1e9), ms), flush=True)
        continue
    hist, pred = D.val_fold(cm, ds, ms)
    torch.save({"hist": hist, "pred": pred}, blob)
    print("[%2d/27] cm%s %03dns ms%03d -> %d samples  (%.0fs)" %
          (i, cm, round(ds * 1e9), ms, hist.shape[0], time.time() - t0), flush=True)
print("cache ready:", OUT)
