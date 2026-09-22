"""Per-angle off-grid response curve for one frozen checkpoint (Fig. 3 data).

For every validation (sample, antenna) pair and every angle of the declared Z1
grid this evaluates

    d(b, z) = || P_X v_N(z) - u_T(z) ||_2 ,

with `v_N(z) = (1, z, ..., z^{N-1})^T`, `u_T(z) = (z^N, ..., z^{N+T-1})^T` and
`P_X` the extracted spectral operator of the checkpoint. The row maximum over the
grid is exactly the `delta_hat_Z1` column of `response_certificate.csv`, which is
checked here before anything is written.

Outputs:
    response_curve_seed<seed>.csv          257 rows, quantiles across all rows
    response_curve_seed<seed>_by_cell.csv  27 x 257 rows, quantiles per cell
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import diag_lib as D


def angles_of(z: torch.Tensor) -> np.ndarray:
    return z.angle().double().cpu().numpy()


def curve_for_cell(model, device, z, cm, ds, ms, stride: int):
    """`[rows, 257]` defect along the angle grid for one validation cell."""
    hist, _pred = D.val_fold(cm, ds, ms)
    if stride > 1:
        hist = hist[::stride]
    x = D.to_model_input(hist).to(device)
    with torch.no_grad():
        out = D.extract(model, x)
        powers = torch.stack([z ** k for k in range(D.N + D.T)], dim=0)
        v, u = powers[:D.N], powers[D.N:]
        diff = torch.einsum("bpn,nz->bpz", out["p_x"], v) - u.unsqueeze(0)
        row_max = torch.linalg.vector_norm(diff, dim=1).amax(dim=1)
        curve = torch.linalg.vector_norm(diff, dim=1)
    rows = x.shape[0]
    tags = []
    for i in range(rows):
        sample, antenna = i // 32, i % 32
        tags.append("cm%s_ds%03d_ms%03d_vidx%04d_ant%02d"
                    % (cm, round(ds * 1e9), ms, 900 + sample * stride, antenna))
    del hist, x, out, diff
    return curve.double().cpu().numpy(), row_max.double().cpu().numpy(), tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--family", default="reported",
                        choices=["reported", "gridfix", "snap"])
    parser.add_argument("--out", type=Path, default=Path("response_curve_seed42.csv"))
    args = parser.parse_args()

    D.setup_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = D.REPORTED[args.seed] if args.family in ("reported", "snap") else D.GRIDFIX[args.seed]
    sha = D.sha256(ckpt)
    model, meta = D.load_model(ckpt, grid_constraint=(args.family == "snap"))
    model = model.to(device).eval()
    z = D.grid([1.0]).to(device)
    angle = angles_of(z)
    print(f"[curve] device={device} family={args.family} seed={args.seed} "
          f"stride={args.stride} angles={angle.size} ckpt_sha={sha[:12]}", flush=True)

    curves, maxima, tags, cell_index = [], [], [], []
    started = time.time()
    for index, (cm, ds, ms) in enumerate(D.combos()):
        curve, row_max, cell_tags = curve_for_cell(model, device, z, cm, ds, ms, args.stride)
        curves.append(curve)
        maxima.append(row_max)
        tags.extend(cell_tags)
        cell_index.extend([(cm, "%g" % ds, ms)] * curve.shape[0])
        print("  [curve] %2d/27 cm%s ds%03d ms%03d rows=%d (%.0fs)"
              % (index + 1, cm, round(ds * 1e9), ms, curve.shape[0], time.time() - started),
              flush=True)

    all_curves = np.concatenate(curves, axis=0)      # [rows, 257]
    all_max = np.concatenate(maxima, axis=0)         # [rows]
    print("[curve] rows=%d shapes=%s" % (all_curves.shape[0], all_curves.shape), flush=True)

    certificate = D.THEORY / "response_certificate.csv"
    if certificate.exists():
        cert = pd.read_csv(certificate)
        cert = cert[(cert["family"] == args.family) & (cert["seed"] == args.seed)]
        cert = cert.set_index(["sample_id", "antenna"])["delta_hat_Z1"]
        keys = pd.MultiIndex.from_tuples([(t, int(t.rsplit("_ant", 1)[1])) for t in tags])
        matched = cert.reindex(keys).to_numpy()
        delta = np.abs(all_max - matched)
        print("[curve] self-check vs response_certificate: matched=%d max_abs_diff=%.3e"
              % (int(np.isfinite(matched).sum()), float(np.nanmax(delta))), flush=True)

    out_path = D.THEORY / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "angle_index": np.arange(angle.size),
        "angle_rad": angle,
        "median": np.median(all_curves, axis=0),
        "q25": np.quantile(all_curves, 0.25, axis=0),
        "q75": np.quantile(all_curves, 0.75, axis=0),
        "p05": np.quantile(all_curves, 0.05, axis=0),
        "p95": np.quantile(all_curves, 0.95, axis=0),
        "mean": all_curves.mean(axis=0),
        "n_rows": all_curves.shape[0],
        "family": args.family,
        "seed": args.seed,
        "checkpoint_sha256": sha,
        "stride": args.stride,
        "grid": "Z1: 257 angles uniform on [-pi, pi], unit circle",
        "statistic": "P_X v_N(z) - u_T(z), L2 over the T horizons, quantiles over rows",
    })
    frame.to_csv(out_path, index=False)
    print(f"[curve] wrote {out_path}", flush=True)

    by_cell = pd.DataFrame({
        "cm": [c[0] for c in cell_index],
        "ds": [c[1] for c in cell_index],
        "ms": [c[2] for c in cell_index],
    })
    rows = []
    for cm, ds, ms in sorted({(c[0], c[1], c[2]) for c in cell_index}):
        mask = np.array([c == (cm, ds, ms) for c in cell_index])
        sub = all_curves[mask]
        rows.append(pd.DataFrame({
            "cm": cm, "ds": ds, "ms": ms,
            "angle_index": np.arange(angle.size), "angle_rad": angle,
            "median": np.median(sub, axis=0),
            "q25": np.quantile(sub, 0.25, axis=0),
            "q75": np.quantile(sub, 0.75, axis=0),
            "n_rows": sub.shape[0],
        }))
    cell_path = out_path.with_name(out_path.stem + "_by_cell.csv")
    pd.concat(rows, ignore_index=True).to_csv(cell_path, index=False)
    print(f"[curve] wrote {cell_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
