"""Run the V1/V3/V5 diagnostics written up in EXPERIMENT_TODO_THEORY_2026-09-21.md.

    python run_diag.py v1v3 --family reported --seeds 42,43,44
    python run_diag.py v1v3 --family gridfix  --seeds 42,43,44 --stride 10
    python run_diag.py v1v3 --family snap     --seeds 42,43,44 --stride 10

`--stride 10` reproduces the V5 common subset (every tenth validation sample,
270 per seed). Results append to operator_identity.csv and
response_certificate.csv, one row per (validation sample, antenna).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

import diag_lib as D

IDENTITY_FIELDS = ["family", "checkpoint_sha256", "seed", "sample_id", "cm", "ds", "ms",
                   "val_index", "antenna", "rel_A", "rel_A_mode", "rel_A_fp64", "rel_B",
                   "rel_full", "rel_gate", "rel_gain_vs_operator", "g1", "g2", "precision"]
CERT_FIELDS = ["family", "checkpoint_sha256", "seed", "sample_id", "cm", "ds", "ms",
               "val_index", "antenna", "delta_hat_Z1", "defect_upper_Z1", "delta_hat_Z2",
               "defect_upper_Z2", "noise_gain_2", "norm_PX_F", "correction_norm",
               "ratio_corr_over_delta", "rel_A", "precision"]

ETA1 = math.pi / 256
ETA2 = max(math.pi / 256, 0.025)


def append(path: Path, fields: list, rows: list) -> None:
    # Any key the row dict carries but the declared schema does not is appended
    # to the header rather than dropped: a mis-declared column must not be able
    # to discard a finished shard.
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_family(family: str, seeds: list, stride: int, shard: int = 0, nshards: int = 1) -> None:
    """Cell-major: each validation cell is read once and scored for every seed.

    The cells are ~1.75 GB on disk, so reading them once per seed (the obvious
    layout) triples the I/O and the jobs end up I/O-bound; the loop below keeps
    all requested checkpoints resident and walks the 27 cells once.
    """
    D.setup_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z1 = D.grid([1.0]).to(device)
    z2 = D.grid([0.90, 0.95, 0.99, 1.00]).to(device)
    lv1, lu1 = D.l_v(1.0), D.l_u(1.0)

    loaded = {}
    for seed in seeds:
        ckpt = D.REPORTED[seed] if family in ("reported", "snap") else D.GRIDFIX[seed]
        model, meta = D.load_model(ckpt, grid_constraint=(family == "snap"))
        loaded[seed] = (ckpt, D.sha256(ckpt), model.to(device))
        print("[%s s%d] ckpt=%s sha=%s grid_ckpt=%s snap=%s" %
              (family, seed, ckpt.name, D.sha256(ckpt)[:12], meta["grid_from_ckpt"], meta["grid_constraint"]),
              flush=True)

    id_rows, cert_rows = [], []
    v2_rows = []
    V, Q = grid_basis(device)
    P0, PLam, lam = reference_operators(V, Q)
    kt257, kt16 = D.T * V.shape[1] / D.N, D.T * D.N / D.N
    Vr, Qr = roots_basis(device)
    P0r = Qr @ torch.linalg.pinv(Vr)
    check = float(torch.linalg.matrix_norm(P0r @ Vr - Qr, ord=2))       # exact: V square
    ls_res = float(torch.linalg.matrix_norm(P0 @ V - Q, ord=2))         # reported, not asserted
    print("[%s] V2 K=%d lambda=%.6g selfcheck(16 roots)=%.3e ls_residual_K257=%.3e" %
          (family, V.shape[1], lam, check, ls_res), flush=True)
    assert check <= 1e-10
    t0 = time.time()
    for cell_index, (cm, ds, ms) in enumerate(D.combos()):
        if cell_index % nshards != shard:
            continue
        hist, _pred = D.val_fold(cm, ds, ms)
        if stride > 1:
            hist = hist[::stride]
        x = D.to_model_input(hist).to(device)
        for seed in seeds:
            ckpt, ckpt_sha, model = loaded[seed]
            with torch.no_grad():
                out = D.extract(model, x)
                d1 = D.defect(out["p_x"], z1)
                d2 = D.defect(out["p_x"], z2)
                up1 = d1 + ETA1 * (out["norm_px_2"] * lv1 + lu1)
                up2 = d2 + ETA2 * (out["norm_px_2"] * lv1 + lu1)
                px64 = out["p_x"].to(torch.complex128)
                dref0 = torch.linalg.matrix_norm(px64 - P0)
                drefl = torch.linalg.matrix_norm(px64 - PLam)
                f2 = out["norm_px_f"] ** 2
                assert bool(torch.all(up1 >= d1 - 1e-9)) and bool(torch.all(up2 >= d2 - 1e-9))
                for i in range(x.shape[0]):
                    sample, ant = i // 32, i % 32
                    vidx = 900 + sample * stride
                    tag = "cm%s_ds%03d_ms%03d_vidx%04d_ant%02d" % (cm, round(ds * 1e9), ms, vidx, ant)
                    id_rows.append({
                        "family": family, "checkpoint_sha256": ckpt_sha, "seed": seed,
                        "sample_id": tag, "cm": cm, "ds": "%g" % ds, "ms": ms,
                        "val_index": vidx, "antenna": ant,
                        "rel_A": "%.10g" % float(out["rel_a"][i]),
                        "rel_A_mode": "%.10g" % float(out["rel_a_mode"][i]),
                        "rel_A_fp64": "%.10g" % float(out["rel_a_fp64"][i]),
                        "rel_B": "%.10g" % float(out["rel_b"][i]),
                        "rel_full": "%.10g" % float(out["rel_full"][i]),
                        "rel_gate": "%.10g" % float(out["rel_gate"][i]),
                        "rel_gain_vs_operator": "%.10g" % float(out["rel_gain_vs_operator"][i]),
                        "g1": "%.10g" % out["g1"], "g2": "%.10g" % out["g2"],
                        "precision": "fp32+fp64_probe"})
                    cert_rows.append({
                        "family": family, "checkpoint_sha256": ckpt_sha, "seed": seed,
                        "sample_id": tag, "cm": cm, "ds": "%g" % ds, "ms": ms,
                        "val_index": vidx, "antenna": ant,
                        "delta_hat_Z1": "%.10g" % float(d1[i]),
                        "defect_upper_Z1": "%.10g" % float(up1[i]),
                        "delta_hat_Z2": "%.10g" % float(d2[i]),
                        "defect_upper_Z2": "%.10g" % float(up2[i]),
                        "noise_gain_2": "%.10g" % float(out["norm_px_2"][i]),
                        "norm_PX_F": "%.10g" % float(out["norm_px_f"][i]),
                        "correction_norm": "%.10g" % float(out["norm_rx_f"][i]),
                        "ratio_corr_over_delta": "%.10g" % float(out["norm_rx_f"][i] / out["norm_px_f"][i]),
                        "rel_A": "%.10g" % float(out["rel_a"][i]),
                        "precision": "fp32+fp64_probe"})
                    v2_rows.append({
                        "family": family, "checkpoint_sha256": ckpt_sha, "seed": seed,
                        "sample_id": tag, "cm": cm, "ds": "%g" % ds, "ms": ms,
                        "val_index": vidx, "antenna": ant,
                        "norm_PX_F2": "%.10g" % float(f2[i]),
                        "KT_over_N_K257": "%.10g" % kt257,
                        "excess_K257": "%.10g" % float(f2[i] - kt257),
                        "excess_K16": "%.10g" % float(f2[i] - kt16),
                        "dist_to_P0_F": "%.10g" % float(dref0[i]),
                        "dist_to_PLam_F": "%.10g" % float(drefl[i]),
                        "selfcheck_P0V_minus_Q": "%.10g" % check,
                        "ls_residual_K257": "%.10g" % ls_res,
                        "lambda_value": "%.10g" % lam, "K_grid": int(V.shape[1]),
                        "precision": "fp32"})
            del out, d1, d2, up1, up2
            del dref0, drefl, f2
        del hist, x
        print("  [%s] cell cm%s ds%03d ms%03d done (%.0fs)" %
              (family, cm, round(ds * 1e9), ms, time.time() - t0), flush=True)
    append(D.THEORY / "operator_identity.csv", IDENTITY_FIELDS, id_rows)
    append(D.THEORY / "response_certificate.csv", CERT_FIELDS, cert_rows)
    append(D.THEORY / "noise_gain.csv", V2_FIELDS, v2_rows)
    r = np.array([float(v["rel_A"]) for v in id_rows])
    print("[%s] seeds=%s rows=%d rel_A median=%.3e max=%.3e | %.1fs" %
          (family, seeds, len(id_rows), np.median(r), r.max(), time.time() - t0), flush=True)


V2_FIELDS = ["family", "checkpoint_sha256", "seed", "sample_id", "cm", "ds", "ms",
             "val_index", "antenna", "norm_PX_F2", "KT_over_N_K257", "excess_K257",
             "excess_K16", "dist_to_P0_F", "dist_to_PLam_F", "selfcheck_P0V_minus_Q",
             "ls_residual_K257", "lambda_value", "K_grid", "precision"]
V4_FIELDS = ["family", "checkpoint_sha256", "seed", "K_modes", "q", "c", "M", "eps",
             "trial", "bound", "true_error", "ratio_bound_over_true", "tight_diag",
             "delta_hat_Z1", "noise_gain_2", "correction_norm", "violation", "precision"]


def grid_basis(device, n_angle=257):
    """V = V_N(z), Q = U_T(z) on the Z1 grid (double precision).

    The V2 self-check `||P0 V - Q||_2 <= 1e-10` is only attainable in double
    precision: in complex64 the same quantity sits near 1e-6 because the pinned
    reference operators are formed through an explicit (pseudo-)inverse.
    """
    theta = torch.linspace(-math.pi, math.pi, n_angle, dtype=torch.float64)
    z = torch.polar(torch.ones_like(theta), theta).to(device)
    powers = torch.stack([z ** k for k in range(D.N + D.T)], dim=0)
    return powers[:D.N], powers[D.N:]


def reference_operators(V, Q, lam_scale=1e-3):
    """P_0 = Q V^dagger and P_lambda = Q Lambda V^H C^{-1} (handoff section 10 / V2)."""
    K = V.shape[1]
    P0 = Q @ torch.linalg.pinv(V)
    Lam = torch.eye(K, dtype=V.dtype, device=V.device) / K
    Vh = V.conj().T
    lam = lam_scale * float(torch.trace(V.conj().T @ V).real) / D.N
    C = V @ Lam @ Vh + lam * torch.eye(D.N, dtype=V.dtype, device=V.device)
    PLam = Q @ Lam @ Vh @ torch.linalg.inv(C)
    return P0, PLam, lam


def roots_basis(device, K=16):
    """The K-th roots of unity: V is [N, K] with K = N here, so V is square.

    The handover's self-check `||P0 V - Q||_2 <= 1e-10` is stated for a mode set
    that is *not* over-determined. On the 257-point Z1 grid the constraint
    P V = Q has 257 columns and only 16 unknowns, so no exact solution exists and
    Q V^dagger is the minimum-norm least-squares solution instead; its residual
    is reported as `ls_residual_K257`. The exact check is therefore run on the
    16th roots of unity, where V is square and invertible.
    """
    theta = torch.arange(K, dtype=torch.float64) * (2 * math.pi / K)
    z = torch.polar(torch.ones_like(theta), theta).to(device)
    powers = torch.stack([z ** k for k in range(D.N + D.T)], dim=0)
    return powers[:D.N], powers[D.N:]


def run_v2(family: str, seeds: list, stride: int) -> None:
    D.setup_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V, Q = grid_basis(device)
    P0, PLam, lam = reference_operators(V, Q)
    check = float(torch.linalg.matrix_norm(P0 @ V - Q, ord=2))
    print(f"[v2] grid K={V.shape[1]} lambda={lam:.6g} selfcheck ||P0 V - Q||_2={check:.3e}", flush=True)
    assert check <= 1e-10, f"V2 self-check failed: {check}"
    KT_over_N_257 = D.T * V.shape[1] / D.N
    KT_over_N_16 = D.T * D.N / D.N

    for seed in seeds:
        ckpt = D.REPORTED[seed] if family in ("reported", "snap") else D.GRIDFIX[seed]
        ckpt_sha = D.sha256(ckpt)
        model, _meta = D.load_model(ckpt, grid_constraint=(family == "snap"))
        model = model.to(device)
        rows = []
        t0 = time.time()
        for cm, ds, ms in D.combos():
            hist, _pred = D.val_fold(cm, ds, ms)
            if stride > 1:
                hist = hist[::stride]
            x = D.to_model_input(hist).to(device)
            with torch.no_grad():
                out = D.extract(model, x)
                p_x = out["p_x"]
                d0 = torch.linalg.matrix_norm(p_x - P0)
                dl = torch.linalg.matrix_norm(p_x - PLam)
                f2 = out["norm_px_f"] ** 2
                for i in range(x.shape[0]):
                    sample, ant = i // 32, i % 32
                    vidx = 900 + sample * stride
                    tag = "cm%s_ds%03d_ms%03d_vidx%04d_ant%02d" % (cm, round(ds * 1e9), ms, vidx, ant)
                    rows.append({
                        "family": family, "checkpoint_sha256": ckpt_sha, "seed": seed,
                        "sample_id": tag, "cm": cm, "ds": "%g" % ds, "ms": ms,
                        "val_index": vidx, "antenna": ant,
                        "norm_PX_F2": "%.10g" % float(f2[i]),
                        "KT_over_N_K257": "%.10g" % KT_over_N_257,
                        "excess_K257": "%.10g" % float(f2[i] - KT_over_N_257),
                        "excess_K16": "%.10g" % float(f2[i] - KT_over_N_16),
                        "dist_to_P0_F": "%.10g" % float(d0[i]),
                        "dist_to_PLam_F": "%.10g" % float(dl[i]),
                        "selfcheck_P0V_minus_Q": "%.10g" % check,
                        "lambda_value": "%.10g" % lam, "K_grid": int(V.shape[1]),
                        "precision": "fp32"})
            del hist, x, out
        append(D.THEORY / "noise_gain.csv", V2_FIELDS, rows)
        print("[v2 %s s%d] rows=%d | %.1fs" % (family, seed, len(rows), time.time() - t0), flush=True)


def run_v4(family: str, seeds: list, trials: int = 200) -> None:
    """Synthetic clean/noisy coverage of the certificate bound (handoff section 10 / V4)."""
    D.setup_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z1 = D.grid([1.0]).to(device)
    for seed in seeds:
        ckpt = D.REPORTED[seed] if family in ("reported", "snap") else D.GRIDFIX[seed]
        ckpt_sha = D.sha256(ckpt)
        model, _meta = D.load_model(ckpt, grid_constraint=(family == "snap"))
        model = model.to(device)
        # recompute the scale from the model's own delay domain (cheap, exact)
        scale_samples = []
        for cm, ds, ms in D.combos()[:9]:
            hist, _ = D.val_fold(cm, ds, ms)
            x = D.to_model_input(hist[::10]).to(device)
            with torch.no_grad():
                xd = model.model.denoiser(x, x.shape[1]) if model.model.use_denoiser else x
                delay = model.model._lift(xd)
                scale_samples.append(torch.linalg.matrix_norm(delay))
            del hist, x, xd, delay
        med = float(torch.cat(scale_samples).median())
        print("[v4 %s s%d] median||X_val||_F=%.4f" % (family, seed, med), flush=True)

        rows = []
        gen = torch.Generator(device="cpu").manual_seed(20260921)
        for K_modes in (1, 4, 16):
            for q in (0.5, 1.0, 2.0):
                for c in (0.1, 0.3):
                    M = q * med
                    eps = c * M
                    S = torch.zeros(trials, D.N, D.L, dtype=torch.complex64)
                    Y = torch.zeros(trials, D.T, D.L, dtype=torch.complex64)
                    E = torch.zeros(trials, D.N, D.L, dtype=torch.complex64)
                    for t in range(trials):
                        theta = torch.rand(K_modes, generator=gen) * 2 * math.pi - math.pi
                        z = torch.polar(torch.ones(K_modes), theta).to(torch.complex64)
                        b = (torch.randn(K_modes, D.L, generator=gen) +
                             1j * torch.randn(K_modes, D.L, generator=gen)).to(torch.complex64)
                        b = b / torch.linalg.matrix_norm(b) * M
                        powers = torch.stack([z ** k for k in range(D.N + D.T)], dim=0)
                        S[t] = torch.einsum("nk,kl->nl", powers[:D.N], b)
                        Y[t] = torch.einsum("nk,kl->nl", powers[D.N:], b)
                        e = (torch.randn(D.N, D.L, generator=gen) +
                             1j * torch.randn(D.N, D.L, generator=gen)).to(torch.complex64)
                        E[t] = e / torch.linalg.matrix_norm(e) * eps
                    X = (S + E).to(device)
                    sub = torch.fft.fft(X, dim=-1)                       # back to the model input domain
                    x_real = torch.view_as_real(sub).reshape(trials, D.N, D.L * 2)
                    with torch.no_grad():
                        out = D.extract(model, x_real)
                        d1 = D.defect(out["p_x"], z1)
                        y_hat = out["pred"]
                        Yd = Y.to(device)
                        true_error = torch.linalg.matrix_norm(y_hat - Yd)
                        bound = M * d1 + eps * out["norm_px_2"] + out["norm_rx_f"]
                        tight = torch.linalg.matrix_norm(
                            torch.einsum("bpn,bnl->bpl", out["p_x"], E.to(device)) + out["r_x"])
                    for t in range(trials):
                        rows.append({
                            "family": family, "checkpoint_sha256": ckpt_sha, "seed": seed,
                            "K_modes": K_modes, "q": q, "c": c, "M": "%.10g" % M,
                            "eps": "%.10g" % eps, "trial": t,
                            "bound": "%.10g" % float(bound[t]),
                            "true_error": "%.10g" % float(true_error[t]),
                            "ratio_bound_over_true": "%.10g" % float(bound[t] / true_error[t].clamp_min(1e-30)),
                            "tight_diag": "%.10g" % float(tight[t]),
                            "delta_hat_Z1": "%.10g" % float(d1[t]),
                            "noise_gain_2": "%.10g" % float(out["norm_px_2"][t]),
                            "correction_norm": "%.10g" % float(out["norm_rx_f"][t]),
                            "violation": int(bool(true_error[t] > bound[t])),
                            "precision": "fp32"})
        append(D.THEORY / "synthetic_coverage.csv", V4_FIELDS, rows)
        viol = sum(r["violation"] for r in rows)
        print("[v4 %s s%d] trials=%d violations=%d" % (family, seed, len(rows), viol), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["v1v3", "v2", "v4"])
    ap.add_argument("--family", required=True, choices=["reported", "gridfix", "snap"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0, help="this shard index (cells with index %% nshards)")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    D.THEORY.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.task == "v1v3":
        run_family(args.family, seeds, args.stride, args.shard, args.nshards)
    elif args.task == "v2":
        run_v2(args.family, seeds, args.stride)
    else:
        run_v4(args.family, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
