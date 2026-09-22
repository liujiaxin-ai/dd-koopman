"""V1b: repeat the V1 operator-identity extraction with the model in float64.

V1 reports `rel_A` (the composed prediction against `sum_j alpha_j P_j A_j`) at a
median of 1.97e-6, above the pre-declared 1e-6 gate, while `rel_full` is exactly
zero and `rel_B` is 2.6e-7. The remaining question is where the 1.97e-6 lives:
in the extraction, in the algebra, or in the model's own float32 arithmetic.

This script answers it by casting the *same* checkpoint to `torch.float64` and
repeating the identical extraction in double precision. The algebra is
untouched: if the identity holds and the residual is float32 arithmetic, then
`rel_A` must collapse to machine precision. If it does not, the extraction is
wrong and no number below should be published.

Writes `results/theory/operator_identity_fp64.csv` with one row per validation
sample and antenna.
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

FIELDS = ["family", "checkpoint_sha256", "seed", "sample_id", "cm", "ds", "ms",
          "val_index", "antenna", "rel_A_fp64", "rel_B_fp64", "rel_full_fp64",
          "g1", "g2", "precision"]


def f_matrix64(device) -> torch.Tensor:
    """`F_N[m, n] = exp(-2i pi m n / N)` in complex128."""
    m = torch.arange(D.N, device=device, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(m, m) / D.N).to(torch.complex128)


def branch_gain64(operator, spectrum: torch.Tensor) -> torch.Tensor:
    """`W[h, m]` exactly as the operator forms it, in double precision."""
    batch = spectrum.shape[0]
    angle, radius = operator.pole_angles(spectrum)
    horizon = torch.arange(1, operator.pred_len + 1, device=spectrum.device,
                           dtype=torch.float64)
    magnitude = radius.unsqueeze(-2) ** horizon.reshape(1, -1, 1)
    advance = torch.polar(magnitude, angle.unsqueeze(-2) * horizon.reshape(1, -1, 1))
    base = torch.complex(operator.gain_real, operator.gain_imag).unsqueeze(0) * advance
    power = torch.log(spectrum.abs().pow(2).mean(dim=-1) + 1e-8)
    modulation = operator.modulator(operator.context_norm(power)).view(
        batch, operator.pred_len, operator.hist_len, 2
    )
    bounded = 1.0 + torch.tanh(torch.view_as_complex(modulation.contiguous()))
    return base * bounded


def extract64(model, x: torch.Tensor) -> dict:
    """The V1 extraction, with every tensor and constant in float64."""
    net = model.model
    steps = x.shape[1]
    xin = net.instance_norm.normalize(x) if net.use_revin else x
    xd = net.denoiser(xin, steps) if net.use_denoiser else xin
    delay = net._lift(xd)
    spec0 = net.spectrum_of(delay)
    out0 = net.operator(spec0)
    branches = [(delay, spec0, out0, branch_gain64(net.operator, spec0))]
    predicted = out0

    mixed = net.shared_mixer(net.arl(delay))
    spec1 = net.spectrum_of(mixed)
    out1 = net.operator(spec1)
    predicted = predicted + net.gate_mix * (out1 - out0)
    branches.append((mixed, spec1, out1, branch_gain64(net.operator, spec1)))

    g2 = 0.0
    if net.use_delay_decomposition:
        _structured, residual = net.decomposition(delay)
        spec2 = net.spectrum_of(residual)
        out2 = net.operator(spec2)
        g2 = float(net.gate_split)
        predicted = predicted + net.gate_split * (out2 - out0)
        branches.append((residual, spec2, out2, branch_gain64(net.operator, spec2)))

    taps = predicted.shape[-1]
    padded = predicted
    if taps < net.num_subcarriers:
        padded = torch.cat([predicted, predicted.new_zeros(
            predicted.shape[0], net.pred_len, net.num_subcarriers - taps)], dim=-1)
    h_pred = torch.fft.fft(padded, dim=-1)[..., : net.num_subcarriers]
    y_full = torch.view_as_real(h_pred).reshape(predicted.shape[0], net.pred_len,
                                               net.num_subcarriers * 2)
    with torch.no_grad():
        y_model = net(x)

    fn = f_matrix64(predicted.device)
    g1 = float(net.gate_mix)
    alphas = [1.0 - g1 - g2, g1, g2]
    recon = torch.zeros_like(predicted)
    for alpha, (a_j, _s_j, _o_j, w_j) in zip(alphas, branches):
        p_j = torch.einsum("bpm,mn->bpn", w_j.to(torch.complex128), fn)
        recon = recon + alpha * torch.einsum("bpn,bnl->bpl", p_j, a_j)

    denom = torch.linalg.matrix_norm(predicted).clamp_min(1e-30)
    rel_a = torch.linalg.matrix_norm(predicted - recon) / denom

    # The same branch expansion the paper states, in double precision.
    p_x = torch.zeros(predicted.shape[0], D.T, D.N, dtype=torch.complex128,
                      device=predicted.device)
    r_x = torch.zeros_like(predicted)
    for alpha, (a_j, _s_j, _o_j, w_j) in zip(alphas, branches):
        p_j = torch.einsum("bpm,mn->bpn", w_j.to(torch.complex128), fn)
        p_x = p_x + alpha * p_j
        r_x = r_x + alpha * torch.einsum("bpn,bnl->bpl", p_j, a_j - delay)
    rel_b = (torch.linalg.matrix_norm(
        recon - (torch.einsum("bpn,bnl->bpl", p_x, delay) + r_x)) / denom)
    rel_full = (torch.linalg.matrix_norm(y_model - y_full) /
                torch.linalg.matrix_norm(y_full).clamp_min(1e-30))
    return {"rel_a": rel_a, "rel_b": rel_b, "rel_full": rel_full, "g1": g1, "g2": g2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--family", default="reported",
                        choices=["reported", "gridfix", "snap"])
    parser.add_argument("--out", type=Path, default=Path("operator_identity_fp64.csv"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    D.setup_env()
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    out_path = D.THEORY / args.out
    print(f"[fp64] device={device} seeds={seeds} stride={args.stride}", flush=True)

    rows = []
    summary = []
    for seed in seeds:
        ckpt = D.REPORTED[seed] if args.family in ("reported", "snap") else D.GRIDFIX[seed]
        sha = D.sha256(ckpt)
        model, _meta = D.load_model(ckpt, grid_constraint=(args.family == "snap"))
        model = model.double().to(device).eval()
        started = time.time()
        rel_a_all, rel_b_all, rel_full_all = [], [], []
        for cm, ds, ms in D.combos():
            hist, _pred = D.val_fold(cm, ds, ms)
            if args.stride > 1:
                hist = hist[::args.stride]
            # The model input stays real; `_lift` performs the complex cast.
            x = D.to_model_input(hist).to(device).to(torch.float64)
            with torch.no_grad():
                out = extract64(model, x)
            rel_a_all.append(out["rel_a"].double().cpu())
            rel_b_all.append(out["rel_b"].double().cpu())
            rel_full_all.append(out["rel_full"].double().cpu())
            for i in range(x.shape[0]):
                sample, antenna = i // 32, i % 32
                vidx = 900 + sample * args.stride
                rows.append({
                    "family": args.family, "checkpoint_sha256": sha, "seed": seed,
                    "sample_id": "cm%s_ds%03d_ms%03d_vidx%04d_ant%02d" % (
                        cm, round(ds * 1e9), ms, vidx, antenna),
                    "cm": cm, "ds": "%g" % ds, "ms": ms,
                    "val_index": vidx, "antenna": antenna,
                    "rel_A_fp64": "%.10g" % float(out["rel_a"][i]),
                    "rel_B_fp64": "%.10g" % float(out["rel_b"][i]),
                    "rel_full_fp64": "%.10g" % float(out["rel_full"][i]),
                    "g1": "%.10g" % out["g1"], "g2": "%.10g" % out["g2"],
                    "precision": "model_and_extraction_float64",
                })
            print("  [fp64] cell cm%s ds%03d ms%03d rows=%d (%.0fs)"
                  % (cm, round(ds * 1e9), ms, x.shape[0], time.time() - started), flush=True)
            del hist, x, out

        rel_a = torch.cat(rel_a_all).numpy()
        rel_b = torch.cat(rel_b_all).numpy()
        rel_full = torch.cat(rel_full_all).numpy()
        summary.append((seed, sha[:12], len(rel_a), float(np.median(rel_a)),
                        float(rel_a.max()), float(np.median(rel_b)), float(rel_b.max()),
                        float(rel_full.max())))
        print("[fp64 s%d] rows=%d rel_A median=%.3e max=%.3e | rel_B median=%.3e max=%.3e "
              "| rel_full max=%.3e | %.0fs"
              % (seed, len(rel_a), np.median(rel_a), rel_a.max(), np.median(rel_b),
                 rel_b.max(), rel_full.max(), time.time() - started), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[fp64] wrote {len(rows)} rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
