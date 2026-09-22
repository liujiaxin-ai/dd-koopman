"""Sanity checks for the exactness ladder (task A and the task B cross-check).

Run from the harness root with `DFTGRID_MODES=16`:

    DFTGRID_MODES=16 python ladder_sanity.py --out sanity_k16.json

Checks, all on synthetic input so they are independent of the data split:

1. `num_modes == 16` reaches the model (no silent default, no truncation);
2. with a full 16-bin support the prediction is the *periodic extension* of the
   delay-domain history: slot 16+h equals slot h, i.e. `pred[h] == D[h]` for
   h = 0..3;
3. the analytic periodic-extension gains `W_per[p, m] = (1/N) z_m^p`,
   `z_m = exp(2 pi i m / N)`, applied to the *unnormalised* time-FFT of the
   delay history reproduce the same output — this is the normalisation check the
   directive asks for, and the object task B's cross-validation uses.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("sanity_k16.json"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--antennas", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    from src.cp.models.baseline.statistical.dftgrid import DFTGRIDMODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = DFTGRIDMODEL().to(device).eval()

    batch, antennas = args.batch, args.antennas
    delay_true = torch.randn(batch, antennas, HIST_LEN, NUM_SUBCARRIERS,
                             dtype=torch.complex64, device=device)
    # the model's input is the subcarrier-domain observation of that history
    x = torch.fft.fft(delay_true, dim=-1, norm="ortho")

    with torch.no_grad():
        y = model(x)
        delay_pred = torch.fft.ifft(y, dim=-1, norm="ortho")

        # the same operator written with the unnormalised time-FFT and the
        # analytic periodic-extension gains W_per[p, m] = z_m^p / N
        spectrum = torch.fft.fft(delay_true, dim=-2)
        m = torch.arange(HIST_LEN, device=device, dtype=torch.float32)
        p = torch.arange(PRED_LEN, device=device, dtype=torch.float32)
        w_per = torch.polar(torch.ones(PRED_LEN, HIST_LEN, device=device),
                            2 * math.pi * torch.outer(p, m) / HIST_LEN) / HIST_LEN
        delay_analytic = torch.einsum("pm,baml->bapl", w_per.to(torch.complex64),
                                      spectrum)
        y_analytic = torch.fft.fft(delay_analytic, dim=-1, norm="ortho")

        topk_identity = bool(
            torch.topk(torch.rand(batch, antennas, HIST_LEN, NUM_SUBCARRIERS,
                                  device=device), HIST_LEN, dim=-2).indices.numel()
            == batch * antennas * HIST_LEN * NUM_SUBCARRIERS
        )
        payload = {
            "num_modes": int(model.num_modes),
            "hist_len": int(model.hist_len),
            "pred_len": int(model.pred_len),
            "topk_of_hist_len_is_identity": topk_identity,
            "periodic_extension_max_abs_err": float(
                (delay_pred - delay_true[:, :, :PRED_LEN]).abs().max()),
            "analytic_vs_harness_max_abs_err": float((y - y_analytic).abs().max()),
            "analytic_predicts_periodic_extension_max_abs_err": float(
                (delay_analytic - delay_true[:, :, :PRED_LEN]).abs().max()),
            "scale_of_prediction": float(delay_pred.abs().mean()),
            "w_per_row0": [[float(w_per[0, i].real), float(w_per[0, i].imag)]
                           for i in range(4)],
            "dtype": str(delay_pred.dtype),
            "device": str(device),
        }
    payload["passed"] = bool(
        payload["num_modes"] == HIST_LEN
        and payload["periodic_extension_max_abs_err"] <= 1e-4
        and payload["analytic_vs_harness_max_abs_err"] <= 1e-4
        and payload["analytic_predicts_periodic_extension_max_abs_err"] <= 1e-4
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit("ladder sanity FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
