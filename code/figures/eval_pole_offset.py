"""Evaluate the reported model under a counterfactual on its pole offsets.

Mechanism note: the v7/v8 operator parameterises each Doppler mode m as

    angle_m(x) = 2*pi*(m + delta_m(x)) / Th,   delta_m(x) = 0.5*tanh(...)

so `delta = 0` places every pole exactly on the 16-point DFT grid and the learned
`delta` is precisely what puts the poles off-grid (89% of them, by 0.05 to 0.34
bin). This wrapper keeps the checkpoint, the denoiser, the delay decomposition
and everything else untouched and only overwrites `delta` with a constant, so the
resulting NMSE is a causal read-out of what the off-grid placement buys.

Three interventions are supported, all read once at import:

  POLE_OFFSET=x   force delta to the constant x (0.00 = the classical grid)
  POLE_QUANT=r    quantise the learned delta to the grid of resolution r,
                  i.e. delta -> round(delta / r) * r, clamped to [-0.5, 0.5]
  POLE_TOPK=k     keep the learned delta of the k modes with the largest mean
                  |delta| in the batch and zero the rest

With no variable set the stock model is evaluated (useful as a sanity check
that the harness path is unchanged).

Usage (harness root, harness python):
    POLE_OFFSET=0.0 python eval_pole_offset.py --model DD_KOOP_TDD --duplex TDD \\
        --test-type regular --limit 0 --out results/run_csv/SNAP_000_full162.csv
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402
import src.cp.models.ours.dd_koop as v7  # noqa: E402

OFFSET = os.environ.get("POLE_OFFSET")
QUANT = os.environ.get("POLE_QUANT")
TOPK = os.environ.get("POLE_TOPK")

if OFFSET is not None:
    _forced_offset = float(OFFSET)
    _original = v7.RadiusSpectralKoopmanOperator.pole_angles

    def _constant_offset(self, doppler):          # noqa: ANN001
        angle, radius = _original(self, doppler)
        bins = torch.arange(self.hist_len, device=angle.device,
                            dtype=angle.dtype)
        angle = (2.0 * math.pi * (bins + _forced_offset)
                 / self.hist_len).unsqueeze(0).expand_as(angle)
        return angle, radius

    v7.RadiusSpectralKoopmanOperator.pole_angles = _constant_offset
    print(f"[snap] pole sub-bin offset forced to {_forced_offset:+.2f} bin",
          flush=True)
elif QUANT is not None:
    _resolution = float(QUANT)
    _original = v7.RadiusSpectralKoopmanOperator.pole_angles

    def _quantised(self, doppler):                # noqa: ANN001
        angle, radius = _original(self, doppler)
        bins = torch.arange(self.hist_len, device=angle.device,
                            dtype=angle.dtype)
        base = 2.0 * math.pi * bins / self.hist_len
        delta = (angle - base.unsqueeze(0)) * self.hist_len / (2.0 * math.pi)
        delta = torch.round(delta / _resolution) * _resolution
        delta = delta.clamp(-0.5, 0.5)
        return base.unsqueeze(0) + delta * (2.0 * math.pi / self.hist_len), radius

    v7.RadiusSpectralKoopmanOperator.pole_angles = _quantised
    print(f"[snap] learned pole offsets quantised to {_resolution:.3f} bin",
          flush=True)
elif TOPK is not None:
    _keep = int(TOPK)
    _original = v7.RadiusSpectralKoopmanOperator.pole_angles
    _state = {"rank": None}

    def _topk(self, doppler):                     # noqa: ANN001
        angle, radius = _original(self, doppler)
        bins = torch.arange(self.hist_len, device=angle.device,
                            dtype=angle.dtype)
        base = 2.0 * math.pi * bins / self.hist_len
        delta = (angle - base.unsqueeze(0)) * self.hist_len / (2.0 * math.pi)
        if _state["rank"] is None or _state["rank"].numel() != delta.shape[-1]:
            importance = delta.abs().mean(dim=0)          # per mode, over batch
            order = torch.argsort(importance, descending=True)
            keep = torch.zeros_like(importance, dtype=torch.bool)
            keep[order[:_keep]] = True
            _state["rank"] = keep
            kept = [int(i) for i in order[:_keep].tolist()]
            print(f"[snap] keeping the learned offset of modes {kept}", flush=True)
        delta = delta * _state["rank"].to(delta.dtype).unsqueeze(0)
        return base.unsqueeze(0) + delta * (2.0 * math.pi / self.hist_len), radius

    v7.RadiusSpectralKoopmanOperator.pole_angles = _topk
    print(f"[snap] keeping the top-{_keep} pole offsets, zeroing the rest",
          flush=True)
else:
    print("[snap] no intervention requested: stock model", flush=True)

sys.argv = ["eval_ours.py"] + sys.argv[1:]
_source = open("eval_ours.py", encoding="utf-8").read()
exec(compile(_source, "eval_ours.py", "exec"), {"__name__": "__main__"})
