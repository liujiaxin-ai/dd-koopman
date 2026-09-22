"""LSGAINS: fixed grid gains fitted by ridge least squares (no training).

The E1 rung of the exactness ladder. Same parameterisation, same input
representation and same 16-bin support as the exactness-constrained operator;
the only difference is that the [T, N] complex gain matrix is fitted to the
training split instead of being pinned by `P0 V = Q`.

    delay    = ifft(x, dim=-1)[..., :taps]        (unnormalised, as in the model)
    spectrum = fft(delay, dim=time)               (unnormalised)
    future   = sum_m W[p, m] spectrum[m, l]
    y        = fft(pad(future))                   (back to the subcarrier domain)

`W` is loaded from the `.npz` written by `ladder_lsgains_fit.py`; the path comes
from `LSGAINS_W_PATH` so the eval entry point stays unchanged.

Input shape:  [batch, num_antennas, hist_len, num_subcarriers] (complex)
Output shape: [batch, num_antennas, pred_len, num_subcarriers] (complex)
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN, TOT_ANTENNAS


class LSGAINSMODEL(nn.Module):
    """Fixed gain matrix over the 16 grid bins, shared across taps and antennas."""

    def __init__(
        self,
        hist_len: int = HIST_LEN,
        pred_len: int = PRED_LEN,
        num_antennas: int = TOT_ANTENNAS,
        num_subcarriers: int = NUM_SUBCARRIERS,
        num_delay_taps: int = NUM_SUBCARRIERS,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = "LSGAINS"
        self.is_separate_antennas = False
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.num_antennas = int(num_antennas)
        self.num_subcarriers = int(num_subcarriers)
        self.num_delay_taps = int(num_delay_taps)

        path = os.environ.get("LSGAINS_W_PATH")
        if not path:
            raise RuntimeError("LSGAINS requires LSGAINS_W_PATH pointing at a .npz "
                               "written by ladder_lsgains_fit.py")
        payload = np.load(path)
        weight = payload["W_fp64"] if "W_fp64" in payload else payload["W"]
        weight = np.asarray(weight, dtype=np.complex128)
        if weight.shape != (self.pred_len, self.hist_len):
            raise ValueError(f"W has shape {weight.shape}, expected "
                             f"{(self.pred_len, self.hist_len)}")
        self.register_buffer("W", torch.from_numpy(weight).to(torch.complex64))
        self.weight_source = path

    def __str__(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError(f"{self.name} expects complex input, got {x.dtype}")
        delay = torch.fft.ifft(x, dim=-1)[..., : self.num_delay_taps]
        spectrum = torch.fft.fft(delay, dim=-2)                       # [B, A, T, L]
        future = torch.einsum("pm,baml->bapl", self.W.to(x.dtype), spectrum)
        if future.shape[-1] < self.num_subcarriers:
            pad = future.new_zeros(future.shape[0], future.shape[1],
                                   future.shape[2],
                                   self.num_subcarriers - future.shape[-1])
            future = torch.cat([future, pad], dim=-1)
        return torch.fft.fft(future, dim=-1)
