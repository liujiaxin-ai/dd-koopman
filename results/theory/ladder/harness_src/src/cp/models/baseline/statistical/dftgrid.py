"""Fixed DFT-grid spectral extrapolator for TDD CSI prediction (no training).

The classical control for the paper's mechanism claim: a predictor whose
Doppler hypotheses are restricted to the 16-point DFT grid of the observation
window.  Per delay tap it keeps the ``num_modes`` strongest grid bins of the
16-slot history, discards the rest, and continues those bins in time.

Because the DFT basis is orthonormal, the least-squares fit restricted to a set
of grid bins is simply the corresponding subset of DFT coefficients, so the
model is exact linear algebra with no learned parameters.

Input shape:  [batch, num_antennas, hist_len, num_subcarriers] (complex)
Output shape: [batch, num_antennas, pred_len, num_subcarriers] (complex)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN, TOT_ANTENNAS


class DFTGRIDMODEL(nn.Module):
    """Top-K fixed-grid Doppler extrapolation, per delay tap."""

    def __init__(
        self,
        num_modes: int = 2,
        hist_len: int = HIST_LEN,
        pred_len: int = PRED_LEN,
        num_antennas: int = TOT_ANTENNAS,
        num_subcarriers: int = NUM_SUBCARRIERS,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = "DFTGRID"
        self.is_separate_antennas = False
        # The grid width is the only knob of this baseline; it is fixed per run
        # through the environment so the harness entry point stays unchanged.
        self.num_modes = int(os.environ.get("DFTGRID_MODES", num_modes))
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.num_antennas = int(num_antennas)
        self.num_subcarriers = int(num_subcarriers)

    def __str__(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError(f"{self.name} expects complex input, got {x.dtype}")
        t_hist = x.shape[-2]
        # delay-domain lift (parameter free)
        delay = torch.fft.ifft(x, dim=-1, norm="ortho")
        # Doppler coefficients of the observation window, one per grid bin
        coeff = torch.fft.fft(delay, dim=-2, norm="ortho")

        # keep the num_modes strongest grid bins per (sample, antenna, tap)
        magnitude = coeff.abs()
        keep = torch.topk(magnitude, self.num_modes, dim=-2).indices
        mask = torch.zeros_like(magnitude, dtype=torch.bool)
        mask.scatter_(-2, keep, True)
        coeff = torch.where(mask, coeff, torch.zeros_like(coeff))

        # continue the retained bins past the observation window: slot 15+h for
        # h = 1..pred_len, i.e. phase exp(j 2 pi k (h-1) / T) after one period.
        bins = torch.arange(t_hist, device=x.device, dtype=torch.float32)
        steps = torch.arange(0, self.pred_len, device=x.device, dtype=torch.float32)
        angles = 2 * torch.pi * torch.outer(steps, bins) / t_hist  # [pred_len, t_hist]
        phase = torch.polar(torch.ones_like(angles), angles).to(coeff.dtype)  # [H, T]

        # inverse of the orthonormal time transform: [H, T] x [B, A, T, L]
        extended = torch.einsum("ht,batl->bahl", phase, coeff) / (t_hist ** 0.5)

        return torch.fft.fft(extended, dim=-1, norm="ortho")
