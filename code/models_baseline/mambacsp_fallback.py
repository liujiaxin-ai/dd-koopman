"""Pure-PyTorch selective scan used in place of the fused `mamba_ssm` kernel.

Why this file exists
--------------------
MambaCSP's `Model` calls `Mamba2(d_model, d_state, d_conv, expand)` from the
`mamba-ssm` package. The evaluation container has torch 2.7.1+cu126 but **no
CUDA toolchain** (`nvcc` is absent), so neither `mamba-ssm` nor `causal-conv1d`
can be compiled and the published kernel cannot be imported.

The substitution replaces the *kernel*, not the architecture: the block that
MambaCSP stacks six times becomes a standard S6 selective scan with the same
arguments, the same residual + LayerNorm wrapper, and the same tensor contract
`(B, L, F) -> (B, L, F)`. The sequence length here is 16, so the explicit
recurrence the fused kernel exists to parallelise costs nothing.

This is disclosed in `design/lightweight-literature.md`; any number produced
from it is labelled as "MambaCSP architecture, pure-PyTorch scan".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PureTorchMamba2(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(
                self.d_inner, 1
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        inner = self.in_proj(x)
        xs, zs = inner.chunk(2, dim=-1)

        xs = xs.transpose(1, 2)
        xs = self.conv1d(xs)[..., :length]
        xs = F.silu(xs.transpose(1, 2))

        dt, b_mat, c_mat = self.x_proj(xs).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        a_mat = -torch.exp(self.A_log)
        d_a = torch.exp(dt.unsqueeze(-1) * a_mat.unsqueeze(0).unsqueeze(0))
        d_b = dt.unsqueeze(-1) * b_mat.unsqueeze(2)

        state = xs.new_zeros(batch, self.d_inner, self.d_state)
        outputs = []
        for step in range(length):
            state = d_a[:, step] * state + d_b[:, step] * xs[:, step].unsqueeze(-1)
            outputs.append((state * c_mat[:, step].unsqueeze(1)).sum(dim=-1))
        y = torch.stack(outputs, dim=1) + self.D * xs
        return self.out_proj(y * F.silu(zs))
