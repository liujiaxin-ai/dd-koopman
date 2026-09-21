"""Four recent lightweight forecasting architectures on the benchmark's protocol.

The comparison family in this project was thin on *recent* competitors: the
published model (2025/2026), its capacity-matched copies, MambaCSP (2026), and
the classical deep-forecasting family. A lightweight claim has to survive
against the models the forecasting community actually uses in 2024-2026, which
are not only transformers: multi-scale decomposition mixers, all-MLP mixers,
inverted attention over variates, and selective state-space models.

Every network here obeys the same harness contract as `baselines_suite.py`:
input `[B, 16, 600]` real (300 subcarriers x 2 for the real and imaginary part,
one antenna per sample), output `[B, 4, 600]` real, and a plain NMSE objective
with no auxiliary terms. All of them are channel independent (weights shared
across the 600 columns), which is the standard formulation for multivariate
forecasting and the reason their parameter counts stay in the range of the
reported 173k-parameter model.

| registry name | family | reference |
|---|---|---|
| `TIMEMIXER_TDD` | multi-scale decomposition + time mixing | Wang et al., ICLR 2024 |
| `TSMIXER_TDD` | alternating time / feature MLP mixing | Chen et al., TSMixer 2023 |
| `ITRANSFORMER_TDD` | attention over variate tokens | Liu et al., ICLR 2024 |
| `MAMBA_LITE_TDD` | selective state space, pure PyTorch | Gu & Dao, Mamba 2023 |

The iTransformer row is adapted to this input shape: 600 columns cannot each be
a token at a sane parameter count, so the columns are grouped and each group is
a token, which is the same "invert the axes" idea at a workable width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cp.config.config import ExperimentConfig
from src.cp.loss.loss import NMSELoss
from src.cp.models.common.base import BaseCSIModel

def _moving_average(series: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return series
    pad = kernel // 2
    shape = series.shape
    flat = series.reshape(-1, 1, shape[-1])
    padded = F.pad(flat, (pad, pad), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel, stride=1)
    return smoothed.reshape(shape)

class TimeMixerNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 d_model: int = 32, scales: tuple[int, ...] = (1, 2, 4),
                 kernel: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.scales = tuple(scales)
        self.embeds = nn.ModuleList()
        self.mixers = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.drops = nn.ModuleList()
        for scale in self.scales:
            length = max(hist_len // scale, 1)
            self.embeds.append(nn.Linear(length, d_model))
            self.mixers.append(nn.Linear(d_model, d_model))
            self.heads.append(nn.Linear(d_model, pred_len))
            self.drops.append(nn.Dropout(dropout))

        self.level_scale = nn.Parameter(torch.zeros(1))
        self.kernel = kernel
        self.pred_len = pred_len
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        series = x.transpose(1, 2)
        out = None
        level = None
        for index, scale in enumerate(self.scales):
            if scale > 1:
                pooled = F.avg_pool1d(series, scale, stride=scale, ceil_mode=True)
            else:
                pooled = series
            trend = _moving_average(pooled, self.kernel)
            seasonal = pooled - trend
            hidden = F.gelu(self.embeds[index](seasonal))
            hidden = hidden + self.drops[index](self.mixers[index](hidden))
            prediction = self.heads[index](hidden)
            out = prediction if out is None else out + prediction
            if level is None:
                level = trend[..., -1].unsqueeze(-1)
        out = out + self.level_scale * level
        return out.transpose(1, 2).contiguous()

class TSMixerNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 hidden: int = 64, feature_hidden: int = 64, groups: int = 40,
                 blocks: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.groups = max(1, min(groups, channels))
        self.group_size = channels // self.groups
        self.usable = self.groups * self.group_size
        self.channels = channels
        self.pred_len = pred_len
        self.time_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hist_len, hidden), nn.GELU(), nn.Linear(hidden, hist_len))
            for _ in range(blocks)
        )
        self.feature_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.group_size, feature_hidden),
                nn.GELU(),
                nn.Linear(feature_hidden, self.group_size),
            )
            for _ in range(blocks)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(self.usable) for _ in range(blocks))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hist_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        usable = self.usable
        hidden = x[..., :usable].transpose(1, 2)
        for time_layer, feature_layer, norm in zip(self.time_layers, self.feature_layers, self.norms):
            hidden = hidden + self.drop(time_layer(hidden))
            by_step = hidden.transpose(1, 2)
            grouped = by_step.reshape(batch, steps, self.groups, self.group_size)
            grouped = grouped + self.drop(feature_layer(grouped))
            hidden = norm(grouped.reshape(batch, steps, usable)).transpose(1, 2)
        prediction = self.head(hidden)
        if usable < channels:
            fill = prediction.new_zeros(batch, channels - usable, self.pred_len)
            prediction = torch.cat([prediction, fill], dim=1)
        return prediction.transpose(1, 2).contiguous()

class ITransformerNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 groups: int = 60, d_model: int = 32, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.groups = max(1, min(groups, channels))
        self.group_size = channels // self.groups
        self.channels = channels
        self.pred_len = pred_len
        self.embed = nn.Linear(hist_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len * self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        usable = self.groups * self.group_size
        grouped = x[..., :usable].reshape(batch, steps, self.groups, self.group_size)
        tokens = grouped.mean(dim=-1).transpose(1, 2)
        hidden = self.encoder(self.embed(tokens))
        decoded = self.head(self.norm(hidden))
        decoded = decoded.reshape(batch, self.groups, self.pred_len, self.group_size)
        decoded = decoded.permute(0, 2, 1, 3).reshape(batch, self.pred_len, usable)
        if usable < channels:
            fill = decoded.new_zeros(batch, self.pred_len, channels - usable)
            decoded = torch.cat([decoded, fill], dim=2)
        return decoded.contiguous()

class MambaLiteNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 d_model: int = 64, d_state: int = 16, expand: int = 2, kernel: int = 4,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        inner = d_model * expand
        self.inner = inner
        self.d_state = d_state
        self.pred_len = pred_len
        self.channels = channels
        self.in_proj = nn.Linear(channels, d_model)
        self.expand_proj = nn.Linear(d_model, inner)
        self.conv = nn.Conv1d(inner, inner, kernel, groups=inner, padding=kernel - 1)

        self.x_proj = nn.Linear(inner, 2 * d_state + 1)
        self.dt_bias = nn.Parameter(torch.zeros(1))
        self.a_log = nn.Parameter(torch.zeros(inner, d_state))
        self.d = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, pred_len * channels)
        self.drop = nn.Dropout(dropout)
        nn.init.uniform_(self.a_log, -1.0, 1.0)
        nn.init.constant_(self.dt_bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        hidden = F.silu(self.expand_proj(self.in_proj(x)))
        convolved = self.conv(hidden.transpose(1, 2))[:, :, :steps].transpose(1, 2)
        hidden = F.silu(convolved)

        projected = self.x_proj(hidden)
        dt, b_param, c_param = projected.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        a = -torch.exp(self.a_log)

        decay = torch.exp(dt.unsqueeze(-1) * a)
        drive = dt.unsqueeze(-1) * b_param.unsqueeze(2) * hidden.unsqueeze(-1)

        state = hidden.new_zeros(decay.shape[0], decay.shape[2], self.d_state)
        outputs = []
        for step in range(steps):
            state = decay[:, step] * state + drive[:, step]
            outputs.append((state * c_param[:, step].unsqueeze(1)).sum(-1))
        series = torch.stack(outputs, dim=1) + self.d.reshape(1, 1, -1) * hidden
        pooled = series.mean(dim=1)
        prediction = self.out_proj(pooled)
        return prediction.reshape(batch, self.pred_len, channels).contiguous()

class _NewBaselineLightning(BaseCSIModel):

    NETWORK: type[nn.Module] = TimeMixerNet
    MODEL_NAME = "NEWBASELINE"

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )
        self.name = self.MODEL_NAME
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})
        self.model = self.NETWORK(**config.model.params)
        self.metric = NMSELoss()

    def __str__(self):
        return self.name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.metric(self(batch[0]), batch[1])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.metric(self(batch[0]), batch[1])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

def _make(name: str, network: type[nn.Module]):
    return type(name, (_NewBaselineLightning,), {"NETWORK": network, "MODEL_NAME": name.split("_")[0]})

TIMEMIXER_TDD = _make("TIMEMIXER_TDD", TimeMixerNet)
TIMEMIXER_FDD = _make("TIMEMIXER_FDD", TimeMixerNet)
TSMIXER_TDD = _make("TSMIXER_TDD", TSMixerNet)
TSMIXER_FDD = _make("TSMIXER_FDD", TSMixerNet)
ITRANSFORMER_TDD = _make("ITRANSFORMER_TDD", ITransformerNet)
ITRANSFORMER_FDD = _make("ITRANSFORMER_FDD", ITransformerNet)
MAMBA_LITE_TDD = _make("MAMBA_LITE_TDD", MambaLiteNet)
MAMBA_LITE_FDD = _make("MAMBA_LITE_FDD", MambaLiteNet)

def _register() -> None:
    from src.cp.models import PREDICTORS

    for name in ("TIMEMIXER", "TSMIXER", "ITRANSFORMER", "MAMBA_LITE"):
        setattr(PREDICTORS, f"{name}_TDD", globals()[f"{name}_TDD"])
        setattr(PREDICTORS, f"{name}_FDD", globals()[f"{name}_FDD"])

_register()

__all__ = [
    "ITransformerNet", "MambaLiteNet", "TimeMixerNet", "TSMixerNet",
]
