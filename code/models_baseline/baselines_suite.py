"""Six modern forecasting baselines on the benchmark's own protocol.

The comparison set in this project was thin: the published model, its
capacity-matched copies, MambaCSP, and the classical NP/AR/Wiener rows from the
published table. A lightweight paper is expected to compare against the standard
deep forecasting family as well, so this file implements that family under the
harness contract - input `[B, 16, 600]` real, output `[B, 4, 600]` real, one
regressor per antenna, exactly what `collect_fn_separate_antennas` produces.

Every model is deliberately small enough to sit in the same parameter range as
the reported model (~170k) or below, because the claim being tested is a
parameter/accuracy trade and a 20M-parameter baseline would not test it.

| registry name | family | reference |
|---|---|---|
| `DLINEAR_TDD` | one linear map per channel, no temporal structure | Zeng et al., DLinear |
| `MLP_TDD` | flattened window through an MLP | the "no structure" control |
| `GRU_TDD` | recurrent encoder over the 16 steps | classic deep baseline |
| `TCN_TDD` | dilated causal convolutions over time | Bai et al. |
| `TRANSFORMER_TDD` | encoder over the 16 steps | Vaswani et al. |
| `PATCHTST_TDD` | patch embedding + encoder, channel independent | Nie et al., PatchTST |
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.cp.config.config import ExperimentConfig
from src.cp.loss.loss import NMSELoss
from src.cp.models.common.base import BaseCSIModel

class DLinearNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600, **kwargs):
        super().__init__()
        self.projection = nn.Linear(hist_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x.transpose(1, 2)).transpose(1, 2)

class MLPNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 hidden: int = 96, depth: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(hist_len * channels, hidden), nn.GELU(),
                                   nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, pred_len * channels))
        self.net = nn.Sequential(*layers)
        self.channels = channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return self.net(x.flatten(1)).reshape(batch, self.pred_len, self.channels)

class GRUNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 hidden: int = 64, layers: int = 1, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.gru = nn.GRU(channels, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, pred_len * channels)
        self.channels = channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1]).reshape(x.shape[0], self.pred_len, self.channels)

class TCNNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 hidden: int = 64, levels: int = 3, kernel: int = 3, dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        blocks = []
        in_channels = channels
        for level in range(levels):
            dilation = 2 ** level
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden, kernel, dilation=dilation,
                              padding=(kernel - 1) * dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = hidden
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden, pred_len * channels)
        self.channels = channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden[:, :, -1]).reshape(x.shape[0], self.pred_len, self.channels)

class TransformerNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 d_model: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.embed = nn.Linear(channels, d_model)
        self.position = nn.Parameter(torch.zeros(1, hist_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, pred_len * channels)
        self.channels = channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.embed(x) + self.position)
        return self.head(hidden[:, -1]).reshape(x.shape[0], self.pred_len, self.channels)

class PatchTSTNet(nn.Module):

    def __init__(self, hist_len: int = 16, pred_len: int = 4, channels: int = 600,
                 patch_len: int = 4, d_model: int = 64, heads: int = 4,
                 layers: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.patch_len = patch_len
        self.patches = hist_len // patch_len
        self.embed = nn.Linear(patch_len * channels, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model * self.patches, pred_len * channels)
        self.channels = channels
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = x.reshape(batch, self.patches, self.patch_len * self.channels)
        hidden = self.encoder(self.embed(tokens) + self.position).flatten(1)
        return self.head(hidden).reshape(batch, self.pred_len, self.channels)

class _BaselineLightning(BaseCSIModel):

    NETWORK: type[nn.Module] = DLinearNet
    MODEL_NAME = "BASELINE"

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
    return type(name, (_BaselineLightning,), {"NETWORK": network, "MODEL_NAME": name.split("_")[0]})

DLINEAR_TDD = _make("DLINEAR_TDD", DLinearNet)
DLINEAR_FDD = _make("DLINEAR_FDD", DLinearNet)
MLP_TDD = _make("MLP_TDD", MLPNet)
MLP_FDD = _make("MLP_FDD", MLPNet)
GRU_TDD = _make("GRU_TDD", GRUNet)
GRU_FDD = _make("GRU_FDD", GRUNet)
TCN_TDD = _make("TCN_TDD", TCNNet)
TCN_FDD = _make("TCN_FDD", TCNNet)
TRANSFORMER_TDD = _make("TRANSFORMER_TDD", TransformerNet)
TRANSFORMER_FDD = _make("TRANSFORMER_FDD", TransformerNet)
PATCHTST_TDD = _make("PATCHTST_TDD", PatchTSTNet)
PATCHTST_FDD = _make("PATCHTST_FDD", PatchTSTNet)

def _register() -> None:
    from src.cp.models import PREDICTORS

    for name in ("DLINEAR", "MLP", "GRU", "TCN", "TRANSFORMER", "PATCHTST"):
        setattr(PREDICTORS, f"{name}_TDD", globals()[f"{name}_TDD"])
        setattr(PREDICTORS, f"{name}_FDD", globals()[f"{name}_FDD"])

_register()

__all__ = [
    "DLinearNet", "GRUNet", "MLPNet", "PatchTSTNet", "TCNNet", "TransformerNet",
]
