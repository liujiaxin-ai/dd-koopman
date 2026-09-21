"""MambaCSP (arXiv 2604.21957) as a baseline on the CSI-4CAST protocol.

The point of this file is a *same-protocol* comparison. MambaCSP reports its
efficiency on DMRS data with its own NMSE/SE protocol; the lightweight claim of
this project ("1/442 of the parameters") can only be tested against it if both
models see identical tensors, the same optimiser budget and the same 162-setting
evaluation. The vendored architecture is used unchanged apart from one import
line, and the fused SSM kernel is replaced by `PureTorchMamba2` because the
container has no CUDA toolchain (see `mambacsp_fallback.py`).

Usage:
    `model.name: MAMBACSP_TDD` in a config, with `params` forwarded to the
    vendored `Model` (`d_model`, `mamba_layers`, `patch_size`, ...).
"""

from __future__ import annotations

import importlib.util
import sys
import types

import torch

from src.cp.config.config import ExperimentConfig
from src.cp.loss.loss import NMSELoss
from src.cp.models.common.base import BaseCSIModel

if importlib.util.find_spec("mamba_ssm") is None:
    from src.cp.models.baseline.mambacsp_fallback import PureTorchMamba2

    _shim = types.ModuleType("mamba_ssm")
    _shim.Mamba2 = PureTorchMamba2
    sys.modules["mamba_ssm"] = _shim

from src.cp.models.baseline.mambacsp_vendor.model import Model as MambaCSPModel

class _MambaCSPLightning(BaseCSIModel):

    MODEL_NAME = "MAMBACSP"

    def __init__(self, config: ExperimentConfig, *args, **kwargs) -> None:
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )
        self.name = self.MODEL_NAME
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})
        params = dict(config.model.params)

        params.setdefault("use_gpu", 0)
        params.setdefault("gpu_id", 0)
        self.model = MambaCSPModel(**params)
        self.metric = NMSELoss()

    def __str__(self) -> str:
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        hist, target = batch[0], batch[1]
        pred = self(hist)
        loss = self.metric(pred, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        hist, target = batch[0], batch[1]
        loss = self.metric(self(hist), target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

class MAMBACSP_TDD(_MambaCSPLightning):

    MODEL_NAME = "MAMBACSP"

class MAMBACSP_FDD(_MambaCSPLightning):

    MODEL_NAME = "MAMBACSP"

def _register() -> None:
    from src.cp.models import PREDICTORS

    PREDICTORS.MAMBACSP_TDD = MAMBACSP_TDD
    PREDICTORS.MAMBACSP_FDD = MAMBACSP_FDD

_register()

__all__ = ["MAMBACSP_FDD", "MAMBACSP_TDD"]
