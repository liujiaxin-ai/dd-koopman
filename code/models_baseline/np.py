"""No-Prediction Baseline Model for CSI Prediction.

Simple baseline that repeats the last observed CSI value for all future time steps.
Useful for establishing baseline performance metrics.

Attribution:
    Modified from: [LLM4CP](https://github.com/liuboxun/LLM4CP/blob/master/models/model.py)

Usage:
    model = NOPREDICTMODEL(pred_len=4)
    predictions = model(historical_csi)
"""

import torch.nn as nn

from src.utils.data_utils import PRED_LEN

class NOPREDICTMODEL(nn.Module):

    def __init__(self, pred_len=PRED_LEN, *args, **kwargs):
        super().__init__()
        self.is_separate_antennas = True
        self.name = "NP"
        self.pred_len = pred_len

    def __str__(self) -> str:
        return self.name

    def forward(self, x):
        last_step = x[:, [-1], :]
        return last_step.repeat(1, self.pred_len, 1)
