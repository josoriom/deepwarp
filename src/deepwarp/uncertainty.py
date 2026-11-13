import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepwarp.correction import join_inputs
from deepwarp.trace_window import TraceWindow


class Uncertainty(nn.Module):
    def __init__(self, settings, points, time_from, time_to):
        super().__init__()
        self.lowest = math.log(settings.uncertainty_min)
        self.highest = math.log(settings.uncertainty_max)
        self.window = TraceWindow(settings, points, time_from, time_to)
        self.linear_1 = nn.Linear(settings.channels * 2 + 2 + self.window.output_size, settings.correction_hidden)
        self.linear_2 = nn.Linear(settings.correction_hidden, 1)
        nn.init.zeros_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, math.log(settings.uncertainty_start))

    def forward(self, whole_run, target_features, target_traces, rt):
        inputs = join_inputs(whole_run, target_features, self.window(target_traces, rt))
        raw = self.linear_2(F.gelu(self.linear_1(inputs))).squeeze(2)
        middle = 0.5 * (self.lowest + self.highest)
        return torch.exp(self.lowest + (self.highest - self.lowest) * torch.sigmoid(raw - middle))
