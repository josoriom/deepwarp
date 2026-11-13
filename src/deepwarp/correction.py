import torch
import torch.nn as nn
import torch.nn.functional as F

from deepwarp.isomer_attention import IsomerAttention
from deepwarp.order_projection import project_order
from deepwarp.trace_window import TraceWindow


def join_inputs(whole_run, target_features, window_features):
    batch, targets, _ = window_features.shape
    return torch.cat(
        [whole_run.unsqueeze(1).expand(-1, targets, -1), target_features.unsqueeze(0).expand(batch, -1, -1), window_features],
        dim=2,
    )


class Correction(nn.Module):
    def __init__(self, settings, points, time_from, time_to):
        super().__init__()
        self.limit = settings.correction_limit
        self.order_projection = settings.order_projection
        self.gap_fraction = settings.gap_fraction
        self.gap_floor = settings.gap_floor
        self.window = TraceWindow(settings, points, time_from, time_to)
        self.linear_1 = nn.Linear(settings.channels * 2 + 2 + self.window.output_size, settings.correction_hidden)
        self.isomers = IsomerAttention(settings.correction_hidden, settings.attention_heads)
        self.linear_2 = nn.Linear(settings.correction_hidden, 1)
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, whole_run, target_features, target_traces, warped, same_trace, reference_rt):
        inputs = join_inputs(whole_run, target_features, self.window(target_traces, warped))
        hidden = self.isomers(F.gelu(self.linear_1(inputs)), same_trace)
        position = warped + self.limit * torch.tanh(self.linear_2(hidden).squeeze(2))
        if self.order_projection:
            return project_order(position, same_trace, reference_rt, self.gap_fraction, self.gap_floor)
        return position
