import torch
import torch.nn as nn
import torch.nn.functional as F

from deepwarp.operations import standardize


class TraceWindow(nn.Module):
    def __init__(self, settings, points, time_from, time_to):
        super().__init__()
        self.points = points
        self.time_from = time_from
        self.scale = (points - 1) / (time_to - time_from)
        self.window = settings.window
        self.pool = settings.window // settings.window_bins
        self.output_size = settings.window_channels * settings.window_bins
        padding = settings.window_kernel // 2
        self.conv_1 = nn.Conv1d(1, settings.window_channels, settings.window_kernel, padding=padding)
        self.conv_2 = nn.Conv1d(settings.window_channels, settings.window_channels, settings.window_kernel, padding=padding)
        half = settings.window // 2
        self.register_buffer("offsets", torch.arange(-half, settings.window - half), persistent=False)

    def cut(self, target_traces, center_rt):
        center = torch.round((center_rt.detach() - self.time_from) * self.scale).long().clamp(0, self.points - 1)
        positions = (center.unsqueeze(-1) + self.offsets).clamp(0, self.points - 1)
        return torch.gather(target_traces, 2, positions).clamp_min(0.0)

    def forward(self, target_traces, center_rt):
        batch, targets, _ = target_traces.shape
        windows = standardize(torch.log1p(self.cut(target_traces, center_rt)), dimension=2, unbiased=True)
        shape = F.gelu(self.conv_2(F.gelu(self.conv_1(windows.reshape(batch * targets, 1, self.window)))))
        return F.max_pool1d(shape, self.pool).reshape(batch, targets, self.output_size)
