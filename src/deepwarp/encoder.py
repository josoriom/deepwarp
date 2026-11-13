import torch.nn as nn
import torch.nn.functional as F

from deepwarp.operations import group_norm


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, norm_groups, dropout):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv_1 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.norm_1 = nn.GroupNorm(norm_groups, channels)
        self.conv_2 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.norm_2 = nn.GroupNorm(norm_groups, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        step = self.dropout(F.gelu(group_norm(self.norm_1, self.conv_1(values))))
        step = self.dropout(F.gelu(group_norm(self.norm_2, self.conv_2(step))))
        return values + step


class Encoder(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.stem = nn.Conv1d(1, settings.channels, settings.kernel, padding=settings.kernel // 2)
        self.blocks = nn.ModuleList(
            ResidualBlock(settings.channels, settings.kernel, dilation, settings.norm_groups, settings.dropout)
            for dilation in settings.dilations
        )

    def forward(self, chromatogram):
        feature_map = self.stem(chromatogram.unsqueeze(1))
        for block in self.blocks:
            feature_map = block(feature_map)
        return feature_map
