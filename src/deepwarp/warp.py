import math

import torch
import torch.nn as nn
import torch.nn.functional as F

slope_shift = math.log(math.e - 1.0)


def build_segment_average(points, segments):
    index = torch.arange(segments)
    first = (index * points) // segments
    last = ((index + 1) * points + segments - 1) // segments
    point = torch.arange(points).unsqueeze(1)
    inside = ((point >= first.unsqueeze(0)) & (point < last.unsqueeze(0))).float()
    return inside / (last - first).float().unsqueeze(0)


class Warp(nn.Module):
    def __init__(self, settings, points, time_from, time_to):
        super().__init__()
        channels = settings.channels
        self.time_from = time_from
        self.segments = settings.segments
        self.segment_width = (time_to - time_from) / settings.segments
        self.start_limit = settings.start_limit
        self.slope_hidden = nn.Linear(channels * 3, settings.warp_hidden)
        self.slope_output = nn.Linear(settings.warp_hidden, 1)
        self.start = nn.Linear(channels * 2, 1)
        for layer in (self.slope_output, self.start):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.register_buffer("segment_average", build_segment_average(points, settings.segments), persistent=False)
        self.register_buffer("sum_before_node", torch.triu(torch.ones(settings.segments, settings.segments + 1), diagonal=1), persistent=False)

    def summarize(self, feature_map):
        whole_run = torch.cat([feature_map.mean(dim=2), feature_map.amax(dim=2)], dim=1)
        per_segment = (feature_map @ self.segment_average).transpose(1, 2)
        return whole_run, per_segment

    def get_nodes(self, whole_run, per_segment):
        whole_run_per_segment = whole_run.unsqueeze(1).expand(-1, self.segments, -1)
        hidden = F.gelu(self.slope_hidden(torch.cat([per_segment, whole_run_per_segment], dim=2)))
        slope = F.softplus(self.slope_output(hidden).squeeze(2) + slope_shift)
        start = self.time_from + self.start_limit * torch.tanh(self.start(whole_run))
        nodes = start + (slope * self.segment_width) @ self.sum_before_node
        return nodes, slope

    def read(self, nodes, reference_rt):
        segment = torch.floor((reference_rt - self.time_from) / self.segment_width).clamp(0, self.segments - 1)
        fraction = (reference_rt - (self.time_from + segment * self.segment_width)) / self.segment_width
        low = nodes.index_select(1, segment.long())
        high = nodes.index_select(1, segment.long() + 1)
        return low + fraction * (high - low)

    def forward(self, feature_map, reference_rt):
        whole_run, per_segment = self.summarize(feature_map)
        nodes, slope = self.get_nodes(whole_run, per_segment)
        return self.read(nodes, reference_rt), nodes, slope, whole_run
