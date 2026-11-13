from dataclasses import dataclass

import torch
import torch.nn as nn

from deepwarp.correction import Correction
from deepwarp.encoder import Encoder
from deepwarp.operations import normalize_chromatogram
from deepwarp.settings import ModelSettings
from deepwarp.uncertainty import Uncertainty
from deepwarp.warp import Warp


@dataclass(frozen=True)
class Layout:
    points: int
    time_from: float
    time_to: float
    mz_mean: float
    mz_std: float


class DeepWarp(nn.Module):
    def __init__(self, layout, settings=ModelSettings()):
        super().__init__()
        self.layout = layout
        self.settings = settings
        self.description = {}
        grid = (layout.points, layout.time_from, layout.time_to)
        self.encoder = Encoder(settings)
        self.warp = Warp(settings, *grid)
        self.correction = Correction(settings, *grid)
        self.uncertainty = Uncertainty(settings, *grid)

    def get_target_features(self, target_mz, reference_rt):
        layout = self.layout
        time = (reference_rt - layout.time_from) / (layout.time_to - layout.time_from)
        mz = (target_mz - layout.mz_mean) / layout.mz_std
        return torch.stack([time, mz], dim=1)

    def forward(self, chromatogram, target_traces, target_mz, reference_rt, same_trace):
        feature_map = self.encoder(normalize_chromatogram(chromatogram))
        warped, nodes, slope, whole_run = self.warp(feature_map, reference_rt)
        target_features = self.get_target_features(target_mz, reference_rt)
        rt = self.correction(whole_run, target_features, target_traces, warped, same_trace, reference_rt)
        uncertainty = self.uncertainty(whole_run, target_features, target_traces, rt)
        return {"rt": rt, "uncertainty": uncertainty, "warped": warped, "nodes": nodes, "slope": slope}

    def count_weights(self):
        return sum(parameter.numel() for parameter in self.parameters())
