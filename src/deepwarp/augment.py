import math

import numpy as np

slope_shift = math.log(math.e - 1.0)


def sample_random_warp(random, time_from, time_to, segments, strength, start_jitter):
    segment_width = (time_to - time_from) / segments
    before = np.linspace(time_from, time_to, segments + 1)
    slope = np.logaddexp(0.0, random.normal(0.0, strength, size=segments) + slope_shift)
    start = time_from + random.normal(0.0, start_jitter)
    after = np.concatenate([[start], start + np.cumsum(slope * segment_width)])
    return before, after


class Augmentation:
    def __init__(self, settings, time_from, time_to, points, seed):
        self.settings = settings
        self.time_from = time_from
        self.time_to = time_to
        self.grid = np.linspace(time_from, time_to, points)
        self.random = np.random.default_rng(seed)

    def warp(self, chromatogram, traces, labels):
        settings = self.settings
        before, after = sample_random_warp(self.random, self.time_from, self.time_to, settings.warp_segments, settings.warp_strength, settings.start_jitter)
        labels = np.interp(labels, before, after)
        read_at = np.clip(np.interp(self.grid, after, before), self.time_from, self.time_to)
        position = (read_at - self.time_from) / (self.time_to - self.time_from) * (len(self.grid) - 1)
        low = np.clip(np.floor(position).astype(np.int64), 0, len(self.grid) - 2)
        weight = position - low
        chromatogram = np.interp(read_at, self.grid, chromatogram)
        traces = traces[:, low] * (1.0 - weight) + traces[:, low + 1] * weight
        return chromatogram, traces, labels

    def __call__(self, chromatogram, traces, labels):
        settings = self.settings
        if self.random.random() < settings.warp_probability:
            chromatogram, traces, labels = self.warp(chromatogram, traces, labels)
        if self.random.random() < settings.scale_probability:
            chromatogram = chromatogram * np.exp(self.random.normal(0.0, settings.scale_sigma))
        if self.random.random() < settings.noise_probability:
            spread = settings.noise_sigma * (chromatogram.std() + 1e-6)
            chromatogram = chromatogram + self.random.normal(0.0, spread, size=chromatogram.shape)
        return chromatogram.astype(np.float32), traces.astype(np.float32), labels.astype(np.float32)
