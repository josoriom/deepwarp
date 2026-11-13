import numpy as np
import pytest

targets = [
    {"id": "first", "mz": 120.0, "rt": 1.2},
    {"id": "second", "mz": 150.0, "rt": 2.0},
    {"id": "isomer_early", "mz": 180.0, "rt": 2.6},
    {"id": "isomer_late", "mz": 180.0, "rt": 2.75},
    {"id": "third", "mz": 210.0, "rt": 3.4},
    {"id": "fourth", "mz": 240.0, "rt": 4.1},
]


def make_data(count, points=512, seed=0):
    random = np.random.default_rng(seed)
    grid = np.linspace(0.5, 5.0, points)
    dataset = np.zeros((count * len(targets), points), dtype=np.float32)
    labels = np.zeros((count, len(targets)), dtype=np.float32)
    for sample in range(count):
        shift = random.normal(0.0, 0.08)
        stretch = 1.0 + random.normal(0.0, 0.02)
        noise = random.uniform(0, 50, size=points)
        labels[sample] = [0.5 + (target["rt"] - 0.5) * stretch + shift for target in targets]
        for row, target in enumerate(targets):
            same_mz = [number for number, other in enumerate(targets) if other["mz"] == target["mz"]]
            peaks = sum(1e5 * np.exp(-0.5 * ((grid - labels[sample, number]) / 0.03) ** 2) for number in same_mz)
            dataset[sample * len(targets) + row] = peaks + noise
    labels[0, 0] = np.nan
    return targets, dataset, labels


@pytest.fixture
def data():
    return make_data(12)
