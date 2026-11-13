from dataclasses import dataclass

import numpy as np
import torch


def group_traces(targets, same_mz):
    trace_mz = []
    first_target = []
    trace_index = []
    for number, target in enumerate(targets):
        match = next((index for index, mz in enumerate(trace_mz) if abs(mz - target["mz"]) < same_mz), None)
        if match is None:
            trace_mz.append(target["mz"])
            first_target.append(number)
            match = len(trace_mz) - 1
        trace_index.append(match)
    return np.array(trace_mz, dtype=np.float64), np.array(first_target, dtype=np.int64), np.array(trace_index, dtype=np.int64)


class TraceLayout:
    def __init__(self, targets, same_mz=0.0005):
        self.targets = [{"id": str(target["id"]), "mz": float(target["mz"]), "rt": float(target["rt"])} for target in targets]
        self.trace_mz, self.first_target, self.trace_index = group_traces(self.targets, same_mz)

    @property
    def target_mz(self):
        return np.array([target["mz"] for target in self.targets], dtype=np.float32)

    @property
    def target_rt(self):
        return np.array([target["rt"] for target in self.targets], dtype=np.float32)

    def get_same_trace(self):
        index = torch.as_tensor(self.trace_index)
        return index.unsqueeze(0) == index.unsqueeze(1)

    def read_traces(self, dataset):
        count = len(self.targets)
        dataset = np.asarray(dataset, dtype=np.float32)
        if dataset.ndim == 2 and dataset.shape[0] % count == 0:
            dataset = dataset.reshape(-1, count, dataset.shape[1])
        if dataset.ndim != 3 or dataset.shape[1] != count:
            raise ValueError(
                f"dataset must hold one trace per target for each sample, in the order of the targets:"
                f" (samples x {count}) x points or samples x {count} x points, got {dataset.shape}"
            )
        if not np.isfinite(dataset).all():
            raise ValueError("dataset has values that are nan or infinite")
        return np.ascontiguousarray(dataset[:, self.first_target])

    def read_labels(self, labels, sample_count):
        target_count = len(self.targets)
        if labels is None:
            return np.full((sample_count, target_count), np.nan, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if labels.size != sample_count * target_count:
            raise ValueError(f"labels must hold one retention time per target for each sample: {sample_count} x {target_count}, got {labels.shape}")
        return labels.reshape(sample_count, target_count)


@dataclass
class SampleSet:
    traces: np.ndarray
    labels: np.ndarray

    @property
    def count(self):
        return self.traces.shape[0]

    @property
    def points(self):
        return self.traces.shape[2]

    def join(self, other):
        return SampleSet(np.concatenate([self.traces, other.traces]), np.concatenate([self.labels, other.labels]))


def get_reference_rt(target_rt, labels):
    reference = target_rt.copy()
    for target in range(labels.shape[1]):
        found = labels[:, target][np.isfinite(labels[:, target])]
        if len(found):
            reference[target] = float(np.median(found))
    return reference


class SamplePool:
    def __init__(self, validation_fraction, seed):
        self.validation_fraction = validation_fraction
        self.random = np.random.default_rng(seed)
        self.training = None
        self.validation = None

    def split(self, count):
        order = self.random.permutation(count)
        validation_count = int(round(count * self.validation_fraction))
        if count < 5 or validation_count == 0:
            return sorted(order.tolist()), []
        return sorted(order[validation_count:].tolist()), sorted(order[:validation_count].tolist())

    def add(self, traces, labels):
        if self.training is not None and traces.shape[2] != self.training.points:
            raise ValueError(f"every sample must have {self.training.points} points, got {traces.shape[2]}")
        training_rows, validation_rows = self.split(len(traces))
        self.training = self.join(self.training, SampleSet(traces[training_rows], labels[training_rows]))
        if validation_rows:
            self.validation = self.join(self.validation, SampleSet(traces[validation_rows], labels[validation_rows]))
        return len(validation_rows) > 0

    def join(self, current, new):
        return new if current is None else current.join(new)

    def get_validation(self):
        return self.validation if self.validation is not None else self.training

    def count_labels(self):
        return int(np.isfinite(self.training.labels).sum()) if self.training is not None else 0
