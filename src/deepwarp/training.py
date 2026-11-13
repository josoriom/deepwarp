import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from deepwarp.augment import Augmentation
from deepwarp.loss import retention_time_loss, uncertainty_loss
from deepwarp.model import DeepWarp, Layout
from deepwarp.samples import SamplePool, TraceLayout, get_reference_rt
from deepwarp.settings import ModelSettings, TrainingSettings, to_dictionary

retention_time_step = "retention time"
uncertainty_step = "uncertainty"


class TraceDataset(Dataset):
    def __init__(self, samples, augmentation=None):
        self.traces = samples.traces
        self.labels = samples.labels
        self.augmentation = augmentation

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, row):
        traces = self.traces[row].astype(np.float32)
        chromatogram = traces.sum(axis=0)
        labels = self.labels[row].astype(np.float32)
        if self.augmentation is not None:
            chromatogram, traces, labels = self.augmentation(chromatogram, traces, labels)
        return torch.from_numpy(chromatogram), torch.from_numpy(traces), torch.from_numpy(labels)


class TargetInputs:
    def __init__(self, trace_layout, reference_rt, device):
        self.trace_index = torch.as_tensor(trace_layout.trace_index, device=device)
        self.mz = torch.as_tensor(trace_layout.target_mz, device=device)
        self.reference_rt = torch.as_tensor(reference_rt, dtype=torch.float32, device=device)
        self.same_trace = trace_layout.get_same_trace().to(device)

    def run(self, model, chromatogram, traces):
        target_traces = traces.index_select(1, self.trace_index)
        return model(chromatogram, target_traces, self.mz, self.reference_rt, self.same_trace)


class BestWeights:
    def __init__(self, model):
        self.reset(model)

    def reset(self, model):
        self.score = float("inf")
        self.weights = copy.deepcopy(model.state_dict())
        self.epoch = 0
        self.epochs_without_gain = 0

    def check(self, score, model, epoch):
        if score < self.score:
            self.score = score
            self.weights = copy.deepcopy(model.state_dict())
            self.epoch = epoch
            self.epochs_without_gain = 0
        else:
            self.epochs_without_gain += 1

    def restore(self, model):
        model.load_state_dict(self.weights)


def select_device(name=""):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_batch(batch, device):
    chromatogram, traces, labels = (part.to(device) for part in batch)
    has_label = torch.isfinite(labels).float()
    return chromatogram, traces, torch.nan_to_num(labels), has_label


def measure(model, loader, target_inputs, device):
    model.eval()
    errors, gaussian_total, label_count = [], 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            chromatogram, traces, labels, has_label = read_batch(batch, device)
            outputs = target_inputs.run(model, chromatogram, traces)
            errors.append((outputs["rt"] - labels).abs()[has_label > 0].cpu().numpy())
            count = float(has_label.sum())
            gaussian_total += uncertainty_loss(outputs, labels, has_label, None).item() * count
            label_count += count
    errors = np.concatenate(errors) if errors else np.zeros(0)
    if errors.size == 0:
        return {"mae": float("nan"), "within_0.05": float("nan"), "gaussian_loss": float("nan"), "count": 0}
    return {
        "mae": float(errors.mean()),
        "within_0.05": float((errors <= 0.05).mean()),
        "gaussian_loss": gaussian_total / max(label_count, 1.0),
        "count": int(errors.size),
    }


def allow_training(model, step):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("uncertainty.") == (step == uncertainty_step)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(trainable, settings, epochs):
    optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def run_epoch(model, step, loader, target_inputs, optimizer, trainable, settings, device):
    model.train(step == retention_time_step)
    loss_function = retention_time_loss if step == retention_time_step else uncertainty_loss
    loss_total, sample_count = 0.0, 0
    for batch in loader:
        chromatogram, traces, labels, has_label = read_batch(batch, device)
        optimizer.zero_grad()
        loss = loss_function(target_inputs.run(model, chromatogram, traces), labels, has_label, settings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, settings.gradient_clip)
        optimizer.step()
        loss_total += loss.item() * len(labels)
        sample_count += len(labels)
    return loss_total / max(sample_count, 1)


def write_epoch(log, step, epoch, loss, metrics):
    log(
        f"{step}  epoch {epoch:3d}  loss {loss:.5f}  validation mae {metrics['mae']:.4f}"
        f"  within 0.05 min {metrics['within_0.05']:.3f}  gaussian loss {metrics['gaussian_loss']:.3f}"
    )


class Training:
    def __init__(self, targets, settings=TrainingSettings(), model_settings=ModelSettings(), log=print):
        self.settings = settings
        self.model_settings = model_settings
        self.log = log or (lambda line: None)
        set_seed(settings.seed)
        self.device = select_device(settings.device)
        self.trace_layout = TraceLayout(targets)
        self.pool = SamplePool(settings.validation_fraction, settings.seed)
        self.model = None
        self.epoch = 0
        self.finished = False

    def add(self, dataset, labels):
        self.check_not_finished()
        traces = self.trace_layout.read_traces(dataset)
        has_new_validation = self.pool.add(traces, self.trace_layout.read_labels(labels, len(traces)))
        if self.model is None:
            self.start()
        elif has_new_validation:
            self.best.reset(self.model)

    def start(self):
        settings = self.settings
        if self.pool.count_labels() == 0:
            raise ValueError("the first data added has no labels to learn from")
        layout = self.trace_layout
        reference_rt = get_reference_rt(layout.target_rt, self.pool.training.labels) if settings.reference_from_labels else layout.target_rt
        self.target_inputs = TargetInputs(layout, reference_rt, self.device)
        mz = layout.target_mz.astype(np.float64)
        model_layout = Layout(self.pool.training.points, settings.time_from, settings.time_to, float(mz.mean()), float(mz.std() + 1e-6))
        self.model = DeepWarp(model_layout, self.model_settings).to(self.device)
        self.augmentation = Augmentation(settings.augmentation, settings.time_from, settings.time_to, model_layout.points, settings.seed)
        self.trainable = allow_training(self.model, retention_time_step)
        self.optimizer, self.scheduler = build_optimizer(self.trainable, settings, settings.epochs)
        self.best = BestWeights(self.model)
        self.log(f"{len(layout.targets)} targets, {self.model.count_weights()} weights, device {self.device}")

    def get_loaders(self):
        training_set = TraceDataset(self.pool.training, self.augmentation)
        sampler = RandomSampler(training_set, replacement=True, num_samples=max(self.settings.samples_per_epoch, len(training_set)))
        train_loader = DataLoader(training_set, batch_size=self.settings.batch_size, sampler=sampler)
        validation_loader = DataLoader(TraceDataset(self.pool.get_validation()), batch_size=self.settings.batch_size)
        return train_loader, validation_loader

    def train(self):
        self.check_not_finished()
        if self.model is None:
            raise RuntimeError("add data before train")
        if self.epoch >= self.settings.epochs:
            raise RuntimeError(f"all {self.settings.epochs} planned epochs are done: use larger groups or more epochs")
        train_loader, validation_loader = self.get_loaders()
        loss = run_epoch(self.model, retention_time_step, train_loader, self.target_inputs, self.optimizer, self.trainable, self.settings, self.device)
        self.scheduler.step()
        self.epoch += 1
        metrics = measure(self.model, validation_loader, self.target_inputs, self.device)
        write_epoch(self.log, retention_time_step, self.epoch, loss, metrics)
        self.best.check(metrics["mae"], self.model, self.epoch)
        return metrics

    def finish(self):
        self.check_not_finished()
        if self.model is None:
            raise RuntimeError("add data before finish")
        self.log(f"{self.pool.training.count} training and {self.pool.get_validation().count} validation samples")
        while self.epoch < self.settings.epochs and self.best.epochs_without_gain < self.settings.patience:
            self.train()
        if self.epoch < self.settings.epochs:
            self.log(f"{retention_time_step}  stopped early at epoch {self.epoch}")
        self.best.restore(self.model)
        self.log(f"{retention_time_step}  best epoch {self.best.epoch}")
        if self.settings.uncertainty_epochs > 0:
            self.train_uncertainty()
        metrics = measure(self.model, self.get_loaders()[1], self.target_inputs, self.device)
        self.log(f"validation  mae {metrics['mae']:.4f} min  within 0.05 min {metrics['within_0.05']:.3f}  of {metrics['count']} labels")
        self.model.description = self.get_description(metrics)
        self.finished = True
        return self.model.eval()

    def train_uncertainty(self):
        settings = self.settings
        trainable = allow_training(self.model, uncertainty_step)
        optimizer, scheduler = build_optimizer(trainable, settings, settings.uncertainty_epochs)
        best = BestWeights(self.model)
        for epoch in range(1, settings.uncertainty_epochs + 1):
            train_loader, validation_loader = self.get_loaders()
            loss = run_epoch(self.model, uncertainty_step, train_loader, self.target_inputs, optimizer, trainable, settings, self.device)
            scheduler.step()
            metrics = measure(self.model, validation_loader, self.target_inputs, self.device)
            write_epoch(self.log, uncertainty_step, epoch, loss, metrics)
            best.check(metrics["gaussian_loss"], self.model, epoch)
            if best.epochs_without_gain >= settings.patience:
                self.log(f"{uncertainty_step}  stopped early at epoch {epoch}")
                break
        best.restore(self.model)
        self.log(f"{uncertainty_step}  best epoch {best.epoch}")

    def get_description(self, metrics):
        reference_rt = self.target_inputs.reference_rt.cpu().numpy()
        return {
            "targets": [{"id": target["id"], "mz": target["mz"], "rt": round(float(rt), 6)} for target, rt in zip(self.trace_layout.targets, reference_rt)],
            "context_mz": [float(mz) for mz in self.trace_layout.trace_mz],
            "eic_ppm": self.settings.eic_ppm,
            "eic_mz_tolerance": self.settings.eic_mz_tolerance,
            "validation": metrics,
            "training": to_dictionary(self.settings),
        }

    def check_not_finished(self):
        if self.finished:
            raise RuntimeError("the training is finished: start a new Training")


def train(targets, dataset, labels, settings=TrainingSettings(), model_settings=ModelSettings(), log=print):
    training = Training(targets, settings, model_settings, log)
    training.add(dataset, labels)
    return training.finish()
