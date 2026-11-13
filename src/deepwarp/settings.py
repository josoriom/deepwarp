from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelSettings:
    channels: int = 64
    kernel: int = 3
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
    norm_groups: int = 8
    dropout: float = 0.1
    segments: int = 15
    start_limit: float = 0.3
    warp_hidden: int = 64
    correction_hidden: int = 64
    correction_limit: float = 0.12
    window: int = 64
    window_channels: int = 16
    window_kernel: int = 9
    window_bins: int = 8
    attention_heads: int = 4
    order_projection: bool = True
    gap_fraction: float = 0.25
    gap_floor: float = 0.002
    uncertainty_min: float = 0.001
    uncertainty_max: float = 0.2
    uncertainty_start: float = 0.05


@dataclass(frozen=True)
class AugmentationSettings:
    warp_probability: float = 0.9
    warp_segments: int = 6
    warp_strength: float = 0.15
    start_jitter: float = 0.05
    scale_probability: float = 0.5
    scale_sigma: float = 0.1
    noise_probability: float = 0.5
    noise_sigma: float = 0.01


@dataclass(frozen=True)
class TrainingSettings:
    time_from: float = 0.5
    time_to: float = 5.0
    eic_ppm: float = 20.0
    eic_mz_tolerance: float = 0.005
    epochs: int = 80
    uncertainty_epochs: int = 80
    patience: int = 80
    batch_size: int = 64
    samples_per_epoch: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.1
    reference_from_labels: bool = True
    huber_delta: float = 0.03
    correction_weight: float = 0.01
    smoothness_weight: float = 1e-3
    seed: int = 0
    device: str = ""
    augmentation: AugmentationSettings = field(default_factory=AugmentationSettings)


def to_dictionary(settings):
    return asdict(settings)


def model_settings_from(dictionary):
    return ModelSettings(**{**dictionary, "dilations": tuple(dictionary["dilations"])})
