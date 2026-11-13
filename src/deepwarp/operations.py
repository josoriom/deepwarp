import torch
import torch.nn.functional as F


def is_exporting():
    return torch.onnx.is_in_onnx_export()


def group_norm(norm, values):
    if not is_exporting():
        return F.group_norm(values, norm.num_groups, norm.weight, norm.bias, norm.eps)
    batch, channels, length = values.shape
    grouped = values.reshape(batch, norm.num_groups, -1)
    mean = grouped.mean(dim=2, keepdim=True)
    variance = ((grouped - mean) ** 2).mean(dim=2, keepdim=True)
    normed = ((grouped - mean) / torch.sqrt(variance + norm.eps)).reshape(batch, channels, length)
    return normed * norm.weight.view(1, -1, 1) + norm.bias.view(1, -1, 1)


def layer_norm(norm, values):
    if not is_exporting():
        return F.layer_norm(values, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    mean = values.mean(dim=-1, keepdim=True)
    variance = ((values - mean) ** 2).mean(dim=-1, keepdim=True)
    return (values - mean) / torch.sqrt(variance + norm.eps) * norm.weight + norm.bias


def standardize(values, dimension, unbiased=False):
    mean = values.mean(dim=dimension, keepdim=True)
    count = values.shape[dimension] - (1 if unbiased else 0)
    spread = torch.sqrt(((values - mean) ** 2).sum(dim=dimension, keepdim=True) / count)
    return (values - mean) / (spread + 1e-6)


def normalize_chromatogram(chromatogram):
    return standardize(torch.log1p(chromatogram.clamp_min(0.0)), dimension=-1)
