import torch.nn.functional as F


def masked_huber(rt, labels, mask, delta):
    per_target = F.huber_loss(rt, labels, reduction="none", delta=delta) * mask
    return per_target.sum() / mask.sum().clamp_min(1.0)


def masked_gaussian(rt, labels, mask, uncertainty):
    per_target = F.gaussian_nll_loss(rt, labels, uncertainty**2, reduction="none") * mask
    return per_target.sum() / mask.sum().clamp_min(1.0)


def warp_smoothness(slope):
    return ((slope[:, 1:] - slope[:, :-1]) ** 2).mean()


def retention_time_loss(outputs, labels, mask, settings):
    correction = ((outputs["rt"] - outputs["warped"]) ** 2).mean()
    return (
        masked_huber(outputs["rt"], labels, mask, settings.huber_delta)
        + settings.correction_weight * correction
        + settings.smoothness_weight * warp_smoothness(outputs["slope"])
    )


def uncertainty_loss(outputs, labels, mask, settings):
    return masked_gaussian(outputs["rt"], labels, mask, outputs["uncertainty"])
