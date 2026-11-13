import torch

lowest = -1e9


def get_order(same_trace, reference_rt):
    later_rt = reference_rt.unsqueeze(1) > reference_rt.unsqueeze(0)
    same_rt = reference_rt.unsqueeze(1) == reference_rt.unsqueeze(0)
    index = torch.arange(reference_rt.shape[0], device=reference_rt.device)
    later_index = index.unsqueeze(1) > index.unsqueeze(0)
    itself = index.unsqueeze(1) == index.unsqueeze(0)
    return same_trace & (later_rt | (same_rt & later_index)), itself


def get_offsets(comes_before, reference_rt, gap_fraction, gap_floor):
    next_rt = torch.where(comes_before, reference_rt.unsqueeze(1), float("inf")).amin(dim=0)
    has_next = comes_before.float().amax(dim=0) > 0
    gap = torch.where(has_next, (gap_fraction * (next_rt - reference_rt)).clamp_min(gap_floor), 0.0)
    return comes_before.float() @ gap


def project_order(position, same_trace, reference_rt, gap_fraction, gap_floor):
    comes_before, itself = get_order(same_trace, reference_rt)
    offsets = get_offsets(comes_before, reference_rt, gap_fraction, gap_floor)
    self_or_before = comes_before | itself
    self_or_after = self_or_before.transpose(0, 1)
    pushed_forward = torch.where(self_or_before, (position - offsets).unsqueeze(1), lowest).amax(dim=2) + offsets
    pushed_back = offsets - torch.where(self_or_after, (offsets - position).unsqueeze(1), lowest).amax(dim=2)
    return 0.5 * (pushed_forward + pushed_back)
