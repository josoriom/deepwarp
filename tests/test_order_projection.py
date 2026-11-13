import torch
import torch.nn.functional as F

from deepwarp.order_projection import project_order


def project_with_loops(position, trace_index, reference_rt, gap_fraction, gap_floor):
    projected = position.clone()
    for trace in torch.unique(trace_index):
        members = torch.nonzero(trace_index == trace).flatten()
        if len(members) < 2:
            continue
        members = members[torch.argsort(reference_rt[members], stable=True)]
        own = position[:, members]
        reference = reference_rt[members]
        gap = (gap_fraction * (reference[1:] - reference[:-1])).clamp(min=gap_floor)
        offset = F.pad(torch.cumsum(gap, dim=0), (1, 0))
        pushed_forward = torch.cummax(own - offset, dim=1).values + offset
        pushed_back = offset - torch.cummax((offset - own).flip(1), dim=1).values.flip(1)
        projected[:, members] = 0.5 * (pushed_forward + pushed_back)
    return projected


def test_matches_the_loop_version():
    generator = torch.Generator().manual_seed(0)
    trace_index = torch.tensor([0, 1, 1, 2, 2, 2, 3, 1, 2])
    reference_rt = torch.rand(9, generator=generator) * 4 + 0.5
    position = reference_rt + torch.randn(16, 9, generator=generator) * 0.3
    same_trace = trace_index.unsqueeze(0) == trace_index.unsqueeze(1)
    expected = project_with_loops(position, trace_index, reference_rt, 0.25, 0.002)
    found = project_order(position, same_trace, reference_rt, 0.25, 0.002)
    assert torch.allclose(found, expected, atol=1e-6)


def test_keeps_the_order_inside_each_trace():
    reference_rt = torch.tensor([2.0, 2.2, 2.4])
    position = torch.tensor([[2.5, 2.1, 2.3]])
    same_trace = torch.ones(3, 3, dtype=torch.bool)
    found = project_order(position, same_trace, reference_rt, 0.25, 0.002)[0]
    assert bool((found[1:] - found[:-1] >= 0.05 - 1e-6).all())
