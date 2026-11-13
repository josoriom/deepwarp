import torch

from deepwarp.model import DeepWarp, Layout
from deepwarp.settings import ModelSettings


def build(order_projection=True, points=1024):
    return DeepWarp(Layout(points, 0.5, 5.0, 300.0, 50.0), ModelSettings(order_projection=order_projection)).eval()


def run(model, batch=2, points=1024):
    target_mz = torch.tensor([260.103, 260.103, 300.0])
    reference_rt = torch.tensor([2.1, 2.3, 3.0])
    same_trace = (target_mz.unsqueeze(0) - target_mz.unsqueeze(1)).abs() < 0.0005
    chromatogram = torch.rand(batch, points) * 1000.0
    target_traces = torch.rand(batch, 3, points) * 1000.0
    with torch.no_grad():
        return model(chromatogram, target_traces, target_mz, reference_rt, same_trace), reference_rt


def test_same_weight_count_as_the_paper_model():
    assert build().count_weights() == 275940


def test_output_shapes():
    outputs, _ = run(build())
    assert outputs["rt"].shape == (2, 3)
    assert outputs["uncertainty"].shape == (2, 3)
    assert outputs["nodes"].shape == (2, 16)
    assert outputs["slope"].shape == (2, 15)


def test_new_model_returns_the_reference_rt():
    outputs, reference_rt = run(build(order_projection=False))
    assert torch.allclose(outputs["rt"], reference_rt.expand(2, -1), atol=1e-5)


def test_new_model_gives_one_uncertainty_inside_the_limits():
    outputs, _ = run(build())
    uncertainty = outputs["uncertainty"]
    assert torch.allclose(uncertainty, uncertainty[0, 0].expand(2, 3))
    assert 0.001 < float(uncertainty[0, 0]) < 0.2


def test_warp_nodes_always_increase():
    model = build()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    outputs, _ = run(model)
    assert bool((outputs["nodes"][:, 1:] > outputs["nodes"][:, :-1]).all())


def test_any_number_of_points():
    outputs, _ = run(build(points=400), points=400)
    assert outputs["rt"].shape == (2, 3)
