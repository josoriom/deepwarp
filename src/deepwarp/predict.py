import numpy as np
import torch

from deepwarp.samples import SampleSet, TraceLayout
from deepwarp.training import TargetInputs, TraceDataset, read_batch


def get_reference_rt(model, targets):
    known = {target["id"]: target["rt"] for target in model.description.get("targets", [])}
    return np.array([known.get(str(target["id"]), target["rt"]) for target in targets], dtype=np.float32)


def predict(model, targets, dataset, batch_size=64):
    trace_layout = TraceLayout(targets)
    traces = trace_layout.read_traces(dataset)
    if traces.shape[2] != model.layout.points:
        raise ValueError(f"the model reads traces of {model.layout.points} points, the dataset has {traces.shape[2]}")
    device = next(model.parameters()).device
    target_inputs = TargetInputs(trace_layout, get_reference_rt(model, targets), device)
    samples = SampleSet(traces, trace_layout.read_labels(None, len(traces)))
    loader = torch.utils.data.DataLoader(TraceDataset(samples), batch_size=batch_size)
    rt, uncertainty = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            chromatogram, traces, _, _ = read_batch(batch, device)
            outputs = target_inputs.run(model, chromatogram, traces)
            rt.append(outputs["rt"].cpu().numpy())
            uncertainty.append(outputs["uncertainty"].cpu().numpy())
    return {"rt": np.concatenate(rt), "uncertainty": np.concatenate(uncertainty)}
