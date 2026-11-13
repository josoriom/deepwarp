import copy
import json

import torch
import torch.nn as nn

input_description = {
    "eics": "rows x (1 + points), m/z then raw intensity; one row per m/z, covering context_mz and every target m/z",
    "targets": "n x 2, m/z and reference RT in minutes",
}
output_description = {
    "rt": "n x 1, minutes",
    "uncertainty": "n x 1, minutes",
    "context_found": "1, how many context_mz have a row in eics",
}


class OnnxWarp(nn.Module):
    def __init__(self, model, same_mz):
        super().__init__()
        self.model = model
        self.same_mz = same_mz
        self.register_buffer("context_mz", torch.tensor(model.description["context_mz"], dtype=torch.float32))

    def match_rows(self, wanted_mz, row_mz):
        return ((wanted_mz.unsqueeze(1) - row_mz.unsqueeze(0)).abs() < self.same_mz).float()

    def pick_traces(self, matches, traces):
        return (matches / matches.sum(dim=1, keepdim=True).clamp_min(1.0)) @ traces

    def forward(self, eics, targets):
        row_mz = eics[:, 0]
        traces = eics[:, 1:]
        target_mz = targets[:, 0]
        reference_rt = targets[:, 1]
        context_matches = self.match_rows(self.context_mz, row_mz)
        chromatogram = self.pick_traces(context_matches, traces).sum(dim=0, keepdim=True)
        target_traces = self.pick_traces(self.match_rows(target_mz, row_mz), traces).unsqueeze(0)
        same_trace = (target_mz.unsqueeze(0) - target_mz.unsqueeze(1)).abs() < self.same_mz
        outputs = self.model(chromatogram, target_traces, target_mz, reference_rt, same_trace)
        context_found = (context_matches.sum(dim=1) > 0).float().sum().reshape(1)
        return outputs["rt"].reshape(-1, 1), outputs["uncertainty"].reshape(-1, 1), context_found


def build_metadata(model, same_mz):
    description = model.description
    return {
        "points": model.layout.points,
        "time_from": model.layout.time_from,
        "time_to": model.layout.time_to,
        "context_mz": description["context_mz"],
        "mz_tolerance": same_mz,
        "eic_ppm": description["eic_ppm"],
        "eic_mz_tolerance": description["eic_mz_tolerance"],
        "targets": description["targets"],
        "inputs": input_description,
        "outputs": output_description,
    }


def build_example(model):
    targets = model.description["targets"]
    context_mz = model.description["context_mz"]
    row_mz = torch.tensor(context_mz, dtype=torch.float32).unsqueeze(1)
    eics = torch.cat([row_mz, torch.rand(len(context_mz), model.layout.points) * 1000.0], dim=1)
    target_values = torch.tensor([[target["mz"], target["rt"]] for target in targets], dtype=torch.float32)
    return eics, target_values


def export_onnx(model, path, same_mz=0.0005, opset=17):
    import onnx

    if not model.description.get("context_mz"):
        raise ValueError("the model has no context_mz: train it with deepwarp.train or set model.description")
    wrapper = OnnxWarp(copy.deepcopy(model).cpu().eval(), same_mz).eval()
    torch.onnx.export(
        wrapper,
        build_example(model),
        path,
        input_names=["eics", "targets"],
        output_names=["rt", "uncertainty", "context_found"],
        dynamic_axes={"eics": {0: "rows"}, "targets": {0: "n"}, "rt": {0: "n"}, "uncertainty": {0: "n"}},
        opset_version=opset,
        dynamo=False,
    )
    saved = onnx.load(path)
    for key, value in build_metadata(model, same_mz).items():
        entry = saved.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)
    onnx.save(saved, path)
    return path
