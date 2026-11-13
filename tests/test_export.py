import numpy as np
import onnxruntime

import deepwarp
from deepwarp.predict import get_reference_rt
from deepwarp.samples import group_traces


def test_onnx_gives_the_same_numbers(tmp_path, data):
    targets, dataset, labels = data
    settings = deepwarp.TrainingSettings(epochs=2, uncertainty_epochs=1, samples_per_epoch=16, batch_size=8, device="cpu")
    model = deepwarp.train(targets, dataset, labels, settings, log=None)
    path = deepwarp.export_onnx(model, str(tmp_path / "model.onnx"))

    sample = dataset.reshape(12, 6, -1)[3]
    expected = deepwarp.predict(model, targets, sample[None])
    trace_mz, first_target, _ = group_traces(targets, 0.0005)
    rows = np.concatenate([trace_mz[:, None], sample[first_target]], axis=1).astype(np.float32)
    target_values = np.array([[target["mz"], rt] for target, rt in zip(targets, get_reference_rt(model, targets))], dtype=np.float32)
    rt, uncertainty, context_found = onnxruntime.InferenceSession(path).run(None, {"eics": rows, "targets": target_values})
    assert np.allclose(rt[:, 0], expected["rt"][0], atol=1e-4)
    assert np.allclose(uncertainty[:, 0], expected["uncertainty"][0], atol=1e-5)
    assert context_found[0] == len(trace_mz)
