import numpy as np
import pytest

import deepwarp
from conftest import make_data
from deepwarp.samples import TraceLayout


def quick_settings(**changes):
    values = {"epochs": 3, "uncertainty_epochs": 2, "samples_per_epoch": 32, "batch_size": 8, "device": "cpu"}
    return deepwarp.TrainingSettings(**{**values, **changes})


def test_train_save_load_predict(tmp_path, data):
    targets, dataset, labels = data
    lines = []
    model = deepwarp.train(targets, dataset, labels, quick_settings(), log=lines.append)
    assert any(line.startswith("retention time  epoch") for line in lines)
    assert any(line.startswith("uncertainty  epoch") for line in lines)
    assert len(model.description["targets"]) == len(targets)

    found = deepwarp.predict(model, targets, dataset)
    assert found["rt"].shape == (12, 6)
    assert np.isfinite(found["rt"]).all()

    path = tmp_path / "model.pt"
    deepwarp.save_model(model, path)
    loaded = deepwarp.load_model(path)
    again = deepwarp.predict(loaded, targets, dataset)
    assert np.allclose(found["rt"], again["rt"], atol=1e-6)
    assert loaded.description == model.description


def test_training_lowers_the_error():
    targets, dataset, labels = make_data(24, seed=1)
    settings = quick_settings(epochs=15, uncertainty_epochs=0, samples_per_epoch=128, batch_size=16, learning_rate=1e-3)
    model = deepwarp.train(targets, dataset, labels, settings, log=None)
    error = np.nanmean(np.abs(deepwarp.predict(model, targets, dataset)["rt"] - labels))
    reference = np.nanmedian(labels, axis=0)
    assert error < np.nanmean(np.abs(reference[None, :] - labels))


def test_loop_of_groups_trains_and_finishes():
    targets, dataset, labels = make_data(24, seed=2)
    groups = np.split(dataset.reshape(24, 6, -1), 3), np.split(labels, 3)
    training = deepwarp.Training(targets, quick_settings(epochs=5, uncertainty_epochs=1), log=None)
    for group_dataset, group_labels in zip(*groups):
        training.add(group_dataset, group_labels)
        training.train()
    assert training.pool.training.count + training.pool.validation.count == 24
    model = training.finish()
    assert training.epoch == 5
    assert np.isfinite(deepwarp.predict(model, targets, dataset)["rt"]).all()


def test_train_is_training_add_finish(data):
    targets, dataset, labels = data
    first = deepwarp.train(targets, dataset, labels, quick_settings(), log=None)
    training = deepwarp.Training(targets, quick_settings(), log=None)
    training.add(dataset, labels)
    second = training.finish()
    assert np.allclose(deepwarp.predict(first, targets, dataset)["rt"], deepwarp.predict(second, targets, dataset)["rt"])


def test_train_past_the_planned_epochs_stops(data):
    targets, dataset, labels = data
    training = deepwarp.Training(targets, quick_settings(epochs=1), log=None)
    training.add(dataset, labels)
    training.train()
    with pytest.raises(RuntimeError):
        training.train()


def test_calls_in_the_wrong_order_stop(data):
    targets, dataset, labels = data
    training = deepwarp.Training(targets, quick_settings(), log=None)
    with pytest.raises(RuntimeError):
        training.train()
    training.add(dataset, labels)
    training.finish()
    with pytest.raises(RuntimeError):
        training.add(dataset, labels)


def test_dataset_as_rows_or_as_samples_gives_the_same_traces(data):
    targets, dataset, _ = data
    layout = TraceLayout(targets)
    assert np.array_equal(layout.read_traces(dataset), layout.read_traces(dataset.reshape(12, 6, -1)))


def test_isomers_share_one_trace(data):
    targets, dataset, _ = data
    layout = TraceLayout(targets)
    assert layout.read_traces(dataset).shape == (12, 5, 512)
    assert layout.trace_index[2] == layout.trace_index[3]


def test_refuses_a_dataset_that_does_not_match_the_targets(data):
    targets, dataset, labels = data
    with pytest.raises(ValueError):
        deepwarp.train(targets, dataset[:-1], labels, quick_settings(), log=None)


def test_refuses_data_without_labels(data):
    targets, dataset, labels = data
    with pytest.raises(ValueError):
        deepwarp.train(targets, dataset, np.full_like(labels, np.nan), quick_settings(), log=None)
