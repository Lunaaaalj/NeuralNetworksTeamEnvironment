"""Paridad de la CNN y exportación del entrenamiento DirectML al evaluador Keras."""

import json
from copy import deepcopy

import keras
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data import build_dataset
from src.directml_checkpoint import export_checkpoint, save_checkpoint
from src.directml_input import INPUT_PIPELINES, batches
from src.model import build_cnn
from src.optim_directml import DirectMLAdam
from src.train_directml import FlowerCNN, copy_from_keras, copy_to_keras, evaluate, run, train_epoch


@pytest.mark.parametrize("size", [32, 112, 224])
def test_pytorch_and_keras_have_same_logits(size):
    original = build_cnn()
    model = FlowerCNN()
    assert sum(p.numel() for p in model.parameters()) == original.count_params() == 480_936
    copy_from_keras(original, model)
    # Cambiar pesos tras importarlos verifica también la exportación del modelo entrenado.
    with torch.no_grad():
        model.dense.bias.add_(0.1)
    exported = build_cnn()
    copy_to_keras(model, exported)
    images = np.random.default_rng(0).uniform(size=(2, size, size, 3)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        actual = model(torch.from_numpy(images).permute(0, 3, 1, 2)).numpy()
    expected = exported(images, training=False).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("cache", [False, "disk", "ram"])
def test_training_checkpoint_loads_in_existing_evaluator(tiny_cfg, cache, monkeypatch):
    from unittest.mock import Mock

    export = Mock(wraps=export_checkpoint)
    monkeypatch.setattr("src.train_directml.export_checkpoint", export)
    tiny_cfg["data"]["cache"] = cache
    summary = run(tiny_cfg, "full", epochs=2, device=torch.device("cpu"))
    model = keras.models.load_model(summary["model_path"])
    val = build_dataset(tiny_cfg["paths"]["tfrecords"], "val", 32, 4)
    labels, predictions = [], []
    for images, batch_labels in val:
        labels.extend(batch_labels.numpy())
        predictions.extend(np.argmax(model(images, training=False).numpy(), axis=1))
    # Guardar el mejor checkpoint debe preservar su accuracy y su F1.
    from src.metrics import f1_macro

    assert round(f1_macro(labels, predictions), 4) == summary["val_f1_macro"]
    assert round(float(np.mean(np.array(labels) == predictions)), 4) == summary["val_accuracy"]
    assert summary["train_images"] == 24
    assert summary["epochs_run"] == 2
    assert export.call_count == 1
    checkpoint = torch.load(summary["checkpoint_path"], map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == summary["best_epoch"]
    assert round(checkpoint["metrics"]["val_f1_macro"], 4) == summary["val_f1_macro"]
    directory = tiny_cfg["paths"]["runs"] / summary["run"]
    history = pd.read_csv(directory / "history.csv")
    assert len(history) == 2
    assert history["val_f1_macro"].notna().all()
    assert json.loads((directory / "summary.json").read_text())["backend"] == "directml"
    if cache == "disk":
        assert len(list((tiny_cfg["paths"]["processed"] / "tfdata_cache").glob("*.index"))) == 2


@pytest.mark.parametrize("pipeline", INPUT_PIPELINES)
def test_input_bridge_preserves_augmented_values_labels_and_partial_batch(tiny_cfg, pipeline):
    # Materializar los mismos aumentos permite exigir igualdad exacta, no solo
    # formas compatibles o distribuciones de píxeles parecidas.
    dataset = list(build_dataset(tiny_cfg["paths"]["tfrecords"], "train", 32, 5,
                                 augment=True, shuffle=True, seed=15))
    actual = list(batches(dataset, torch.device("cpu"), pipeline=pipeline))
    assert len(actual[-1][0]) == 4
    for (images, labels), (expected_images, expected_labels) in zip(actual, dataset):
        assert images.is_contiguous()
        assert labels.dtype == torch.int64
        np.testing.assert_array_equal(images.numpy(), expected_images.numpy().transpose(0, 3, 1, 2))
        np.testing.assert_array_equal(labels.numpy(), expected_labels.numpy())


def test_prefetch_propagates_reader_failure():
    import tensorflow as tf

    def source():
        yield tf.zeros((1, 32, 32, 3)), tf.zeros((1,), tf.int32)
        raise OSError("TFRecord read failed")

    iterator = batches(source(), torch.device("cpu"))
    next(iterator)
    with pytest.raises(OSError, match="TFRecord read failed"):
        next(iterator)


@pytest.mark.parametrize("interrupted", [False, True])
def test_final_export_keeps_best_epoch_even_after_worse_weights(tiny_cfg, monkeypatch, interrupted):
    from src.config import run_name

    expected = {}
    calls = 0

    def train(model, *args, **kwargs):
        nonlocal calls
        calls += 1
        with torch.no_grad():
            model.logits.bias[0].add_(1.0)
        if calls == 1:
            expected.update({key: value.detach().clone() for key, value in model.state_dict().items()})
        elif interrupted:
            raise KeyboardInterrupt
        return {"loss": 1.0, "accuracy": 0.5}, 24, 6

    def validation(*args, **kwargs):
        return {"val_loss": 1.0, "val_accuracy": 0.5, "val_f1_macro": 1.0 / calls}

    monkeypatch.setattr("src.train_directml.train_epoch", train)
    monkeypatch.setattr("src.train_directml.evaluate", validation)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            run(tiny_cfg, "full", epochs=2, device=torch.device("cpu"))
    else:
        summary = run(tiny_cfg, "full", epochs=2, device=torch.device("cpu"))
        assert summary["best_epoch"] == 1
    path = tiny_cfg["paths"]["models"] / (run_name("full", None, 32, 0) + "_directml")
    checkpoint = torch.load(path.with_suffix(".pt"), weights_only=True)
    for key, value in expected.items():
        torch.testing.assert_close(checkpoint["model_state"][key], value, rtol=0, atol=0)
    exported = keras.models.load_model(path.with_suffix(".keras"))
    restored = FlowerCNN(3)
    copy_from_keras(exported, restored)
    for key, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    model = FlowerCNN(3)
    save_checkpoint(model, path, epoch=1, metrics={"val_f1_macro": 0.5}, img_size=32)

    def failed_save(checkpoint, temporary):
        temporary.write_bytes(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(model, path, epoch=2, metrics={"val_f1_macro": 0.8}, img_size=32)
    assert torch.load(path, weights_only=True)["epoch"] == 1
    assert not path.with_suffix(".pt.tmp").exists()


def test_failed_export_preserves_previous_keras_file(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    destination = tmp_path / "best.keras"
    save_checkpoint(FlowerCNN(3), path, epoch=1, metrics={"val_f1_macro": 0.5}, img_size=32)
    build_cnn(3).save(destination)
    previous = destination.read_bytes()

    def failed_export(model, temporary, *args, **kwargs):
        temporary.write_bytes(b"incomplete")
        raise OSError("export failed")

    monkeypatch.setattr(keras.Model, "save", failed_export)
    with pytest.raises(OSError, match="export failed"):
        export_checkpoint(path, destination)
    assert destination.read_bytes() == previous
    assert not destination.with_name("best.tmp.keras").exists()


def test_directml_validation_matches_keras(tiny_cfg):
    original = build_cnn(3)
    model = FlowerCNN(3)
    copy_from_keras(original, model)
    val = build_dataset(tiny_cfg["paths"]["tfrecords"], "val", 32, 4)
    from src.metrics import predict_labels, f1_macro

    true, predicted, loss = predict_labels(original, val)
    actual = evaluate(model, val, torch.device("cpu"))
    assert actual["val_loss"] == pytest.approx(loss, abs=1e-5)
    assert actual["val_accuracy"] == float(np.mean(true == predicted))
    assert actual["val_f1_macro"] == f1_macro(true, predicted)


@pytest.mark.parametrize("scale", [1.0, 1e-8])
def test_directml_adam_matches_standard_adam_and_restores_state(scale):
    generator = torch.Generator().manual_seed(12)
    original = [torch.nn.Parameter(torch.randn(3, 4, generator=generator)),
                torch.nn.Parameter(torch.randn(4, generator=generator))]
    candidate = [torch.nn.Parameter(value.detach().clone()) for value in original]
    reference = torch.optim.Adam(original, lr=0.002, betas=(0.8, 0.95), eps=1e-7, foreach=False)
    actual = DirectMLAdam(candidate, lr=0.002, betas=(0.8, 0.95), eps=1e-7)
    for step in range(20):
        for index, (left, right) in enumerate(zip(original, candidate)):
            grad = torch.randn(left.shape, generator=generator) * scale
            # Un parámetro sin gradiente no debe avanzar sus momentos ni contador.
            left.grad = grad if index == 0 or step % 3 else None
            right.grad = None if left.grad is None else grad.clone()
        reference.step()
        actual.step()
        if step == 9:
            restored = DirectMLAdam(candidate)
            restored.load_state_dict(deepcopy(actual.state_dict()))
            actual = restored
        for left, right in zip(original, candidate):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)


def test_epoch_metrics_match_with_one_transfer_and_partial_batch():
    import tensorflow as tf

    rng = np.random.default_rng(0)
    images = rng.uniform(size=(5, 32, 32, 3)).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, np.array([0, 1, 2, 0, 1], np.int32))).batch(2)
    model = FlowerCNN(3)
    reference = deepcopy(model)
    optimizer = DirectMLAdam(model.parameters())
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001, eps=1e-7, foreach=False)
    torch.manual_seed(25)
    expected, expected_count, expected_steps = train_epoch(
        reference, dataset, reference_optimizer, torch.device("cpu"), sync_each_batch=True)
    torch.manual_seed(25)
    actual, count, steps = train_epoch(model, dataset, optimizer, torch.device("cpu"))
    assert (count, steps) == (expected_count, expected_steps) == (5, 3)
    assert actual["accuracy"] == expected["accuracy"]
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-5)


@pytest.mark.parametrize("cache", [True, "disk"])
def test_cache_preserves_images_but_does_not_freeze_augmentation(tiny_cfg, tmp_path, cache):
    args = (tiny_cfg["paths"]["tfrecords"], "train", 32, 4)
    plain = list(build_dataset(*args).as_numpy_iterator())
    cached = list(build_dataset(*args, cache=cache, cache_dir=tmp_path).as_numpy_iterator())
    for (left, labels_left), (right, labels_right) in zip(plain, cached):
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(labels_left, labels_right)
    augmented = build_dataset(*args, cache=cache, cache_dir=tmp_path, augment=True, seed=13)
    first_epoch = list(augmented.as_numpy_iterator())
    second_epoch = list(augmented.as_numpy_iterator())
    assert any(not np.array_equal(left[0], right[0]) for left, right in zip(first_epoch, second_epoch))
