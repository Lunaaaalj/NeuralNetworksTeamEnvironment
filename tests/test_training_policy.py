"""Equivalencia numérica de AdamW y cambios de LR sobre evaluaciones reales."""

from copy import deepcopy
import math

import keras
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.model import build_cnn, compile_cnn
from src.optim_directml import DirectMLAdamW, build_optimizer
from src.training_policy import keras_scheduler, torch_scheduler


@pytest.mark.parametrize("scale", [1.0, 1e-8])
@pytest.mark.parametrize("decay", [0.0, 0.03])
def test_adamw_matches_pytorch_with_missing_gradients_lr_changes_and_restored_state(scale, decay):
    rng = torch.Generator().manual_seed(14)
    left = [torch.nn.Parameter(torch.randn(3, 4, generator=rng)),
            torch.nn.Parameter(torch.randn(4, generator=rng))]
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    reference = torch.optim.AdamW(left, lr=0.002, betas=(0.8, 0.95), eps=1e-7,
                                 weight_decay=decay, foreach=False)
    actual = DirectMLAdamW(right, lr=0.002, betas=(0.8, 0.95), eps=1e-7, weight_decay=decay)
    for step in range(20):
        for index, (p, q) in enumerate(zip(left, right)):
            gradient = torch.randn(p.shape, generator=rng) * scale
            p.grad = gradient if index == 0 or step % 3 else None
            q.grad = None if p.grad is None else gradient.clone()
        reference.step()
        actual.step()
        if step == 9:
            restored = DirectMLAdamW(right)
            restored.load_state_dict(deepcopy(actual.state_dict()))
            actual = restored
            reference.param_groups[0]["lr"] *= 0.5
            actual.param_groups[0]["lr"] *= 0.5
        for p, q in zip(left, right):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=1e-5)


def test_decay_applies_to_weights_and_excludes_biases():
    model = torch.nn.Linear(3, 2)
    optimizer = build_optimizer(model, {"optimizer": "adamw", "weight_decay": 0.1, "lr": 0.01})
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.fill_(1)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 0.999))
    torch.testing.assert_close(model.bias, torch.ones_like(model.bias))


def test_scheduler_backends_agree_and_skip_epochs_without_validation():
    config = {"lr": 0.001, "lr_schedule": {"factor": 0.5, "patience": 1, "min_lr": 0.00025}}
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=config["lr"])
    reference = torch_scheduler(optimizer, config)
    model = compile_cnn(build_cnn(3))
    callback = keras_scheduler(config)
    callback.set_model(model)
    callback.on_train_begin()
    for epoch, loss in enumerate([1.0, 0.9, math.nan, None, 0.95, 0.96, 0.97, 0.98, 0.99]):
        if loss is not None and math.isfinite(loss):
            reference.step(loss)
        callback.on_epoch_end(epoch, {"val_loss": loss})
        assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(optimizer.param_groups[0]["lr"])
    assert optimizer.param_groups[0]["lr"] == 0.00025


def test_directml_history_records_lr_used_and_next_lr(tiny_cfg, monkeypatch):
    import src.train_directml as trainer

    tiny_cfg["train"].update(optimizer="adamw", weight_decay=0.01, patience=12,
                             lr_schedule={"type": "plateau", "factor": 0.5,
                                          "patience": 1, "min_lr": 0.0001})
    losses = iter([1.0, 0.9, 0.95, 0.96, 0.97])
    scores = iter([0.4, 0.5, 0.5, 0.5, 0.6])
    used = []

    def train(model, dataset, optimizer, device, **kwargs):
        used.append(optimizer.param_groups[0]["lr"])
        return {"loss": 1.0, "accuracy": 0.5}, 24, 6

    def evaluate(*args, **kwargs):
        return {"val_loss": next(losses), "val_accuracy": 0.5, "val_f1_macro": next(scores)}

    monkeypatch.setattr(trainer, "train_epoch", train)
    monkeypatch.setattr(trainer, "evaluate", evaluate)
    summary = trainer.run(tiny_cfg, "full", epochs=5, device=torch.device("cpu"))
    history = pd.read_csv(tiny_cfg["paths"]["runs"] / summary["run"] / "history.csv")
    np.testing.assert_allclose(used, [0.001, 0.001, 0.001, 0.001, 0.0005])
    np.testing.assert_allclose(history.learning_rate, used)
    np.testing.assert_allclose(history.next_learning_rate, [0.001, 0.001, 0.001, 0.0005, 0.0005])
    assert summary["optimizer"] == "DirectMLAdamW"
    assert summary["best_epoch"] == 5
    assert "adamw_plateau" in summary["run"]
    checkpoint = torch.load(summary["checkpoint_path"], weights_only=True)
    assert checkpoint["metrics"]["learning_rate"] == 0.0005
    assert keras.models.load_model(summary["model_path"]).output_shape[-1] == 3


def test_tensorflow_runs_new_policy_with_existing_evaluator(tiny_cfg):
    from src.train import run

    tiny_cfg["train"].update(optimizer="adamw", weight_decay=0.01,
                             lr_schedule={"type": "plateau", "factor": 0.5,
                                          "patience": 1, "min_lr": 0.0001})
    summary = run(tiny_cfg, "full", epochs=2)
    restored = keras.models.load_model(summary["model_path"])
    assert summary["optimizer"] == "adamw"
    assert isinstance(restored.optimizer, keras.optimizers.AdamW)
    assert restored.optimizer.weight_decay == pytest.approx(0.01)
    assert "adamw_plateau" in summary["run"]
