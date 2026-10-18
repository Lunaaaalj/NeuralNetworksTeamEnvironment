"""Comprueba el registro de F1 personalizado y el cierre de corridas sin credenciales."""

import math

import pytest

from src.tracking import WandbEpochLogger, track_training
from src.train import run


def test_epoch_logger_skips_missing_validation():
    class CaptureRun:
        def log(self, metrics):
            self.metrics = metrics

    captured = CaptureRun()
    callback = WandbEpochLogger(captured)
    callback.on_epoch_begin(1)
    callback.on_epoch_end(1, {"loss": 0.5, "accuracy": 0.75, "val_f1_macro": math.nan})
    assert captured.metrics["epoch"] == 2
    assert captured.metrics["loss"] == 0.5
    assert "val_f1_macro" not in captured.metrics


def test_tracking_disabled_does_not_initialize_wandb(tiny_cfg, monkeypatch):
    wandb = pytest.importorskip("wandb")

    def unexpected_init(**kwargs):
        pytest.fail("Training without --wandb must not initialize W&B")

    monkeypatch.setattr(wandb, "init", unexpected_init)
    with track_training(tiny_cfg, "disabled", "full", None, 0, 1) as tracked:
        assert tracked is None


def test_offline_training_logs_validation_and_finishes(tiny_cfg, monkeypatch):
    wandb = pytest.importorskip("wandb")
    tiny_cfg["wandb"] = {"enabled": True, "project": "cnn-tests", "mode": "offline"}
    logged = []
    original_log = wandb.Run.log

    def capture_log(self, metrics, *args, **kwargs):
        logged.append(dict(metrics))
        return original_log(self, metrics, *args, **kwargs)

    monkeypatch.setattr(wandb.Run, "log", capture_log)
    summary = run(tiny_cfg, "full", epochs=2)
    assert [row["epoch"] for row in logged] == [1, 2]
    assert all({"loss", "accuracy", "val_loss", "val_accuracy", "val_f1_macro"} <= row.keys()
               for row in logged)
    assert all(0 <= row["val_f1_macro"] <= 1 for row in logged)
    assert wandb.run is None
    assert list(tiny_cfg["paths"]["runs"].parent.glob("wandb/offline-run-*/run-*.wandb"))
    assert tiny_cfg["paths"]["models"].joinpath(f"{summary['run']}.keras").exists()


def test_offline_tracking_finishes_after_failure(tiny_cfg):
    wandb = pytest.importorskip("wandb")
    tiny_cfg["wandb"] = {"enabled": True, "project": "cnn-tests", "mode": "offline"}
    with pytest.raises(RuntimeError, match="training failed"):
        with track_training(tiny_cfg, "failure", "full", None, 0, 1):
            raise RuntimeError("training failed")
    assert wandb.run is None
