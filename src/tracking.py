"""Registro opcional de métricas por época en Weights & Biases."""

from contextlib import contextmanager
import math
from pathlib import Path
import time

import keras


class WandbEpochLogger(keras.callbacks.Callback):
    """Se ejecuta después de ValidationMonitor para incluir sus métricas de val."""

    def __init__(self, run):
        super().__init__()
        self.run = run

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        metrics = {key: float(value) for key, value in (logs or {}).items()
                   if value is not None and math.isfinite(float(value))}
        metrics.update(epoch=epoch + 1, epoch_seconds=time.perf_counter() - self.started)
        self.run.log(metrics)


@contextmanager
def track_training(cfg, name, data, ipc, seed, epochs):
    """Abre una corrida por entrenamiento y la cierra incluso si fit falla."""
    options = cfg.get("wandb", {})
    if not options.get("enabled", False):
        yield None
        return

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Instala W&B con: python -m pip install wandb") from exc

    # Mantener los archivos del SDK dentro del directorio local de reportes.
    log_dir = Path(cfg["paths"]["runs"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=options.get("project", "flower-cnn"),
        entity=options.get("entity"),
        mode=options.get("mode", "online"),
        name=name,
        dir=str(log_dir),
        tags=[data, f"res{cfg['data']['img_size']}"],
        config={
            "architecture": "flower_cnn", "data_mode": data,
            "ipc": ipc if data != "full" else None, "seed": seed, "epochs": epochs,
            "data": cfg["data"], "train": cfg["train"],
            "runtime": cfg.get("runtime", {"backend": "tensorflow"}),
            "distill": cfg.get("distill", {}) if data == "distilled" else {},
        },
    ) as run:
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        yield run
