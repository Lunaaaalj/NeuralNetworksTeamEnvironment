"""La CNN del reporte y sus variantes en PyTorch/DirectML para Windows.

Usa los mismos TFRecords y aumentos de datos; exporta el mejor modelo a Keras
para que src.evaluate siga funcionando con los checkpoints nuevos.
"""

import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from src.config import run_name, set_seed
from src.data import build_dataset
from src.directml_checkpoint import export_checkpoint, save_checkpoint
from src.directml_input import batches
from src.metrics import f1_macro
from src.model import (BATCHNORM_EPSILON, BATCHNORM_MOMENTUM, EXTRA_CONV_STAGES,
                       FILTERS, build_cnn, model_architecture)
from src.optim_directml import build_optimizer
from src.tracking import track_training
from src.training_policy import torch_scheduler, training_suffix


class FlowerCNN(nn.Module):
    """CNN original, deep o cuatro bloques Conv/BatchNorm/ReLU/Pool."""

    def __init__(self, num_classes=104, *, architecture="baseline"):
        super().__init__()
        if architecture not in EXTRA_CONV_STAGES:
            raise ValueError(f"Arquitectura desconocida: {architecture}")
        self.architecture = architecture
        channels = (3, 32, 64, 128, 256)
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, bias=architecture != "batchnorm")
            for i in range(4)
        ])
        self.extra_convs = nn.ModuleDict({
            str(i): nn.Conv2d(FILTERS[i - 1], FILTERS[i - 1], 3, padding=1)
            for i in EXTRA_CONV_STAGES[architecture]
        })
        # PyTorch momentum es la fracción NUEVA, al contrario de Keras.
        self.batchnorms = nn.ModuleDict({
            str(i): nn.BatchNorm2d(filters, eps=BATCHNORM_EPSILON,
                                   momentum=1 - BATCHNORM_MOMENTUM)
            for i, filters in enumerate(FILTERS, 1) if architecture == "batchnorm"
        })
        self.pool = nn.MaxPool2d(2)
        self.dense = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.logits = nn.Linear(256, num_classes)

    def forward(self, images):
        for index, conv in enumerate(self.convs, 1):
            images = conv(images)
            if str(index) in self.batchnorms:
                images = self.batchnorms[str(index)](images)
            images = self.pool(torch.relu(images))
            if str(index) in self.extra_convs:
                images = torch.relu(self.extra_convs[str(index)](images))
        features = images.mean(dim=(2, 3))
        return self.logits(self.dropout(torch.relu(self.dense(features))))

    def named_convolutions(self):
        for index, conv in enumerate(self.convs, 1):
            yield f"conv{index}", conv
            if str(index) in self.extra_convs:
                yield f"conv{index}_extra", self.extra_convs[str(index)]

    def named_batchnorms(self):
        for index, layer in self.batchnorms.items():
            yield f"bn{index}", layer


def _check_architecture(keras_model, torch_model):
    expected = {name for name, _ in torch_model.named_convolutions()}
    actual = {layer.name for layer in keras_model.layers if layer.name.startswith("conv")}
    expected_bn = {name for name, _ in torch_model.named_batchnorms()}
    actual_bn = {layer.name for layer in keras_model.layers if layer.name.startswith("bn")}
    if actual != expected or actual_bn != expected_bn:
        raise ValueError("Las arquitecturas Keras y PyTorch no coinciden.")
    for name, conv in torch_model.named_convolutions():
        if keras_model.get_layer(name).use_bias != (conv.bias is not None):
            raise ValueError(f"Las arquitecturas Keras y PyTorch no coinciden en {name} (bias).")


def copy_from_keras(source, target):
    """Conserva también la inicialización de Keras, no solo la arquitectura."""
    _check_architecture(source, target)
    with torch.no_grad():
        for name, conv in target.named_convolutions():
            weights = source.get_layer(name).get_weights()
            conv.weight.copy_(torch.from_numpy(weights[0].transpose(3, 2, 0, 1)))
            if conv.bias is not None:
                conv.bias.copy_(torch.from_numpy(weights[1]))
        for name, layer in target.named_batchnorms():
            gamma, beta, mean, variance = source.get_layer(name).get_weights()
            for destination, value in zip((layer.weight, layer.bias, layer.running_mean, layer.running_var),
                                           (gamma, beta, mean, variance)):
                destination.copy_(torch.from_numpy(value))
            # Con momentum fijo, este contador no afecta las medias importadas.
            layer.num_batches_tracked.zero_()
        for name in ("dense", "logits"):
            kernel, bias = source.get_layer(name).get_weights()
            getattr(target, name).weight.copy_(torch.from_numpy(kernel.T))
            getattr(target, name).bias.copy_(torch.from_numpy(bias))


def copy_to_keras(source, target):
    """Convierte pesos y estadísticas de BatchNorm para inferencia en Keras."""
    _check_architecture(target, source)
    for name, conv in source.named_convolutions():
        weights = [conv.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)]
        if conv.bias is not None:
            weights.append(conv.bias.detach().cpu().numpy())
        target.get_layer(name).set_weights(weights)
    for name, layer in source.named_batchnorms():
        target.get_layer(name).set_weights([
            value.detach().cpu().numpy()
            for value in (layer.weight, layer.bias, layer.running_mean, layer.running_var)
        ])
    for name in ("dense", "logits"):
        layer = getattr(source, name)
        target.get_layer(name).set_weights([
            layer.weight.detach().cpu().numpy().T, layer.bias.detach().cpu().numpy(),
        ])


def select_device(index=None):
    import torch_directml

    devices = [(i, torch_directml.device_name(i).rstrip("\x00"))
               for i in range(torch_directml.device_count())]
    if not devices:
        raise RuntimeError("DirectML no detectó una GPU. Revisa el controlador de AMD.")
    for i, name in devices:
        print(f"DirectML adapter {i}: {name}", flush=True)
    if index is None:
        # Preferir la Radeon dedicada frente a la GPU integrada del procesador.
        index = next((i for i, name in devices if "Radeon RX" in name), 0)
    if index not in dict(devices):
        raise ValueError(f"GPU {index} no existe. Adaptadores disponibles: {devices}")
    device = torch_directml.device(index)
    # Fuerza una operación real: si el controlador falla, no seguir en CPU.
    probe = torch.ones(1, device=device) + 1
    if probe.cpu().item() != 2:
        raise RuntimeError("Falló la comprobación de DirectML.")
    print(f"Training on GPU: {dict(devices)[index]} ({device})", flush=True)
    return device, dict(devices)[index]


def evaluate(model, dataset, device, *, input_pipeline="dlpack_prefetch"):
    model.eval()
    total_loss, count = torch.zeros((), device=device), 0
    true, predicted = [], []
    with torch.no_grad():
        for images, labels in batches(dataset, device, pipeline=input_pipeline):
            logits = model(images)
            total_loss.add_(nn.functional.cross_entropy(logits, labels, reduction="sum"))
            count += len(labels)
            true.append(labels.cpu().numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())
    y_true, y_pred = np.concatenate(true), np.concatenate(predicted)
    return {"val_loss": total_loss.cpu().item() / count, "val_accuracy": float(np.mean(y_true == y_pred)),
            "val_f1_macro": f1_macro(y_true, y_pred)}


def train_epoch(model, dataset, optimizer, device, *, sync_each_batch=False,
                input_pipeline="dlpack_prefetch"):
    """Acumula loss/accuracy en GPU y las copia una sola vez al terminar la época.

    sync_each_batch conserva el comportamiento anterior para medir el baseline.
    """
    model.train()
    loss_sum, correct = torch.zeros((), device=device), torch.zeros((), device=device)
    host_loss, host_correct, count, steps = 0.0, 0.0, 0, 0
    for images, labels in batches(dataset, device, pipeline=input_pipeline):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batch_loss = loss.detach() * len(labels)
        batch_correct = (logits.detach().argmax(dim=1) == labels).float().sum()
        if sync_each_batch:
            host_loss += batch_loss.cpu().item()
            host_correct += batch_correct.cpu().item()
        else:
            loss_sum.add_(batch_loss)
            correct.add_(batch_correct)
        count += len(labels)
        steps += 1
    if count == 0:
        raise ValueError("El conjunto de entrenamiento está vacío.")
    if not sync_each_batch:
        host_loss, host_correct = torch.stack((loss_sum, correct)).cpu().tolist()
    return {"loss": host_loss / count, "accuracy": host_correct / count}, count, steps


def run(cfg, data, ipc=None, seed=0, epochs=None, val_ds=None, *, device=None):
    # La importación diferida comparte exactamente el cargador de src.train.
    from src.train import load_train_data

    cfg = deepcopy(cfg)
    if device is None:
        device, device_name = select_device(cfg.get("runtime", {}).get("device_index"))
    else:
        device_name = str(device)  # dispositivo explícito para pruebas de paridad en CPU
    cfg.setdefault("runtime", {}).update(backend="directml", device=device_name,
                                         metrics_sync="epoch",
                                         checkpoint_format="pytorch_then_keras")
    input_pipeline = cfg["runtime"].setdefault("input_pipeline", "dlpack_prefetch")
    architecture = model_architecture(cfg)
    set_seed(seed)
    torch.manual_seed(seed)
    tcfg, dcfg = cfg["train"], cfg["data"]
    small = data != "full"
    epochs = epochs or (tcfg["epochs_small"] if small else tcfg["epochs_full"])
    every = tcfg["eval_every_small"] if small else 1
    suffix = "" if architecture == "baseline" else f"_{architecture}"
    name = run_name(data, ipc, dcfg["img_size"], seed) + suffix + training_suffix(tcfg) + "_directml"
    run_dir = Path(cfg["paths"]["runs"]) / name
    model_path = Path(cfg["paths"]["models"]) / f"{name}.keras"
    checkpoint_path = model_path.with_suffix(".pt")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds, n_images, extra = load_train_data(cfg, data, ipc, seed)
    if val_ds is None:
        val_ds = build_dataset(cfg["paths"]["tfrecords"], "val", dcfg["img_size"], 128,
                               cache=dcfg.get("cache", False),
                               cache_dir=Path(cfg["paths"]["processed"]) / "tfdata_cache")
    initial = build_cnn(dcfg["num_classes"], architecture=architecture)
    model = FlowerCNN(dcfg["num_classes"], architecture=architecture)
    copy_from_keras(initial, model)
    del initial
    model.to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"CNN {architecture}: {len(tuple(model.named_convolutions()))} convolutions, "
          f"{parameters:,} parameters, input {dcfg['img_size']}x{dcfg['img_size']}", flush=True)
    optimizer = build_optimizer(model, tcfg)
    scheduler = torch_scheduler(optimizer, tcfg)
    cfg["runtime"]["optimizer"] = type(optimizer).__name__
    best_f1, best_accuracy, best_epoch, bad_evals = -1.0, None, -1, 0
    train_seconds = 0.0
    checkpoint_seconds = 0.0
    fields = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "val_f1_macro",
              "learning_rate", "next_learning_rate"]

    with track_training(cfg, name, data, ipc, seed, epochs) as tracked:
        try:
            with (run_dir / "history.csv").open("w", newline="") as history_file:
                writer = csv.DictWriter(history_file, fieldnames=fields)
                writer.writeheader()
                for epoch in range(epochs):
                    started = time.perf_counter()
                    learning_rate = optimizer.param_groups[0]["lr"]
                    training_metrics, count, steps = train_epoch(
                        model, train_ds, optimizer, device, input_pipeline=input_pipeline)
                    train_seconds += time.perf_counter() - started
                    n_images = count
                    metrics = {**training_metrics, "learning_rate": learning_rate,
                               "val_loss": math.nan, "val_accuracy": math.nan, "val_f1_macro": math.nan}
                    should_eval = epoch == 0 or (epoch + 1) % every == 0 or epoch + 1 == epochs
                    if should_eval:
                        metrics.update(evaluate(model, val_ds, device, input_pipeline=input_pipeline))
                        if metrics["val_f1_macro"] > best_f1:
                            saving_started = time.perf_counter()
                            save_checkpoint(model, checkpoint_path, epoch=epoch + 1, metrics=metrics,
                                            img_size=dcfg["img_size"])
                            checkpoint_seconds += time.perf_counter() - saving_started
                            best_f1, best_accuracy = metrics["val_f1_macro"], metrics["val_accuracy"]
                            best_epoch, bad_evals = epoch + 1, 0
                        else:
                            bad_evals += 1
                        if scheduler is not None and math.isfinite(metrics["val_loss"]):
                            scheduler.step(metrics["val_loss"])
                    metrics["next_learning_rate"] = optimizer.param_groups[0]["lr"]
                    writer.writerow({"epoch": epoch, **metrics})
                    history_file.flush()
                    elapsed = time.perf_counter() - started
                    print(f"Epoch {epoch + 1}/{epochs} - {steps} batches - {elapsed:.1f}s - "
                          + " - ".join(f"{key}: {value:.4f}" for key, value in metrics.items()
                                       if math.isfinite(value) and "learning_rate" not in key)
                          + f" - lr: {learning_rate:.2e}"
                          + (f" -> {metrics['next_learning_rate']:.2e}"
                             if metrics["next_learning_rate"] != learning_rate else ""), flush=True)
                    if tracked is not None:
                        tracked.log({"epoch": epoch + 1, "epoch_seconds": elapsed,
                                     **{key: value for key, value in metrics.items() if math.isfinite(value)}})
                    if should_eval and tcfg.get("patience") is not None and bad_evals >= tcfg["patience"]:
                        print(f"Parada temprana: {bad_evals} evaluaciones sin mejora.")
                        break
        finally:
            if best_epoch > 0:
                export_started = time.perf_counter()
                export_checkpoint(checkpoint_path, model_path)
                export_seconds = time.perf_counter() - export_started
                print(f"Mejor época {best_epoch}: exportada a {model_path} ({export_seconds:.2f}s).",
                      flush=True)

        summary = {
            "run": name, "data": data, "ipc": ipc if small else None, "seed": seed,
            "img_size": dcfg["img_size"], "train_images": n_images, "epochs_run": epoch + 1,
            "best_epoch": best_epoch, "val_f1_macro": round(best_f1, 4),
            "val_accuracy": round(best_accuracy, 4), "train_seconds": round(train_seconds, 1),
            "distill_seconds": None, "distill_iterations": None, "model_path": str(model_path),
            "backend": "directml", "device": device_name, **extra,
            "checkpoint_path": str(checkpoint_path), "input_pipeline": input_pipeline,
            "checkpoint_seconds": round(checkpoint_seconds, 3),
            "export_seconds": round(export_seconds, 3),
            "architecture": architecture, "parameters": parameters,
            "optimizer": type(optimizer).__name__, "weight_decay": tcfg.get("weight_decay", 0.0),
            "lr_schedule": tcfg.get("lr_schedule"), "final_learning_rate": optimizer.param_groups[0]["lr"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if tracked is not None:
            tracked.summary.update({"best_val_f1_macro": summary["val_f1_macro"],
                                    "best_val_accuracy": summary["val_accuracy"],
                                    "best_epoch": best_epoch, "train_seconds": summary["train_seconds"],
                                    "epochs_run": epoch + 1, "train_images": n_images,
                                    "checkpoint_seconds": round(checkpoint_seconds, 3),
                                    "export_seconds": round(export_seconds, 3),
                                    "final_learning_rate": optimizer.param_groups[0]["lr"], **extra})
    print(json.dumps(summary, indent=2))
    return summary
