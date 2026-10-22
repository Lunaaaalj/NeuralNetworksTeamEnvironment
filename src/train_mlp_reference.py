"""Fully connected reference using the existing data and DirectML training loop.

This is a reconstructed comparison, not an undocumented Avance 3 checkpoint.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import keras
import numpy as np
import torch
from torch import nn
import yaml

from src.config import load_config, set_seed
from src.data import build_dataset
from src.optim_directml import build_optimizer
from src.train_directml import evaluate, select_device, train_epoch


class FlowerMLP(nn.Module):
    def __init__(self, img_size=112, num_classes=104):
        super().__init__()
        self.dense = nn.Linear(img_size * img_size * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.logits = nn.Linear(256, num_classes)

    def forward(self, images):
        # NHWC flatten order agrees with Keras, including when exporting weights.
        images = images.permute(0, 2, 3, 1).contiguous().flatten(1)
        return self.logits(self.dropout(torch.relu(self.dense(images))))


def build_keras_mlp(img_size=112, num_classes=104):
    return keras.Sequential([
        keras.Input((img_size, img_size, 3)),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(256, activation="relu", name="dense"),
        keras.layers.Dropout(0.5, name="dropout"),
        keras.layers.Dense(num_classes, name="logits"),
    ], name="flower_mlp_reference")


def run(config, destination, *, seed=0, epochs=150, lr=None, patience=None, min_epochs=0):
    cfg = load_config(config)
    if lr is not None:
        cfg["train"]["lr"] = lr
    if patience is not None:
        cfg["train"]["patience"] = patience
    destination = Path(destination).resolve()
    name = f"full_res{cfg['data']['img_size']}_s{seed}_mlp_reference_directml"
    run_dir = destination / "runs/mlp_referencia" / name
    model_dir = destination / "models/mlp_referencia"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    effective = {**cfg, "model": {"architecture": "mlp_reference", "hidden_units": 256},
                 "train": {**cfg["train"], "min_epochs": min_epochs}}
    effective["paths"] = {key: str(value) for key, value in cfg["paths"].items()}
    effective["paths"].update(runs=str(destination / "runs/mlp_referencia"), models=str(model_dir),
                              figures=str(destination / "figures/mlp_referencia"))
    (destination / "mlp_referencia.yaml").write_text(
        yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8")
    set_seed(seed)
    torch.manual_seed(seed)
    device, device_name = select_device()
    size, classes = cfg["data"]["img_size"], cfg["data"]["num_classes"]
    model = FlowerMLP(size, classes)
    initial = build_keras_mlp(size, classes)
    with torch.no_grad():
        for layer in ("dense", "logits"):
            kernel, bias = initial.get_layer(layer).get_weights()
            getattr(model, layer).weight.copy_(torch.from_numpy(kernel.T))
            getattr(model, layer).bias.copy_(torch.from_numpy(bias))
    del initial
    model.to(device)
    train_ds = build_dataset(cfg["paths"]["tfrecords"], "train", size, cfg["data"]["batch_size"],
                             augment=True, shuffle=True, seed=seed, cache="ram")
    val_ds = build_dataset(cfg["paths"]["tfrecords"], "val", size, 128, cache="ram")
    optimizer = build_optimizer(model, cfg["train"])
    checkpoint_path = model_dir / f"{name}.pt"
    model_path = model_dir / f"{name}.keras"
    best_f1, best_epoch, bad_evals = -1.0, 0, 0
    train_seconds = 0.0
    fields = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "val_f1_macro",
              "learning_rate", "next_learning_rate"]
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for epoch in range(epochs):
            started = time.perf_counter()
            training, count, steps = train_epoch(model, train_ds, optimizer, device)
            train_seconds += time.perf_counter() - started
            validation = evaluate(model, val_ds, device)
            metrics = {**training, **validation, "learning_rate": cfg["train"]["lr"],
                       "next_learning_rate": cfg["train"]["lr"]}
            if metrics["val_f1_macro"] > best_f1:
                best_f1, best_epoch, bad_evals = metrics["val_f1_macro"], epoch + 1, 0
                torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "epoch": best_epoch, "metrics": metrics, "architecture": "mlp_reference",
                            "img_size": size, "num_classes": classes}, checkpoint_path)
            else:
                bad_evals += 1
            writer.writerow({"epoch": epoch, **metrics})
            output.flush()
            print(f"MLP epoch {epoch + 1}/{epochs} - {steps} batches - "
                  + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()), flush=True)
            if epoch + 1 >= min_epochs and bad_evals >= cfg["train"]["patience"]:
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    exported = build_keras_mlp(size, classes)
    for layer in ("dense", "logits"):
        exported.get_layer(layer).set_weights([
            checkpoint["model_state"][f"{layer}.weight"].numpy().T,
            checkpoint["model_state"][f"{layer}.bias"].numpy(),
        ])
    exported.save(model_path)
    summary = {
        "run": name, "data": "full", "ipc": None, "seed": seed, "img_size": size,
        "train_images": count, "epochs_run": epoch + 1, "best_epoch": best_epoch,
        "val_accuracy": checkpoint["metrics"]["val_accuracy"], "val_f1_macro": best_f1,
        "train_seconds": round(train_seconds, 1), "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path), "architecture": "mlp_reference",
        "parameters": sum(p.numel() for p in model.parameters()), "device": device_name,
        "optimizer": type(optimizer).__name__, "weight_decay": 0.0,
        "learning_rate": cfg["train"]["lr"], "patience": cfg["train"]["patience"], "min_epochs": min_epochs,
        "provenance": "Reconstructed fully connected reference, not the original Avance 3 model",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-epochs", type=int, default=0)
    options = parser.parse_args()
    run(options.config, options.output, epochs=options.epochs, lr=options.lr,
        patience=options.patience, min_epochs=options.min_epochs)
