"""Entrena la CNN de la Tabla 3 con el conjunto completo, un subconjunto real aleatorio o el destilado.

La validación siempre se hace sobre el conjunto real de val, a la resolución `data.img_size`.

Uso:
    python -m src.train --config configs/local.yaml --data distilled --ipc 10 --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np

from src.config import get_strategy, load_config, run_name, set_seed
from src.data import ClassCache, arrays_dataset, build_dataset, cache_path
from src.distill import load_synthetic, synthetic_path
from src.metrics import ValidationMonitor
from src.model import build_cnn, compile_cnn


def load_train_data(cfg, data, ipc, seed):
    """Devuelve (dataset de entrenamiento, número de imágenes, metadatos extra)."""
    img_size, batch = cfg["data"]["img_size"], cfg["data"]["batch_size"]
    paths = cfg["paths"]
    if data == "full":
        ds = build_dataset(paths["tfrecords"], "train", img_size, batch, augment=True, shuffle=True, seed=seed)
        return ds, None, {}
    if data == "random":
        cache = ClassCache(cache_path(paths["processed"], img_size))
        images, labels = cache.random_subset(ipc, np.random.default_rng(seed))
        return arrays_dataset(images, labels, batch, seed=seed), len(labels), {}
    if data == "distilled":
        images, labels, meta = load_synthetic(synthetic_path(paths["processed"], ipc, img_size))
        if meta["img_size"] != img_size:
            raise ValueError(f"El conjunto destilado es de {meta['img_size']}px y el config usa {img_size}px")
        return arrays_dataset(images, labels, batch, seed=seed), len(labels), {
            "distill_seconds": meta["distill_seconds"], "distill_iterations": meta["iterations"]}
    raise ValueError(f"data desconocido: {data}")


def run(cfg, data, ipc=None, seed=0, epochs=None, val_ds=None):
    """Entrena una corrida completa y devuelve un resumen (también se guarda en summary.json)."""
    set_seed(seed)
    tcfg = cfg["train"]
    img_size = cfg["data"]["img_size"]
    name = run_name(data, ipc, img_size, seed)
    run_dir = Path(cfg["paths"]["runs"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg["paths"]["models"]) / f"{name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    small = data != "full"
    epochs = epochs or (tcfg["epochs_small"] if small else tcfg["epochs_full"])
    every = tcfg["eval_every_small"] if small else 1

    train_ds, n_images, extra = load_train_data(cfg, data, ipc, seed)
    if val_ds is None:
        val_ds = build_dataset(cfg["paths"]["tfrecords"], "val", img_size, 128)

    with get_strategy().scope():
        model = compile_cnn(build_cnn(cfg["data"]["num_classes"]), tcfg["lr"])

    monitor = ValidationMonitor(val_ds, model_path, every=every, patience=tcfg["patience"])
    callbacks = [monitor, keras.callbacks.CSVLogger(str(run_dir / "history.csv"))]

    start = time.perf_counter()
    history = model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=2, shuffle=False)
    total = time.perf_counter() - start

    summary = {
        "run": name, "data": data, "ipc": ipc if small else None, "seed": seed, "img_size": img_size,
        "train_images": n_images, "epochs_run": len(history.epoch),
        "best_epoch": monitor.best_epoch, "val_f1_macro": round(monitor.best_f1, 4),
        "val_accuracy": round(monitor.best_accuracy, 4), "train_seconds": round(total - monitor.eval_seconds, 1),
        "distill_seconds": None, "distill_iterations": None,  # columnas fijas para resultados.csv
        "model_path": str(model_path), **extra,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--data", choices=["full", "random", "distilled"], required=True)
    parser.add_argument("--ipc", type=int, help="imágenes por clase (random / distilled)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="sobrescribe las épocas del config")
    args = parser.parse_args()
    if args.data != "full" and not args.ipc:
        parser.error("--ipc es obligatorio con --data random/distilled")
    run(load_config(args.config), args.data, args.ipc, args.seed, args.epochs)


if __name__ == "__main__":
    main()
