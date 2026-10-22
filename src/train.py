"""Entrena la CNN de la Tabla 3 con el conjunto completo, un subconjunto real aleatorio o el destilado.

La validación siempre se hace sobre el conjunto real de val, a la resolución `data.img_size`.

Uso:
    python -m src.train --config configs/local.yaml --data distilled --ipc 10 --seed 0
"""

import argparse
import faulthandler
import json
import signal
import time
from pathlib import Path

import keras
import numpy as np

from src.config import get_strategy, load_config, run_name, set_seed
from src.data import (ClassCache, arrays_dataset, balanced_class_weight, build_dataset, cache_path,
                      class_counts)
from src.distill import load_synthetic, synthetic_path
from src.metrics import ValidationMonitor
from src.model import build_model, compile_cnn, model_architecture
from src.tracking import WandbEpochLogger, track_training
from src.training_policy import keras_scheduler, optimizer_options, training_suffix


def load_train_data(cfg, data, ipc, seed):
    """Devuelve (dataset de entrenamiento, número de imágenes, metadatos extra)."""
    img_size, batch = cfg["data"]["img_size"], cfg["data"]["batch_size"]
    paths = cfg["paths"]
    if data == "full":
        ds = build_dataset(paths["tfrecords"], "train", img_size, batch,
                           augment=cfg["data"].get("augment", True), shuffle=True,
                           seed=seed, cache=cfg["data"].get("cache", False),
                           cache_dir=Path(paths["processed"]) / "tfdata_cache")
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
    if cfg.get("runtime", {}).get("backend") == "directml":
        try:
            from src.train_directml import run as run_directml
        except ImportError as exc:
            raise RuntimeError("Instala DirectML con: python -m pip install -r requirements-directml.txt") from exc
        return run_directml(cfg, data, ipc, seed, epochs, val_ds)
    set_seed(seed)
    tcfg = cfg["train"]
    architecture = model_architecture(cfg)
    img_size = cfg["data"]["img_size"]
    name = run_name(data, ipc, img_size, seed)
    if architecture != "baseline":
        name += f"_{architecture}"
    name += training_suffix(tcfg)
    run_dir = Path(cfg["paths"]["runs"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(cfg["paths"]["models"]) / f"{name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    small = data != "full"
    epochs = epochs or (tcfg["epochs_small"] if small else tcfg["epochs_full"])
    every = tcfg["eval_every_small"] if small else 1

    train_ds, n_images, extra = load_train_data(cfg, data, ipc, seed)
    if val_ds is None:
        val_ds = build_dataset(cfg["paths"]["tfrecords"], "val", img_size, 128,
                               cache=cfg["data"].get("cache", False),
                               cache_dir=Path(cfg["paths"]["processed"]) / "tfdata_cache")

    optimizer, decay = optimizer_options(tcfg)
    smoothing = float(tcfg.get("label_smoothing", 0.0))
    with get_strategy().scope():
        model = compile_cnn(build_model(cfg), tcfg["lr"], optimizer=optimizer, weight_decay=decay,
                            label_smoothing=smoothing)

    class_weight = None
    if tcfg.get("class_weight") == "balanced":
        if data != "full":
            raise ValueError("class_weight solo aplica a --data full (los subconjuntos ya están balanceados)")
        class_weight = balanced_class_weight(
            class_counts(cfg["paths"]["tfrecords"], "train", cfg["data"]["num_classes"]))
    elif tcfg.get("class_weight") not in (None, "none"):
        raise ValueError("train.class_weight debe ser balanced o none")

    train_eval_ds = None
    if tcfg.get("train_eval_images"):
        # Primeras N imágenes de train en orden fijo, sin aumentos: mismo tamaño que val.
        # take() antes de cache(): cachear el split completo y cortarlo deja la caché a medias,
        # TF la descarta y vuelve a decodificar todo en cada época.
        train_eval_ds = build_dataset(cfg["paths"]["tfrecords"], "train", img_size, 128,
                                      ).unbatch().take(int(tcfg["train_eval_images"])).batch(128)
        if cfg["data"].get("cache"):
            train_eval_ds = train_eval_ds.cache()

    monitor = ValidationMonitor(val_ds, model_path, every=every, patience=tcfg["patience"],
                                train_eval_ds=train_eval_ds)
    callbacks = [monitor]
    schedule = keras_scheduler(tcfg)
    if schedule is not None:
        callbacks.append(schedule)
    callbacks.append(keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True))
    (run_dir / "history.csv").unlink(missing_ok=True)

    fine_tune = tcfg.get("fine_tune")
    with track_training(cfg, name, data, ipc, seed, epochs) as tracking_run:
        if tracking_run is not None:
            callbacks.append(WandbEpochLogger(tracking_run))

        start = time.perf_counter()
        if fine_tune:
            # Fase 1: solo la cabeza nueva. Fase 2: últimas capas de la red base con LR menor;
            # BatchNorm sigue congelada para no romper las estadísticas de ImageNet.
            backbone = model.get_layer("backbone")
            backbone.trainable = False
            model = compile_cnn(model, tcfg["lr"], optimizer=optimizer, weight_decay=decay,
                                label_smoothing=smoothing)
            frozen = min(int(fine_tune["frozen_epochs"]), epochs)
            first = model.fit(train_ds, epochs=frozen, callbacks=callbacks, verbose=2, shuffle=False,
                              class_weight=class_weight)
            epochs_done = len(first.epoch)
            if not model.stop_training:
                backbone.trainable = True
                for layer in backbone.layers[:-int(fine_tune["unfreeze_last"])]:
                    layer.trainable = False
                for layer in backbone.layers:
                    if isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = False
                model = compile_cnn(model, float(fine_tune["lr"]), optimizer=optimizer, weight_decay=decay,
                                    label_smoothing=smoothing)
                second = model.fit(train_ds, initial_epoch=frozen, epochs=epochs, callbacks=callbacks,
                                   verbose=2, shuffle=False, class_weight=class_weight)
                epochs_done += len(second.epoch)
        else:
            history = model.fit(train_ds, epochs=epochs, callbacks=callbacks, verbose=2, shuffle=False,
                                class_weight=class_weight)
            epochs_done = len(history.epoch)
        total = time.perf_counter() - start

        summary = {
            "run": name, "data": data, "ipc": ipc if small else None, "seed": seed, "img_size": img_size,
            "train_images": n_images, "epochs_run": epochs_done,
            "best_epoch": monitor.best_epoch, "val_f1_macro": round(monitor.best_f1, 4),
            "val_accuracy": round(monitor.best_accuracy, 4), "train_seconds": round(total - monitor.eval_seconds, 1),
            "distill_seconds": None, "distill_iterations": None,  # columnas fijas para resultados.csv
            "architecture": architecture, "parameters": model.count_params(),
            "optimizer": optimizer, "weight_decay": decay, "lr_schedule": tcfg.get("lr_schedule"),
            "final_learning_rate": float(model.optimizer.learning_rate.numpy()),
            "augment": cfg["data"].get("augment", True), "dropout": cfg.get("model", {}).get("dropout"),
            "l2": cfg.get("model", {}).get("l2", 0.0), "label_smoothing": smoothing,
            "class_weight": tcfg.get("class_weight"), "lr": tcfg["lr"],
            "model_path": str(model_path), **extra,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        if tracking_run is not None:
            tracking_run.summary.update({
                "best_val_f1_macro": summary["val_f1_macro"],
                "best_val_accuracy": summary["val_accuracy"],
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "train_seconds": summary["train_seconds"],
                "train_images": n_images,
                **extra,
            })
    print(json.dumps(summary, indent=2))
    return summary


def main():
    # SIGUSR1 imprime la pila de todos los hilos: run_final lo usa antes de matar una corrida colgada.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--data", choices=["full", "random", "distilled"], required=True)
    parser.add_argument("--ipc", type=int, help="imágenes por clase (random / distilled)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="sobrescribe las épocas del config")
    parser.add_argument("--backend", choices=["tensorflow", "directml"], help="DirectML usa GPUs AMD/Intel en Windows")
    parser.add_argument("--device-index", type=int, help="índice del adaptador DirectML; por defecto prefiere Radeon RX")
    parser.add_argument("--wandb", action="store_true", help="registra métricas y gráficas en Weights & Biases")
    parser.add_argument("--wandb-project", help="proyecto de W&B (por defecto flower-cnn)")
    parser.add_argument("--wandb-entity", help="usuario o equipo de W&B")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], help="modo de W&B")
    args = parser.parse_args()
    if args.data != "full" and not args.ipc:
        parser.error("--ipc es obligatorio con --data random/distilled")
    cfg = load_config(args.config)
    if args.backend is not None:
        cfg.setdefault("runtime", {})["backend"] = args.backend
    if args.device_index is not None:
        cfg.setdefault("runtime", {})["device_index"] = args.device_index
    tracking = cfg.setdefault("wandb", {})
    if args.wandb:
        tracking["enabled"] = True
    for key in ("project", "entity", "mode"):
        value = getattr(args, f"wandb_{key}")
        if value is not None:
            tracking[key] = value
    run(cfg, args.data, args.ipc, args.seed, args.epochs)


if __name__ == "__main__":
    main()
