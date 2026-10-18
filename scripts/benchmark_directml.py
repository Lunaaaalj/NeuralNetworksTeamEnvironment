"""Compara el baseline y batches 32/64/128 sin sobrescribir modelos ni activar W&B."""

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import statistics
import time

import keras
import torch

from src.config import load_config, set_seed
from src.data import build_dataset
from src.directml_checkpoint import export_checkpoint, save_checkpoint
from src.directml_input import INPUT_PIPELINES
from src.model import build_cnn
from src.optim_directml import DirectMLAdam
from src.train_directml import FlowerCNN, copy_from_keras, copy_to_keras, evaluate, select_device, train_epoch


def available_memory():
    """RAM libre en Windows; None si la plataforma no expone este contador."""
    if not hasattr(ctypes, "windll"):
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "total_page", "available_page",
                "total_virtual", "available_virtual", "available_extended")]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.available_physical


def run_case(cfg, device, batch_size, *, optimized=True, cache=False, epochs=3,
             input_pipeline="numpy"):
    keras.backend.clear_session()
    gc.collect()
    set_seed(0)
    torch.manual_seed(0)
    img_size = cfg["data"]["img_size"]
    cache_dir = Path(cfg["paths"]["processed"]) / "tfdata_cache"
    train = build_dataset(cfg["paths"]["tfrecords"], "train", img_size, batch_size,
                          augment=True, shuffle=True, seed=0, cache=cache, cache_dir=cache_dir)
    val = build_dataset(cfg["paths"]["tfrecords"], "val", img_size, 128, cache=cache, cache_dir=cache_dir)
    model = FlowerCNN(cfg["data"]["num_classes"])
    initial = build_cnn(cfg["data"]["num_classes"])
    copy_from_keras(initial, model)
    del initial
    model.to(device)
    optimizer = (DirectMLAdam(model.parameters(), lr=cfg["train"]["lr"], eps=1e-7) if optimized
                 else torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], eps=1e-7, foreach=False))
    cache_label = "disk" if cache == "disk" else str(int(cache))
    label = f"{'fast' if optimized else 'baseline'}_b{batch_size}_cache{cache_label}_{input_pipeline}"
    measurements = []
    for epoch in range(epochs):
        start = time.perf_counter()
        metrics, images, steps = train_epoch(model, train, optimizer, device,
                                            sync_each_batch=not optimized, input_pipeline=input_pipeline)
        trained = time.perf_counter()
        metrics.update(evaluate(model, val, device, input_pipeline=input_pipeline))
        finished = time.perf_counter()
        measurement = {"epoch": epoch + 1, "warmup": epoch == 0, "images": images, "batches": steps,
                       "train_seconds": trained - start, "validation_seconds": finished - trained,
                       "epoch_seconds": finished - start, **metrics}
        measurements.append(measurement)
        print(f"{label} epoch {epoch + 1}/{epochs}: {finished - start:.2f}s "
              f"(train {trained - start:.2f}s; val {finished - trained:.2f}s; "
              f"{images} images; {'warmup' if epoch == 0 else 'measured'})", flush=True)
    return {"case": label, "batch_size": batch_size, "optimized": optimized, "cache": cache,
            "input_pipeline": input_pipeline,
            "median_epoch_seconds": statistics.median(row["epoch_seconds"] for row in measurements[1:]),
            "epochs": measurements}


def compare_saving(cfg, device, directory):
    """Mide guardados reales en GPU, en una carpeta aislada de los modelos del usuario."""
    model = FlowerCNN(cfg["data"]["num_classes"]).to(device)
    exported = build_cnn(cfg["data"]["num_classes"])
    old, native = [], []
    for repeat in range(4):
        started = time.perf_counter()
        copy_to_keras(model, exported)
        exported.save(directory / "saving_legacy.keras")
        old.append(time.perf_counter() - started)
        started = time.perf_counter()
        save_checkpoint(model, directory / "saving_native.pt", epoch=repeat + 1,
                        metrics={"val_f1_macro": 0.0}, img_size=cfg["data"]["img_size"])
        native.append(time.perf_counter() - started)
    started = time.perf_counter()
    export_checkpoint(directory / "saving_native.pt", directory / "saving_native.keras")
    return {"legacy_median_seconds": statistics.median(old[1:]),
            "native_median_seconds": statistics.median(native[1:]),
            "final_export_seconds": time.perf_counter() - started,
            "legacy_samples_seconds": old, "native_samples_seconds": native,
            "note": "Primera repetición excluida; exportación final fuera del bucle de épocas."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--epochs", type=int, default=3, help="primera época de warmup + al menos una medida")
    parser.add_argument("--no-cache", action="store_true", help="no medir variantes que guardan imágenes en RAM")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-cache", action="store_true", help="compara sin caché/RAM/disco con el batch del config")
    comparison.add_argument("--compare-input", action="store_true", help="compara NumPy/DLPack/prefetch y mide los guardados")
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("--epochs requiere al menos 2 (warmup + medición)")
    cfg = load_config(args.config)
    device, name = select_device()
    free_ram = available_memory()
    # 12,753 train + 3,712 val en float32 RGB; dejar 2 GiB adicionales libres.
    estimated_cache = 16465 * cfg["data"]["img_size"] ** 2 * 3 * 4
    cache_allowed = not args.no_cache and free_ram is not None and free_ram > estimated_cache + 2 * 1024 ** 3
    if args.compare_input:
        cache = cfg["data"].get("cache", False)
        if cache in (True, "ram"):
            if not cache_allowed:
                parser.error("La comparación de entrada requiere RAM suficiente para el modo configurado; usa cache: disk o libera RAM.")
            cache = True
        jobs = [(cfg["data"]["batch_size"], True, cache)]
    elif args.compare_cache:
        batch = cfg["data"]["batch_size"]
        jobs = [(batch, True, False)]
        if cache_allowed:
            jobs.append((batch, True, True))
        storage = Path(cfg["paths"]["processed"])
        while not storage.exists():
            storage = storage.parent
        if shutil.disk_usage(storage).free > estimated_cache + 2 * 1024 ** 3:
            jobs.append((batch, True, "disk"))
        else:
            print("Disk cache skipped: insufficient free disk space.", flush=True)
    else:
        jobs = [(32, False, False)] + [(batch, True, False) for batch in (32, 64, 128)]
        if cache_allowed:
            jobs += [(batch, True, True) for batch in (32, 64, 128)]
    if not cache_allowed:
        print("RAM cache cases skipped (disabled or insufficient free RAM).", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(cfg["paths"]["runs"]) / "directml_benchmark" / stamp / "results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {"device": name, "img_size": cfg["data"]["img_size"], "learning_rate": cfg["train"]["lr"],
              "free_ram_bytes": free_ram, "estimated_cache_bytes": estimated_cache,
              "comparison": "input" if args.compare_input else "cache" if args.compare_cache else "batches",
              "note": "Primera época excluida. Incluye validación; excluye exportación Keras y W&B. "
                      "Estas corridas cortas miden velocidad, no calidad final ni equivalencia entre batches.",
              "cases": []}
    print(f"Benchmark report: {destination}", flush=True)
    for batch, optimized, cache in jobs:
        pipelines = INPUT_PIPELINES if args.compare_input else (cfg.get("runtime", {}).get("input_pipeline", "numpy"),)
        for pipeline in pipelines:
            result = run_case(cfg, device, batch, optimized=optimized, cache=cache,
                              epochs=args.epochs, input_pipeline=pipeline)
            report["cases"].append(result)
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.compare_input:
        report["saving"] = compare_saving(cfg, device, destination.parent)
        print(f"Checkpoint saving: {report['saving']}", flush=True)
    fastest = min((row for row in report["cases"] if row["optimized"]),
                  key=lambda row: row["median_epoch_seconds"])
    report["fastest"] = {key: fastest[key] for key in ("case", "batch_size", "cache", "input_pipeline", "median_epoch_seconds")}
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in report["cases"]:
        print(f"{row['case']}: {row['median_epoch_seconds']:.2f}s/epoch", flush=True)
    print(f"Fastest measured: {report['fastest']}", flush=True)
    print(f"Results: {destination}", flush=True)


if __name__ == "__main__":
    main()
