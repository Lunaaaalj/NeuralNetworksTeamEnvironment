"""Experimentos del reporte final: cada uno cambia una sola cosa respecto a la referencia.

Escribe un YAML por experimento (rutas relativas, portables) en
reports/final/experimentos/configs/, entrena con src.train y evalúa el mejor checkpoint
con src.evaluate. Las corridas terminadas (con eval_report.json) se conservan.

Uso:
    python -m scripts.run_final                     # los 7 del reporte (REPORTE)
    python -m scripts.run_final --all               # también las ablaciones extra
    python -m scripts.run_final --only referencia mlp
    python -m scripts.run_final --list

Si una corrida deja de escribir en el log más de --timeout segundos (p. ej. la GPU Metal
se cuelga), se termina y se pasa a la siguiente.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import yaml

from src.config import ROOT

BASE = "configs/final.yaml"
OUTPUT = Path("reports/final/experimentos")

# (etiqueta, grupo, descripción, cambios respecto a configs/final.yaml)
EXPERIMENTS = [
    ("referencia", "referencia", "CNN original + dropout + aumento + AdamW + LR adaptable", {}),
    ("mlp", "arquitectura", "MLP: 112x112x3 aplanada -> 256 -> 104 (LR 1e-4)",
     {"model": {"architecture": "mlp"}, "train": {"lr": 1.0e-4, "lr_schedule": {"min_lr": 1.0e-6}}}),
    ("cnn_ampliada", "arquitectura", "6 convoluciones (2 extra), misma política",
     {"model": {"architecture": "deep"}}),
    ("sin_regularizacion", "regularizacion", "Sin dropout, sin aumento y sin weight decay",
     {"model": {"dropout": 0.0}, "data": {"augment": False}, "train": {"optimizer": "adam", "weight_decay": 0.0}}),
    ("class_weight", "desbalance", "Pérdida ponderada por clase (balanced)",
     {"train": {"class_weight": "balanced"}}),
    ("lr_3e-4", "hiperparametros", "Tasa de aprendizaje 3e-4 (en vez de 1e-3)",
     {"train": {"lr": 3.0e-4}}),
    ("dropout_0.3", "hiperparametros", "Dropout 0.3 (en vez de 0.5)",
     {"model": {"dropout": 0.3}}),
    ("res_160", "hiperparametros", "Imágenes de 160x160 (en vez de 112x112)",
     {"data": {"img_size": 160, "cache": False}}),
    ("sin_dropout", "regularizacion", "Sin dropout", {"model": {"dropout": 0.0}}),
    ("sin_aumento", "regularizacion", "Sin aumento de datos", {"data": {"augment": False}}),
    ("sin_weight_decay", "regularizacion", "Adam sin weight decay",
     {"train": {"optimizer": "adam", "weight_decay": 0.0}}),
    ("l2", "regularizacion", "Adam + L2 5e-4 en la pérdida (en vez de AdamW)",
     {"model": {"l2": 5.0e-4}, "train": {"optimizer": "adam", "weight_decay": 0.0}}),
    ("label_smoothing", "regularizacion", "Label smoothing 0.1",
     {"train": {"label_smoothing": 0.1}}),
    ("efficientnetb0", "transfer", "EfficientNetB0 (ImageNet) a 224x224: 3 épocas congelada + 5 de fine-tuning",
     {"model": {"architecture": "efficientnetb0", "dropout": 0.2},
      "data": {"img_size": 224, "batch_size": 32, "cache": False},
      "train": {"epochs_full": 8, "patience": 6,
                "fine_tune": {"frozen_epochs": 3, "unfreeze_last": 40, "lr": 1.0e-4}}}),
]

# Corridas que usa el reporte final; las demás quedan disponibles con --all.
REPORTE = ("referencia", "mlp", "cnn_ampliada", "sin_regularizacion", "class_weight", "lr_3e-4",
           "dropout_0.3")


def merge(base, changes):
    out = deepcopy(base)
    for key, value in changes.items():
        out[key] = merge(out.get(key, {}), value) if isinstance(value, dict) else value
    return out


def run_watched(command, timeout):
    """Corre command reenviando su salida; lo termina si pasa timeout segundos sin imprimir nada."""
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = [time.monotonic()]

    def pump():
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            last[0] = time.monotonic()

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    while proc.poll() is None:
        if time.monotonic() - last[0] > timeout:
            proc.send_signal(signal.SIGUSR1)  # src.train vuelca su pila al log
            time.sleep(5)
            proc.kill()
            proc.wait()
            print(f"Sin salida en {timeout} s: corrida terminada", flush=True)
            return False
        time.sleep(5)
    reader.join()
    return proc.returncode == 0


def write_configs(output):
    base = yaml.safe_load((ROOT / BASE).read_text(encoding="utf-8"))
    (ROOT / output / "configs").mkdir(parents=True, exist_ok=True)
    configs = []
    for label, group, description, changes in EXPERIMENTS:
        cfg = merge(base, changes)
        cfg["paths"].update(runs=str(output / "runs" / label), models=f"models/final/{label}",
                            figures=f"models/final/figures/{label}")
        cfg["experiment"] = {"label": label, "group": group, "description": description}
        path = output / "configs" / f"{label}.yaml"
        (ROOT / path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        configs.append((label, path))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="relativo a la raíz del repo")
    parser.add_argument("--only", nargs="*", help="etiquetas a correr")
    parser.add_argument("--all", action="store_true", help="todas las de EXPERIMENTS, no solo REPORTE")
    parser.add_argument("--epochs", type=int, help="tope de épocas (solo para pruebas rápidas)")
    parser.add_argument("--timeout", type=int, default=600, help="segundos sin salida antes de abortar")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for label, group, description, _ in EXPERIMENTS:
            print(f"{'*' if label in REPORTE else ' '} {label:20s} {group:16s} {description}")
        return
    EXPERIMENTS_ALL = [label for label, *_ in EXPERIMENTS]
    for label, config in write_configs(args.output):
        if label not in (args.only or (EXPERIMENTS_ALL if args.all else REPORTE)):
            continue
        run_dir = ROOT / args.output / "runs" / label
        if list(run_dir.glob("*/eval_report.json")):
            print(f"Ya terminado: {label}", flush=True)
            continue
        print(f"\n=== {label} ===", flush=True)
        # Procesos separados: cada corrida libera la caché en RAM y la memoria de la GPU.
        command = [sys.executable, "-u", "-m", "src.train", "--config", str(config), "--data", "full", "--seed", "0"]
        if args.epochs:
            command += ["--epochs", str(args.epochs)]
        if not run_watched(command, args.timeout):
            print(f"Falló: {label}", flush=True)
            continue
        summary = json.loads(next(run_dir.glob("*/summary.json")).read_text())
        run_watched([sys.executable, "-u", "-m", "src.evaluate", "--config", str(config),
                     "--model", summary["model_path"], "--confusion-matrix"], args.timeout)


if __name__ == "__main__":
    main()
