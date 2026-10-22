"""Figura de aciertos y errores de validación para un experimento del reporte final.

Seis aciertos y seis errores elegidos con semilla fija; la proporción es ilustrativa, no
la tasa de error. Escribe reports/final/figures/predicciones_<experimento>.{pdf,png}.

Uso:
    python -m scripts.examples_final referencia
    python -m scripts.examples_final referencia --presentacion   # 2x6, 16:9, fondo transparente
"""

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scripts.analyze_final import PLAIN, save
from src.classes import CLASSES
from src.config import ROOT, load_config
from src.data import build_dataset

OK, WRONG = "#12805c", "#b52b32"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final/figures")
    parser.add_argument("--presentacion", action="store_true",
                        help="2x6 en 16:9 con fondo transparente, en figures/presentacion/")
    args = parser.parse_args()

    experiments = ROOT / "reports/final/experimentos"
    cfg = load_config(experiments / "configs" / f"{args.experiment}.yaml")
    summary = json.loads(next((experiments / "runs" / args.experiment).glob("*/summary.json")).read_text())
    model = keras.models.load_model(summary["model_path"], compile=False)

    images, y_true, y_pred, confidence = [], [], [], []
    for x, y in build_dataset(cfg["paths"]["tfrecords"], "val", cfg["data"]["img_size"], 128):
        probs = tf.nn.softmax(model(x, training=False)).numpy()
        images.append((np.clip(x.numpy(), 0, 1) * 255).astype(np.uint8))
        y_true.append(y.numpy())
        y_pred.append(probs.argmax(1))
        confidence.append(probs.max(1))
    images, y_true, y_pred, confidence = map(np.concatenate, (images, y_true, y_pred, confidence))

    rng = np.random.default_rng(0)
    good, bad = np.flatnonzero(y_true == y_pred), np.flatnonzero(y_true != y_pred)
    selected = np.r_[rng.choice(good, 6, replace=False), rng.choice(bad, 6, replace=False)]

    rows, cols, size, font = (2, 6, (12, 6.75), 10) if args.presentacion else (3, 4, (7.2, 6.4), 7)
    fig, axes = plt.subplots(rows, cols, figsize=size)
    for ax, index in zip(axes.flat, selected):
        correct = y_true[index] == y_pred[index]
        ax.imshow(images[index])
        title = (f"{'✓' if correct else '✗'} Real: {CLASSES[y_true[index]]}\nPred.: {CLASSES[y_pred[index]]}"
                 + ("\n" if args.presentacion else " ") + f"({confidence[index]:.0%})")
        ax.set_title(title, color=OK if correct else WRONG, fontsize=font)
        ax.axis("off")
    fig.tight_layout()
    if args.presentacion:
        (args.out / "presentacion").mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out / "presentacion" / "predicciones.png", dpi=200, transparent=True)
        plt.close(fig)
    else:
        save(fig, args.out, f"predicciones_{args.experiment}")
    print(f"{PLAIN[args.experiment]}: {len(good)} aciertos, {len(bad)} errores en validación")


if __name__ == "__main__":
    main()
