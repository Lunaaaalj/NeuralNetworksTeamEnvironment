"""Matrices completas de validación, con filas reales y columnas predichas."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_tables(y_true, y_pred, num_classes):
    counts = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    return counts, normalized


def save_confusion_artifacts(y_true, y_pred, class_names, directory, figure_directory, *, title):
    """CSV sin perder la diagonal y figuras de las 104 clases; no reentrena el modelo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    directory, figure_directory = Path(directory), Path(figure_directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    counts, normalized = confusion_tables(y_true, y_pred, len(class_names))
    labels = [f"{index}: {name}" for index, name in enumerate(class_names)]
    for name, values in (("confusion_matrix", counts), ("confusion_matrix_normalized", normalized)):
        table = pd.DataFrame(values, index=labels, columns=labels)
        table.index.name = "clase_real / clase_predicha"
        table.to_csv(directory / f"{name}.csv", encoding="utf-8-sig")
    pd.DataFrame({"sample_index": np.arange(len(y_true)), "true_label": y_true,
                  "predicted_label": y_pred, "correct": np.asarray(y_true) == np.asarray(y_pred)}).to_csv(
        directory / "validation_predictions.csv", index=False)
    accuracy = np.trace(counts) / counts.sum()
    artifacts = {}
    for name, values in (("confusion_matrix", counts), ("confusion_matrix_normalized", normalized)):
        is_normalized = name.endswith("normalized")
        fig, ax = plt.subplots(figsize=(22, 20), layout="constrained")
        plot = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if is_normalized else None,
                         interpolation="nearest")
        colorbar = fig.colorbar(plot, ax=ax, fraction=0.035, pad=0.015)
        colorbar.set_label("Proporción dentro de la clase real" if is_normalized else "Número de imágenes")
        if is_normalized:
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks, labels=labels, rotation=90, fontsize=7)
        ax.set_yticks(ticks, labels=labels, fontsize=7)
        ax.set_xlabel("Clase predicha", fontsize=13)
        ax.set_ylabel("Clase real", fontsize=13)
        description = "normalizada por fila" if is_normalized else "conteos"
        ax.set_title(f"{title}\nMatriz de confusión: {description} | Validación, n={len(y_true):,} | "
                     f"Exactitud={accuracy:.2%}", fontsize=16, pad=15)
        for row, col in np.argwhere(counts > 0):
            if is_normalized and row != col:
                continue
            value = values[row, col]
            text = f"{value:.0%}" if is_normalized else str(value)
            limit = 0.5 if is_normalized else counts.max() / 2
            ax.text(col, row, text, ha="center", va="center", fontsize=4 if is_normalized else 5.5,
                    color="white" if value > limit else "black")
        paths = []
        for extension in ("png", "pdf"):
            path = figure_directory / f"{name}.{extension}"
            fig.savefig(path, dpi=180)
            paths.append(str(path.resolve()))
        plt.close(fig)
        artifacts[name] = paths
    return {"samples": int(counts.sum()), "correct_predictions": int(np.trace(counts)),
            "accuracy": float(accuracy), "matrices": artifacts,
            "csv_directory": str(directory.resolve())}
