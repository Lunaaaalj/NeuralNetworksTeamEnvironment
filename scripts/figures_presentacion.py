"""Figuras 16:9 para la presentación final, con los mismos datos que el reporte.

PNG con fondo transparente (para colocarlas sobre el diseño de Canva), 2400x1350 px, letra
grande. Paleta validada para daltonismo: azul/naranja para entrenamiento/validación; la
tercera serie (verde) va siempre con línea punteada y etiqueta directa.

Uso:
    python -m scripts.figures_presentacion        # escribe reports/final/figures/presentacion/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analyze_final import PLAIN, at_best, load_runs, results_table
from src.config import ROOT

BLUE, ORANGE, GREEN = "#1d6fb5", "#cc5a2e", "#2e8b57"
INK, INK_2, MUTED, GRID = "#1e3a2f", "#4a5c54", "#8a9a92", "#d5ddd8"
SIZE = (12, 6.75)  # 16:9

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 17, "axes.titlecolor": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": MUTED, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.6,
    "axes.axisbelow": True, "axes.facecolor": "none", "figure.facecolor": "none", "savefig.transparent": True,
})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color, fontsize=13,
                va="center", fontweight="bold")


def curves(runs, labels, out, name, common_scale=False):
    """Pérdida, exactitud y F1: entrenamiento (sin aumento) contra validación, una fila por modelo.

    common_scale fija los ejes de exactitud y F1 en 0-0.9, como las CNN: sin eso, el zoom automático
    hace que las curvas de la MLP parezcan aprender."""
    labels = [label for label in labels if label in runs]
    fig, axes = plt.subplots(len(labels), 3, figsize=SIZE, squeeze=False)
    panels = [("Pérdida", "train_eval_loss", "val_loss"), ("Exactitud", "train_eval_accuracy", "val_accuracy"),
              ("F1 macro", "train_f1_macro", "val_f1_macro")]
    for row, label in enumerate(labels):
        history, best = runs[label]["history"], runs[label]["summary"]["best_epoch"]
        epochs = history["epoch"] + 1
        for col, (title, train_key, val_key) in enumerate(panels):
            ax = axes[row, col]
            ax.plot(epochs, history[train_key], color=BLUE)
            ax.plot(epochs, history[val_key], color=ORANGE)
            ax.axvline(best, color=INK_2, linewidth=1.2, linestyle=":")
            ax.set_xlim(0, epochs.max() * 1.22)
            if common_scale:
                ax.set_ylim((0, 4.5) if col == 0 else (0, 0.9))
            if col == 2:  # etiquetas directas una sola vez por fila; se separan si las curvas terminan juntas
                end_train, end_val = history[train_key].iloc[-1], history[val_key].iloc[-1]
                low, high = ax.get_ylim()
                dy = 9 if abs(end_train - end_val) < 0.08 * (high - low) else 0
                label_end(ax, epochs.iloc[-1], end_train, "Entr.", BLUE, dy=dy)
                label_end(ax, epochs.iloc[-1], end_val, "Val.", ORANGE, dy=-dy)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(PLAIN[label], color=INK, fontsize=15, fontweight="bold")
            if row == len(labels) - 1:
                ax.set_xlabel("Época")
        gap = at_best(runs[label])
        y_text, va = (0.95, "top") if common_scale else (0.05, "bottom")  # donde no hay curvas
        axes[row, 2].text(0.97, y_text, f"checkpoint: época {best}\nbrecha F1 = "
                          f"{gap.train_f1_macro - runs[label]['summary']['val_f1_macro']:.3f}",
                          transform=axes[row, 2].transAxes, ha="right", va=va, fontsize=12, color=INK_2)
    fig.tight_layout()
    save(fig, out, name)


def imbalance(counts, out):
    train = counts["train"].sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(np.arange(len(train)), train, width=0.75, color=BLUE, linewidth=0)
    median = counts["train"].median()
    ax.axhline(median, color=ORANGE, linewidth=2, linestyle="--")
    ax.text(len(train) - 1, median + 18, f"mediana = {median:.0f}", ha="right", color=ORANGE, fontsize=15,
            fontweight="bold")
    ordered = counts.sort_values("train", ascending=False)
    for position, (_, row) in enumerate(ordered.head(3).iterrows()):
        ax.annotate(f"{row['name']} ({row.train})", (position, row.train), xytext=(14, -2 - 20 * position),
                    textcoords="offset points", fontsize=13, color=INK_2)
    stats = (f"mínimo {train.min()} · mediana {median:.0f} · máximo {train.max()}\n"
             f"razón máx/mín = {train.max() / train.min():.0f} : 1 · 24 clases con menos de 30 imágenes")
    ax.text(0.98, 0.72, stats, transform=ax.transAxes, ha="right", fontsize=15, color=INK)
    ax.set(xlim=(-1, len(train)), xticks=[], xlabel="104 clases, ordenadas de mayor a menor",
           ylabel="Imágenes de entrenamiento")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "desbalance")


def per_class(runs, counts, out, k=8):
    report = pd.DataFrame(runs["referencia"]["eval"]["per_class"]).set_index("label")
    report["n"] = counts["train"]
    ordered = report.sort_values(["f1", "n"], ascending=[False, False])
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharex=True)
    for ax, rows, color, title in ((axes[0], ordered.head(k), BLUE, f"Las {k} mejores"),
                                   (axes[1], ordered.tail(k), ORANGE, f"Las {k} peores")):
        rows = rows.iloc[::-1] if title.startswith("Las 8 mejores") else rows
        y = np.arange(len(rows))
        ax.barh(y, rows.f1, height=0.6, color=color, linewidth=0)
        for yi, (_, row) in zip(y, rows.iterrows()):
            ax.text(max(row.f1, 0) + 0.02, yi, f"{row.f1:.2f}", va="center", fontsize=13, color=INK_2)
        ax.set_yticks(y, [f"{row['name']} (n={row.n})" for _, row in rows.iterrows()], fontsize=13)
        ax.set_title(title)
        ax.set_xlim(0, 1.15)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("F1 en validación")
    fig.text(0.5, 0.01, "n = imágenes de entrenamiento de la clase · CNN de referencia", ha="center",
             fontsize=12, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "f1_por_clase")


def comparison(table, out):
    t = table[table.experimento != "mlp"].sort_values("val_f1_macro")
    names = [PLAIN[k] + ("†" if interrupted else "") for k, interrupted in zip(t.experimento, t.interrumpida)]
    colors = [ORANGE if k == "referencia" else BLUE for k in t.experimento]
    fig, (left, right) = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    y = np.arange(len(t))
    for ax, column, title, fmt in ((left, "val_f1_macro", "F1 macro de validación", "{:.3f}"),
                                   (right, "brecha_f1", "Brecha F1 (entr. − val.)", "{:.3f}")):
        ax.barh(y, t[column], height=0.6, color=colors, linewidth=0)
        for yi, value in zip(y, t[column]):
            ax.text(value + 0.008, yi, fmt.format(value), va="center", fontsize=13, color=INK_2)
        reference = t.set_index("experimento").loc["referencia", column]
        ax.axvline(reference, color=ORANGE, linewidth=1.5, linestyle="--")
        ax.set_title(title)
        ax.set_xlim(0, t[column].max() * 1.2)
        ax.grid(axis="y", visible=False)
    left.set_yticks(y, names)
    mlp = table.set_index("experimento").loc["mlp"] if "mlp" in set(table.experimento) else None
    note = "Naranja: CNN de referencia. "
    if mlp is not None:
        note += f"MLP (fuera de escala): F1 {mlp.val_f1_macro:.3f}, brecha {mlp.brecha_f1:.3f}. "
    if t.interrumpida.any():
        note += "† corrida interrumpida; se usa su mejor checkpoint."
    fig.text(0.5, 0.01, note, ha="center", fontsize=12, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "comparacion_experimentos")


def regularization(runs, out):
    series = [("referencia", BLUE, "-"), ("sin_regularizacion", ORANGE, "-"), ("dropout_0.3", GREEN, "--")]
    series = [s for s in series if s[0] in runs]
    fig, (left, right) = plt.subplots(1, 2, figsize=SIZE)
    for label, color, style in series:
        history = runs[label]["history"]
        epochs = history["epoch"] + 1
        gap = history["train_f1_macro"] - history["val_f1_macro"]
        left.plot(epochs, history["val_f1_macro"], color=color, linestyle=style)
        right.plot(epochs, gap, color=color, linestyle=style)
        label_end(right, epochs.iloc[-1], gap.iloc[-1], PLAIN[label], color)
    left.set(title="F1 macro de validación", xlabel="Época")
    right.set(title="Brecha F1 (entrenamiento − validación)", xlabel="Época")
    right.set_xlim(0, max(runs[s[0]]["history"].epoch.max() for s in series) * 1.45)
    right.axhline(0, color=INK_2, linewidth=1)
    left.legend([plt.Line2D([], [], color=c, linestyle=s) for _, c, s in series],
                [PLAIN[label] for label, *_ in series], loc="lower right", fontsize=13)
    fig.tight_layout()
    save(fig, out, "regularizacion")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final/figures/presentacion")
    args = parser.parse_args()
    experiments = ROOT / "reports/final/experimentos"
    runs = load_runs(experiments)
    counts = pd.read_csv(experiments / "conteo_por_clase.csv").set_index("label")
    table = results_table(runs, counts)

    curves(runs, ["referencia", "cnn_ampliada"], args.out, "curvas_cnn")
    curves(runs, ["mlp"], args.out, "curvas_mlp", common_scale=True)
    imbalance(counts, args.out)
    per_class(runs, counts, args.out)
    comparison(table, args.out)
    regularization(runs, args.out)
    print(f"Figuras en {args.out}")


if __name__ == "__main__":
    main()
