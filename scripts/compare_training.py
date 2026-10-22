"""Compara las tres corridas full locales usando sus checkpoints elegidos por F1."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/runs/directml_regularized")
    parser.add_argument("--figures", type=Path, default=ROOT / "reports/figures/directml_regularized")
    args = parser.parse_args()
    jobs = [
        ("Original", "directml_fast", "full_res112_s0_directml", "#2463a7"),
        ("6 convoluciones", "directml_deep", "full_res112_s0_deep_directml", "#7d8794"),
        ("AdamW + LR adaptativo", "directml_regularized", "full_res112_s0_adamw_plateau_directml", "#12805c"),
    ]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for label, profile, name, color in jobs:
        directory = ROOT / "reports/runs" / profile / name
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        history = pd.read_csv(directory / "history.csv")
        epochs = history["epoch"] + 1  # CSVLogger y DirectML guardan índices desde cero.
        best = history.loc[epochs == summary["best_epoch"]].iloc[0]
        rows.append({"model": label, "f1_macro": summary["val_f1_macro"],
                     "accuracy": summary["val_accuracy"], "best_epoch": summary["best_epoch"],
                     "epochs_run": summary["epochs_run"], "train_seconds": summary["train_seconds"],
                     "checkpoint": summary["model_path"], "summary": str(directory / "summary.json")})
        for ax, metric in zip(axes[0], ("val_f1_macro", "val_accuracy")):
            ax.plot(epochs, history[metric], color=color, label=label, linewidth=1.5)
            ax.scatter([summary["best_epoch"]], [best[metric]], color=color, marker="*", s=110, zorder=3)
        if profile == "directml_regularized":
            axes[1, 0].plot(epochs, history["loss"], label="Entrenamiento", color=color)
            axes[1, 0].plot(epochs, history["val_loss"], label="Validación", color="#c47929")
            axes[1, 1].step(epochs, history["learning_rate"], where="post", color=color)
            axes[1, 1].set_yscale("log")
    titles = ["F1 macro de validación", "Exactitud de validación",
              "Pérdida — AdamW + LR adaptativo", "Learning rate usado en cada época"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("F1 macro")
    axes[0, 1].set_ylabel("Exactitud")
    axes[1, 0].set_ylabel("Entropía cruzada")
    axes[1, 1].set_ylabel("Learning rate")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Clasificación de flores — validación real, resolución 112, semilla 0\n"
                 "Las estrellas marcan el checkpoint elegido por F1; una corrida por estrategia.", fontsize=13)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "comparison.csv", index=False)
    report = {"runs": rows,
              "f1_change_vs_original": round(rows[2]["f1_macro"] - rows[0]["f1_macro"], 4),
              "accuracy_change_vs_original": round(rows[2]["accuracy"] - rows[0]["accuracy"], 4),
              "note": "Semilla única. Cambian optimizador, regularización, scheduler y paciencia conjuntamente."}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for extension in ("png", "pdf"):
        fig.savefig(args.figures / f"training_comparison.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
