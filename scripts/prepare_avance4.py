"""Produce auditable metrics, figures, LaTeX sections and the Avance 4 notebook."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src.config import ROOT

LABELS = {
    "mlp_referencia": "Totalmente conectada (referencia)",
    "cnn_original": "CNN original (4 convoluciones)",
    "cnn_dos_capas_extra": "CNN ampliada (6 convoluciones)",
    "cnn_regularizada": "CNN original + AdamW + LR adaptable",
}
COLORS = {"mlp_referencia": "#7564a5", "cnn_original": "#2463a7",
          "cnn_dos_capas_extra": "#c06a20", "cnn_regularizada": "#12805c"}


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def architecture_figures(output):
    common = [
        ("Entrada RGB", "112 × 112 × 3"),
        ("Conv1: 3×3, 32 + ReLU + Pool", "56 × 56 × 32"),
        ("Conv2: 3×3, 64 + ReLU + Pool", "28 × 28 × 64"),
        ("Conv3: 3×3, 128 + ReLU + Pool", "14 × 14 × 128"),
        ("Conv4: 3×3, 256 + ReLU + Pool", "7 × 7 × 256"),
        ("Global Average Pooling", "256 características"),
        ("Densa + ReLU", "256 neuronas"),
        ("Dropout", "p = 0.5"),
        ("Densa de salida (logits)", "104 clases"),
    ]
    deep = common[:4] + [("Conv extra 1: 3×3, 128 + ReLU", "14 × 14 × 128")]
    deep += common[4:5] + [("Conv extra 2: 3×3, 256 + ReLU", "7 × 7 × 256")] + common[5:]
    mlp = [("Entrada RGB", "112 × 112 × 3"), ("Flatten (sin convoluciones)", "37,632 valores"),
           ("Densa + ReLU", "256 neuronas"), ("Dropout", "p = 0.5"),
           ("Densa de salida (logits)", "104 clases")]
    specs = [("mlp_referencia", mlp), ("cnn_original", common), ("cnn_dos_capas_extra", deep)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 11), layout="constrained")
    for ax, (key, blocks) in zip(axes, specs):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        diagram_titles = {"mlp_referencia": "MLP\nReferencia reconstruida",
                          "cnn_original": "CNN original\n4 convoluciones",
                          "cnn_dos_capas_extra": "CNN ampliada\n6 convoluciones"}
        ax.set_title(diagram_titles[key], fontsize=16, fontweight="bold", pad=16)
        height = min(0.068, (0.85 / (len(blocks) - 1)) * 0.65)
        positions = np.linspace(0.93, 0.08, len(blocks))
        for index, ((title, shape), y) in enumerate(zip(blocks, positions)):
            extra = "extra" in title
            box = FancyBboxPatch((0.02, y-height/2), 0.96, height,
                                 boxstyle="round,pad=0.008", linewidth=1.3,
                                 edgecolor="#c06a20" if extra else COLORS[key],
                                 facecolor="#fff0dc" if extra else "#f0f4f8")
            ax.add_patch(box)
            ax.text(0.5, y, f"{title}\n{shape}", ha="center", va="center", fontsize=14)
            if index:
                ax.annotate("", xy=(0.5, y + height/2 + 0.009),
                            xytext=(0.5, positions[index-1] - height/2 - 0.009),
                            arrowprops={"arrowstyle": "->", "color": COLORS[key]})
        if key == "mlp_referencia":
            ax.text(0.5, 0.005, "Referencia reconstruida; no es el checkpoint del Avance 3.",
                    ha="center", fontsize=9)
    fig.suptitle("Modelos: misma entrada y mismas 104 clases\nPool = MaxPooling 2×2; convoluciones con padding same", fontsize=17)
    save_figure(fig, output / "figures/arquitecturas")
    for key, blocks in specs + [("cnn_regularizada", common)]:
        fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
        ax.axis("off")
        table = ax.table(cellText=[[i+1, title, shape] for i, (title, shape) in enumerate(blocks)],
                         colLabels=["Paso", "Operación", "Salida"], loc="center",
                         colWidths=[0.08, 0.64, 0.28], cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.4)
        ax.set_title(LABELS[key], fontsize=17, pad=20)
        save_figure(fig, output / "figures" / key / "arquitectura")


def evaluate_run(output, key, summary, cfg):
    import keras
    import torch
    from src.classes import CLASSES
    from src.confusion import save_confusion_artifacts
    from src.data import build_dataset
    from src.directml_input import batches
    from src.evaluate import error_report
    from src.train_directml import FlowerCNN, select_device
    from src.train_mlp_reference import FlowerMLP

    run_dir = output / "runs" / key / summary["run"]
    evaluation_path = run_dir / "eval_report.json"
    if evaluation_path.exists():
        return json.loads(evaluation_path.read_text(encoding="utf-8"))
    device, _ = select_device()
    checkpoint = torch.load(summary["checkpoint_path"], map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    model = (FlowerMLP(checkpoint["img_size"], checkpoint["num_classes"])
             if architecture == "mlp_reference" else FlowerCNN(checkpoint["num_classes"], architecture=architecture))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    ds = build_dataset(cfg["paths"]["tfrecords"], "val", summary["img_size"], 128)
    y_true, y_pred, confidence, examples = [], [], [], []
    loss_sum, total = 0.0, 0
    probe = None
    with torch.no_grad():
        for images, labels in batches(ds, device):
            logits = model(images)
            probabilities = logits.softmax(1)
            predictions = logits.argmax(1)
            image_cpu = images.cpu().numpy().transpose(0, 2, 3, 1)
            if probe is None:
                probe = (image_cpu[:4], logits[:4].cpu().numpy())
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            confidence.extend(probabilities.max(1).values.cpu().numpy().tolist())
            examples.extend(image_cpu)
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").cpu().item()
            total += len(labels)
    y_true, y_pred, confidence = np.array(y_true), np.array(y_pred), np.array(confidence)
    # Verify that the portable Keras model and GPU checkpoint encode the same model.
    exported = keras.models.load_model(summary["model_path"])
    portable_logits = exported(probe[0], training=False).numpy()
    np.testing.assert_allclose(portable_logits, probe[1], atol=0.005, rtol=0.005)
    assert np.array_equal(portable_logits.argmax(1), probe[1].argmax(1))
    report = error_report(y_true, y_pred, len(CLASSES))
    report.update(save_confusion_artifacts(
        y_true, y_pred, CLASSES, run_dir / "evaluation", output / "figures" / key,
        title=LABELS[key]))
    report.update(val_loss=loss_sum / total, split="val", seed=summary["seed"],
                  checkpoint_epoch=checkpoint["epoch"], keras_export_verified=True,
                  model_path=summary["model_path"])
    from src.confusion import confusion_tables
    counts, normalized = confusion_tables(y_true, y_pred, len(CLASSES))
    errors = counts.copy()
    np.fill_diagonal(errors, 0)
    focus = np.sort(np.argsort(errors.sum(0) + errors.sum(1))[-12:])
    view = normalized[np.ix_(focus, focus)]
    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    plot = ax.imshow(view, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(focus)), [CLASSES[c] for c in focus], rotation=60, ha="right")
    ax.set_yticks(range(len(focus)), [CLASSES[c] for c in focus])
    ax.set(xlabel="Clase predicha", ylabel="Clase real", title=LABELS[key] +
           "\n12 clases con más participación en errores (submatriz)")
    for row in range(len(focus)):
        for col in range(len(focus)):
            if view[row, col] > 0:
                ax.text(col, row, f"{view[row,col]:.0%}", ha="center", va="center",
                        color="white" if view[row,col] > 0.5 else "black", fontsize=10)
    fig.colorbar(plot, ax=ax, label="Fracción sobre TODOS los casos reales de la clase")
    fig.supxlabel("Las filas pueden sumar menos de 100%: se omiten las otras 92 clases.", fontsize=10)
    save_figure(fig, output / "figures" / key / "confusiones_principales")
    report["confusion_focus_labels"] = focus.tolist()
    pd.DataFrame({"sample_index": np.arange(total), "true_label": y_true, "predicted_label": y_pred,
                  "confidence": confidence, "correct": y_true == y_pred}).to_csv(
                      run_dir / "evaluation/validation_predictions.csv", index=False)
    # Balanced display of actual successes and failures, selected deterministically.
    rng = np.random.default_rng(0)
    good, bad = np.flatnonzero(y_true == y_pred), np.flatnonzero(y_true != y_pred)
    selected = np.r_[rng.choice(good, min(6, len(good)), replace=False),
                     rng.choice(bad, min(6, len(bad)), replace=False)]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
    selected_rows = []
    for ax, index in zip(axes.flat, selected):
        ax.imshow(np.clip(examples[index], 0, 1))
        correct = y_true[index] == y_pred[index]
        ax.set_title(f"Real: {CLASSES[y_true[index]]}\nPred.: {CLASSES[y_pred[index]]}\n"
                     f"Confianza: {confidence[index]:.1%}",
                     color="#12805c" if correct else "#b52b32", fontsize=12)
        ax.axis("off")
        selected_rows.append({"sample_index": int(index), "correct": bool(correct)})
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle(f"{LABELS[key]}: aciertos y errores de validación\n"
                 "Selección ilustrativa equilibrada; no representa la frecuencia de errores", fontsize=15)
    save_figure(fig, output / "figures" / key / "predicciones_ejemplo")
    report["displayed_examples"] = selected_rows
    evaluation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def plot_results(output, records):
    rows = []
    fig, axes = plt.subplots(len(records), 2, figsize=(14, 3.7*len(records)), layout="constrained", squeeze=False)
    for row, (key, summary, evaluation, history) in enumerate(records):
        epochs = history.epoch + 1
        best = summary["best_epoch"]
        for column, metric in enumerate(("accuracy", "loss")):
            ax = axes[row, column]
            ax.plot(epochs, history[metric], label="Entrenamiento (con aumentos y dropout)", color=COLORS[key])
            ax.plot(epochs, history[f"val_{metric}"], label="Validación (sin aumentos)", color="#c43d43")
            ax.axvline(best, ls=":", color="black", alpha=0.6, label=f"Checkpoint: época {best}")
            ax.set(title=LABELS[key], xlabel="Época", ylabel="Exactitud" if metric == "accuracy" else "Entropía cruzada")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        single, single_axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, metric in zip(single_axes, ("accuracy", "loss")):
            ax.plot(epochs, history[metric], label="Entrenamiento", color=COLORS[key])
            ax.plot(epochs, history[f"val_{metric}"], label="Validación", color="#c43d43")
            ax.axvline(best, ls=":", color="black", label=f"Mejor F1: época {best}")
            ax.set(xlabel="Época", ylabel="Exactitud" if metric == "accuracy" else "Entropía cruzada")
            ax.grid(alpha=0.2)
            ax.legend()
        single.suptitle(LABELS[key])
        save_figure(single, output / "figures" / key / "curvas_entrenamiento_validacion")
        rate_fig, rate_ax = plt.subplots(figsize=(10, 3), layout="constrained")
        rate_ax.step(epochs, history.learning_rate, where="post", color=COLORS[key])
        rate_ax.set(xlabel="Época", ylabel="Tasa de aprendizaje", yscale="log", title=LABELS[key])
        rate_ax.grid(alpha=0.2)
        save_figure(rate_fig, output / "figures" / key / "tasa_aprendizaje")
        best_row = history.iloc[best-1]
        rows.append({"experiment": key, "model": LABELS[key], "architecture": summary["architecture"],
                     "parameters": summary["parameters"], "train_images": summary["train_images"],
                     "validation_images": evaluation["samples"], "epochs_run": summary["epochs_run"],
                     "best_epoch": best, "val_accuracy": evaluation["accuracy"],
                     "val_f1_macro": evaluation["f1_macro"], "val_loss": evaluation["val_loss"],
                     "train_accuracy_at_checkpoint": best_row.accuracy,
                     "train_loss_at_checkpoint": best_row.loss,
                     "accuracy_gap_at_checkpoint": best_row.accuracy - evaluation["accuracy"],
                     "train_seconds": summary["train_seconds"], "seed": summary["seed"],
                     "model_path": summary["model_path"]})
    save_figure(fig, output / "figures/comparacion_curvas")
    record_map = {record[0]: record for record in records}
    for filename, keys in (("comparacion_original_ampliada", ("cnn_original", "cnn_dos_capas_extra")),
                           ("comparacion_referencia_regularizada", ("mlp_referencia", "cnn_regularizada"))):
        pair_fig, pair_axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
        for index, key in enumerate(keys):
            _, summary, _, history = record_map[key]
            for ax, metric in zip(pair_axes[index], ("accuracy", "loss")):
                ax.plot(history.epoch+1, history[metric], color=COLORS[key], label="Entrenamiento")
                ax.plot(history.epoch+1, history[f"val_{metric}"], color="#c43d43", label="Validación")
                ax.axvline(summary["best_epoch"], ls=":", color="black", label="Checkpoint")
                ax.set(title=LABELS[key], xlabel="Época", ylabel="Exactitud" if metric == "accuracy" else "Pérdida")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=9)
        save_figure(pair_fig, output / "figures" / filename)
    table = pd.DataFrame(rows)
    table.to_csv(output / "resultados.csv", index=False, encoding="utf-8-sig")
    table.to_json(output / "resultados.json", orient="records", indent=2, force_ascii=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    short = {"mlp_referencia": "MLP (referencia)", "cnn_original": "CNN original",
             "cnn_dos_capas_extra": "CNN +2 convoluciones", "cnn_regularizada": "CNN + AdamW + LR"}
    for ax, metric, title in zip(axes, ("val_accuracy", "val_f1_macro"), ("Exactitud de validación", "F1 macro de validación")):
        bars = ax.bar([short[key] for key in table.experiment], table[metric],
                      color=[COLORS[key] for key in table.experiment])
        ax.bar_label(bars, labels=[f"{value:.2%}" if metric == "val_accuracy" else f"{value:.4f}"
                                  for value in table[metric]], padding=4)
        ax.set(ylim=(0, min(1, table[metric].max()+0.15)), title=title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output / "figures/comparacion_metricas")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagrams-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    architecture_figures(output)
    historical = output / "figures/dm_historico"
    historical.mkdir(parents=True, exist_ok=True)
    for ipc in (1, 10):
        source = ROOT / f"reports/figures/synthetic_ipc{ipc}_res112.png"
        if source.exists():
            shutil.copy2(source, historical / source.name)
    (historical / "PROCEDENCIA.txt").write_text(
        "Figuras DM históricas del repositorio (añadidas el 12 de octubre de 2026).\n"
        "No fueron generadas por las nuevas CNN entrenadas en este avance.\n"
        "IPC 1: 104 imágenes sintéticas; IPC 10: 1,040.\n"
        "Ambos mosaicos muestran solo la primera imagen sintética de cada clase.\n",
        encoding="utf-8")
    if args.diagrams_only:
        return
    from src.config import load_config
    configs = {key: load_config(output / f"{key}.yaml") for key in LABELS if key != "mlp_referencia"}
    configs["mlp_referencia"] = load_config(output / "mlp_referencia.yaml")
    records = []
    for key in LABELS:
        summaries = list((output / "runs" / key).glob("*/summary.json"))
        if len(summaries) != 1:
            raise RuntimeError(f"Expected one completed {key} experiment, found {len(summaries)}")
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        evaluation = evaluate_run(output, key, summary, configs[key])
        history = pd.read_csv(summaries[0].parent / "history.csv")
        records.append((key, summary, evaluation, history))
    table = plot_results(output, records)
    print(table[["experiment", "val_accuracy", "val_f1_macro", "parameters"]].to_string(index=False))


if __name__ == "__main__":
    main()
