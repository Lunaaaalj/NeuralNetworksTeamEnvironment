"""Recalculate historical MLP/CNN evidence, without importing training frameworks.

python -m scripts.analyze_mlp_avance4
Writes derived diagnostics only to reports/entregable_4/correcciones_etapa2/.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import importlib.metadata
import importlib.util

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
A4 = Path("reports/entregable_4/avance4_2026-10-21")
FINAL = Path("reports/final/experimentos")
OUTPUT = ROOT / "reports/entregable_4/correcciones_etapa2"
SPECS = [
    ("A4_MLP_lr1e4", "MLP histórica A4", A4, "mlp_referencia",
     "full_res112_s0_mlp_reference_directml", 0.0001),
    ("A4_CNN_original", "CNN original A4", A4, "cnn_original", "full_res112_s0_directml", 0.001),
    ("A4_CNN_deep", "CNN ampliada A4", A4, "cnn_dos_capas_extra", "full_res112_s0_deep_directml", 0.001),
    ("A4_CNN_AdamW", "CNN AdamW + LR A4", A4, "cnn_regularizada", "full_res112_s0_adamw_plateau_directml", 0.001),
    ("F_MLP_AdamW", "MLP final (otra corrida)", FINAL, "mlp", "full_res112_s0_mlp_adamw_plateau", 0.0001),
]


def classification_metrics(true, predicted, num_classes=104):
    """Use an explicit label universe, including classes never predicted."""
    true, predicted = np.asarray(true), np.asarray(predicted)
    if (true.ndim != 1 or true.shape != predicted.shape or len(true) == 0
            or not np.isin(true, range(num_classes)).all()
            or not np.isin(predicted, range(num_classes)).all()):
        raise ValueError("Invalid labels, empty data or incompatible shapes")
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=range(num_classes), zero_division=0)
    return {"accuracy": float(accuracy_score(true, predicted)),
            "f1_macro": float(f1.mean()), "predicted_classes": int(len(np.unique(predicted))),
            "zero_f1_classes": int((f1 == 0).sum()),
            "per_class": pd.DataFrame({"label": range(num_classes), "precision": precision,
                                       "recall": recall, "f1": f1, "support": support})}


def validate_saved_predictions(predictions, matrix, report, summary, history, num_classes=104):
    true = predictions.true_label.to_numpy()
    predicted = predictions.predicted_label.to_numpy()
    metrics = classification_metrics(true, predicted, num_classes)
    if not np.array_equal(matrix, confusion_matrix(true, predicted, labels=range(num_classes))):
        raise ValueError("Saved matrix does not match individual predictions")
    for value, key in [(metrics["accuracy"], "accuracy"), (metrics["f1_macro"], "f1_macro")]:
        if abs(value - report[key]) > .000051 or abs(value - summary[f"val_{key}"]) > .000051:
            raise ValueError(f"Stored {key} does not match predictions")
    if len(history) != summary["epochs_run"]:
        raise ValueError("Epoch count does not match history")
    best = history.loc[history.epoch + 1 == summary["best_epoch"]]
    if len(best) != 1 or abs(float(best.iloc[0].val_f1_macro) - metrics["f1_macro"]) > .000051:
        raise ValueError("Checkpoint epoch does not match predictions")
    if abs(float(history.val_f1_macro.max()) - metrics["f1_macro"]) > .000051:
        raise ValueError("Selected checkpoint does not maximize recorded F1")
    return metrics


def save_figure(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=160, bbox_inches="tight",
                    metadata={"Creator": "scripts.analyze_mlp_avance4"})
    plt.close(fig)


def main():
    plt.switch_backend("Agg")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    text_sources_lf = {}

    def source(relative):
        path = ROOT / relative
        key, data = path.relative_to(ROOT).as_posix(), path.read_bytes()
        sources[key] = hashlib.sha256(data).hexdigest()
        if path.suffix in (".csv", ".json", ".yaml", ".py"):
            text_sources_lf[key] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        return path

    counts = pd.read_csv(source(FINAL / "conteo_por_clase.csv")).set_index("label")
    assert counts.train.sum() == 12753 and counts.val.sum() == 3712
    rows, profiles, per_class_rows, concentration_rows = [], {}, [], []
    shared_labels = None
    for identifier, name, family, variant, run, lr in SPECS:
        base = family / "runs" / variant / run
        summary = json.loads(source(base / "summary.json").read_text(encoding="utf-8"))
        report = json.loads(source(base / "eval_report.json").read_text(encoding="utf-8"))
        history = pd.read_csv(source(base / "history.csv"))
        predictions = pd.read_csv(source(base / "evaluation/validation_predictions.csv"))
        matrix = pd.read_csv(source(base / "evaluation/confusion_matrix.csv"), index_col=0).to_numpy()
        source(family / (f"{variant}.yaml" if family == A4 else f"configs/{variant}.yaml"))
        assert len(predictions) == 3712 and np.array_equal(predictions.sample_index, np.arange(3712))
        assert np.array_equal(np.bincount(predictions.true_label, minlength=104), counts.val)
        if shared_labels is not None:
            assert np.array_equal(shared_labels, predictions.true_label)
        shared_labels = predictions.true_label.to_numpy()
        metrics = validate_saved_predictions(predictions, matrix, report, summary, history)
        best = history.iloc[summary["best_epoch"] - 1]
        class_table = metrics.pop("per_class").set_index("label").join(counts[["name", "train"]])
        predicted_counts = np.bincount(predictions.predicted_label, minlength=104)
        class_table["predicted_count"] = predicted_counts
        class_table["experiment_id"] = identifier
        per_class_rows.append(class_table.reset_index())
        top = np.argsort(-predicted_counts, kind="stable")[:5]
        concentration = float(predicted_counts[top].sum() / len(predictions))
        correlation = float(spearmanr(class_table.train, class_table.f1).statistic)
        majority_label = int(counts.train.idxmax())
        majority = classification_metrics(predictions.true_label,
                                          np.full(3712, majority_label))
        row = {"experiment_id": identifier, "model": name,
               "backend_provenance": "PyTorch/DirectML RX6700S (historical)" if family == A4
                                     else "TensorFlow/Metal (historical documentation)",
               "architecture": summary["architecture"], "parameters": summary["parameters"],
               "seed": summary["seed"], "train_images_protocol": 12753, "val_images": 3712,
               "epochs_run": summary["epochs_run"], "checkpoint_epoch": summary["best_epoch"],
               "initial_lr": lr, "optimizer": summary["optimizer"], **metrics,
               "val_loss_saved": report["val_loss"], "train_accuracy_online": float(best.accuracy),
               "train_loss_online": float(best.loss), "top5_prediction_share": concentration,
               "spearman_train_count_vs_class_f1": correlation,
               "train_eval_accuracy_saved": float(best.get("train_eval_accuracy", np.nan)),
               "train_eval_f1_saved": float(best.get("train_f1_macro", np.nan)),
               "train_eval_loss_saved": float(best.get("train_eval_loss", np.nan)),
               "majority_control_label": majority_label,
               "majority_control_accuracy": majority["accuracy"],
               "majority_control_f1": majority["f1_macro"],
               "history": (base / "history.csv").as_posix()}
        if "confidence" in predictions:
            row["median_predicted_confidence"] = float(predictions.confidence.median())
        rows.append(row)
        for rank, label in enumerate(np.argsort(-predicted_counts, kind="stable"), 1):
            concentration_rows.append({"experiment_id": identifier, "rank": rank, "label": int(label),
                                       "name": counts.loc[label, "name"], "predicted_count": int(predicted_counts[label]),
                                       "validation_support": int(counts.loc[label, "val"]),
                                       "train_support": int(counts.loc[label, "train"])})
        profiles[identifier] = (name, history, summary, class_table)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "comparacion_modelos.csv", index=False)
    pd.concat(per_class_rows, ignore_index=True).to_csv(OUTPUT / "metricas_por_clase.csv", index=False)
    pd.DataFrame(concentration_rows).to_csv(OUTPUT / "concentracion_predicciones.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"train": "#2463a7", "val": "#c06a20"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for index, identifier in enumerate(("A4_MLP_lr1e4", "A4_CNN_original")):
        name, history, summary, _ = profiles[identifier]
        for ax, field, ylabel in zip(axes[index], ("loss", "accuracy", "val_f1_macro"),
                                     ("Entropía cruzada", "Exactitud", "F1 macro")):
            if field != "val_f1_macro":
                ax.plot(history.epoch + 1, history[field], color=colors["train"],
                        label="Train online: aumentos + dropout")
                ax.plot(history.epoch + 1, history[f"val_{field}"], color=colors["val"],
                        ls="--", label="Validación sin aumentos")
            else:
                ax.plot(history.epoch + 1, history[field], color=colors["val"], ls="--", label="Validación")
            ax.axvline(summary["best_epoch"], color="#555555", ls=":", label="Checkpoint por F1")
            ax.set(title=name, xlabel="Época (base 1)", ylabel=ylabel)
            ax.set_xlim(1, 87)
            ax.set_xticks([1, 20, 40, 60, 87])
            if field in ("accuracy", "val_f1_macro"):
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 5)
            ax.grid(alpha=.2)
        axes[index, 0].legend(fontsize=8)
    fig.suptitle("MLP frente a CNN: históricos A4, seed 0, validación de 3,712 imágenes")
    save_figure(fig, "curvas_mlp_cnn")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharex=True)
    for ax, identifier in zip(axes, ("A4_MLP_lr1e4", "A4_CNN_original")):
        name, _, _, classes = profiles[identifier]
        selected = classes.sort_values("predicted_count", ascending=False).head(10).iloc[::-1]
        positions = np.arange(len(selected))
        ax.barh(positions - .18, selected.support, height=.34, color="#777777", label="Casos reales de val")
        ax.barh(positions + .18, selected.predicted_count, height=.34, color="#2463a7", label="Predicciones")
        ax.set_yticks(positions, selected["name"])
        ax.set(title=name, xlabel="Imágenes (validación n=3,712)")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Diez clases más predichas por modelo: frecuencia real y predicha")
    save_figure(fig, "predicciones_mlp_cnn")

    # Keep failed and interrupted historical trials separate; no merged epochs or claims of retraining.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, relative, color, style in [
        ("LR 1e-3, fallido (6 épocas)", A4 / "diagnosticos/mlp_lr1e3/runs/full_res112_s0_mlp_reference_directml/history.csv", "#777777", ":"),
        ("LR 1e-4, interrumpido (25 épocas)", A4 / "diagnosticos/mlp_lr1e4_interrumpido/runs/full_res112_s0_mlp_reference_directml/history.csv", "#c06a20", "--"),
        ("LR 1e-4, definitivo (87 épocas)", A4 / "runs/mlp_referencia/full_res112_s0_mlp_reference_directml/history.csv", "#2463a7", "-"),
    ]:
        history = pd.read_csv(source(relative))
        for ax, field in zip(axes, ("val_loss", "val_f1_macro")):
            ax.plot(history.epoch + 1, history[field], color=color, ls=style, label=name)
            ax.set(xlabel="Época (base 1)", ylabel="Loss val" if field == "val_loss" else "F1 macro val")
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("MLP A4: tres ensayos históricos separados, no etapas concatenadas")
    save_figure(fig, "diagnosticos_lr_mlp")

    weight_path = OUTPUT / "pesos_mlp.json"
    if weight_path.exists():
        weight_data = json.loads(weight_path.read_text(encoding="utf-8"))
        for record in weight_data["models"].values():
            actual = source(Path(record["source"]))
            assert sources[actual.relative_to(ROOT).as_posix()] == record["sha256"]
        source(weight_path.relative_to(ROOT))
    for relative in ("src/train_mlp_reference.py", "src/model.py", "src/data.py", "src/metrics.py",
                     "scripts/analyze_mlp_avance4.py", "scripts/inspect_mlp_weights.py",
                     "tests/test_avance4_evidence.py"):
        source(Path(relative))
    packages = {package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "matplotlib", "scikit-learn", "scipy")}
    result = {"purpose": "Stage 2: derived historical diagnostics, no new training/inference",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "python": sys.version.split()[0],
              "packages": packages,
              "local_execution_readiness": {
                  "configured_tfrecords": "data/raw/tpu-getting-started/tfrecords-jpeg-224x224",
                  "configured_tfrec_files": len(list((ROOT / "data/raw/tpu-getting-started/tfrecords-jpeg-224x224").rglob("*.tfrec"))),
                  "available_modules": {module: importlib.util.find_spec(module) is not None
                                        for module in ("tensorflow", "keras", "torch", "torch_directml")},
                  "external_dataset": "Not located; a path was requested from the user"},
              "sources_sha256": sources, "text_sources_lf_sha256": text_sources_lf,
              "experiments": table.fillna("not_available").to_dict("records"),
              "not_evaluated": ["real-image activations/gradients", "new train-inference metrics A4",
                                "new training", "checkpoint inference on TFRecords"],
              "comparison_limitations": "Same declared splits and seed; MLP/CNN differ in LR, patience and capacity. "
                                        "Final MLP is a separate TensorFlow/Metal historical experiment."}
    (OUTPUT / "diagnostico.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(table[["experiment_id", "accuracy", "f1_macro", "predicted_classes", "top5_prediction_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
