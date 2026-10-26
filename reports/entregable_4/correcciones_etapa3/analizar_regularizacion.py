"""Analyze existing final-report runs; never train or modify original evidence.

python reports/entregable_4/correcciones_etapa3/analizar_regularizacion.py
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_mlp_avance4 import validate_saved_predictions

OUT = Path(__file__).resolve().parent
FIELDS = ["train_eval_accuracy", "train_eval_loss", "train_f1_macro",
          "val_accuracy", "val_loss", "val_f1_macro", "learning_rate"]
LABELS = {"referencia": "F/con regularización", "sin_regularizacion": "F/sin regularización"}
COLORS = {"referencia": "#2463a7", "sin_regularizacion": "#c06a20"}


def snapshot(history, epoch):
    rows = history.loc[history.epoch + 1 == epoch]
    if len(rows) != 1:
        raise ValueError(f"Missing or duplicate exact epoch {epoch}; no interpolation permitted")
    row = rows.iloc[0]
    if not np.isfinite(row[FIELDS].to_numpy(dtype=float)).all():
        raise ValueError(f"Incomplete metrics at epoch {epoch}")
    return {"epoch": epoch, **{field: float(row[field]) for field in FIELDS},
            "gap_f1": float(row.train_f1_macro - row.val_f1_macro),
            "gap_accuracy": float(row.train_eval_accuracy - row.val_accuracy),
            "gap_loss": float(row.val_loss - row.train_eval_loss)}


def load_evidence(root=ROOT):
    runs, sources = {}, {}
    for variant in LABELS:
        base = root / "reports/final/experimentos"
        candidates = list((base / "runs" / variant).glob("*/summary.json"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one completed run for {variant}")
        directory = candidates[0].parent
        summary = json.loads(candidates[0].read_text(encoding="utf-8"))
        config_path = base / "configs" / f"{variant}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        history = pd.read_csv(directory / "history.csv")
        if history.epoch.duplicated().any():
            raise ValueError("Duplicate history epochs")
        assert config["train"]["train_eval_images"] == 3712
        assert summary["seed"] == 0 and summary["architecture"] == "baseline"
        assert summary["parameters"] == 480936 and config["data"]["img_size"] == 112
        assert config["data"]["batch_size"] == 64 and config["train"]["patience"] == 12
        pred = pd.read_csv(directory / "evaluation/validation_predictions.csv")
        matrix = pd.read_csv(directory / "evaluation/confusion_matrix.csv", index_col=0).to_numpy()
        report = json.loads((directory / "eval_report.json").read_text(encoding="utf-8"))
        validate_saved_predictions(pred, matrix, report, summary, history)
        assert len(pred) == 3712
        for path in [config_path, *[directory / name for name in ("summary.json", "history.csv", "eval_report.json")],
                     directory / "evaluation/validation_predictions.csv", directory / "evaluation/confusion_matrix.csv"]:
            data = path.read_bytes()
            sources[path.relative_to(root).as_posix()] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "text_lf_sha256": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}
        runs[variant] = {"history": history, "summary": summary, "config": config}
    common = sorted(set(runs["referencia"]["history"].epoch + 1)
                    & set(runs["sin_regularizacion"]["history"].epoch + 1))
    if 34 not in common:
        raise ValueError("Epoch 34 is not recorded in both histories")
    rows = []
    for context in ("selected_checkpoint", "common_epoch_34"):
        for variant, run in runs.items():
            epoch = run["summary"]["best_epoch"] if context == "selected_checkpoint" else 34
            rows.append({"comparison": context, "experiment_id": "F/" + variant,
                         "run": run["summary"]["run"], "epochs_run": run["summary"]["epochs_run"],
                         **snapshot(run["history"], epoch)})
    return runs, pd.DataFrame(rows), common, sources


def main():
    plt.switch_backend("Agg")
    runs, table, common, sources = load_evidence()
    table.to_csv(OUT / "comparacion_regularizacion.csv", index=False)
    histories = []
    for variant, run in runs.items():
        h = run["history"].copy()
        h["epoch"] += 1
        h["experiment_id"] = "F/" + variant
        h["gap_f1"] = h.train_f1_macro - h.val_f1_macro
        histories.append(h)
    pd.concat(histories).to_csv(OUT / "curvas_regularizacion.csv", index=False)

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for variant, run in runs.items():
        h, color, name = run["history"], COLORS[variant], LABELS[variant]
        x = h.epoch + 1
        axes[0, 0].plot(x, h.val_f1_macro, color=color, label=name)
        for ax, tr, val in [(axes[0, 1], "train_eval_accuracy", "val_accuracy"),
                            (axes[1, 0], "train_eval_loss", "val_loss")]:
            ax.plot(x, h[tr], color=color, label=name + ": train limpio")
            ax.plot(x, h[val], color=color, ls="--", label=name + ": val")
        axes[1, 1].plot(x, h.train_f1_macro - h.val_f1_macro, color=color, label=name)
        axes[0, 0].scatter([run["summary"]["best_epoch"]],
                           [snapshot(h, run["summary"]["best_epoch"])["val_f1_macro"]], color=color, marker="o")
    for ax, title, ylabel in zip(axes.flat,
            ["F1 macro de validación", "Exactitud: train limpio y validación",
             "Pérdida: train limpio y validación", "Brecha F1: train limpio menos validación"],
            ["F1 macro", "Exactitud", "Entropía cruzada", "Diferencia F1"]):
        ax.axvline(34, color="#555555", ls=":", lw=1)
        ax.set(title=title, xlabel="Época (base 1)", ylabel=ylabel, xlim=(1, 103))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Familia F histórica: TensorFlow/Metal; seed 0; línea vertical = época común 34")
    save(fig, "curvas_comparadas")

    for context, filename, title in [
        ("selected_checkpoint", "mejores_checkpoints", "Checkpoints seleccionados: F/con 91; F/sin 34"),
        ("common_epoch_34", "epoca_comun_34", "Comparación complementaria: época 34 de ambas corridas")]:
        t = table[table.comparison == context].reset_index(drop=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        for ax, tr, val, label in zip(axes,
                ["train_f1_macro", "train_eval_accuracy", "train_eval_loss"],
                ["val_f1_macro", "val_accuracy", "val_loss"], ["F1 macro", "Exactitud", "Entropía cruzada"]):
            for i, variant in enumerate(LABELS):
                ax.bar(i - .17, t.loc[i, tr], width=.32, color=COLORS[variant], label="Train limpio" if i == 0 else None)
                ax.bar(i + .17, t.loc[i, val], width=.32, color=COLORS[variant], alpha=.45,
                       hatch="//", label="Validación" if i == 0 else None)
                ax.text(i, max(t.loc[i, tr], t.loc[i, val]) + .02,
                        f"val={t.loc[i, val]:.4f}", ha="center", fontsize=8)
            ax.set_xticks([0, 1], ["F/con", "F/sin"])
            ax.set(ylabel=label, ylim=(0, 1.05 if tr != "train_eval_loss" else 2.45))
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(title + "\nMisma arquitectura y semilla; intervención conjunta, una corrida por variante")
        save(fig, filename)

    best = table[table.comparison == "selected_checkpoint"].reset_index(drop=True)
    common_table = table[table.comparison == "common_epoch_34"].reset_index(drop=True)
    delta = {field: float(best.loc[0, field] - best.loc[1, field])
             for field in ["val_f1_macro", "val_accuracy", "gap_f1"]}
    manifest = {"family": "F: complementary historical final-report experiments",
                "backend_provenance": "TensorFlow/Metal according to repository documentation, not newly detected",
                "common_epochs": common, "common_epoch": 34,
                "train_protocol": "First 3712 fixed train images, no augmentation, training=False; metrics from histories",
                "sources": sources, "best_delta_con_minus_sin": delta,
                "snapshots": table.to_dict("records"),
                "limitations": ["One seed; no statistical significance", "Joint dropout/augmentation/optimizer/WD intervention",
                                "Both retain early stopping and adaptive LR", "Train metrics not independently re-inferred",
                                "Common epoch does not equal common optimization trajectory"]}
    (OUT / "fuentes.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    md = f'''# Etapa 3: regularización con evidencia histórica

Se reutilizan F/referencia y F/sin_regularizacion del reporte final, sin modificar sus archivos ni entrenar. Ambas usan CNN baseline de 480,936 parámetros, seed 0, imágenes 112×112, batch 64, train 12,753 y val 3,712. El backend TensorFlow/Metal procede de documentación histórica. No son las corridas A4/DirectML.

## Método

Se verifican predicciones/matrices y summary/history. Train limpio: primeras 3,712 imágenes fijas de train sin aumentos y en inferencia, según código/configuración; estos valores solo se verifican documentalmente, no mediante nueva inferencia. Las pérdidas también son históricas. La brecha F1 se resta con precisión completa dentro de la misma fila, sin usar cifras redondeadas del summary. Los historiales comparten épocas {common[0]}–{common[-1]}; la época 34 contiene todas las métricas y se extrae exactamente, sin interpolación.

## Mejores checkpoints

| Variante F | Época / ejecutadas | F1 train limpio | F1 val | Exactitud val | Loss val | Brecha F1 |
|---|---|---:|---:|---:|---:|---:|
'''
    for _, r in best.iterrows():
        md += f"| {r.experiment_id} | {r.epoch} / {r.epochs_run} | {r.train_f1_macro:.4f} | {r.val_f1_macro:.4f} | {100*r.val_accuracy:.2f}% | {r.val_loss:.4f} | {r.gap_f1:.4f} |\n"
    md += f'''
La intervención conjunta mejora F1 val en {delta['val_f1_macro']:.6f} y exactitud en {100*delta['val_accuracy']:.4f} puntos porcentuales; reduce la brecha F1 en {-delta['gap_f1']:.6f}. Sin regularización explícita la red ajusta más train, pero su mayor brecha y loss val son compatibles con mayor sobreajuste. No es una atribución causal a un regularizador individual.

## Comparación complementaria en época 34

| Variante F | F1 train limpio | F1 val | Exactitud train | Exactitud val | Loss train | Loss val | Brecha F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for _, r in common_table.iterrows():
        md += f"| {r.experiment_id} | {r.train_f1_macro:.4f} | {r.val_f1_macro:.4f} | {100*r.train_eval_accuracy:.2f}% | {100*r.val_accuracy:.2f}% | {r.train_eval_loss:.4f} | {r.val_loss:.4f} | {r.gap_f1:.4f} |\n"
    md += '''
En época 34 la variante sin regularización tiene mayor F1 y exactitud de validación, mientras la regularizada tiene menor pérdida y menor brecha. Esta diferencia temporal impide afirmar que la regularización domina en todas las épocas. La versión regularizada sigue mejorando después y su checkpoint seleccionado supera al mejor de la otra variante.

## Interpretación y límites

La intervención cambia simultáneamente dropout 0.5→0, aumentos activados→desactivados y AdamW/WD 0.01→Adam/WD 0. Ambas conservan LR inicial 1e-3, scheduler de val_loss ×0.5/paciencia 2/mínimo 1e-5, parada temprana de 12 evaluaciones y tope de 150. Las trayectorias reales de LR difieren: comparar época 34 no controla el número efectivo de pasos a una misma LR ni elimina otros efectos de la política. Una brecha pequeña también puede acompañar infraajuste; debe leerse junto con F1/loss de ambos splits. No hay significancia estadística con una sola semilla ni test independiente.

## Reproducción y fuentes

Desde la raíz: `python reports/entregable_4/correcciones_etapa3/analizar_regularizacion.py`. Requiere numpy, pandas, matplotlib, scipy, scikit-learn y PyYAML, sin frameworks de entrenamiento. `fuentes.json` registra todas las rutas y SHA-256 crudo/LF. `comparacion_regularizacion.csv` contiene las cuatro filas y todas las métricas; `curvas_regularizacion.csv` contiene cada época. PNG y PDF identifican familia y protocolo. El script rechaza épocas ausentes/duplicadas o valores no finitos.
'''
    (OUT / "ANALISIS_REGULARIZACION.md").write_text(md, encoding="utf-8")
    print(table[["comparison", "experiment_id", "epoch", "val_f1_macro", "val_accuracy", "gap_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
