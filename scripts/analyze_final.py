"""Tablas, figuras y resultados.csv del reporte final a partir de las corridas guardadas.

No entrena nada: lee history.csv, summary.json y eval_report.json de cada experimento de
scripts/run_final.py. Escribe:
    reports/final/resultados.csv       una fila por experimento
    reports/final/resumen.json         cifras citadas en el texto
    reports/final/tablas/*.tex         tablas que \\input-ean las secciones
    reports/final/figures/*.{pdf,png}  figuras

Uso:
    python -m scripts.analyze_final
"""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_final import EXPERIMENTS
from src.config import ROOT

# Paleta categórica de referencia (modo claro, impresión): orden fijo, nunca cíclico.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TRAIN, VAL = SERIES[0], SERIES[1]
MUTED, INK, INK_2, GRID = "#9b9a94", "#0b0b0b", "#52514e", "#e6e5e0"

NAMES = {
    "referencia": "CNN de referencia", "mlp": "MLP", "cnn_ampliada": "CNN ampliada (6 conv.)",
    "sin_regularizacion": "Sin regularización", "class_weight": "Pesos por clase",
    "lr_3e-4": "LR $3\\times10^{-4}$", "dropout_0.3": "Dropout 0.3", "res_160": "160$\\times$160",
    "sin_dropout": "Sin dropout", "sin_aumento": "Sin aumento", "sin_weight_decay": "Sin weight decay",
    "l2": "L2 (Adam)", "label_smoothing": "Label smoothing 0.1", "efficientnetb0": "EfficientNetB0",
}
PLAIN = {key: value.replace("$\\times$", "×").replace("$3\\times10^{-4}$", "3e-4") for key, value in NAMES.items()}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "axes.titlesize": 10, "axes.titlecolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
})


def load_runs(experiments_dir):
    runs = {}
    for label, group, description, _ in EXPERIMENTS:
        directory = Path(experiments_dir) / "runs" / label
        summaries = list(directory.glob("*/summary.json"))
        if not summaries:
            continue
        run_dir = summaries[0].parent
        report = run_dir / "eval_report.json"
        runs[label] = {
            "group": group, "description": description, "dir": run_dir,
            "summary": json.loads(summaries[0].read_text()),
            "history": pd.read_csv(run_dir / "history.csv"),
            "eval": json.loads(report.read_text()) if report.exists() else None,
        }
    return runs


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def at_best(run):
    """Fila del history en la época del checkpoint (best_epoch es 1-indexada)."""
    history = run["history"]
    return history.iloc[run["summary"]["best_epoch"] - 1]


def results_table(runs, counts):
    small = counts["train"].nsmallest(10).index
    large = counts["train"].nlargest(10).index
    rows = []
    for label, run in runs.items():
        s, best = run["summary"], at_best(run)
        per_class = pd.DataFrame(run["eval"]["per_class"]).set_index("label") if run["eval"] else None
        rows.append({
            "experimento": label, "grupo": run["group"], "descripcion": run["description"],
            "arquitectura": s["architecture"], "parametros": s["parameters"], "img_size": s["img_size"],
            "lr": s.get("lr"), "dropout": s.get("dropout"), "aumento": s.get("augment"),
            "optimizador": s["optimizer"], "weight_decay": s["weight_decay"], "l2": s.get("l2"),
            "label_smoothing": s.get("label_smoothing"), "class_weight": s.get("class_weight"),
            "epocas": s["epochs_run"], "mejor_epoca": s["best_epoch"],
            "val_exactitud": s["val_accuracy"], "val_f1_macro": s["val_f1_macro"],
            "val_perdida": round(float(best["val_loss"]), 4),
            "train_exactitud": round(float(best["train_eval_accuracy"]), 4),
            "train_f1_macro": round(float(best["train_f1_macro"]), 4),
            "train_perdida": round(float(best["train_eval_loss"]), 4),
            "brecha_exactitud": round(float(best["train_eval_accuracy"] - s["val_accuracy"]), 4),
            "brecha_f1": round(float(best["train_f1_macro"] - s["val_f1_macro"]), 4),
            "f1_10_clases_menores": round(float(per_class.loc[small, "f1"].mean()), 4) if per_class is not None else None,
            "f1_10_clases_mayores": round(float(per_class.loc[large, "f1"].mean()), 4) if per_class is not None else None,
            "clases_f1_cero": int((per_class["f1"] == 0).sum()) if per_class is not None else None,
            "segundos_entrenamiento": s["train_seconds"],
            "interrumpida": bool(s.get("interrumpida")),
            "history_csv": relative(run["dir"] / "history.csv"),
        })
    return pd.DataFrame(rows)


# --- Tablas LaTeX ----------------------------------------------------------------------------

def pct(x):
    return f"{100 * x:.2f}"


def write_table(path, caption, label, columns, header, rows, note=None, wide=True):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small", f"\\begin{{tabular}}{{{columns}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += [row + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["\\par\\smallskip", f"\\parbox{{0.95\\textwidth}}{{\\footnotesize {note}}}"]
    lines.append(f"\\end{{{env}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bold_best(values, fmt, higher=True):
    best = max(values) if higher else min(values)
    return [f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v) for v in values]


def tables(table, counts, runs, out):
    out.mkdir(parents=True, exist_ok=True)
    t = table.set_index("experimento")

    def rows_for(labels, cells):
        labels = [label for label in labels if label in t.index]
        f1s = bold_best([t.loc[k, "val_f1_macro"] for k in labels], lambda v: f"{v:.4f}")
        mark = lambda k: NAMES[k] + ("$^\\dagger$" if t.loc[k, "interrumpida"] else "")
        return [" & ".join([mark(k), *cells(t.loc[k]), f1]) for k, f1 in zip(labels, f1s)]

    models = ["mlp", "referencia", "cnn_ampliada"]
    write_table(
        out / "modelos.tex", "Arquitecturas comparadas con los mismos datos de validación (3,712 imágenes).",
        "tab:modelos", "lrrrrrrr",
        "Modelo & Parámetros & Época & Exact. (\\%) & F1 entr. & Brecha F1 & Tiempo (min) & F1 macro",
        rows_for(models, lambda r: [f"{r.parametros:,}", f"{r.mejor_epoca}/{r.epocas}", pct(r.val_exactitud),
                                    f"{r.train_f1_macro:.4f}", f"{r.brecha_f1:+.4f}",
                                    f"{r.segundos_entrenamiento / 60:.1f}"]),
        "Época: checkpoint elegido por F1 macro / épocas ejecutadas. F1 entr.: F1 macro sobre 3,712 imágenes "
        "fijas de entrenamiento sin aumento ni dropout, en la misma época. Brecha F1 = F1 entr. $-$ F1 macro "
        "de validación. Todas a 112$\\times$112, entrenadas desde cero con semilla 0. $^\\dagger$La corrida se interrumpió en la época 50 por un bloqueo de la GPU; "
        "se reporta su mejor checkpoint.")

    reg = ["referencia", "sin_dropout", "sin_aumento", "sin_weight_decay", "sin_regularizacion", "l2",
           "label_smoothing"]
    yes = lambda b: "sí" if b else "no"
    write_table(
        out / "regularizacion.tex", "La misma CNN con y sin regularización (semilla 0, mismas épocas máximas).",
        "tab:regularizacion", "lcccrrrrr",
        "Variante & Dropout & Aumento & Pesos & Época & Exact. (\\%) & F1 entr. & Brecha F1 & F1 macro",
        rows_for(reg, lambda r: [f"{r.dropout:g}", yes(r.aumento),
                                 ("WD 0.01" if r.optimizador == "adamw" else f"L2 {r.l2:g}" if r.l2 else "--")
                                 + (", LS 0.1" if r.label_smoothing else ""),
                                 f"{r.mejor_epoca}/{r.epocas}", pct(r.val_exactitud), f"{r.train_f1_macro:.4f}",
                                 f"{r.brecha_f1:+.4f}"]),
        "Pesos: WD = weight decay desacoplado de AdamW; -- = Adam sin decaimiento. Ambas comparten "
        "arquitectura, tasa de aprendizaje inicial, reducción de LR en meseta, paciencia de 12 evaluaciones "
        "y tope de 150 épocas.")

    hyper = ["referencia", "lr_3e-4", "dropout_0.3", "res_160"]
    write_table(
        out / "hiperparametros.tex", "Efecto de un hiperparámetro a la vez, con el conjunto completo.",
        "tab:hiperparametros", "lcccrrrrr",
        "Configuración & LR & Dropout & Resolución & Época & Exact. (\\%) & Brecha F1 & Tiempo (min) & F1 macro",
        rows_for(hyper, lambda r: [f"{r.lr:g}", f"{r.dropout:g}", f"{r.img_size}", f"{r.mejor_epoca}/{r.epocas}",
                                   pct(r.val_exactitud), f"{r.brecha_f1:+.4f}",
                                   f"{r.segundos_entrenamiento / 60:.1f}"]),
        "Cada fila cambia un solo valor respecto a la referencia (LR $10^{-3}$, dropout 0.5, 112$\\times$112).")

    imb = ["referencia", "class_weight"]
    write_table(
        out / "desbalance.tex", "Pérdida ponderada por clase frente a la pérdida sin ponderar.",
        "tab:desbalance", "lrrrrr",
        "Entrenamiento & Exact. (\\%) & F1 10 menores & F1 10 mayores & Clases F1 = 0 & F1 macro",
        rows_for(imb, lambda r: [pct(r.val_exactitud), f"{r.f1_10_clases_menores:.4f}",
                                 f"{r.f1_10_clases_mayores:.4f}", f"{r.clases_f1_cero}"]),
        "F1 10 menores/mayores: F1 promedio de las 10 clases con menos (18--19) y más (460--782) imágenes de "
        "entrenamiento. Pesos $w_c = N/(K\\,n_c)$: van de "
        f"{counts['train'].sum() / (104 * counts['train'].max()):.2f} (iris) a "
        f"{counts['train'].sum() / (104 * counts['train'].min()):.2f} (las clases de 18 imágenes).",
        wide=False)

    avance4 = ROOT / "reports/entregable_4/avance4_2026-10-21/resultados.csv"
    if avance4.exists():
        old = pd.read_csv(avance4, encoding="utf-8-sig").set_index("experiment")
        pairs = [("referencia", "cnn_regularizada", "AdamW + LR adaptable"),
                 ("cnn_ampliada", "cnn_dos_capas_extra", "Adam, LR fijo"),
                 ("mlp", "mlp_referencia", "Adam, LR fijo")]
        rows = []
        for new, previous, policy in pairs:
            if new not in t.index:
                continue
            o, r = old.loc[previous], t.loc[new]
            rows.append(f"{NAMES[new]} & {policy} & {o.val_f1_macro:.4f} & {o.accuracy_gap_at_checkpoint:+.4f} & "
                        f"{r.val_f1_macro:.4f} & {r.brecha_exactitud:+.4f} & {r.brecha_f1:+.4f}")
        write_table(
            out / "avance4.tex", "Avance 4 frente al reporte final: la brecha medida con aumento subestima el "
            "sobreajuste.", "tab:avance4", "llrrrrr",
            "Modelo & Política en Avance 4 & F1 A4 & Brecha exact. A4 & F1 final & Brecha exact. & Brecha F1",
            rows,
            "A4: PyTorch/DirectML; su exactitud de entrenamiento se midió durante el ciclo, con aumento y dropout "
            "activos. Final: TensorFlow/Metal, todas con AdamW + LR adaptable; brechas medidas sobre 3,712 "
            "imágenes de entrenamiento sin aumento y en modo inferencia, en la época del checkpoint.")

    if "referencia" in runs:
        run = runs["referencia"]
        best, last = at_best(run), run["history"].iloc[-1]
        metrics = [("Exactitud de entrenamiento (\\%)", "train_eval_accuracy", pct),
                   ("F1 macro de entrenamiento", "train_f1_macro", lambda v: f"{v:.4f}"),
                   ("Pérdida de entrenamiento", "train_eval_loss", lambda v: f"{v:.3f}"),
                   ("Exactitud de validación (\\%)", "val_accuracy", pct),
                   ("F1 macro de validación", "val_f1_macro", lambda v: f"{v:.4f}"),
                   ("Pérdida de validación", "val_loss", lambda v: f"{v:.3f}")]
        write_table(
            out / "ultima_epoca.tex", "CNN de referencia en la época elegida y al terminar el entrenamiento.",
            "tab:ultima-epoca", "lrr",
            f"Métrica & Época elegida ({int(best.epoch) + 1}) & Última época ({int(last.epoch) + 1})",
            [f"{name} & {fmt(best[key])} & {fmt(last[key])}" for name, key, fmt in metrics],
            "Métricas de entrenamiento sobre 3,712 imágenes fijas sin aumento, en modo inferencia.", wide=False)

    distill = ROOT / "reports/resultados.csv"
    if distill.exists():
        dm = pd.read_csv(distill)
        dm = dm[dm.data.isin(["random", "distilled"])]
        rows = []
        for (ipc, data), g in dm.groupby(["ipc", "data"], sort=True):
            name = "Sintéticas (DM)" if data == "distilled" else "Reales aleatorias"
            extra = f"{g.distill_seconds.iloc[0] / 60:.0f}" if data == "distilled" else "--"
            rows.append(f"{int(ipc)} & {int(ipc) * 104:,} & {name} & {pct(g.val_accuracy.mean())} & "
                        f"{g.val_f1_macro.mean():.4f} $\\pm$ {g.val_f1_macro.std():.4f} & {extra}")
        write_table(
            out / "dm.tex", "Selección aleatoria frente a destilación con Distribution Matching (DM).",
            "tab:dm", "rrlrrr",
            "IPC & Imágenes & Conjunto & Exact. (\\%) & F1 macro & Destilación (min)", rows,
            "Promedio $\\pm$ desviación estándar de tres semillas de entrenamiento por condición. "
            "Misma CNN, Adam $10^{-3}$, $112\\times112$. Datos: \\path{reports/resultados.csv}.", wide=False)

    if "referencia" in runs and runs["referencia"]["eval"]:
        per_class = pd.DataFrame(runs["referencia"]["eval"]["per_class"]).set_index("label")
        per_class["train"] = counts["train"]
        ordered = per_class.sort_values(["f1", "train"], ascending=[False, False])
        best, worst = ordered.head(10), ordered.tail(10).iloc[::-1]
        rows = []
        for (_, b), (_, w) in zip(best.iterrows(), worst.iterrows()):
            rows.append(f"{b['name']} & {b.train} & {b.f1:.3f} & {w['name']} & {w.train} & {w.f1:.3f}")
        write_table(
            out / "clases.tex", "F1 por clase de la CNN de referencia: las 10 mejores y las 10 peores.",
            "tab:clases", "lrrlrr",
            "Mejores & $n_{\\text{entr}}$ & F1 & Peores & $n_{\\text{entr}}$ & F1", rows,
            "$n_{\\text{entr}}$: imágenes de esa clase en entrenamiento. Desempates por tamaño de clase.")


# --- Figuras ---------------------------------------------------------------------------------

def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def class_distribution(counts, out):
    train = counts["train"].sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(np.arange(len(train)), train, width=0.8, color=TRAIN, linewidth=0)
    median = counts["train"].median()
    ax.axhline(median, color=INK_2, linewidth=1, linestyle="--")
    ax.text(len(train) * 0.45, median + 15, f"mediana = {median:.0f}", ha="center", color=INK_2, fontsize=8)
    ax.set_xlim(-1, len(train))
    ax.set_xticks([])
    ax.set_xlabel("Clases ordenadas por número de imágenes (104)")
    ax.set_ylabel("Imágenes de entrenamiento")
    ax.grid(axis="x", visible=False)
    ordered = counts.sort_values("train", ascending=False)
    for position, (_, row) in enumerate(ordered.head(3).iterrows()):
        ax.annotate(f"{row['name']} ({row.train})", (position, row.train), xytext=(10, -4 - 10 * position),
                    textcoords="offset points", fontsize=7, color=INK_2)
    ax.annotate(f"{ordered.iloc[-1]['name']} y otras: {ordered.train.min()}",
                (len(train) - 1, ordered.train.min()), xytext=(-150, 40), textcoords="offset points",
                fontsize=7, color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    save(fig, out, "distribucion_clases")


def curves(runs, labels, out, name):
    labels = [label for label in labels if label in runs]
    fig, axes = plt.subplots(len(labels), 3, figsize=(7.2, 1.9 * len(labels) + 0.4), sharex="row",
                             squeeze=False)
    panels = [("Pérdida (entropía cruzada)", "train_eval_loss", "val_loss", "loss"),
              ("Exactitud", "train_eval_accuracy", "val_accuracy", "accuracy"),
              ("F1 macro", "train_f1_macro", "val_f1_macro", None)]
    for row, label in enumerate(labels):
        history, best = runs[label]["history"], runs[label]["summary"]["best_epoch"]
        epochs = history["epoch"] + 1
        for col, (title, train_key, val_key, fit_key) in enumerate(panels):
            ax = axes[row, col]
            if fit_key:
                ax.plot(epochs, history[fit_key], color=MUTED, linewidth=1, linestyle=":",
                        label="Entrenamiento (con aumento y dropout)")
            ax.plot(epochs, history[train_key], color=TRAIN, label="Entrenamiento (sin aumento)")
            ax.plot(epochs, history[val_key], color=VAL, label="Validación")
            ax.axvline(best, color=INK_2, linewidth=0.8, linestyle="--")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(PLAIN[label], color=INK, fontsize=9)
            if row == len(labels) - 1:
                ax.set_xlabel("Época")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, name)


def regularization_curves(runs, out):
    labels = [k for k in ("referencia", "sin_dropout", "sin_aumento", "sin_regularizacion") if k in runs]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for color, label in zip(SERIES, labels):
        history = runs[label]["history"]
        epochs = history["epoch"] + 1
        left.plot(epochs, history["val_f1_macro"], color=color, label=PLAIN[label])
        right.plot(epochs, history["train_f1_macro"] - history["val_f1_macro"], color=color, label=PLAIN[label])
    left.set(title="F1 macro de validación", xlabel="Época")
    right.set(title="Brecha F1 (entrenamiento − validación)", xlabel="Época")
    right.axhline(0, color=INK_2, linewidth=0.8)
    left.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "curvas_regularizacion")


def per_class_figure(runs, counts, out):
    if "referencia" not in runs or not runs["referencia"]["eval"]:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for color, label in zip((TRAIN, VAL), ("referencia", "class_weight")):
        if label not in runs or not runs[label]["eval"]:
            continue
        per_class = pd.DataFrame(runs[label]["eval"]["per_class"]).set_index("label")
        ax.scatter(counts["train"], per_class["f1"], s=22, color=color, edgecolor="white", linewidth=0.8,
                   label=PLAIN[label], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Imágenes de entrenamiento de la clase (escala log)")
    ax.set_ylabel("F1 de la clase (validación)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out, "f1_vs_tamano_clase")


def overview(table, out):
    t = table.sort_values("val_f1_macro")
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(t) + 0.8))
    reference = table.set_index("experimento").loc["referencia", "val_f1_macro"] \
        if "referencia" in set(table.experimento) else None
    colors = [VAL if k == "referencia" else TRAIN for k in t.experimento]
    ax.barh(np.arange(len(t)), t.val_f1_macro, height=0.7, color=colors, linewidth=0)
    for y, value in enumerate(t.val_f1_macro):
        ax.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(np.arange(len(t)), [PLAIN[k] for k in t.experimento])
    if reference is not None:
        ax.axvline(reference, color=INK_2, linewidth=0.8, linestyle="--")
    ax.set_xlabel("F1 macro de validación")
    ax.set_xlim(0, min(1, t.val_f1_macro.max() + 0.08))
    ax.grid(axis="y", visible=False)
    save(fig, out, "comparacion_experimentos")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, default=ROOT / "reports/final/experimentos")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/final")
    args = parser.parse_args()
    runs = load_runs(args.experiments)
    counts = pd.read_csv(args.experiments / "conteo_por_clase.csv").set_index("label")
    table = results_table(runs, counts)
    table.to_csv(args.report / "resultados.csv", index=False)

    figures = args.report / "figures"
    class_distribution(counts, figures)
    curves(runs, ["mlp", "referencia", "cnn_ampliada"], figures, "curvas_modelos")
    regularization_curves(runs, figures)
    per_class_figure(runs, counts, figures)
    overview(table, figures)
    tables(table, counts, runs, args.report / "tablas")

    resumen = {
        "clases": {"min": int(counts.train.min()), "max": int(counts.train.max()),
                   "mediana": float(counts.train.median()), "media": round(float(counts.train.mean()), 1),
                   "val_min": int(counts.val.min()), "val_max": int(counts.val.max())},
        "experimentos": table.set_index("experimento").drop(columns=["history_csv"]).to_dict("index"),
    }
    if "referencia" in runs and runs["referencia"]["eval"]:
        resumen["pares_confundidos_referencia"] = runs["referencia"]["eval"]["confused_pairs"][:8]
        # src.evaluate deja la matriz completa en models/final/figures/<experimento>/<modelo>/.
        model = Path(runs["referencia"]["summary"]["model_path"])
        source = model.parent.parent / "figures" / "referencia" / model.stem
        for extension in ("pdf", "png"):
            if (source / f"confusion_matrix_normalized.{extension}").exists():
                shutil.copy(source / f"confusion_matrix_normalized.{extension}",
                            figures / f"confusion_referencia.{extension}")
    (args.report / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table[["experimento", "val_f1_macro", "val_exactitud", "brecha_f1", "mejor_epoca", "epocas"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
