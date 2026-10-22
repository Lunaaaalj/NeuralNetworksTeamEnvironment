"""Write the report sections and an executed notebook from completed experiments."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import nbformat
import pandas as pd

from src.config import ROOT

SHORT = {"mlp_referencia": "MLP de referencia", "cnn_original": "CNN original",
         "cnn_dos_capas_extra": "CNN + 2 convoluciones", "cnn_regularizada": "CNN + AdamW + LR"}


def tex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def write_sections(output, table):
    by_key = table.set_index("experiment")
    baseline, deep, regularized, mlp = [by_key.loc[key] for key in
        ("cnn_original", "cnn_dos_capas_extra", "cnn_regularizada", "mlp_referencia")]
    best = table.sort_values("val_f1_macro", ascending=False).iloc[0]
    figure_root = output.relative_to(ROOT / "reports").as_posix() + "/figures"
    histories, evaluations = {}, {}
    for key in table.experiment:
        directory = next((output / "runs" / key).glob("*/summary.json")).parent
        histories[key] = pd.read_csv(directory / "history.csv")
        evaluations[key] = json.loads((directory / "eval_report.json").read_text(encoding="utf-8"))
    diagnosis = []
    for key, history in histories.items():
        row = by_key.loc[key]
        minimum = history.loc[history.val_loss.idxmin()]
        last = history.iloc[-1]
        diagnosis.append(
            f"\\item \\textbf{{{SHORT[key]}}}: la menor pérdida de validación fue "
            f"{minimum.val_loss:.3f} en la época {int(minimum.epoch)+1}. Al terminar, "
            f"la pérdida de entrenamiento fue {last.loss:.3f} y la de validación "
            f"{last.val_loss:.3f}; la exactitud de entrenamiento fue {last.accuracy*100:.2f}\\% "
            f"y la de validación {last.val_accuracy*100:.2f}\\%. "
            + ("La reducción de la pérdida de entrenamiento junto con una pérdida de validación "
               "mayor que su mínimo es evidencia de sobreajuste en la fase final."
               if last.val_loss > minimum.val_loss * 1.02 and last.loss < last.val_loss
               else "Estas curvas deben interpretarse junto con F1 macro; no se concluye "
                    "sobreajuste solo por una diferencia entre exactitudes."))
    regularization = r"""\section{Regularización}

\subsection{Diagnóstico con curvas de entrenamiento y validación}

La Figura~\ref{fig:curvas-av4} compara exactitud y entropía cruzada para cada
modelo. Se muestran todas las épocas ejecutadas y se marca la época seleccionada
por F1 macro. No se confunde la última época con el checkpoint final evaluado.
En una fase de sobreajuste, el modelo sigue reduciendo su error de entrenamiento
sin obtener la misma mejora en validación; en particular, una pérdida de validación
que aumenta revela errores más confiados aunque la exactitud cambie poco.

\begin{itemize}
""" + "\n".join(diagnosis) + r"""
\end{itemize}

\begin{figure*}[p]
\centering
\includegraphics[width=\textwidth,keepaspectratio]{""" + figure_root + r"""/comparacion_original_ampliada.pdf}
\caption{CNN original y CNN ampliada: curvas reales por época. El entrenamiento usa aumentos y dropout; la
validación no. Las líneas verticales marcan el mejor checkpoint por F1 macro.}
\label{fig:curvas-av4}
\end{figure*}

\begin{figure*}[p]
\centering
\includegraphics[width=\textwidth,keepaspectratio]{""" + figure_root + r"""/comparacion_referencia_regularizada.pdf}
\caption{Curvas de la referencia totalmente conectada y la CNN con la intervención
combinada de regularización. Se mantienen separados sus checkpoints y métricas.}
\label{fig:curvas-ref-reg-av4}
\end{figure*}

\subsection{Técnicas aplicadas y efecto medido}

Todos los modelos usan dropout de 0.5 antes de la salida, aumentos de datos
solo en entrenamiento y parada temprana con restauración del mejor checkpoint.
No se entrenó una ablación sin dropout ni sin aumentos; por eso no se atribuye
por separado una mejora numérica a estas dos técnicas.

La intervención adicional conserva la CNN de cuatro convoluciones, pero utiliza
AdamW con decaimiento de pesos de 0.01 sobre matrices de pesos, excluyendo sesgos.
Se trata de decaimiento desacoplado y no de añadir un término L2 a la pérdida
mostrada. Además, reduce la tasa de aprendizaje a la mitad cuando la pérdida de
validación no mejora durante dos evaluaciones, hasta un mínimo de $10^{-5}$.
La paciencia de parada temprana pasa de 5 a 12 evaluaciones para permitir actuar
a esa reducción. Esta es una intervención combinada: no permite aislar cuánto
aportó el decaimiento, el ajuste de tasa o la paciencia.

"""
    accuracy_change = 100 * (regularized.val_accuracy - baseline.val_accuracy)
    f1_change = regularized.val_f1_macro - baseline.val_f1_macro
    regularization += (
        f"Frente a la CNN original, la variante regularizada cambia la exactitud de "
        f"{baseline.val_accuracy*100:.2f}\\% a {regularized.val_accuracy*100:.2f}\\% "
        f"({accuracy_change:+.2f} puntos porcentuales) y F1 macro de "
        f"{baseline.val_f1_macro:.4f} a {regularized.val_f1_macro:.4f} "
        f"({f1_change:+.4f}). Ambos valores se obtienen de los respectivos checkpoints "
        "seleccionados por F1 macro sobre la misma validación.\n\n"
        + ("La diferencia de F1 macro es pequeña. Con una sola semilla no se considera "
           "evidencia de una mejora robusta o estadísticamente significativa.\n\n"
           if abs(f1_change) < 0.01 else "") +
        "Una mejora de exactitud acompañada de una caída de F1 macro no se interpreta "
        "como mejora general: el criterio prioritario es el desempeño equilibrado entre "
        "las 104 clases. Tampoco se afirma que el sobreajuste desaparezca; se compara "
        "la evidencia de las curvas y se conserva el checkpoint que generaliza mejor "
        "según la validación disponible.\n")
    regularization += (
        "\n\\begin{figure*}[htbp]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{figure_root}/cnn_regularizada/tasa_aprendizaje.pdf}}\n"
        "\\caption{Tasa de aprendizaje realmente utilizada en cada época de la variante "
        "regularizada. Las reducciones responden al estancamiento de la pérdida de validación.}\n"
        "\\label{fig:lr-av4}\n\\end{figure*}\n")
    results = r"""\section{Resultados y evaluación}

\subsection{Comparación de modelos}

La Tabla~\ref{tab:resultados-av4} contiene exclusivamente las nuevas ejecuciones
de este avance. Todos los resultados se calcularon sobre 3,712 imágenes reales
de validación, sin aumentos, y corresponden a un checkpoint identificado. Se
conservan los históricos de entrenamiento y las predicciones por imagen.

\begin{table*}[htbp]
\centering
\caption{Resultados nuevos; el checkpoint de cada modelo se selecciona por F1 macro.}
\label{tab:resultados-av4}
\small
\begin{tabular}{lrrrrrr}
\toprule
Modelo & Parámetros & Época & Exactitud (\%) & F1 macro & Pérdida & Tiempo (s) \\
\midrule
"""
    for row in table.itertuples(index=False):
        results += (f"{SHORT[row.experiment]} & {row.parameters:,} & {row.best_epoch} & "
                    f"{row.val_accuracy*100:.2f} & {row.val_f1_macro:.4f} & "
                    f"{row.val_loss:.3f} & {row.train_seconds:.1f} \\\\\n")
    results += r"""\bottomrule
\end{tabular}
\par\smallskip
\parbox{0.95\textwidth}{\footnotesize Época: checkpoint seleccionado, no número total
de épocas ejecutadas. Tiempo: entrenamiento, excluyendo evaluación y generación
de figuras. La MLP es una referencia reconstruida; no un resultado histórico
atribuible al Avance 3. Una sola semilla por modelo.}
\end{table*}

\begin{figure*}[htbp]
\centering
\includegraphics[width=\textwidth]{""" + figure_root + r"""/comparacion_metricas.png}
\caption{Exactitud y F1 macro sobre la misma validación. El modelo elegido se
determina por F1 macro y puede diferir del que maximiza la exactitud.}
\label{fig:metricas-av4}
\end{figure*}

\subsection{Puente entre la red totalmente conectada y la CNN}

"""
    results += (
        f"La CNN original obtuvo {baseline.val_accuracy*100:.2f}\\% de exactitud frente "
        f"a {mlp.val_accuracy*100:.2f}\\% de la red totalmente conectada de referencia "
        f"({(baseline.val_accuracy-mlp.val_accuracy)*100:+.2f} puntos porcentuales). "
        f"F1 macro cambió de {mlp.val_f1_macro:.4f} a {baseline.val_f1_macro:.4f}. "
        f"La MLP tiene {int(mlp.parameters):,} parámetros frente a "
        f"{int(baseline.parameters):,} de la CNN.\n\n"
        "El resultado es compatible con la ventaja de compartir filtros y preservar "
        "vecindarios de píxeles: una CNN puede reutilizar patrones locales sin asignar "
        "un peso independiente a cada píxel de cada posición. GAP también limita los "
        "parámetros de la etapa densa. Esta explicación es una interpretación "
        "arquitectónica del resultado, no una prueba causal aislada; ambas redes "
        "tienen distinta capacidad y pueden requerir otros hiperparámetros.\n\n"
        "La comparación solicitada con el modelo exacto entregado en el Avance 3 "
        "queda limitada por la ausencia de ese artefacto identificado. El experimento "
        "reconstruido permite comparar familias de modelos en condiciones comunes, "
        "sin inventar resultados históricos. El primer intento con Adam a $10^{-3}$ "
        "predijo una sola clase (5.82\\% de exactitud, F1 macro 0.0011); se conserva "
        "como diagnóstico. La referencia final utiliza $10^{-4}$ y mayor paciencia. "
        "Esta adaptación de optimización se declara para no atribuir exclusivamente "
        "a la arquitectura todas las diferencias observadas.\n\n"
        "\\subsection{Efecto de agregar dos convoluciones}\n\n"
        f"La versión de seis convoluciones obtuvo {deep.val_accuracy*100:.2f}\\% "
        f"de exactitud y F1 macro {deep.val_f1_macro:.4f}. Frente a la CNN original, "
        f"la diferencia es {(deep.val_accuracy-baseline.val_accuracy)*100:+.2f} puntos "
        f"de exactitud y {deep.val_f1_macro-baseline.val_f1_macro:+.4f} de F1 macro. "
        f"Los parámetros aumentan de {int(baseline.parameters):,} a {int(deep.parameters):,}. "
        "Agregar profundidad incrementa la capacidad de representación, pero no "
        "garantiza mejorar validación: también modifica la optimización y puede "
        "facilitar el sobreajuste. La conclusión corresponde a esta configuración y "
        "semilla, sin afirmar superioridad universal de una profundidad.\n\n"
        "\\subsection{Modelo elegido y análisis de errores}\n\n"
        f"Se conserva \\textbf{{{SHORT[best.experiment]}}}, por tener el mayor F1 macro "
        f"de esta comparación: {best.val_f1_macro:.4f}, con exactitud "
        f"{best.val_accuracy*100:.2f}\\%, pérdida {best.val_loss:.3f} y checkpoint en "
        f"la época {int(best.best_epoch)}. La Figura~\\ref{{fig:confusion-av4}} muestra "
        "la matriz de confusión normalizada por clase real. La diagonal representa "
        "recall por clase; los valores fuera de ella son errores. Se entregan además "
        "la matriz de conteos, los CSV de ambas matrices y las predicciones individuales.\n\n")
    selected_eval = evaluations[best.experiment]
    results += (
        "\\begin{figure*}[p]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{figure_root}/{best.experiment}/confusiones_principales.pdf}}\n"
        "\\caption{Vista legible de las 12 clases con mayor participación en errores. "
        "Es una submatriz de la matriz completa: los porcentajes se calculan sobre "
        "todos los ejemplos reales de cada clase y las filas mostradas pueden sumar "
        "menos de 100\\%.}\n\\label{fig:confusiones-foco-av4}\n\\end{figure*}\n\n")
    results += (
        "\\begin{figure*}[p]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth,height=0.85\\textheight,keepaspectratio]"
        f"{{{figure_root}/{best.experiment}/confusion_matrix_normalized.pdf}}\n"
        "\\caption{Matriz de confusión del modelo elegido sobre validación. "
        "Filas: clase real; columnas: clase predicha. Normalización por fila. "
        "Se conserva la versión de alta resolución con las 104 clases.}\n"
        "\\label{fig:confusion-av4}\n\\end{figure*}\n\n"
        "Algunos de los pares más confundidos son:\n\\begin{itemize}\n")
    for pair in selected_eval["confused_pairs"][:5]:
        results += f"\\item {tex_escape(pair['true'])} predicha como {tex_escape(pair['pred'])}: {pair['count']} imágenes.\n"
    results += (
        "\\end{itemize}\nLa Figura~\\ref{fig:predicciones-av4} ilustra aciertos y "
        "errores del checkpoint elegido. Se muestran seis de cada tipo, seleccionados "
        "con semilla fija, por lo que esa figura no estima la tasa de error global.\n\n"
        "\\begin{figure*}[p]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{figure_root}/{best.experiment}/predicciones_ejemplo.png}}\n"
        "\\caption{Ejemplos reales de validación con etiqueta verdadera, predicción "
        "y confianza softmax. El conjunto mostrado equilibra aciertos y errores "
        "solo para facilitar la inspección visual.}\n\\label{fig:predicciones-av4}\n\\end{figure*}\n\n"
        "\\subsection{Conclusiones y limitaciones}\n\n"
        "La comparación documenta el cambio de una referencia totalmente conectada "
        "a un extractor convolucional, el efecto de aumentar dos convoluciones y "
        "el efecto observado de una intervención combinada de regularización. "
        "La elección se basa en F1 macro para evitar que las clases numerosas "
        "dominen la decisión. Las curvas permiten diagnosticar el comportamiento "
        "durante el entrenamiento y justifican conservar checkpoints anteriores "
        "al final cuando la validación deja de mejorar.\n\n"
        "La validación se utiliza también para selección, los resultados son de una "
        "semilla y no se dispone de etiquetas de prueba locales. Harían falta varias "
        "semillas, ablaciones controladas y una evaluación independiente para "
        "cuantificar incertidumbre y separar el efecto de cada técnica.\n")
    sections = ROOT / "reports/secciones"
    (sections / "07_regularizacion.tex").write_text(regularization, encoding="utf-8")
    (sections / "08_resultados.tex").write_text(results, encoding="utf-8")
    selected = {"experiment": best.experiment, "selection_metric": "val_f1_macro",
                "model_path": best.model_path, "accuracy": best.val_accuracy,
                "f1_macro": best.val_f1_macro, "split": "val", "samples": 3712}
    (output / "modelo_elegido.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    abstract = r"""\begin{abstract}
Se comparan una red totalmente conectada de referencia, una CNN original de cuatro
convoluciones, una variante con dos convoluciones adicionales y una intervención
de regularización con AdamW y ajuste de tasa. Se utilizan 12,753 imágenes reales
de entrenamiento y 3,712 de validación de 104 categorías de flores, con entrada
$112\times112$. Se incluyen representaciones de las arquitecturas, curvas de
entrenamiento y validación, matrices de confusión y predicciones ilustrativas.
""" + (f"El modelo elegido por F1 macro de validación es {SHORT[best.experiment]}, "
             f"con F1 macro {best.val_f1_macro:.4f} y exactitud {best.val_accuracy*100:.2f}\\%. ") + r"""
La red totalmente conectada es una referencia reconstruida, no un checkpoint
histórico identificado del Avance 3. Los resultados corresponden a una sola
semilla y no se presentan como evaluación sobre prueba independiente.
\end{abstract}

\begin{keywords}
Clasificación de imágenes \sep Redes neuronales convolucionales \sep
Regularización \sep F1 macro \sep Flores
\end{keywords}
"""
    (sections / "01_resumen.tex").write_text(abstract, encoding="utf-8")
    return selected


def write_notebook(output, selected):
    nbf = nbformat.v4
    root_relative = output.relative_to(ROOT).as_posix()
    cells = [nbf.new_markdown_cell("""# equipo_proyecto_avance4

Comparación de una red totalmente conectada de referencia, la CNN original y la CNN con dos convoluciones adicionales. Diagnóstico de sobreajuste y regularización.

**Procedencia:** los experimentos se ejecutaron desde este repositorio con DirectML en una Radeon RX 6700S. La MLP es una referencia reconstruida y no el modelo histórico identificado del Avance 3. Las métricas corresponden a validación, no a un test independiente.

Las figuras se conservan en PNG y PDF; los históricos, matrices y predicciones se guardan en CSV. Para volver a entrenar, use una carpeta nueva de salida y active `RUN_TRAINING`."""),
        nbf.new_code_cell(f'''from pathlib import Path
import sys, json, inspect, subprocess
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/model.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "{root_relative}"
RUN_TRAINING = False
print("Proyecto:", ROOT)
print("Evidencia de ejecuciones:", OUTPUT)'''),
        nbf.new_markdown_cell("""## 1. Datos y protocolo

Se usan 12,753 imágenes de entrenamiento y 3,712 de validación, 104 clases, resolución efectiva 112×112, lote 64 y semilla 0. Las etiquetas son enteras y los píxeles se normalizan a [0,1]. Solo entrenamiento usa aumentos. El mejor checkpoint se selecciona por F1 macro. Las dos CNN sin la intervención adicional usan Adam a 1e-3, dropout 0.5 y paciencia 5. La MLP usa Adam a 1e-4, paciencia 10 y mínimo 10 épocas, tras el colapso del primer intento a 1e-3. Ese ensayo se conserva en diagnósticos. La intervención combinada usa AdamW, reducción del LR y paciencia 12.

Una sola semilla no permite afirmar significancia estadística. La validación usada para selección no es un test final independiente."""),
        nbf.new_code_cell('''from src.config import load_config
cfg = load_config(OUTPUT / "cnn_original.yaml")
display({"data": cfg["data"], "train": cfg["train"], "runtime": cfg["runtime"]})
for key in ["cnn_original", "cnn_dos_capas_extra", "cnn_regularizada", "mlp_referencia"]:
    profile = load_config(OUTPUT / f"{key}.yaml")
    display({"experiment": key, "model": profile.get("model", {"architecture": "baseline"}), "train": profile["train"]})
manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
print("Commit:", manifest["git_commit"])
print("Checkpoint elegido por:", manifest["selection_metric"])'''),
        nbf.new_markdown_cell("""## 2. Código y representaciones de los modelos

La versión ampliada agrega una convolución después de pool3 y otra después de pool4. Se muestran las implementaciones reales utilizadas para entrenar, no una arquitectura ilustrativa distinta."""),
        nbf.new_code_cell('''from src.train_directml import FlowerCNN, train_epoch, evaluate
from src.train_mlp_reference import FlowerMLP
print(inspect.getsource(FlowerCNN))
print(inspect.getsource(FlowerMLP))
display(Image(filename=str(OUTPUT / "figures/arquitecturas.png"), width=1100))'''),
        nbf.new_markdown_cell("""## 3. Entrenamiento reproducible

Estos comandos utilizan el mismo código que produjo los resultados guardados. Se usa una carpeta nueva para conservar las ejecuciones anteriores. Cambie `RUN_TRAINING=True` para ejecutarlos; una revisión normal del notebook carga los resultados existentes.

La CNN original y la ampliada se entrenan primero. Se añade la variante regularizada y se entrena la referencia totalmente conectada con el cargador y el ciclo existentes."""),
        nbf.new_code_cell('''NEW_OUTPUT = ROOT / "reports" / "avance4_repeticion"
if RUN_TRAINING:
    subprocess.run([sys.executable, "-u", "-m", "scripts.run_avance4", "--output", str(NEW_OUTPUT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-u", "-m", "src.train_mlp_reference", "--config", str(NEW_OUTPUT / "cnn_original.yaml"), "--output", str(NEW_OUTPUT), "--lr", "0.0001", "--patience", "10", "--min-epochs", "10"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-u", "-m", "scripts.prepare_avance4", "--output", str(NEW_OUTPUT)], cwd=ROOT, check=True)
    OUTPUT = NEW_OUTPUT
else:
    print("Se cargan las ejecuciones completas; RUN_TRAINING=False.")'''),
        nbf.new_code_cell('''print("Ciclo de entrenamiento compartido:")
print(inspect.getsource(train_epoch))
print("Evaluación sobre imágenes sin aumentos:")
print(inspect.getsource(evaluate))'''),
        nbf.new_markdown_cell("""## 4. Resultados finales y comparación

La exactitud y F1 macro pertenecen al mismo checkpoint. Se compara explícitamente MLP frente a CNN y CNN original frente a CNN con dos capas adicionales. La convolución comparte filtros y utiliza vecindarios espaciales; la MLP recibe píxeles aplanados. El aumento de profundidad no garantiza una mejora de validación."""),
        nbf.new_code_cell('''results = pd.read_csv(OUTPUT / "resultados.csv")
display(results[["model", "parameters", "best_epoch", "epochs_run", "val_accuracy", "val_f1_macro", "val_loss", "train_seconds"]])
display(Image(filename=str(OUTPUT / "figures/comparacion_metricas.png"), width=1100))
indexed = results.set_index("experiment")
for reference, candidate in [("mlp_referencia", "cnn_original"), ("cnn_original", "cnn_dos_capas_extra"), ("cnn_original", "cnn_regularizada")]:
    print(reference, "->", candidate)
    print("Cambio de exactitud (puntos porcentuales):", round(100*(indexed.loc[candidate,"val_accuracy"]-indexed.loc[reference,"val_accuracy"]),2))
    print("Cambio de F1 macro:", round(indexed.loc[candidate,"val_f1_macro"]-indexed.loc[reference,"val_f1_macro"],4))'''),
        nbf.new_markdown_cell("""## 5. Diagnóstico de sobreajuste y regularización

Se comparan pérdida y exactitud de entrenamiento frente a validación. Una pérdida de entrenamiento que disminuye mientras validación se estanca o empeora es evidencia de sobreajuste. Entrenamiento se mide con aumentos y dropout, por lo que una diferencia puntual de exactitudes no basta para diagnosticarlo.

Dropout, aumentos y parada temprana están presentes en todos los modelos. La intervención adicional combina AdamW (weight decay 0.01, sin sesgos), reducción del LR por factor 0.5 con paciencia 2, mínimo 1e-5 y parada temprana con paciencia 12. El experimento mide el efecto combinado, no el aporte aislado de cada técnica."""),
        nbf.new_code_cell('''display(Image(filename=str(OUTPUT / "figures/comparacion_curvas.png"), width=1000))
display(results[["model", "train_accuracy_at_checkpoint", "val_accuracy", "accuracy_gap_at_checkpoint"]])
for row in results.itertuples():
    run_dir = next((OUTPUT / "runs" / row.experiment).glob("*/summary.json")).parent
    history = pd.read_csv(run_dir / "history.csv")
    best_loss = history.loc[history.val_loss.idxmin()]
    print(row.model, "| mínima pérdida val:", round(best_loss.val_loss,4), "en época", int(best_loss.epoch)+1,
          "| última pérdida train/val:", round(history.iloc[-1].loss,4), round(history.iloc[-1].val_loss,4))'''),
        nbf.new_markdown_cell("""## 6. Modelo elegido, matrices de confusión y errores

Se elige el mayor F1 macro de validación. La matriz normalizada muestra recall en la diagonal (filas reales, columnas predichas). Los ejemplos muestran seis aciertos y seis errores elegidos con semilla fija; esa selección ilustrativa no representa la frecuencia de errores del modelo."""),
        nbf.new_code_cell('''selected = results.sort_values("val_f1_macro", ascending=False).iloc[0]
key = selected.experiment
print("Modelo elegido:", selected.model, "| F1 macro:", selected.val_f1_macro, "| exactitud:", selected.val_accuracy)
run_dir = next((OUTPUT / "runs" / key).glob("*/summary.json")).parent
evaluation = json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8"))
display(pd.DataFrame(evaluation["confused_pairs"]))
display(Image(filename=str(OUTPUT / "figures" / key / "confusion_matrix_normalized.png"), width=1100))
display(Image(filename=str(OUTPUT / "figures" / key / "predicciones_ejemplo.png"), width=1000))'''),
        nbf.new_markdown_cell("""## 7. Comprobación de las métricas y artefactos

Se recalculan exactitud y F1 desde las predicciones individuales, se comprueba que la matriz contiene 3,712 casos y se verifica que cada corrida tiene sus figuras. La evaluación también compara los logits del checkpoint de GPU y su exportación a Keras."""),
        nbf.new_code_cell('''import numpy as np
from sklearn.metrics import accuracy_score, f1_score
for row in results.itertuples():
    run_dir = next((OUTPUT / "runs" / row.experiment).glob("*/summary.json")).parent
    pred = pd.read_csv(run_dir / "evaluation/validation_predictions.csv")
    cm = pd.read_csv(run_dir / "evaluation/confusion_matrix.csv", index_col=0)
    np.testing.assert_allclose(accuracy_score(pred.true_label, pred.predicted_label), row.val_accuracy, atol=1e-9)
    np.testing.assert_allclose(f1_score(pred.true_label, pred.predicted_label, average="macro"), row.val_f1_macro, atol=0.000051)
    assert cm.to_numpy().sum() == len(pred) == 3712
    assert cm.shape == (104,104)
    for name in ["curvas_entrenamiento_validacion", "confusion_matrix", "confusion_matrix_normalized", "predicciones_ejemplo"]:
        assert (OUTPUT / "figures" / row.experiment / f"{name}.png").exists()
    print(row.model, "OK:", len(pred), "predicciones y matriz 104×104")'''),
        nbf.new_markdown_cell("""## 8. Conclusiones y entrega

La tabla y las curvas sustentan la elección por F1 macro. Las métricas corresponden a una semilla y al conjunto usado para seleccionar los modelos. No se atribuye a Avance 3 la referencia MLP reconstruida, ni a DM los resultados de entrenamiento con todas las imágenes reales.

Se corrigieron resolución, lote, descripción de arquitecturas, separación entre destilación y clasificación, selección del checkpoint y alcance de validación. Se incorporaron las representaciones solicitadas. No se afirma atender comentarios de retroalimentación cuyo texto no fue proporcionado.

El PDF actualizado contiene las secciones 1–8. Los archivos de entrega se llaman `equipo_proyecto_avance4.pdf` y `equipo_proyecto_avance4.ipynb`. Las figuras originales se conservan para reutilizarlas en la presentación."""),
    ]
    notebook = nbf.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3 (proyecto)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}})
    path = ROOT / "notebooks/equipo_proyecto_avance4.ipynb"
    nbformat.write(notebook, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    table = pd.read_csv(output / "resultados.csv")
    selected = write_sections(output, table)
    notebook = write_notebook(output, selected)
    (output / "report_build.json").write_text(json.dumps({"updated_at": datetime.now().astimezone().isoformat(),
        "notebook": str(notebook), "selected": selected}, indent=2), encoding="utf-8")
    print(notebook)


if __name__ == "__main__":
    main()
