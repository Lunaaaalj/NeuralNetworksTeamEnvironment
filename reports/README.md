# reports/

Cada entrega del proyecto vive en su propio subdirectorio con fuentes LaTeX autocontenidas
(`main.tex`, `secciones/`, `references.bib`, `figures/`) y el PDF entregado.

| Directorio | Contenido | Compilar |
|---|---|---|
| `entregable_1/` | Primer reporte: metodología y destilación DM (`main.pdf`). Fuentes recuperadas del commit `cd73c96`. Usa biblatex + biber. | `cd reports/entregable_1 && latexmk -pdf main.tex` |
| `entregable_4/` | Avance 4: comparación MLP / CNN / CNN ampliada / CNN regularizada (`equipo_proyecto_avance4.pdf`). Usa natbib. | `cd reports/entregable_4 && tectonic equipo_proyecto_avance4.tex` |
| `entregable_4/avance4_2026-10-21/` | Evidencia del Avance 4: configs, corridas, modelos, figuras y métricas generadas por `scripts/run_avance4.py` y `scripts/prepare_avance4.py`. | — |
| `final/` | Reporte final (pendiente). | — |

## Salidas del pipeline (fuera de los entregables)

Los configs de `configs/*.yaml` escriben aquí, así que estas rutas no se movieron:

- `runs/`: `history.csv`, `summary.json` y registros por corrida (excluido de Git).
- `figures/`: imágenes sintéticas DM, curvas y matrices de confusión de los perfiles DirectML.
- `resultados.csv`: una fila por corrida de `scripts/run_experiments.py`.

Para usar una figura del pipeline en un reporte, cópiala al `figures/` del entregable.
