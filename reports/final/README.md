# Reporte final

Fuentes LaTeX del reporte final (`main.tex`, `secciones/`, `references.bib`) y los datos que lo respaldan.

```bash
python -m scripts.run_final        # entrena y evalúa los 7 experimentos del reporte (≈2.5 h, Mac M4)
python -m scripts.analyze_final    # regenera resultados.csv, resumen.json, tablas/ y figures/
latexmk -pdf main.tex              # desde esta carpeta
```

El notebook [`notebooks/reporte_final.ipynb`](../../notebooks/reporte_final.ipynb) hace los dos últimos pasos y
muestra tablas y figuras.

| Ruta | Contenido |
|---|---|
| `resultados.csv` | una fila por experimento: configuración, métricas de validación, F1 de entrenamiento y brecha |
| `resumen.json` | cifras citadas en el texto (desbalance, pares más confundidos) |
| `tablas/*.tex` | tablas generadas que `secciones/07_resultados.tex` incluye con `\input` |
| `figures/` | figuras generadas (`analyze_final`) y fijas (diagramas y ejemplos) |
| `experimentos/configs/` | YAML exacto de cada experimento, derivado de `configs/final.yaml` |
| `experimentos/runs/<exp>/<run>/` | `history.csv` por época, `summary.json`, `eval_report.json` (F1 por clase) y matriz de confusión |
| `experimentos/conteo_por_clase.csv` | imágenes por clase en train y val |
| `kaggle/` | predicciones de test de la referencia en formato de Kaggle (`id,label`) |

## Experimentos

Cada uno cambia una sola cosa respecto a `referencia` (`python -m scripts.run_final --list`):
`referencia`, `mlp`, `cnn_ampliada`, `sin_regularizacion`, `class_weight`, `lr_3e-4` y `dropout_0.3`.
`run_final.py` define además corridas que no se usaron (`res_160`, `sin_dropout`, `sin_aumento`,
`sin_weight_decay`, `l2`, `label_smoothing`, `efficientnetb0`); se corren con `--all`.

Notas:

- `class_weight` se colgó en la época 64 en su primer intento y se volvió a correr.
- `cnn_ampliada` se interrumpió en la época 50 porque la GPU Metal se colgó. Se evaluó su mejor
  checkpoint (época 41) y su `summary.json` lo indica en el campo `interrumpida`. Desde entonces,
  `run_final` termina cualquier corrida que pase 10 min sin escribir en el log.
- Los checkpoints `.keras` quedan en `models/final/` y no se versionan.
