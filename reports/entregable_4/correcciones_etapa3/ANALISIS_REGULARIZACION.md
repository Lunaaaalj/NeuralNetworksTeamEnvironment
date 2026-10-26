# Etapa 3: regularización con evidencia histórica

Se reutilizan F/referencia y F/sin_regularizacion del reporte final, sin modificar sus archivos ni entrenar. Ambas usan CNN baseline de 480,936 parámetros, seed 0, imágenes 112×112, batch 64, train 12,753 y val 3,712. El backend TensorFlow/Metal procede de documentación histórica. No son las corridas A4/DirectML.

## Método

Se verifican predicciones/matrices y summary/history. Train limpio: primeras 3,712 imágenes fijas de train sin aumentos y en inferencia, según código/configuración; estos valores solo se verifican documentalmente, no mediante nueva inferencia. Las pérdidas también son históricas. La brecha F1 se resta con precisión completa dentro de la misma fila, sin usar cifras redondeadas del summary. Los historiales comparten épocas 1–46; la época 34 contiene todas las métricas y se extrae exactamente, sin interpolación.

## Mejores checkpoints

| Variante F | Época / ejecutadas | F1 train limpio | F1 val | Exactitud val | Loss val | Brecha F1 |
|---|---|---:|---:|---:|---:|---:|
| F/referencia | 91 / 103 | 0.8138 | 0.5784 | 64.06% | 1.4830 | 0.2354 |
| F/sin_regularizacion | 34 / 46 | 0.8767 | 0.5332 | 59.64% | 2.0395 | 0.3435 |

La intervención conjunta mejora F1 val en 0.045155 y exactitud en 4.4181 puntos porcentuales; reduce la brecha F1 en 0.108023. Sin regularización explícita la red ajusta más train, pero su mayor brecha y loss val son compatibles con mayor sobreajuste. No es una atribución causal a un regularizador individual.

## Comparación complementaria en época 34

| Variante F | F1 train limpio | F1 val | Exactitud train | Exactitud val | Loss train | Loss val | Brecha F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F/referencia | 0.5793 | 0.4851 | 64.22% | 56.98% | 1.2247 | 1.6289 | 0.0942 |
| F/sin_regularizacion | 0.8767 | 0.5332 | 87.34% | 59.64% | 0.4434 | 2.0395 | 0.3435 |

En época 34 la variante sin regularización tiene mayor F1 y exactitud de validación, mientras la regularizada tiene menor pérdida y menor brecha. Esta diferencia temporal impide afirmar que la regularización domina en todas las épocas. La versión regularizada sigue mejorando después y su checkpoint seleccionado supera al mejor de la otra variante.

## Interpretación y límites

La intervención cambia simultáneamente dropout 0.5→0, aumentos activados→desactivados y AdamW/WD 0.01→Adam/WD 0. Ambas conservan LR inicial 1e-3, scheduler de val_loss ×0.5/paciencia 2/mínimo 1e-5, parada temprana de 12 evaluaciones y tope de 150. Las trayectorias reales de LR difieren: comparar época 34 no controla el número efectivo de pasos a una misma LR ni elimina otros efectos de la política. Una brecha pequeña también puede acompañar infraajuste; debe leerse junto con F1/loss de ambos splits. No hay significancia estadística con una sola semilla ni test independiente.

## Reproducción y fuentes

Desde la raíz: `python reports/entregable_4/correcciones_etapa3/analizar_regularizacion.py`. Requiere numpy, pandas, matplotlib, scipy, scikit-learn y PyYAML, sin frameworks de entrenamiento. `fuentes.json` registra todas las rutas y SHA-256 crudo/LF. `comparacion_regularizacion.csv` contiene las cuatro filas y todas las métricas; `curvas_regularizacion.csv` contiene cada época. PNG y PDF identifican familia y protocolo. El script rechaza épocas ausentes/duplicadas o valores no finitos.
