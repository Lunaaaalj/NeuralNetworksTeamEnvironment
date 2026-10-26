# Etapa 2 — Baseline MLP y comparación con CNN

Fecha de revisión: 26 de octubre de 2026. Rama: `fix/avance4-retroalimentacion`.
Punto de partida: auditoría `e414f8f` y plan `dc2543f`, ya subidos a GitHub.

## Resultado y alcance

Se confirmó que la MLP histórica es totalmente conectada, se recalcularon sus métricas desde las predicciones guardadas, se examinó su distribución de predicciones, curvas, tasa de aprendizaje y pesos disponibles, y se contrastó con las CNN del Avance 4. Se conserva íntegra la evidencia anterior. No se entrenó ninguna red nueva ni se repitió inferencia sobre imágenes: faltan los TFRecords y los frameworks de entrenamiento. Las pérdidas y las métricas de train son históricas, no mediciones nuevas.

El diagnóstico es **ajuste insuficiente con indicios de problemas de optimización y limitaciones de representación**, no una demostración de que cualquier MLP sea incapaz de aprender flores. La comparación histórica es válida como referencia de las ejecuciones documentadas, pero no demuestra un efecto puramente arquitectónico ni que la MLP esté bien optimizada. Se conserva esta limitación explícita en lugar de presentar el baseline como óptimo.

## Arquitectura comprobada

`src/train_mlp_reference.py::FlowerMLP` implementa `Flatten NHWC → Linear(37,632,256) → ReLU → Dropout(0.5) → Linear(256,104)`; no contiene convoluciones ni pooling. La permutación NCHW→NHWC antes de aplanar conserva el orden espacial de la exportación Keras. `src/model.py::build_mlp` implementa la misma topología para la familia TensorFlow del reporte final, con parámetros de dropout configurables.

La lectura directa de `config.json` y `model.weights.h5` dentro de los archivos Keras confirma capas `InputLayer, Flatten, Dense, Dropout, Dense`, dimensiones 37,632×256 y 256×104, y **9,660,776 parámetros**. La primera capa concentra 9,634,048, el **99.72%**. Los dos checkpoints MLP Keras inspeccionados contienen pesos finitos. Se usó h5py, sin cargar objetos Keras ni deserializar pickle. No se comprobó nuevamente paridad de logits entre frameworks: sigue siendo una comprobación histórica.

## Comparación verificada de validación

Mismas particiones declaradas de Kaggle: 12,753 imágenes de train y 3,712 de val, 104 clases, seed 0, RGB 112×112, x/255, batch 64. Se comprobó que las secuencias de etiquetas de val y sus soportes son iguales. Sin TFRecords/IDs de imagen no se certifica nuevamente la identidad de cada imagen. F1/exactitud se recalculan de clases predichas; loss procede de evaluación guardada.

| Experimento histórico A4 | Parámetros | Checkpoint / épocas | Exactitud val | F1 macro val | Loss val guardada |
|---|---:|---:|---:|---:|---:|
| MLP, Adam LR 1e-4 | 9,660,776 | 77 / 87 | 15.52% | 0.0280 | 3.4032 |
| CNN original, Adam LR 1e-3 | 480,936 | 53 / 58 | 63.25% | 0.5740 | 1.5196 |
| CNN ampliada, Adam LR 1e-3 | 1,218,600 | 46 / 51 | 62.23% | 0.5502 | 1.7857 |
| CNN original, AdamW + LR adaptable | 480,936 | 61 / 73 | 64.44% | 0.5745 | 1.5075 |

Fuente exacta de cada fila: `avance4_2026-10-21/runs/<variante>/<run>/`, identificada con rutas y hashes en `correcciones_etapa2/diagnostico.json`. Derivado de precisión completa: `comparacion_modelos.csv`.

La CNN original supera a la MLP en **47.7371 puntos porcentuales** de exactitud y **0.545941** de F1 macro, con aproximadamente 20 veces menos parámetros. Las redes difieren en capacidad, LR y paciencia: MLP LR=1e-4/paciencia 10/mínimo 10 épocas; CNN original LR=1e-3/paciencia 5. Ambas usan aumentos y dropout 0.5. La ventaja observada no se atribuye exclusivamente a las convoluciones.

Como control analítico, se eligió la clase más frecuente de **train** (`iris`, etiqueta 67), sin optimizarla con val. Predecirla siempre en las etiquetas de val produce 6.1422% de exactitud y F1 macro 0.001113. La MLP A4 supera ese control, pero sigue lejos de un clasificador útil de las 104 clases. Es un control calculado, no un modelo entrenado ni un resultado de Kaggle.

## Comportamiento de las predicciones

| Indicador | MLP A4 | CNN original A4 |
|---|---:|---:|
| Clases distintas predichas | 21 de 104 | 102 de 104 |
| Clases con F1 cero | 87 | 4 |
| Fracción de predicciones en sus cinco clases más predichas | 85.24% | 29.55% |
| Mediana de confianza softmax guardada | 7.30% | 72.25% |

La MLP nunca predice 83 clases, y otras cuatro sí predichas no tienen verdaderos positivos. Esto explica por qué la exactitud de 15.52% convive con F1 macro 0.0280: los aciertos se concentran en pocas clases.

| Clase | Predicciones MLP A4 | Casos reales de val | Imágenes train |
|---|---:|---:|---:|
| iris | 1,033 | 228 | 782 |
| rose | 831 | 134 | 460 |
| wild rose | 666 | 216 | 743 |
| sunflower | 376 | 134 | 460 |
| wild geranium | 258 | 205 | 703 |

La concentración en clases abundantes es observable. La correlación de Spearman entre tamaño de train por clase y F1 es 0.5305 en la MLP A4 y 0.1058 en la CNN original; es una descripción de estas 104 clases, no causalidad ni significancia estadística. La confianza softmax no es una métrica de calibración ni permite concluir que los errores estén bien calibrados.

## Curvas, normalización y optimización

**MLP A4 a LR 1e-4.** La loss de train online baja de 4.4906 a 3.5948; val_loss baja de 4.2509 a 3.3452. En el checkpoint 77: exactitud de train online 12.06%, val 15.52%, loss online 3.6163 y val_loss 3.4032 en el historial. La LR permanece en 1e-4. El máximo F1 se selecciona en 77; la menor loss de val ocurre en 76. No deben confundirse selección por F1, mínimo de loss y última época 87.

Estas curvas no muestran memorización intensa con alta exactitud de train y deterioro sostenido de val. Sugieren ajuste insuficiente en las condiciones ensayadas. Sin embargo, train online tiene dropout y aumentos y no es una medida limpia de ajuste al conjunto de entrenamiento: no se puede calcular una brecha F1 limpia para esta MLP A4 a partir de los históricos actuales.

**Ensayo A4 a LR 1e-3.** El historial tiene seis épocas y el mejor checkpoint es la primera, con F1 0.0010575 y exactitud 5.8190%. Su summary registra colapso a una sola clase; no se guardaron predicciones individuales de ese diagnóstico para volver a verificar esa declaración. Se preserva como fallo de optimización documentado, no se sustituye por la corrida definitiva.

**Ensayo A4 a LR 1e-4 interrumpido.** Tiene 25 épocas completas y un PT de época 25, sin summary final. El estado registra fallo durante escritura de checkpoint y reinicio desde cero. No se concatenaron sus curvas/épocas con la MLP definitiva ni se tomó como repetición independiente que estime variabilidad.

**MLP del reporte final, distinta corrida.** Mismo tamaño de red, AdamW y LR adaptable, F1 val 0.0177734/exactitud 14.3588%, checkpoint 62 de 74. Su histórico sí mide train limpio sobre 3,712 imágenes fijas en inferencia: F1 0.0192126, exactitud 15.7597% y loss 3.5532; val_loss 3.5817. Ese bajo ajuste limpio respalda infraajuste en **esa corrida**. La LR llega a 1e-6 en la época 57: revisar su reducción temprana es una hipótesis razonable, no una causa aislada demostrada. No se aplica su brecha a A4 ni se combina su train con el val de A4.

**Preprocesamiento y orden espacial.** `src/data.py` especifica float32/255, redimensionamiento bilineal con antialias a 112 y aumentos solo en train. El flatten NHWC del código PyTorch concuerda con Keras. No se identificó una convolución oculta ni un error estático de orden espacial; no se probó el pipeline con imágenes reales en esta etapa. Faltan pruebas sobre distribución efectiva de entradas, activaciones y gradientes antes de descartar fallos en runtime.

## Qué aportó la inspección de pesos

En la primera capa, 252/256 neuronas de la MLP definitiva y 256/256 del ensayo LR=1e-3 tienen suma de pesos entrantes más sesgo negativa. La mediana de la suma de pesos es −19.56 y −190.52, respectivamente. Esto motiva investigar el centrado de entradas y la actividad ReLU, especialmente en el ensayo colapsado.

No prueba neuronas muertas: para cada neurona se calculó el límite superior exacto `bias + sum(max(weight,0))` en el dominio completo x∈[0,1]^37632. Ninguna de las 256 neuronas queda certificada como permanentemente inactiva por ese criterio. Sin imágenes reales no se midió la fracción de activaciones cero, ni los gradientes, ni logits. No se usaron imágenes aleatorias o sintéticas para fabricar métricas de entrenamiento.

## Por qué una CNN es una referencia adecuada

La conectividad local preserva vecindarios de píxeles y permite aprender patrones espaciales. Compartir cada filtro entre posiciones reduce parámetros y reutiliza un detector en distintas regiones. Max pooling reduce dimensiones y aporta cierta tolerancia a desplazamientos; GAP resume cada mapa antes de la cabeza densa. La MLP sobre píxeles aplanados asigna un peso distinto a cada posición y concentra casi todos sus parámetros en la primera capa.

Estas propiedades constituyen una explicación arquitectónica plausible de la diferencia observada. No garantizan invariancia completa ni superioridad universal, y los resultados disponibles no separan su efecto del de optimización, capacidad y regularización.

## Mejoras propuestas y condición de ejecución

No se ejecuta una variante nueva sin datos. El orden recomendado para evitar más corridas poco informativas es:

1. Recuperar la misma copia de TFRecords y verificar hashes del manifiesto A4, particiones y 104 etiquetas. Preparar un entorno compatible y comprobar dispositivo; el hardware local es RTX 4050, no la Radeon histórica.
2. Evaluar el checkpoint A4 en train sin aumentos/dropout y en val, registrando IDs/hashes del subconjunto y métricas de los dos splits. Medir activaciones ReLU, distribución de logits y gradientes sobre imágenes reales, sin actualizar el checkpoint histórico.
3. Comprobar si una MLP puede ajustar una muestra pequeña **real** de train con dropout/aumentos desactivados en un diagnóstico identificado. Ese control de optimización no sustituye entrenamiento completo ni sus métricas cuentan como resultado final.
4. Elegir un solo cambio según evidencia: LR/calendario menos agresivo si los gradientes son válidos y el progreso lento; centrado de entrada si hay desplazamiento persistente de preactivaciones; o una activación con gradiente negativo si se comprueba inactividad de ReLU. No aplicar todo junto y atribuir el efecto a una sola técnica.
5. Entrenar la variante seleccionada con el train completo, mismo split/seed/resolución, checkpoint por F1 y evaluación limpia de train/val; registrar parámetros, condiciones, tiempos y curvas en una carpeta nueva. Preservar A4 histórico y declarar diferencias de backend.

No se fuerza a la MLP a igualar a la CNN. Una sola variante/semilla permitiría describir un resultado observado, no significancia estadística. Los candidatos anteriores son propuestas, sin métricas ni promesa de mejora.

## Archivos y reproducción

- `scripts/inspect_mlp_weights.py`: lectura de pesos Keras/HDF5, sin frameworks; genera `pesos_mlp.json`.
- `scripts/analyze_mlp_avance4.py`: recálculo/validación de cinco corridas históricas, tablas y curvas, sin entrenamiento.
- `tests/test_avance4_evidence.py`: controles ligeros sobre evidencia, incluidos rechazo de predicciones alteradas, métricas mezcladas y sustitución del checkpoint por la última época.
- `correcciones_etapa2/diagnostico_mlp_cnn.ipynb`: notebook complementario de diagnóstico con visualizaciones editables y comprobación de fuentes. No reemplaza todavía el notebook de entrega.
- `correcciones_etapa2/{comparacion_modelos,metricas_por_clase,concentracion_predicciones}.csv`: derivados de cinco corridas (cuatro A4 y una MLP F), identificadas por familia.
- `correcciones_etapa2/{curvas_mlp_cnn,predicciones_mlp_cnn,diagnosticos_lr_mlp}.{png,pdf}`: curvas y distribuciones de evidencia histórica.
- `correcciones_etapa2/diagnostico.json`: manifiesto de inputs/código y entorno de análisis, con SHA-256 crudo y equivalente LF para texto. Los checkpoints requieren hash binario exacto.

Desde la raíz, con dependencias ligeras numpy/pandas/scipy/scikit-learn/matplotlib:

```powershell
python -m scripts.analyze_mlp_avance4
python -m unittest tests.test_avance4_evidence -v
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 reports/entregable_4/correcciones_etapa2/diagnostico_mlp_cnn.ipynb
```

La inspección de pesos es opcional para volver a abrir el notebook, porque se guardó el diagnóstico con hashes. Para regenerarla: instalar h5py/numpy en entorno aislado y ejecutar `python -m scripts.inspect_mlp_weights` antes del analizador. En esta sesión se usó `.venv` local ignorado por Git (h5py 3.16.0 y numpy 2.5.4) para esa lectura; las métricas/gráficas se calcularon con el Python del sistema, cuyas versiones constan en `diagnostico.json`. No se sube ese entorno ni se modificaron los paquetes del Python global.

## Verificación y límites de la etapa

El analizador se ejecutó sobre las cinco corridas: métricas, matrices, soportes, épocas y checkpoint coinciden a tolerancia de redondeo. Las seis pruebas de evidencia pasaron con `unittest`. No son pruebas de entrenamiento ni sustituyen la suite completa. La suite pytest sigue requiriendo TensorFlow en `tests/conftest.py`; no se afirma que haya pasado.

El entorno activo sigue sin TensorFlow/Keras/PyTorch/DirectML ni TFRecords en la ruta del proyecto, y no hay credenciales Kaggle convencionales presentes. Se solicitó al usuario una ruta de datos externa si ya existe. No se descargaron datos, no se solicitaron tokens y no se entrenó. Las correcciones de señales Windows/rutas y controles generales DirectML identificados en auditoría siguen pendientes para el flujo de entrenamiento; no se ejecutó una nueva CLI de entrenamiento ni se afirma que esos problemas estén reparados.

El notebook complementario se ejecutó de principio a fin: siete celdas de código sin errores y dos gráficas PNG embebidas, inspeccionadas visualmente junto con las tres figuras del analizador. La ejecución inicial se bloqueó en `asyncio/socketpair` por el aislamiento de conexiones locales; se completó fuera de ese aislamiento usando el mismo Python. No fue un entrenamiento. Las figuras son legibles y conservan escalas comunes por métrica; el esquema nbformat y los hashes de las fuentes se verificaron.

El código de análisis/pruebas quedó registrado en `749ddd0`; el manifiesto de los derivados apunta a ese commit y añade los hashes exactos de sus inputs. La compilación del reporte y el notebook de entrega siguen reservados para las etapas 4–5. El siguiente paso propuesto, tras revisión del usuario, es la etapa 3: recuperar la comparación histórica conjunta de regularización sin repetir entrenamientos.
