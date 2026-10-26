# Trazabilidad y plan experimental del Avance 4 — etapa 1

Fecha: 26 de octubre de 2026. Base: `c08f38796f47ed8744f6668c454da4af31bc4175` (`main`).
Rama: `fix/avance4-retroalimentacion`. Auditoría previa: `AUDITORIA_CORRECCIONES.md`, commit `e414f8f`.

Este documento registra corridas históricas y propone trabajo posterior. Ningún resultado de las tablas se obtuvo entrenando en esta tarea. Las etapas 2–5 requieren instrucción posterior del usuario. No se modifican experimentos ni el reporte final.

## Convenciones del registro

Cada identificador incluye familia, variante y nombre de corrida. Un nombre como `full_res112_s0_adamw_plateau` no es único fuera de su directorio: varias variantes lo comparten. A4 y F son familias distintas; H es evidencia histórica de A3/DM. Los alias de este documento no renombran archivos.

Rutas abreviadas:

- `A = reports/entregable_4/avance4_2026-10-21`.
- `F = reports/final/experimentos`.
- `H = reports/entregable_3`.

En A4/F, cada directorio `runs/<variante>/<run>/` contiene `summary.json`, `history.csv`, `eval_report.json`, `evaluation/validation_predictions.csv`, `confusion_matrix.csv` y `confusion_matrix_normalized.csv`. El YAML A4 está en `A/<variante>.yaml`; el final en `F/configs/<variante>.yaml`. Esas reglas, más las tablas, especifican los archivos de evidencia de cada experimento.

**Verificable** significa: F1/exactitud recalculados desde predicciones y consistentes con matrices/resúmenes. No significa nueva inferencia. **Documental** significa: métricas solo cotejadas con históricos/resúmenes. Pérdidas: documentales en ambos casos, pues los CSV no guardan los 104 logits. `history.epoch` es base 0; épocas de tablas/checkpoints, base 1. La columna exactitud está en porcentaje; F1 y pérdida, en su escala original.

## Condiciones comunes y arquitecturas

Todas las corridas principales A4 y F declaran train original Kaggle de 12,753 imágenes reales y val original de 3,712, 104 clases, semilla 0, entrada efectiva RGB 112×112, batch 64, entrenamiento desde cero y selección del checkpoint por máximo F1 macro de val. No se afirma evaluación de test independiente. Muchos `summary.train_images` F son null: el tamaño se sostiene en protocolo/conteos, no en ese campo. Las once secuencias de etiquetas de val coinciden; falta comprobación de imágenes originales y hashes de shards finales.

Preprocesamiento P: TFRecords JPEG de origen 224 px → RGB float32/255 → bilineal con antialias a 112 px. Aumentos G: flip horizontal, rotación factor 0.05, zoom 0.1 y contraste 0.1, recorte [0,1], solo train y después de caché. MLP aplana NHWC; no usa convoluciones ni pooling.

| Arquitectura | Definición | Parámetros entrenables |
|---|---|---:|
| MLP / mlp_reference | Flatten 37,632 → Dense256/ReLU → Dropout → Dense104 logits | 9,660,776 |
| baseline | Conv3×3 32/64/128/256, ReLU y MaxPool2×2 en cada bloque; GAP → Dense256/ReLU → Dropout → Dense104 | 480,936 |
| deep | baseline + Conv128 tras pool3 y Conv256 tras pool4, ReLU, sin pool adicional | 1,218,600 |

Los conteos concuerdan con resúmenes, definiciones y dimensiones de capas; no se instanciaron redes en esta sesión. No hay BatchNorm en estas corridas. L2 y label smoothing son cero en F; class_weight es `none` salvo la variante indicada.

## Familia A4: PyTorch/DirectML, Radeon RX 6700S

Procedencia: manifiesto `A/manifest.json`, creado el 21 de octubre, commit registrado `857e6839023076d18bb4f68dc253c844b94d4dc6`; código empaquetado con hashes adicionales y versiones TF 2.21.0, Keras 3.15.1, torch 2.4.1, torch-directml 0.2.5.dev240914, numpy 1.26.4. El commit no representa por sí solo todos los cambios locales de aquella ejecución; conservar también hashes. La GPU histórica no es la GPU local actual.

Todos usan P, G y dropout 0.5. CNN original/deep: Adam específico DirectML, LR fijo 0.001, weight decay 0, paciencia 5. MLP: mismo cargador y optimizador, LR 0.0001, paciencia 10 y mínimo 10 épocas. Regularizada: DirectMLAdamW, weight decay 0.01 excluyendo sesgos, LR inicial 0.001, reducción en meseta de val_loss ×0.5, paciencia del scheduler 2, mínimo 1e-5; parada temprana 12. Tope 150 en las cuatro.

| ID completo (familia/variante/run) | Arquitectura / parámetros | Épocas ejecutadas | Checkpoint | F1 macro val | Exactitud val (%) | Pérdida val evaluada | Verificación |
|---|---|---:|---:|---:|---:|---:|---|
| A4/mlp_referencia/full_res112_s0_mlp_reference_directml | MLP / 9,660,776 | 87 | 77 | 0.0280126128 | 15.51724138 | 3.4031510682 | Predicciones + matriz + historial |
| A4/cnn_original/full_res112_s0_directml | baseline / 480,936 | 58 | 53 | 0.5739536942 | 63.25431034 | 1.5195830649 | Predicciones + matriz + historial |
| A4/cnn_dos_capas_extra/full_res112_s0_deep_directml | deep / 1,218,600 | 51 | 46 | 0.5501506186 | 62.23060345 | 1.7856977644 | Predicciones + matriz + historial |
| A4/cnn_regularizada/full_res112_s0_adamw_plateau_directml | baseline / 480,936 | 73 | 61 | 0.5744742346 | 64.43965517 | 1.5074768930 | Predicciones + matriz + historial |

F1/exactitud anteriores se recalcularon desde predicciones sin entrenar. Las pérdidas pertenecen a `eval_report.json`; pueden diferir en últimas cifras de `history.csv` por reducción numérica/batch de evaluación, no se sustituyen para forzar igualdad. Los resúmenes CNN redondean F1/exactitud a cuatro decimales; `resultados.csv` A4 usa F1 redondeado y exactitud de evaluación sin redondear.

Checkpoints: `A/models/<variante>/<run>.pt` y `.keras`, presentes. Metadatos PT de época y arquitectura coherentes; contenedores Keras corresponden a las capas descritas. La comprobación histórica de paridad logits DirectML/Keras está marcada en los JSON; no se repitió sin datos/frameworks. Las cuatro evaluaciones A4 no incluyen el bloque `per_class`, pero se puede derivar íntegro desde los CSV. No hay evaluación limpia de train en esta familia: pérdida/exactitud de train se midieron durante actualizaciones con G y dropout.

### Diagnósticos MLP conservados por separado

| ID / ruta de evidencia | Condiciones y procedencia | Ejecución / checkpoint | Métricas de validación | Uso previsto |
|---|---|---|---|---|
| A4-D/mlp_lr1e3, `A/diagnosticos/mlp_lr1e3/runs/full_res112_s0_mlp_reference_directml/` | MLP 9,660,776, P/G, dropout 0.5, DirectMLAdam, WD 0, LR 0.001, seed 0, mismo dataset | 6 épocas / 1, `.pt` y `.keras` presentes | F1 0.0010574965; exactitud 5.81896552%; loss 4.55856692 en historial | Ensayo fallido de optimización. No hay CSV de predicciones para comprobar independientemente la declaración de una sola clase; conservar como diagnóstico documentado. |
| A4-D/mlp_lr1e4_interrumpido, `A/diagnosticos/mlp_lr1e4_interrumpido/` | MLP 9,660,776, config P/G, dropout 0.5 según implementación, Adam LR 0.0001, seed 0; mismo backend/datos | 25 épocas completas / 25 en PT; sin summary ni Keras final | Mejor F1 registrado 0.0215383396; exactitud 14.78987069%; loss 3.54749561 | `status.json`: interrupción en escritura de checkpoint, reinicio desde cero. No sumar sus épocas a las 87 de la referencia definitiva. Métricas documentales. |

## Familia F: TensorFlow/Keras, Metal según documentación histórica

Procedencia: `reports/final/README.md`, configuraciones F, resúmenes con rutas `/Users/angelluna/...`, scripts `run_final.py`/`analyze_final.py` y `tablas/avance4.tex`. El backend/hardware no está dentro de cada summary: TensorFlow/Metal se atribuye a esa documentación, no a una detección nueva ni a una prueba de ejecución. No hay manifiesto equivalente ni versiones completas por corrida. Checkpoints `models/final/.../*.keras` referidos en los resúmenes **no están en el checkout**.

Todas usan P y semilla 0. Comparten tope 150, paciencia 12, scheduler por val_loss factor 0.5/paciencia 2. LR mínimo 1e-5, excepto MLP 1e-6. Optimizador AdamW WD=0.01 excluyendo sesgos, dropout 0.5 y aumentos G salvo indicación. La brecha se mide sobre las primeras 3,712 imágenes fijas de train, sin G y con `training=False`, y sobre las 3,712 de val al final de la misma época. No es la totalidad de train ni una muestra estratificada; sus etiquetas no están guardadas, de modo que ese F1 de train solo se verifica documentalmente.

| ID (F/variante/run) | Arquitectura / parámetros | Cambios de configuración | Épocas / checkpoint | F1 val recalculado | Exactitud val (%) | Loss val en historial |
|---|---|---|---|---:|---:|---:|
| F/referencia/full_res112_s0_adamw_plateau | baseline / 480,936 | LR inicial 0.001 | 103 / 91 | 0.5783668706 | 64.06250000 | 1.48297236 |
| F/sin_regularizacion/full_res112_s0_adam_plateau | baseline / 480,936 | dropout 0; G desactivado; Adam, WD 0; LR 0.001 | 46 / 34 | 0.5332117946 | 59.64439655 | 2.03954392 |
| F/dropout_0.3/full_res112_s0_adamw_plateau | baseline / 480,936 | dropout 0.3; LR 0.001 | 73 / 61 | 0.5596392797 | 61.96120690 | 1.62719963 |
| F/lr_3e-4/full_res112_s0_adamw_plateau | baseline / 480,936 | LR inicial 0.0003 | 110 / 98 | 0.4605070457 | 54.17564655 | 1.78270933 |
| F/mlp/full_res112_s0_mlp_adamw_plateau | MLP / 9,660,776 | LR 0.0001 y mínimo 1e-6, además de arquitectura | 74 / 62 | 0.0177733755 | 14.35883621 | 3.58174774 |
| F/cnn_ampliada/full_res112_s0_deep_adamw_plateau | deep / 1,218,600 | arquitectura deep | 49 completas / 41 | 0.5663130969 | 63.76616379 | 1.57424094 |
| F/class_weight/full_res112_s0_adamw_plateau | baseline / 480,936 | pérdida ponderada `balanced`, LR 0.001 | 82 / 70 | 0.4880910468 | 48.86853448 | 1.94067015 |

Las siete son verificables desde predicciones/matrices; sus 104 entradas de precisión, recall, F1 y soporte también concuerdan con el recálculo. La loss de evaluación y la de historial se cotejan como evidencia histórica. La CNN ampliada se bloqueó durante la época 50: no declarar 50 épocas terminadas ni parada temprana normal; su tiempo de entrenamiento es estimado. Reinicios previos de `class_weight` se mencionan en el texto final, pero no tienen historiales separados versionados: no tratarlos como réplicas verificadas.

Los YAML `sin_dropout`, `sin_aumento`, `sin_weight_decay`, `l2`, `label_smoothing`, `res_160`, `efficientnetb0` existen, pero carecen de corridas completas y métricas en F. Son configuraciones propuestas, no resultados. `configs/directml_batchnorm.yaml` también implementa una alternativa sin evidencia de entrenamiento completo versionado. No son necesarios para resolver la rúbrica actual.

## Familia H: contexto A3 / destilación

Fuente: `H/config_avance3.yaml` (comentario de procedencia commit `e327742`), `H/runs/<run>/{summary.json,history.csv}`, `H/resultados.csv`, `reevaluacion.csv` y notebook A3. Arquitectura baseline 480,936 por diseño/configuración; summaries antiguos no guardan ese conteo. Backend TensorFlow por pipeline y documentación; rutas históricas macOS, sin dispositivo/versión por run. No afirmar hardware exacto a partir de la ruta. Checkpoints referidos en `models/` no están disponibles aquí.

P, dropout 0.5, Adam LR 0.001, batch 32, paciencia de 5 evaluaciones. Completo: G, máximo 50 y val cada época. Aleatorio/destilado: arrays normalizados [0,1] de caché uint8 a 112, DSA y máximo 300, val cada diez épocas (también primera/última). Aleatorio muestrea train por clase; DM optimiza píxeles con extractor aleatorio renovado, Adam LR de píxeles 0.01 durante 500 iteraciones. Destilación una vez por IPC, reutilizada entre semillas; tiempo de preparación separado. Train reducido: 104 imágenes (IPC1) o 1,040 (IPC10); val siempre 3,712 reales. Estos experimentos no son baseline MLP ni controles sin regularización del conjunto completo.

Cada ID es `H/<run>`; la semilla es el sufijo `s0/s1/s2`. Métricas siguientes son documentales de historial/resumen; `reevaluacion.csv` registra una reevaluación histórica, no repetida aquí.

| Run | Train / IPC | Semilla | Épocas / checkpoint | F1 val (summary) | Exactitud val (%) | Loss val (history) |
|---|---|---:|---|---:|---:|---:|
| full_res112_s0 | completo / — | 0 | 43 / 38 | 0.5538 | 61.6110 | 1.510179 |
| random_ipc1_res112_s0 | aleatorio / 1 | 0 | 280 / 230 | 0.0519 | 5.3341 | 14.949248 |
| random_ipc1_res112_s1 | aleatorio / 1 | 1 | 230 / 180 | 0.0483 | 5.4418 | 9.718845 |
| random_ipc1_res112_s2 | aleatorio / 1 | 2 | 300 / 270 | 0.0540 | 6.4925 | 16.282252 |
| random_ipc10_res112_s0 | aleatorio / 10 | 0 | 270 / 220 | 0.2641 | 26.8588 | 6.471684 |
| random_ipc10_res112_s1 | aleatorio / 10 | 1 | 300 / 280 | 0.2602 | 26.2392 | 8.259096 |
| random_ipc10_res112_s2 | aleatorio / 10 | 2 | 300 / 270 | 0.2483 | 24.4881 | 8.363102 |
| distilled_ipc1_res112_s0 | DM / 1 | 0 | 230 / 180 | 0.0625 | 9.9946 | 7.843553 |
| distilled_ipc1_res112_s1 | DM / 1 | 1 | 300 / 260 | 0.0662 | 8.8362 | 11.762148 |
| distilled_ipc1_res112_s2 | DM / 1 | 2 | 290 / 240 | 0.0728 | 10.7220 | 12.028281 |
| distilled_ipc10_res112_s0 | DM / 10 | 0 | 300 / 260 | 0.2170 | 21.0668 | 8.031603 |
| distilled_ipc10_res112_s1 | DM / 10 | 1 | 230 / 180 | 0.2188 | 22.1983 | 6.846629 |
| distilled_ipc10_res112_s2 | DM / 10 | 2 | 220 / 170 | 0.1773 | 18.3728 | 6.469477 |

`H/runs/distill_ipc{1,10}_res112/loss.csv` son pérdidas de preparación DM, sin checkpoint clasificador ni F1/exactitud de clasificación propios. No se inventan métricas para ellos. Las cuatro variantes de `H/hiperparametros/` son alias byte a byte de F/referencia, F/sin_regularizacion, F/lr_3e-4 y F/dropout_0.3 en summary/history: sus arquitecturas, parámetros, dataset, optimizador, épocas y métricas son las de F. No aumentan n ni cuentan como entrenamientos nuevos.

## Regularización: respuesta a la pregunta central

**Sí: los archivos almacenados permiten demostrar cuantitativamente el efecto observado de la intervención conjunta sin entrenar otra vez.** Se debe usar exclusivamente F/referencia frente a F/sin_regularizacion para esa tabla. No se debe completar con brechas de A4, aunque sus nombres sean similares.

| Variante histórica F | Checkpoint / épocas | F1 train limpio | F1 val | Exactitud train limpia (%) | Exactitud val (%) | Loss train limpia | Loss val | Brecha F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| referencia: dropout 0.5 + G + AdamW WD 0.01 | 91 / 103 | 0.81379643 | 0.57836687 | 83.00107759 | 64.06250000 | 0.55713650 | 1.48297236 | 0.23542956 |
| sin dropout, G ni WD; Adam | 34 / 46 | 0.87666447 | 0.53321179 | 87.33836207 | 59.64439655 | 0.44344508 | 2.03954392 | 0.34345267 |

Usando históricos sin redondear: F1 val mejora +0.04515508, exactitud +4.41810345 puntos porcentuales; la brecha F1 disminuye 0.10802312. Las cifras publicadas 0.5784/64.06%/0.2354 y 0.5332/59.64%/0.3435 son coherentes. `analyze_final.py` calcula brechas con F1 val redondeado del summary, introduciendo diferencias ~0.00003; para derivados nuevos conviene restar las dos columnas sin redondear del mismo history y redondear al mostrar. No corregir los archivos históricos para ocultar esa diferencia.

Comparabilidad: misma arquitectura/partición declarada/semilla/resolución/batch/LR inicial/regla de scheduler/paciencia/tope. Cambian conjuntamente dropout, aumentos y decaimiento/optimizador. Las trayectorias reales de LR y las épocas de selección difieren como resultado de la política adaptativa. No se aísla la contribución individual de dropout, G o WD, ni se mantiene un calendario numérico idéntico. Ambas usan parada temprana y reducción de LR: “sin regularización” significa sin esas tres intervenciones explícitas, no ausencia de todo mecanismo de selección o sesgo inductivo.

Limitación: train limpio solo está sustentado por los históricos y el código de inferencia; faltan predicciones de ese subconjunto y pesos finales locales para recalcularlo. Esto permite reutilización documental transparente, pero no declarar una nueva validación independiente de esa brecha. No hay significancia estadística con una semilla, y la validación se usó para elegir modelos.

## Comparaciones permitidas y límites

| Comparación | Uso válido | Condiciones que impiden una interpretación más fuerte |
|---|---|---|
| A4 MLP vs A4 CNN original | Baseline totalmente conectado real frente a CNN, mismas imágenes declaradas y backend. | LR, paciencia, capacidad y optimización difieren; MLP infraajusta/puede estar mal optimizada. No aislar causalidad arquitectónica. |
| A4 original vs A4 deep | Comparación principal de profundidad: Adam fijo, seed, batch, G, dropout y paciencia iguales. | Una semilla y capacidad distinta; distinta duración por parada temprana. |
| A4 original vs A4 regularizada | Intervención adicional de optimizador/WD/scheduler/paciencia. | Ambas usan G/dropout; no es comparación con/sin regularización. |
| F referencia vs F sin_regularizacion | Efecto conjunto observado y brecha limpia, tabla central de regularización. | Varias intervenciones, una semilla, train_eval no estratificado, sin pesos locales. |
| F referencia vs F dropout_0.3 | Variación de tasa dropout dentro de la misma política. | No equivale a quitar dropout; una semilla y scheduler adaptativo. |
| F referencia vs F MLP | Evidencia complementaria separada por procedencia. | También cambia LR inicial/mínimo; no mezclar con la MLP A4 de F1 0.0280. |
| F referencia vs F deep | Evidencia complementaria de profundidad. | Deep interrumpida, 49 épocas completas; no comparar sus tiempos como si fueran mediciones completas equivalentes. |
| A4 vs F / H full | Contexto histórico identificado. | Backend, políticas, inicialización/aleatoriedad, batch/paciencia según familia; no son repeticiones de la misma corrida ni brechas intercambiables. |
| H DM vs H random, mismo IPC | Experimento independiente de reducción de datos con tres semillas de clasificador. | No mide MLP vs CNN ni regularización con train completo; una destilación compartida, no tres destilaciones independientes. |

## Decisión sobre repetir entrenamientos

| Experimento / tarea | Decisión | Condición y razón |
|---|---|---|
| A4 CNN original, ampliada y AdamW+LR | No repetir | Hay predicciones, matrices, históricos y checkpoints coherentes. |
| MLP histórica A4, fallida e interrumpida | Conservar; no repetir por defecto | Repetir idéntico no resolvería el diagnóstico. Mantener el ensayo interrumpido fuera de comparación. |
| F referencia y sin_regularizacion | No repetir para cubrir regularización | Comparación y curvas ya disponibles; explicitar límites documentales del train limpio. |
| F resto de siete corridas | No repetir | Contexto complementario, fuera del mínimo requerido; class_weight no es necesario en el A4. |
| H completo/DM/random y copias hiperparámetros | No repetir | Son historia de entregas anteriores, no el foco de la corrección. |
| Métricas por clase A4 y matriz de confusión | Derivar de CSV | No requiere inferencia ni entrenamiento. |
| Brecha limpia de checkpoints A4 | Solo inferencia, si se necesita | Requiere restaurar entorno y TFRecords; definir subconjunto/IDs fijos. No inventar F1 train actual. |
| Nueva MLP razonablemente ajustada | Condicional, etapa 2 | Tras diagnóstico limpio y recursos; conservar histórico, cambiar una condición por variante y salida nueva. Si no es ejecutable, documentar causas e incertidumbre. |
| Ablaciones individuales G/dropout/WD | Opcionales, etapa 3 | No obligatorias para efecto conjunto solicitado. Solo tras validar opciones del backend y recursos, y autorización de etapa. No usar YAML existentes como resultados. |

No hay un entrenamiento que deba repetirse obligatoriamente con la evidencia actual. La única posible necesidad de una nueva corrida principal es una MLP mejor diagnosticada; aún no se decidió ni se ejecutó.

## Trabajo propuesto, secuencial y sujeto a revisión

### Etapa 2: diagnóstico de la MLP

Primero confirmar las conclusiones de esta auditoría con el usuario. Después, inspeccionar estado del entorno y preparar un flujo de análisis que separe carga de evidencia de entrenamiento. Resolver guardias de señales Windows y rutas sin sobrescribir fuentes históricas. Analizar las 21 clases predichas de la MLP A4, soportes, concentración en clases frecuentes, curvas/LR y comparación con los 11 tipos predichos por F/MLP; no confundir ambas.

Con datos/dependencias, medir MLP A4 en train limpio y val usando su checkpoint, revisar activaciones ReLU, logits, gradientes y normalización. Verificar primero que puede ajustar una muestra real de train; ese control no sustituye entrenamiento/evaluación del conjunto completo. Conservar P, particiones, seed y criterio de selección. Elegir un cambio justificado por el diagnóstico (LR/calendario o centrado de entrada, por ejemplo), registrarlo como variante distinta y limitar el presupuesto antes de entrenar. No prometer que la MLP se acerque a la CNN. Si cambia backend, identificarlo y no denominarlo réplica DirectML.

Salida esperada: explicación sustentada de optimización/infraajuste frente a sobreajuste, conteo de parámetros, curvas y métricas históricas o nuevas claramente separadas. Commit verificable y detenerse.

### Etapa 3: regularización

Revisar resultado de etapa 2. Recuperar F/referencia y F/sin_regularizacion en un análisis de solo lectura, producir derivados bajo entregable_4 con referencias/hashes a sus fuentes. Curvas de train limpio/val para pérdida, exactitud y F1, checkpoint y duración; tabla anterior calculada con precisión completa. Mantener aparte curva online. No modificar final ni convertir esa reutilización en “entrenamiento nuevo”. Una ablación adicional solo si aporta una pregunta concreta y hay backend/configuración validada. Commit y detenerse.

### Etapa 4: reporte

Revisar resultados previos; actualizar secciones 1–8 del A4 con los comentarios de la profesora y nombres de familia, tabla MLP/CNN, profundidad, tabla conjunta de regularización, métricas por clase, matriz, límites y enlace al notebook/repositorio. Derivar anexo completo de 104 clases del seleccionado A4 desde sus CSV. Mantener el seleccionado histórico dentro de la comparación A4; si se cambia la selección, justificar el conjunto de candidatos y su procedencia sin intercambiar métricas. Actualizar generadores para que no reviertan las correcciones. Commit y detenerse.

### Etapa 5: entrega

Revisar etapas previas; organizar notebook con modo evidencia que no requiera GPU ni entrene, y modo entrenamiento explícito. Evitar rutas absolutas y escritura a historiales originales. Ejecutar pruebas pertinentes con dependencias, compilar PDF y revisar todas sus páginas y referencias. Regenerar copias PDF/ipynb/ZIP desde fuentes canónicas verificando contenido; excluir residuos y modelos/datos nuevos grandes. Documentar comandos/versiones y cualquier ejecución bloqueada. Preparar PR hacia main sin fusionar, solo en esta etapa. No se publica PR en etapas 0–1.

## Bloqueos de ejecución y aceptación

Recursos detectados en la auditoría: RTX 4050 Laptop ~6 GiB, Intel UHD, i7-13620H, RAM ~15.64 GiB con ~2.51 GiB disponible al medir; no TFRecords ni frameworks de entrenamiento instalados en el Python activo ni credenciales Kaggle convencionales presentes. NVIDIA detectada por driver no implica TensorFlow GPU nativo Windows ni DirectML funcional. Elegir backend después de una prueba real de dispositivo, y registrar cualquier cambio de plataforma. Caché RAM completa deja poco margen; preferir disco cuando se autorice ejecución.

No se descargan datos ni se instalan dependencias durante estas etapas. Las pérdidas y brechas train históricas no podrán confirmarse mediante nueva inferencia sin recuperar datos y, para F, checkpoints externos. Si no se recuperan, reportar esos límites sin reemplazar resultados por pruebas sintéticas. No solicitar tokens personales ni incluirlos en Git.

## Comprobación de auditoría reproducible sin entrenar

Ejecutar desde la raíz con Python que tenga numpy/scikit-learn; este bloque no importa `src` ni modifica archivos. Reproduce la comprobación central hecha en etapas 0–1:

```python
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

roots = [Path("reports/entregable_4/avance4_2026-10-21/runs"),
         Path("reports/final/experimentos/runs")]
checked = 0
for root in roots:
    for path in sorted(root.rglob("validation_predictions.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        true = np.array([int(r["true_label"]) for r in rows])
        pred = np.array([int(r["predicted_label"]) for r in rows])
        run = path.parents[1]
        report = json.loads((run / "eval_report.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8-sig")))
        best = history[summary["best_epoch"] - 1]
        assert len(rows) == 3712 and len(set(true)) == 104
        assert len(history) == summary["epochs_run"]
        assert int(best["epoch"]) + 1 == summary["best_epoch"]
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, labels=range(104), average="macro", zero_division=0)
        for value, key in [(acc, "accuracy"), (f1, "f1_macro")]:
            assert abs(value - report[key]) <= 0.000051
        assert abs(f1 - summary["val_f1_macro"]) <= 0.000051
        assert abs(f1 - float(best["val_f1_macro"])) <= 0.000051
        counts = np.loadtxt(path.with_name("confusion_matrix.csv"), delimiter=",",
                            skiprows=1, usecols=range(1, 105), dtype=int)
        assert np.array_equal(counts, confusion_matrix(true, pred, labels=range(104)))
        checked += 1
        print(path, acc, f1)
assert checked == 11
```

Otros controles realizados: 29 summary/history coherentes, matrices normalizadas, métricas por clase de siete F, esquema nbformat válido y AST de Python válido. `pytest --collect-only` bloqueado por falta de TensorFlow; no se declara suite aprobada ni notebook reejecutado. PDF A4 inspeccionado textualmente y una página renderizada; compilación/revisión visual completa pendientes de etapa 5.

## Punto de detención

Etapas 0 y 1 concluyen con documentos de diagnóstico y planificación, sin cambios a experimentos, reporte, entregas anteriores o final. Revisar especialmente A01–A08 de la auditoría y las decisiones de reutilización anteriores. Esperar instrucción explícita del usuario antes de iniciar etapa 2; luego volver a detenerse al terminar cada etapa.
