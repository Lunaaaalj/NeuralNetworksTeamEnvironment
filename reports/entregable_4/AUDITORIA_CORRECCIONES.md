# Auditoría de correcciones del Avance 4 — etapa 0

Fecha: 26 de octubre de 2026. Repositorio: `Lunaaaalj/NeuralNetworksTeamEnvironment`.
Base inspeccionada: `main`, commit `c08f38796f47ed8744f6668c454da4af31bc4175`.
Rama de trabajo: `fix/avance4-retroalimentacion`.

## Alcance y criterio de verificación

Esta auditoría analiza evidencia preexistente; no ejecuta entrenamientos, no cambia métricas históricas ni reescribe el reporte. La calificación 75/100 y las observaciones proceden del enunciado del usuario; no se encontró una rúbrica original adjunta. El estado de `main` ya incorpora correcciones posteriores: no debe confundirse el documento evaluado con la versión actual.

Se distinguen tres niveles: (1) métrica recalculada desde etiquetas/predicciones guardadas; (2) dato consistente entre historial, resumen y configuración; (3) declaración histórica que requiere datos o pesos para confirmarse de nuevo. Recalcular F1 desde un CSV no verifica que el modelo haya producido esas predicciones sobre los TFRecords originales. La pérdida tampoco se recupera de etiquetas y clases predichas: necesita logits/probabilidades de la clase real.

## Estructura y flujo revisados

El inventario Git inicial contiene 460 archivos. No se encontraron instrucciones `AGENTS.md` en el checkout. Se revisaron las rutas solicitadas, los perfiles de `configs/`, README y guía, módulos auxiliares de datos/optimizadores/checkpoints, scripts de ejecución y empaquetado, y estructura/código de pruebas.

| Área | Contenido y relación con el Avance 4 |
|---|---|
| `reports/entregable_1/` | 11 archivos; propuesta de CNN a 224 px, batch 32, secciones de contexto/datos/algoritmos/metodología; no aporta una MLP entrenada. PDF de 6 páginas. |
| `reports/entregable_3/` | 70 archivos; CNN completa, 12 corridas aleatorias/DM con semillas 0–2, notebook de 19 celdas, curvas y cuatro configuraciones de hiperparámetros. PDF de 9 páginas. |
| `reports/entregable_4/` | 156 archivos antes de esta auditoría; fuentes de secciones 1–8, PDF de 13 páginas, evidencia fechada, modelos, diagnósticos y residuos de construcción/pruebas. |
| `notebooks/equipo_proyecto_avance4.ipynb` | 18 celdas, 9 de código con contadores 1–9 y sin errores guardados. Incluye comandos de entrenamiento desactivados y comprobación de predicciones. |
| `entregables/avance4/` | PDF, notebook, LEEME y ZIP de 166 entradas. Copias sueltas sincronizadas; notebook dentro del ZIP desactualizado respecto a ellas. |
| `reports/final/` | 112 archivos; siete corridas con historial/resumen/evaluación/predicciones/matrices, 14 YAML de experimentos y tablas/figuras. Fuente reutilizable de evidencia, sin modificarla. |
| `src/`, `scripts/`, `tests/` | Entrenamiento TensorFlow y DirectML, MLP específica, evaluación, DM, seguimiento W&B, generación de reportes y pruebas. |

Flujo reconstruido:

1. `scripts/download_data.py` autentica con Kaggle, selecciona TFRecords 224 px y verifica destinos de descarga/extracción. `download_data.sh` ofrece la vía shell. Los datos no están versionados.
2. `src/data.py` enumera shards ordenados por split, decodifica JPEG RGB, convierte a float32 y divide por 255; redimensiona a 112 px con interpolación bilineal y antialias. Caché antes de shuffle/aumentos; validación determinista. Aumentos: flip horizontal, rotación 0.05, zoom 0.1, contraste 0.1 y recorte a [0,1].
3. Ruta principal: train real 12,753 / val 3,712, sin nueva partición. Ruta DM independiente: caché por clase, optimización de imágenes sintéticas, entrenamiento con IPC 1/10 o control aleatorio. No son datos usados por las CNN principales del A4.
4. `src/train.py` selecciona backend. TensorFlow usa `build_model` y `ValidationMonitor`; DirectML usa `FlowerCNN`, inicialización Keras, puente NHWC→NCHW, Adam/AdamW y exportación `.pt`→`.keras`. `src/train_mlp_reference.py` conserva flatten en orden NHWC y su propia MLP reconstruida.
5. Checkpoint seleccionado por F1 macro de validación. `history.epoch` empieza en 0; `best_epoch` y metadatos del checkpoint empiezan en 1. `src/evaluate.py` y `scripts/prepare_avance4.py` producen evaluación y matrices. `build_avance4_report.py` escribe secciones/notebook; `package_avance4.py` crea copias y ZIP. No se ejecutaron esos generadores en esta auditoría.
6. El reporte final usa otra familia: TensorFlow/Metal según documentación, AdamW y LR adaptable, con evaluación limpia de las primeras 3,712 imágenes de train al final de cada época. Esa evaluación es distinta de la métrica online con dropout/aumentos del A4.

## Comparación entre entregas y retroalimentación

El A3 actual ya es un trabajo de CNN/DM: no contiene una MLP histórica identificada. Sus cuatro carpetas `hiperparametros/{referencia,sin_regularizacion,lr_3e-4,dropout_0.3}` tienen `summary.json` e `history.csv` idénticos byte a byte a los del reporte final. Se añadieron al A3 en el commit `c9ee366` del 25 de octubre, después del A4 del 21: no son cuatro repeticiones independientes ni evidencia cronológicamente anterior por estar en esa carpeta.

| Observación | Estado en la base auditada | Pendiente |
|---|---|---|
| Documento identificado como Avance 3 | Corregido: portada y pie dicen Avance 4; secciones 1–8 presentes. | Conservar identificación al recompilar. |
| Algoritmos reducidos | Hay subsecciones CNN, métodos tradicionales, alternativas profundas y aplicaciones. Metodología explica filtros compartidos y MLP. | Desarrollar operaciones y justificación MLP/CNN/pooling; evitar superlativos sin evidencia. |
| Notebook/repositorio ausentes | Notebook versionado y copia de entrega presentes. | Hacer explícito el enlace/ruta de reproducción en el reporte; reparar portabilidad y ZIP. |
| Comparación sin regularización | A4 compara Adam con AdamW+LR; ambos ya usan dropout y aumentos. | Incorporar evidencia final con y sin intervención conjunta en una tabla identificada. |
| Tabla F1/exactitud/brecha | No hay tabla con/sin y brecha F1 limpia en el A4. | Recuperarla de los históricos finales; no usar brecha online como equivalente. |
| MLP frente a CNN | Ya existe MLP reconstruida a LR 1e-4 y comparación explícita. | Diagnóstico de ajuste/optimización de la MLP; no presentarla como MLP del A3. |
| F1 CNN ampliada / épocas | Tabla actual: F1 0.5502, checkpoint 46; historial 51 épocas. | Mantener total y selección explícitos; distinguir la CNN ampliada del final (41/49). |
| Métricas por clase | A4 tiene matrices/predicciones y peores clases, pero no `per_class` completo ni tabla/anexo. | Derivar precisión, recall, F1 y soporte de los CSV, sin entrenamiento. |
| Rendimiento inferior CNN profunda | Se advierte mayor capacidad y posible sobreajuste. | Vincular la explicación a mínimos de pérdida, curvas y protocolo; formular causas como hipótesis. |

## Hallazgos priorizados

P1: afecta directamente la rúbrica o impide reproducción; P2: trazabilidad/entrega; P3: mantenimiento.

| ID / prioridad | Problema y criterio afectado | Evidencia concreta / archivos implicados | Solución propuesta | ¿Nuevo entrenamiento? |
|---|---|---|---|---|
| A01 / P1 | Falta control sin regularización y tabla solicitada (regularización 25 puntos). | `secciones/07_regularizacion.tex`: todos usan dropout 0.5 y aumentos; compara 0.5740 con 0.5745. `reports/final/experimentos/runs/{referencia,sin_regularizacion}/` sí contiene el control. | Usar el par final como evidencia histórica adicional, con nombre/backend propios, curvas y brecha limpia. | No para el efecto conjunto. |
| A02 / P1 | Brecha de entrenamiento del A4 no comparable con validación. | `src/train_directml.py::train_epoch` activa `.train()` y dataset aumentado; `prepare_avance4.py::plot_results` resta esa exactitud a la de inferencia. No hay `train_f1_macro` limpio en esas cuatro corridas. | Etiquetar métrica online; usar `train_eval_*`/`train_f1_macro` del par final. Re-evaluar checkpoints A4 solo si se requiere su brecha propia. | No; inferencia requiere datos y dependencias. |
| A03 / P1 | Bajo rendimiento MLP aún no explica qué falló (baseline 25 puntos). | `runs/mlp_referencia/.../history.csv`: al checkpoint train online 12.06%, val 15.52%, F1 0.0280; predice 21 clases. Final MLP distinta: F1 0.0178 y 11 clases. Diagnóstico LR 1e-3 colapsado; intento LR 1e-4 interrumpido. | Conservar ambas procedencias; revisar activaciones, distribución de predicción, LR, normalización y ajuste limpio antes de nuevas variantes. | Condicional, después del diagnóstico y autorización etapa 2. |
| A04 / P1 | Métricas por clase incompletas en la entrega (CNN/resultados). | Cuatro `eval_report.json` A4 carecen de `per_class`; predicciones y matrices 104×104 sí existen. `src/evaluate.py` actual ya calcula esas métricas, pero `prepare_avance4.py::evaluate_run` devuelve el JSON antiguo si existe. | Generar tabla/anexo desde predicciones y corregir regeneración para no omitir campos por caché antigua. | No. |
| A05 / P1 | Rutas absolutas y anteriores a reorganización rompen reproducción. | YAML/resúmenes A4 apuntan a `C:\Users\andre\...\reports\avance4_2026-10-21`; notebook actual apunta correctamente a `reports/entregable_4/...`. Final refiere `/Users/angelluna/...`. `mlp_reference` no es un nombre aceptado por `src.model.model_architecture`. | Resolver rutas relativas con mapeo explícito y mantener originales como procedencia; ejecutar MLP histórica mediante su módulo específico. | No. |
| A06 / P1 | CLI no portable en Windows. | `src/train.py::main` usa `signal.SIGUSR1` sin guardia; `scripts/run_final.py::run_watched` también lo usa. Python Windows carece de esa señal. Se deduce por revisión estática; no se lanzó entrenamiento. | Guardar registro/envío de señales por plataforma y comprobar CLI sin entrenar antes de futuras ejecuciones. | No. |
| A07 / P1 | DirectML no implementa controles generales del final. | `FlowerCNN.dropout` fijo en 0.5; `run` construye `build_cnn` sin dropout/L2 de config; no registra train limpio ni admite MLP/transfer como TensorFlow. `FlowerMLP` histórico también fija dropout/aumentos/caché. | No ejecutar una ablación YAML con DirectML creyendo que desactiva dropout; implementar/validar opciones o usar TensorFlow compatible. | No para reparar; solo condicional para una ablación nueva. |
| A08 / P2 | Riesgo de mezclar familias y contar copias como réplicas. | F1 A4 regularizada 0.5745 (DirectML, 61/73) frente a final referencia 0.5784 (TensorFlow/Metal, 91/103); A3 hiperparámetros duplica final. | Identificadores con familia+config+run, backend y epoch; nunca combinar F1 de una con brecha de otra. | No. |
| A09 / P2 | ZIP desincronizado. | PDF suelto A4 y PDF entrega SHA-256 idéntico; notebook suelto y entrega idénticos. Notebook dentro del ZIP difiere y contiene `reports/avance4_2026-10-21`. ZIP íntegro estructuralmente, 166 entradas. | Regenerar copias desde una fuente canónica al final, verificando hashes de contenido y rutas internas. | No. |
| A10 / P2 | PDF anterior a las rutas actuales. | PDF A4 p. 6 cita `reports/avance4_2026-10-21`; fuente actual `06_protocolo.tex` cita `reports/entregable_4/avance4_2026-10-21`. | Recompilar en etapa 5 y verificar correspondencia PDF/fuentes/notebook. | No. |
| A11 / P2 | Explicación de CNN profunda aún general y final interrumpida. | A4 deep: mínimo val_loss 1.658 en época 31, checkpoint F1 46 y pérdida final 2.024 en 51; final deep: 49 épocas completas, bloqueo durante 50, tiempo 1,029 s estimado a 21 s/época según summary. | Analizar A4 como comparación principal de profundidad; marcar interrupción/tiempo estimado si se cita final. No sustituir resultados entre ellas. | No. |
| A12 / P2 | Procedencia no basta para reproducción exacta. | Manifest A4 registra commit `857e683...`, versiones y hashes de 32 shards train/val, pero código actual difiere de varios hashes (otros solo por CRLF/LF); final no tiene manifiesto equivalente y sus checkpoints no están versionados. CSV usan sample_index, sin ID estable de imagen. | Preservar manifiesto histórico; registrar commit/config/versiones/hashes de derivados y aclarar límites de verificar imágenes/orden. | No; datos para verificar hashes o inferencia. |
| A13 / P2 | Generadores pueden sobrescribir evidencia o contenido corregido. | `build_avance4_report.py` reescribe 01,07,08 y notebook; `run_final` genera YAML antes de omitir corridas; `src.train` elimina history previo; `prepare_avance4` reutiliza eval existente. Nombres de corrida no codifican todos los hiperparámetros. | Separar salida de reproducción, no ejecutar generadores globales sobre final; actualizar generación junto al reporte cuando se autorice. | No. |
| A14 / P2 | Dependencias no fijadas y entorno ausente. | `requirements.txt` Windows permite TF>=2.16, numpy sin tope; DirectML exige numpy<2 y torch-directml fijo; manifest A4 guarda otras versiones concretas. No hay entorno instalado de entrenamiento. | Entorno aislado compatible, registrar versiones; no instalar/entrenar automáticamente en esta etapa. | No. |
| A15 / P3 | Residuos y duplicados innecesarios. | A4 incluye `pytest-temp/`, `pytest-atomic-temp/`, `jupyter-kernels/avance4/kernel.json`, `latex-build/`; hay `.DS_Store`, copias PNG/PDF intencionales, ZIP y modelos ya versionados. Final tiene `main.*-SAVE-ERROR`. | No borrar en esta fase. Excluir residuos de futuros paquetes; PNG/PDF son formatos deliberados. No agregar checkpoints grandes. | No. |
| A16 / P2 | Alcance de las conclusiones requiere cautela. | Final secciones 07/08 atribuye causas fuertes a regularización/arquitectura con una semilla; cabecera de `run_final` dice que cada experimento cambia una cosa, pero `sin_regularizacion` cambia varias y MLP también LR/min_lr. | En A4 describir intervención conjunta y observación de una semilla; no afirmar significancia ni causalidad individual. Conservar final intacto. | No. |

## Verificaciones realizadas y límites

Se recalcularon exactitud y F1 macro sobre las 104 clases para 11 CSV de predicciones (cuatro A4 y siete final), cada uno con 3,712 filas. Coinciden con resumen/evaluación a tolerancia de redondeo 0.000051. Se reconstruyeron las once matrices de conteos y se comprobaron sus normalizaciones. Las secuencias de etiquetas verdaderas son idénticas entre las once corridas; esto apoya la consistencia del orden, pero no demuestra identidad de imágenes sin IDs/datos. Las 104 métricas por clase de las siete evaluaciones finales también coinciden.

Se comprobaron 29 pares summary/history: número de filas, máximo F1 y checkpoint seleccionado coinciden; incluyen cuatro duplicados A3/final. Las cuatro filas de `avance4_2026-10-21/resultados.csv` y siete de `reports/final/resultados.csv` concuerdan con sus fuentes. Las pérdidas se cotejaron documentalmente, sin nueva inferencia. Las cifras iniciales aportadas por el usuario quedan verificadas como resultados históricos, no obtenidos entrenando en esta tarea.

Se inspeccionaron contenedores Keras (configuración/metadatos) y PT (metadatos con `pickletools`, sin deserializar/ejecutar pickle). Los checkpoints A4 declaran épocas 77, 53, 46 y 61, coherentes con los resúmenes; capas Keras confirman MLP sin convoluciones y CNN de 4/6 convoluciones, dropout 0.5. No se cargaron tensores ni se repitió la paridad de logits: `keras_export_verified=true` sigue siendo una comprobación histórica.

Los dos notebooks solicitados validan contra el esquema nbformat y no tienen errores almacenados. No se ejecutaron: requieren frameworks incluso con `RUN_TRAINING=False`. Se analizó sintaxis de los Python de src/scripts/tests con AST, sin importarlos. `python -m pytest --collect-only -q` falló al cargar `tests/conftest.py` por `ModuleNotFoundError: tensorflow`; no se ejecutó ninguna prueba ni entrenamiento sintético. La suite cubre datos/DM, métricas, arquitecturas, exportación, checkpoint atómico, LR, tracking y entrenamiento mínimo; queda pendiente ejecutarla con dependencias.

Se extrajo texto de todas las páginas de PDFs A1/A3/A4 y `references/main-39.pdf`; no se encontraron `??` en esa extracción. La referencia PDF es contenido de propuesta, no la rúbrica de la profesora. Se renderizó e inspeccionó p. 8 del A4: tabla y curvas legibles; no constituye revisión visual completa de las 13 páginas ni compilación nueva. Figuras citadas de metodología/resultados existen en disco. Revisión completa de enlaces/figuras/tablas y compilación quedan para etapa 5.

## Recursos observados

En el Python 3.12 del sistema: numpy, pandas, scikit-learn, PyYAML, pytest y nbformat disponibles; TensorFlow, Keras, torch y torch_directml ausentes. Runtime PDF separado disponible con pypdf/Poppler; no se encontró tectonic/latexmk/pdflatex en PATH. No se alteraron dependencias.

No hay `.tfrec` dentro del checkout ni archivos de datos en la ruta relativa configurada. No se encontró `~/.kaggle/kaggle.json`, `KAGGLE_KEY` ni `KAGGLE_API_TOKEN` en este proceso (solo se comprobó presencia, sin leer valores). No se descargaron datos ni se buscaron credenciales fuera de esas ubicaciones convencionales. Una copia externa no declarada sigue sin verificar.

Hardware: Intel Core i7-13620H, 16 procesadores lógicos; GPU NVIDIA RTX 4050 Laptop de 6,141 MiB según `nvidia-smi`, driver 616.92; también Intel UHD. Memoria RAM total ~15.64 GiB, disponible ~2.51 GiB en la consulta; disco C libre ~656 GiB. El dato CIM de VRAM se evitó como capacidad definitiva por su límite de representación. DirectML/TF no se probaron porque faltan paquetes. La GPU histórica RX 6700S no es el hardware actual; la disponibilidad de NVIDIA no demuestra un backend funcional instalado.

La caché float32 de train+val a 112 px requiere ~2.31 GiB solo en imágenes, además de etiquetas, prefetch, frameworks, pesos, activaciones y optimizador. La MLP tiene ~36.85 MiB de pesos float32 y necesita memoria adicional para gradientes/estados Adam. La RAM libre es ajustada: preferir caché de disco y comprobación real de memoria antes de futuros entrenamientos, no prometer tiempos/capacidad con el estado actual.

## Cierre de etapa 0

El mayor ahorro posible es recuperar la comparación de regularización existente, y derivar métricas por clase de predicciones guardadas. No hay evidencia de que sea necesario repetir las CNN para satisfacer la rúbrica. La necesidad de una MLP nueva depende del diagnóstico de la etapa 2. El registro y las decisiones detalladas se presentan en `PLAN_EXPERIMENTAL.md` (etapa 1). Solo se agregan documentos; las entregas anteriores, reporte final, notebooks, código, resultados y checkpoints quedan sin cambios.
