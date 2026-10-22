# Guía de uso

## 1. Entorno

### Local (Mac con Apple Silicon)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import src, tensorflow as tf; print(tf.config.list_physical_devices())"   # debe listar GPU:0
```

`requirements.txt` fija `tensorflow~=2.18.0` + `tensorflow-metal==1.2.0` en macOS. Es la única combinación que funciona: con TF 2.20 el plugin de Metal no carga.

### Kaggle

Abre `notebooks/kaggle_runner.ipynb` en Kaggle, agrega el dataset de la competencia *tpu-getting-started*, activa Internet y elige una GPU (T4/P100). El notebook clona este repo y corre todo con `configs/kaggle.yaml`. No instales `requirements.txt` en Kaggle, porque TensorFlow ya viene preinstalado.

### Windows con GPU AMD / DirectML

Para la Radeon RX 6700S, usa PyTorch con DirectML en Windows. CUDA corresponde
a NVIDIA; instalar CUDA no acelera esta Radeon. El plugin TensorFlow-DirectML
antiguo está descontinuado y requiere versiones anteriores de Python/TensorFlow.
Referencias: [Microsoft DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)
y [estado del plugin TensorFlow](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin).

Desde CMD, en la raíz del repositorio:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-directml.txt
.venv\Scripts\python.exe -m src.train --config configs/local.yaml --data full --epochs 1 --backend directml
```

Al iniciar se enumeran las GPUs y se imprime `Training on GPU: AMD Radeon RX 6700S`.
La GPU Radeon RX dedicada se selecciona automáticamente; puedes elegir otro
adaptador con `--device-index N`. Si DirectML no puede usar la GPU, el entrenamiento
falla con un mensaje en lugar de seguir silenciosamente en CPU.

Para el entrenamiento completo con gráficas de W&B:

```bat
.venv\Scripts\python.exe -m src.train --config configs/local.yaml --data full --epochs 50 --backend directml --wandb
```

La arquitectura, resolución, normalización y aumentos de datos son los del config
y de la CNN original. TensorFlow se utiliza para leer y aumentar imágenes en CPU;
PyTorch ejecuta la CNN y sus gradientes en la GPU mediante DirectML. Las secuencias
aleatorias y la implementación de Adam difieren entre frameworks, por lo que las
curvas no son idénticas a las de TensorFlow. Registra este backend en el reporte.
El backend también puede entrenar `random` y conjuntos `distilled` ya generados;
la destilación de `src.distill` continúa usando TensorFlow.

Con DirectML puede seguir apareciendo el aviso de TensorFlow sin GPU: corresponde
al lector de datos en CPU. La línea `Training on GPU: ... (privateuseone:0)` indica
el dispositivo que ejecuta la CNN. `DirectMLAdam` utiliza las ecuaciones estándar
de Adam con multiplicación y suma para evitar el fallback a CPU de `aten::lerp`.
Las métricas de entrenamiento se acumulan en GPU y se copian a CPU al terminar
la época; sus cálculos incluyen correctamente el último lote incompleto.

La entrada de DirectML usa DLPack para compartir los tensores de TensorFlow con
PyTorch sin pasar por `.numpy()`. Conserva la decodificación, normalización,
resolución, mezclado y aumentos originales. Todavía es necesario preparar NCHW
y transferir cada lote a la GPU. `runtime.input_pipeline` permite comparar
`numpy` (puente anterior), `dlpack` y `dlpack_prefetch` (prepara un lote adicional
en CPU mientras trabaja la GPU; la cola está limitada para no duplicar la caché).

Durante el entrenamiento, cada mejora del F1 se guarda como `.pt` con pesos en
CPU, época, métricas y resolución. El reemplazo del checkpoint es atómico.
Al finalizar o al interrumpir normalmente con Ctrl+C, se exporta una sola vez
el mejor checkpoint a `.keras`, compatible con `src.evaluate`. Una terminación
forzada puede impedir esa exportación; el `.pt` ya guardado se puede convertir:

```bat
.venv\Scripts\python.exe -m src.directml_checkpoint models/directml_fast/full_res112_s0_directml.pt
```

El checkpoint contiene los mejores pesos, no el estado necesario para reanudar
Adam. `summary.json` registra `checkpoint_path`, `checkpoint_seconds`,
`export_seconds` e `input_pipeline`; W&B conserva sus gráficas por época.
Antes de estas optimizaciones, en una prueba de una época con 12,753 imágenes, batch 32 y resolución 112,
la RX 6700S tardó 20.7 s incluyendo validación (16.7 s de entrenamiento), frente
a los 80 s de la corrida anterior de TensorFlow/CPU. Son mediciones de corridas
individuales con frameworks distintos, no una garantía de tiempo por época.

Para medir el baseline anterior y las variantes optimizadas con batches 32/64/128:

```bat
.venv\Scripts\python.exe -m scripts.benchmark_directml --config configs/local.yaml --epochs 3
```

Cada variante usa una época de calentamiento y dos medidas. El resultado compara
la mediana de entrenamiento más validación, sin exportación de modelos ni W&B.
Se guarda en `reports/runs/directml_benchmark/FECHA/results.json`, sin escribir
los checkpoints ni historiales de tus corridas. Ejecuta la medición con la GPU
libre de otros entrenamientos. Las corridas cortas comparan velocidad; la calidad
de cada batch debe comprobarse con entrenamientos completos y F1 macro.

`data.cache` acepta `false` (sin caché), `true`/`ram` (RAM) o `disk` (disco).
La caché guarda imágenes decodificadas antes de mezclar y aumentar los datos,
conservando los aumentos aleatorios de cada época. A resolución 112, train y val
requieren aproximadamente 2.5 GB de RAM o disco, según el modo elegido.
La opción `disk` guarda archivos en `data/processed/tfdata_cache/` y los reutiliza
entre ejecuciones. Su huella incluye rutas, tamaño/fecha de los TFRecords,
resolución, versión del preprocesamiento y TensorFlow. Si estos cambian se genera
otra caché. Las cachés antiguas permanecen en disco hasta que las elimines.

La primera pasada completa crea la caché de disco; las siguientes evitan repetir
la decodificación y el redimensionamiento. El mezclado y los aumentos se vuelven
a ejecutar. Es el comportamiento de [`Dataset.cache`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache).
Para comparar los modos con el batch de tu configuración:

```bat
.venv\Scripts\python.exe -m scripts.benchmark_directml --config configs/directml_fast.yaml --compare-cache --epochs 3
```

Se compara sin caché contra disco y, si hay RAM suficiente, contra RAM. La primera
época se reporta separada como calentamiento y las dos siguientes determinan la
mediana. El benchmark verifica espacio disponible antes de probar disco, y deja
un margen de RAM antes de probar RAM; `--no-cache` permite excluir el modo RAM.

El perfil `configs/directml_fast.yaml` usa batch 64, lr 0.001 y caché en RAM,
con hasta 150 épocas y parada temprana. Guarda modelos y corridas en subcarpetas
`directml_fast`, separadas de los entrenamientos anteriores. Para usarlo:

```bat
.venv\Scripts\python.exe -m src.train --config configs/directml_fast.yaml --data full --wandb
```

Para comparar los tres puentes de entrada con el mismo batch y caché, y medir
por separado el guardado anterior frente al checkpoint de PyTorch:

```bat
.venv\Scripts\python.exe -m scripts.benchmark_directml --config configs/directml_fast.yaml --compare-input --epochs 4
```

La primera época se excluye y se toma la mediana de las tres siguientes. Los
archivos de la comparación de guardados quedan en la carpeta del benchmark,
sin sobrescribir los checkpoints del entrenamiento. La exportación final se
mide aparte; los tiempos por época de esta comparación excluyen guardado y W&B.

En la RX 6700S, con batch 64, resolución 112 y caché en RAM, se midieron 8.58
s/época con NumPy, 8.42 con DLPack y 8.40 con DLPack y preparación anticipada
(medianas de tres épocas después del calentamiento). El perfil rápido usa
`input_pipeline: dlpack_prefetch`; su mejora frente a NumPy fue aproximadamente
2%, y la diferencia entre las dos variantes DLPack fue muy pequeña. Guardar una
mejora tardó 27.4 ms con la exportación Keras anterior frente a 5.4 ms con `.pt`;
la exportación Keras final tardó 69.8 ms. El ahorro de guardado es pequeño en
tiempo absoluto y solo corresponde a las épocas que mejoran el F1. El reporte
está en `reports/runs/directml_fast/directml_benchmark/20261018T034302639111Z/results.json`.

La medición local en RX 6700S dio estas medianas de dos épocas después del calentamiento:

| Variante | Batch | Segundos/época |
|---|---:|---:|
| Adam estándar + métricas por lote | 32 | 16.23 |
| DirectMLAdam + métricas por época | 32 | 10.95 |
| DirectMLAdam + métricas por época | 64 | 10.40 |
| DirectMLAdam + métricas por época | 128 | 10.68 |

Se eligió 64: 64 y 128 dieron tiempos cercanos, y 64 usa menos memoria y conserva
más actualizaciones por época. Estos tiempos incluyen validación, excluyen exportar
checkpoints y W&B, y pueden variar con la carga y temperatura del equipo. No se
activó caché en esa comparación porque el equipo no tenía suficiente RAM libre. Cambiar el batch puede
cambiar la convergencia y el F1; el benchmark corto no establece la calidad final.

En una comparación posterior con batch 64, sin caché se midieron 10.07 s/época y
con caché en SSD 9.58 s/época (medianas de dos épocas después del calentamiento),
aproximadamente un 5% menos.
Se verificaron 12,753 imágenes y etiquetas de train y 3,712 de val en los archivos
persistentes, que ocupan aproximadamente 2.48 GB en total. Se conserva float32 y
el mismo redimensionamiento; no se cuantizan las imágenes ni se congelan los aumentos.

Después de liberar RAM y cerrar la otra ventana de VS Code, quedaron 5.26 GiB
disponibles antes de cargar TensorFlow y PyTorch. Una nueva comparación con batch
64 midió 9.81 s/época sin caché, 9.14 s con disco y 8.92 s con RAM. Son medianas
de dos épocas después del calentamiento, con entrenamiento y validación; la
ventaja de RAM frente a disco es pequeña (aproximadamente 2%). El perfil rápido
queda con `cache: ram`; la caché se vuelve a llenar en la primera época de cada
ejecución. Mantén cerrada la otra ventana mientras entrenas; si necesitas abrir
otras aplicaciones pesadas, cambia a `cache: disk` para dejar más RAM disponible.
Los resultados están en
`reports/runs/directml_fast/directml_benchmark/20261018T032901446546Z/results.json`.

Se guardan `history.csv` y `summary.json` en `reports/runs/full_res112_s0_directml/`,
y el mejor checkpoint en `models/full_res112_s0_directml.keras`, compatible con:

```bat
.venv\Scripts\python.exe -m src.evaluate --config configs/local.yaml --model models/full_res112_s0_directml.keras
```

Para usar DirectML también en `scripts.run_experiments`, agrega
`runtime: {backend: directml}` al YAML de tus experimentos y usa un archivo de
resultados nuevo para conservar la información del backend. Sin `--backend`
ni esa configuración se mantiene el entrenamiento original de TensorFlow.

### Variante de seis convoluciones

`configs/directml_deep.yaml` selecciona `model.architecture: deep`. La CNN
original de cuatro convoluciones sigue siendo el valor por defecto (`baseline`).
La variante añade una Conv 3x3 de 128 filtros después de pool3 y otra de 256
después de pool4, ambas con ReLU y padding same. Conserva los cuatro poolings,
GAP, Dense 256, Dropout 0.5, los 104 logits y las imágenes a 112x112. La nueva
arquitectura tiene 1,218,600 parámetros; la original, 480,936. El extractor usado
para destilar datos con DM conserva la arquitectura original.

| Etapa | Original | Deep |
|---|---|---|
| 1 | Conv32 → ReLU → Pool | Igual |
| 2 | Conv64 → ReLU → Pool | Igual |
| 3 | Conv128 → ReLU → Pool | Conv128 → ReLU → Pool → **Conv128 → ReLU** |
| 4 | Conv256 → ReLU → Pool | Conv256 → ReLU → Pool → **Conv256 → ReLU** |
| Clasificador | GAP → Dense256/ReLU → Dropout → 104 logits | Igual |

Para entrenarla con DirectML y gráficas de W&B:

```bat
.venv\Scripts\python.exe -m src.train --config configs/directml_deep.yaml --data full --wandb
```

Usa hasta 150 épocas, batch 64, lr 0.001, caché RAM y la misma parada temprana
de cinco evaluaciones. El nombre es `full_res112_s0_deep_directml`; los modelos
quedan en `models/directml_deep/` y los historiales en
`reports/runs/directml_deep/`. W&B registra la arquitectura `flower_cnn_deep`
y la etiqueta `deep`. El `.pt` guarda la arquitectura y la exportación final
conserva las seis convoluciones; los `.pt` anteriores sin ese campo se siguen
leyendo como la arquitectura original.

Una prueba de tres épocas sobre las 12,753 imágenes en RX 6700S completó la
validación y la exportación, con 11.3 s en la primera época y 9.0–9.1 s en las
dos siguientes. Se comprobaron las salidas de los pesos `.pt` frente al `.keras`
y el registro offline de W&B. Es una prueba de funcionamiento, no una medida
de calidad convergida ni una comparación controlada de velocidad. Sus archivos
están en `reports/runs/directml_deep/smoke_20261018T045028Z/`.

La corrida original de DirectML compartida antes de añadir capas terminó en la
época 74 y eligió la 69 por F1 macro. Su resumen está en
`reports/runs/directml_fast/full_res112_s0_directml/summary.json`.
Comparada con la corrida full del reporte proporcionado:

| Medida | Reporte | Corrida original en DirectML |
|---|---:|---:|
| F1 macro de validación | 0.5538 | 0.6051 |
| Accuracy del checkpoint elegido | 61.61% | 63.82% |
| Mejor época | 38 | 69 |
| Épocas ejecutadas | 43 | 74 |
| Batch | 32 | 64 |
| Épocas máximas | 50 | 150 |
| Tiempo de entrenamiento, sin validación | 581.5 s | 577.6 s |

La mejora observada es 0.0513 de F1 macro y 2.21 puntos porcentuales de accuracy.
Ambas corridas usan resolución 112 y el mismo conjunto de validación de 3,712
imágenes, pero cambian el batch, el presupuesto de épocas y el backend. Una sola
corrida por configuración no permite atribuir esa mejora a una optimización
concreta ni establecer superioridad estadística. Estos números corresponden a
la CNN original; el desempeño de deep debe medirse con un entrenamiento completo.
Más capas aumentan el cálculo y su beneficio en validación no está garantizado.

### Experimento con AdamW y learning rate adaptativo

`configs/directml_regularized.yaml` vuelve a la CNN original de cuatro
convoluciones y 480,936 parámetros. Entrena desde cero con resolución 112,
batch 64 y los mismos aumentos. Cambia la política de optimización:

- AdamW con weight decay de 0.01 sobre matrices de pesos; los sesgos quedan excluidos.
- Learning rate inicial de 0.001, multiplicado por 0.5 al acumular tres evaluaciones
  sin una mejora de al menos 0.0001 en val_loss (patience del scheduler = 2).
- Learning rate mínimo de 0.00001. El mejor checkpoint sigue elegido por F1 macro.
- Máximo de 150 épocas y parada tras 12 evaluaciones sin mejorar el F1, para dar
  tiempo a las reducciones del learning rate.

```bat
.venv\Scripts\python.exe -m src.train --config configs/directml_regularized.yaml --data full --wandb
```

El nombre de corrida es `full_res112_s0_adamw_plateau_directml`. Los checkpoints
se guardan en `models/directml_regularized/`; los historiales, en
`reports/runs/directml_regularized/`. El manifiesto de la corrida local está en
`reports/runs/directml_regularized/experiment_manifest.json` y conserva la
configuración y las huellas de los checkpoints de referencia.

`learning_rate` registra el valor usado en una época y `next_learning_rate` el
que se usará en la siguiente. Ambos están en CSV y W&B. La pérdida de validación
controla el scheduler; el F1 controla el guardado y la parada temprana. En conjuntos
pequeños, las épocas sin validación no consumen paciencia del scheduler.

`DirectMLAdamW` conserva el decay separado de los momentos de Adam y evita
el operador lerp que causa fallback a CPU en DirectML. Sus actualizaciones se
comparan con `torch.optim.AdamW`, incluyendo cambios de LR, gradientes ausentes
y restauración del estado. TensorFlow admite la misma configuración con AdamW
y un callback que ajusta el LR únicamente después de evaluaciones válidas.

Este experimento cambia conjuntamente el optimizador, la regularización, el
scheduler y la paciencia. Su comparación mide esa estrategia completa; una sola
corrida no permite atribuir un cambio del F1 exclusivamente a AdamW.

La corrida local de semilla 0 terminó tras 69 épocas y eligió la época 57.
Obtuvo F1 macro de 0.5855 y accuracy de 64.36% (2,389 aciertos de 3,712 imágenes
de validación), confirmados al evaluar el `.keras` exportado. Frente a la CNN
original (F1 0.6051, accuracy 63.82%), aumentó la accuracy en 0.54 puntos
porcentuales pero bajó el F1 en 0.0196. Por la métrica principal del proyecto,
la CNN original continúa siendo el mejor de estos tres checkpoints.

La primera reducción de LR ocurrió al terminar la época 46 y la última llegó
a 0.00001 al terminar la 67. El tiempo de entrenamiento sin validación fue de
540.2 s. Los checkpoints originales y profundos conservaron sus huellas.
La corrida y las gráficas están en
[W&B](https://forge.coreweave.com/wandb/a00838626-tecnol-gico-de-monterrey/flower-cnn/runs/qh3nx933).

Para regenerar la comparación y la matriz:

```bat
.venv\Scripts\python.exe -m scripts.compare_training
.venv\Scripts\python.exe -m src.evaluate --config configs/directml_regularized.yaml --model models/directml_regularized/full_res112_s0_adamw_plateau_directml.keras --confusion-matrix
```

La comparación en CSV/JSON está en `reports/runs/directml_regularized/` y sus
curvas PNG/PDF en `reports/figures/directml_regularized/training_comparison.*`.
La matriz y sus porcentajes por fila están bajo esa carpeta de figuras, en
`full_res112_s0_adamw_plateau_directml/`; los conteos y predicciones están bajo
`reports/runs/directml_regularized/full_res112_s0_adamw_plateau_directml/evaluation/`.

### Nuevo experimento pendiente: Batch Normalization

`configs/directml_batchnorm.yaml` selecciona `model.architecture: batchnorm`.
Cada bloque usa Conv 3x3 → BatchNorm → ReLU → MaxPool; conserva los filtros
32/64/128/256, GAP, Dense 256, Dropout 0.5 y los 104 logits. Las convoluciones
anteriores a BatchNorm omiten el sesgo porque la normalización ya incluye un
desplazamiento entrenable.

El perfil conserva resolución 112, batch 64, Adam con LR fijo 0.001, máximo de
150 épocas y parada después de cinco evaluaciones sin mejora del F1. Así se
compara la normalización con la CNN original sin añadir simultáneamente AdamW
ni un scheduler. Se entrena desde cero y guarda sus archivos en subcarpetas
`directml_batchnorm`, con nombre `full_res112_s0_batchnorm_directml`.

```bat
.venv\Scripts\python.exe -m src.train --config configs/directml_batchnorm.yaml --data full --wandb
```

La exportación copia gamma, beta, media y varianza acumuladas desde PyTorch a
Keras. Ambos modelos usan epsilon 0.001; el momentum de Keras es 0.9 y el de
PyTorch 0.1 porque sus convenciones son inversas. PyTorch y Keras actualizan la
varianza de entrenamiento con convenciones distintas; la exportación usa
directamente la varianza acumulada por PyTorch para la inferencia en Keras.
Los buffers también se conservan en el checkpoint `.pt`.

Esta variante se implementó sin ejecutar pruebas, evaluaciones ni entrenamiento,
por petición del usuario. Su compatibilidad de ejecución, exactitud y F1 quedan
pendientes de la corrida del equipo; no hay una mejora medida que reportar.

## 2. Datos

```bash
bash scripts/download_data.sh                                # → data/raw/tpu-getting-started/
python -m src.data --config configs/local.yaml --build-cache # → data/processed/train_cache_res112.npz
```

Para usar la API de Kaggle necesitas `~/.kaggle/kaggle.json` (Kaggle → Settings → Create New Token) y haber aceptado las reglas de la competencia.

La caché guarda todo train como `uint8`, ordenado por etiqueta, para que DM y el subconjunto aleatorio puedan muestrear lotes de una sola clase. Ocupa ~480 MB a 112 px y ~1.9 GB a 224 px.

## 3. Destilación de datos (Distribution Matching)

```bash
python -m src.distill --config configs/local.yaml --ipc 10 [--iterations 500]
```

Cada iteración recorre las 104 clases en grupos de `classes_per_step`. Por cada grupo:

1. Se muestrea una red con **pesos aleatorios nuevos**: la pila convolucional de la Tabla 3 hasta GAP. Esta red nunca se entrena.
2. Se toman `real_batch` imágenes reales de cada clase del grupo, junto con sus `ipc` imágenes sintéticas.
3. Se aplica la **misma** augmentación diferenciable (DSA: color, escala, rotación, traslación, volteo) a las reales y a las sintéticas de cada clase.
4. La pérdida es `Σ_c ‖media f(reales_c) − media f(sintéticas_c)‖²`. Solo se actualizan los píxeles sintéticos, con Adam y recortados a [0, 1].

Salidas:
- `data/processed/synthetic_ipc{N}_res{R}.npz`: las imágenes sintéticas con sus etiquetas.
- `reports/figures/synthetic_ipc{N}_res{R}.png`: una imagen por clase.
- `reports/runs/distill_ipc{N}_res{R}/loss.csv`: la pérdida y el tiempo.

Las imágenes sintéticas se inicializan con fotos reales de su clase. Por eso la pérdida empieza baja y fluctúa: cada paso usa una red aleatoria distinta. Lo que importa es el F1 al entrenar con ellas (sección 5), no el valor absoluto de la pérdida.

**Hiperparámetros** (`distill:` en el YAML):

| Parámetro | Local | Kaggle | Efecto |
|---|---|---|---|
| `ipc` | 10 | 50 | imágenes por clase (1 / 10 / 50 en los experimentos) |
| `iterations` | 1000 | 5000 | pasadas sobre las 104 clases |
| `real_batch` | 64 | 128 | imágenes reales por clase en cada paso |
| `classes_per_step` | 8 | 8 | clases procesadas juntas (memoria GPU) |
| `lr_img` | 0.01 | 0.01 | cambio aproximado de píxel por paso (Adam) |

### Diferencias respecto al paper de DM

- **Arquitectura:** se usa la red de este proyecto (sin normalización), no la ConvNet con *instance norm* del paper. Con esta red los gradientes por píxel son ~1e-5, así que el SGD del paper (lr 1, momentum 0.5) casi no movía las imágenes. Se usa **Adam por píxel**, con un contador de pasos por clase.
- **Resolución:** en local se destila a 112 px. Como la CNN termina en GAP, la misma arquitectura funciona a 112 y a 224 px. La evaluación siempre se hace a la resolución del config.

## 4. Entrenamiento

```bash
python -m src.train --config configs/local.yaml --data {full|random|distilled} [--ipc N] [--seed S] [--epochs E]
```

| Modo | Datos | Épocas | Validación |
|---|---|---|---|
| `full` | los 12,753 TFRecords de train con augmentación Keras (flip, rotación, zoom, contraste) | `epochs_full` (50) | cada época |
| `random` | `ipc` fotos reales por clase al azar + DSA | `epochs_small` | cada `eval_every_small` épocas |
| `distilled` | conjunto destilado + DSA | `epochs_small` | cada `eval_every_small` épocas |

- La validación es **siempre** el conjunto real de val. El callback `ValidationMonitor` calcula `val_loss`, `val_accuracy` y `val_f1_macro`.
- Se guarda el modelo con mejor F1 macro en `models/{run}.keras`.
- El entrenamiento se detiene tras `patience` evaluaciones sin mejora.
- `train_seconds` excluye el tiempo de evaluación.

Cada corrida escribe `reports/runs/{run}/history.csv` y `summary.json`.

### Gráficas con Weights & Biases (Windows / CMD)

Desde la raíz del repositorio, instala el SDK en el entorno del proyecto e inicia sesión:

```bat
.venv\Scripts\python.exe -m pip install "wandb>=0.19,<1"
.venv\Scripts\wandb.exe login
```

Crea una cuenta siguiendo la [guía oficial de W&B](https://docs.wandb.ai/quickstart).
En **User settings → API keys → New key**, crea y copia una clave, y pégala cuando
`wandb login` la solicite. Esta clave es distinta al token de Kaggle.

Prueba una época y después ejecuta el entrenamiento completo:

```bat
.venv\Scripts\python.exe -m src.train --config configs/local.yaml --data full --epochs 1 --wandb
.venv\Scripts\python.exe -m src.train --config configs/local.yaml --data full --wandb
```

El terminal imprime el enlace a cada corrida del proyecto `flower-cnn`. En su pestaña
**Workspace** aparecen `loss`, `accuracy`, `val_loss`, `val_accuracy` y `val_f1_macro`
contra `epoch`, además del tiempo por época y los hiperparámetros. Para comparar
entrenamientos, selecciona las corridas en el mismo proyecto. Las métricas de val
en conjuntos pequeños se registran solo cuando se evalúa, sin rellenar los huecos.
El resumen incluye el mejor F1, su accuracy y su época.

Usa `--wandb-project NOMBRE` para otro proyecto y `--wandb-entity EQUIPO` para
seleccionar tu equipo de W&B. El mismo `--wandb` funciona con `random` y `distilled`.
Para registrar todas las corridas de `scripts.run_experiments`, cambia
`wandb.enabled` a `true` en el YAML que uses.

Sin conexión o sin iniciar sesión puedes registrar localmente:

```bat
.venv\Scripts\python.exe -m src.train --config configs/local.yaml --data full --epochs 1 --wandb --wandb-mode offline
```

Con `configs/local.yaml`, los registros del SDK se guardan en `reports/wandb/`;
con `configs/directml_fast.yaml`, en `reports/runs/wandb/` (ambos excluidos de Git). Para ver
una corrida offline en la web, inicia sesión y ejecuta
`.venv\Scripts\wandb.exe sync reports\wandb\offline-run-NOMBRE`
usando el nombre de la carpeta que imprimió el SDK. La integración registra métricas
y configuración; el checkpoint sigue guardándose en `models/`.

## 5. Experimentos y evaluación

```bash
python -m scripts.run_experiments --config configs/local.yaml
python -m src.evaluate --config configs/kaggle.yaml --model models/full_res224_s0.keras --make-submission
```

`run_experiments` destila los IPC que falten, entrena cada combinación de la sección `experiments:` del YAML y agrega una fila por corrida a `reports/resultados.csv`.

**Cómo leer los resultados:**

- **DM contra aleatorio con el mismo IPC** es la comparación clave. Si el destilado no supera a fotos reales elegidas al azar, la destilación no aporta.
- **DM contra el conjunto completo** muestra cuánto F1 se sacrifica a cambio de velocidad.
- **Costo total** = `distill_seconds` + `train_seconds`. La destilación se paga una vez; conviene cuando se van a hacer muchos entrenamientos sobre el mismo conjunto, como una búsqueda de hiperparámetros o una comparación de arquitecturas.

`src.evaluate` agrega las 10 clases con peor F1 y los pares más confundidos (se esperan, por ejemplo, daisy/osteospermum e iris/yellow iris). Con `--make-submission` escribe `submission.csv` (`id,label`).

## 6. Problemas conocidos

- **La ReLU de la capa densa desaparece en modo grafo con tensorflow-metal 1.2:** el plugin fusiona `MatMul + BiasAdd + Relu` y el kernel fusionado omite la activación. Esto afecta a `fit`, `predict` y `evaluate`, pero no a la llamada eager, así que se entrenaba una red y se validaba otra. `src/__init__.py` apaga el meta-optimizador de Grappler en macOS (sin costo de velocidad medible), y `tests/test_model.py::test_graph_matches_eager` lo detecta si regresa. **Importa `src` antes de construir cualquier modelo.** Además, con Keras 3.15 el modo grafo también difería en CPU, por eso Keras está fijado a `<3.8` en macOS.
- **Gradientes de `x[i]` en tensorflow-metal 1.2:** el gradiente de StridedSlice se copia a todas las filas. En `src/distill.py` se usa `tf.unstack` para separar las clases, y `tests/test_distill.py::test_class_gradients_do_not_leak_between_classes` lo detecta si vuelve a aparecer. Evita indexar tensores con `x[i]` dentro de un `GradientTape` en Mac.
- **`ModuleNotFoundError: distutils`:** TF 2.18 importa `distutils`, que en Python 3.12 provee `setuptools<70`. macOS marca como ocultos los `.pth` del venv (y vuelve a hacerlo aunque se quite el flag), y Python ≥ 3.12.10 los ignora. `src/__init__.py` importa `setuptools` para activar el shim, así que en scripts y notebooks haz `import src` **antes** de `import tensorflow`.
- **TPU:** `get_strategy()` detecta la TPU, pero el bucle de DM y el callback de validación corren fuera de `strategy.run`. En Kaggle usa GPU.
