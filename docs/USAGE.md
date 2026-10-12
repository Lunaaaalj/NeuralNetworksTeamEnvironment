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
