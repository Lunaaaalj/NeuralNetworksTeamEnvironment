# Clasificación de 104 tipos de flores con CNN + destilación de datos

Proyecto del curso de Redes Neuronales y Aprendizaje Profundo (TC2035.201) basado en la competencia
[Petals to the Metal](https://www.kaggle.com/competitions/tpu-getting-started) de Kaggle. El planteamiento
completo está en [references/main-39.pdf](references/main-39.pdf).

- **Datos:** 23,847 imágenes RGB de 104 clases en TFRecords (train 12,753 · val 3,712 · test 7,382), con desbalance entre clases.
- **Modelo:** la CNN de la Tabla 3 del reporte: 4 bloques Conv 3×3 (32→64→128→256) + MaxPool, GAP, Dense 256, Dropout 0.5 y 104 logits. Se entrena con Adam (lr 1e-3, batch 32) y entropía cruzada dispersa.
- **Métrica:** F1 macro, la misma que usa Kaggle.
- **Preprocesamiento con destilación de datos:** *Distribution Matching* (DM; Zhao & Bilen, WACV 2023) condensa el conjunto de entrenamiento en unas pocas imágenes sintéticas por clase (IPC) para entrenar más rápido y con menos cómputo.

## Inicio rápido

```bash
# Python 3.12 (TensorFlow aún no soporta 3.14)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
pytest -q                                                   # pruebas con datos sintéticos

bash scripts/download_data.sh                               # requiere token de Kaggle
python -m src.data    --config configs/local.yaml --build-cache
python -m src.distill --config configs/local.yaml --ipc 10
python -m src.train   --config configs/local.yaml --data distilled --ipc 10
python -m src.train   --config configs/local.yaml --data random    --ipc 10   # control
python -m src.evaluate --config configs/local.yaml --model models/distilled_ipc10_res112_s0.keras
python -m scripts.run_experiments --config configs/local.yaml               # barrido completo
```

En [docs/USAGE.md](docs/USAGE.md) está la guía completa: local vs. Kaggle, cómo funciona DM, el diseño de experimentos y los problemas conocidos.

Para ver gráficas de entrenamiento en **Weights & Biases**, ejecuta `wandb login` en
el entorno del proyecto y agrega `--wandb` al comando de `src.train`. El proyecto
predeterminado es `flower-cnn`; los pasos para Windows/CMD están en la guía de uso.

En Windows con GPU AMD/Intel, instala `requirements-directml.txt` y agrega
`--backend directml` al comando de entrenamiento. Usa la misma arquitectura y
exporta checkpoints `.keras`; los pasos están en `docs/USAGE.md`.
El perfil `configs/directml_fast.yaml` usa el batch medido de 64 y las optimizaciones
de Adam, métricas y entrada DLPack para la Radeon RX 6700S. Guarda el mejor modelo
en PyTorch durante el entrenamiento y lo exporta una vez a `.keras` al terminar.

## Estructura

```
├── configs/            # local.yaml (Mac, 112 px) y kaggle.yaml (GPU, 224 px)
├── data/               # raw/ (TFRecords) y processed/ (cachés y conjuntos destilados); no se versionan
├── docs/USAGE.md       # guía de uso y decisiones de diseño
├── models/             # checkpoints .keras (no se versionan)
├── notebooks/          # 01_eda.ipynb y kaggle_runner.ipynb
├── references/         # reporte del proyecto
├── reports/            # figures/ (imágenes destiladas), runs/ (logs) y resultados.csv
├── scripts/            # download_data.sh y run_experiments.py
├── src/
│   ├── data.py         # TFRecord → tf.data, x/255, augmentación, caché por clase
│   ├── model.py        # CNN de la Tabla 3 y extractor para DM
│   ├── augment.py      # augmentación diferenciable siamesa (DSA)
│   ├── distill.py      # Distribution Matching
│   ├── train.py        # entrenamiento: full | random | distilled
│   ├── evaluate.py     # F1 macro, errores por clase, submission.csv
│   └── metrics.py      # F1 macro y callback de validación
└── tests/              # pytest
```

## Contribuir

1. Trabaja en una rama y abre un PR hacia `main`.
2. Agrega pruebas en `tests/` para la funcionalidad nueva y corre `pytest -q` antes de subir cambios.
3. No subas datos ni modelos: el `.gitignore` ya excluye `data/raw`, `data/processed` y `models/`.
