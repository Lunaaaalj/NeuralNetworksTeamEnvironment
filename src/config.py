"""Carga de configuración YAML y utilidades de entorno (semillas, estrategia de distribución)."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    """Lee un YAML de `configs/` y resuelve las rutas relativas contra la raíz del repo."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def set_seed(seed):
    """Fija las semillas de Python, NumPy y TensorFlow."""
    import keras

    keras.utils.set_random_seed(seed)


@lru_cache(maxsize=1)
def get_strategy():
    """TPUStrategy si hay TPU disponible (Kaggle); si no, la estrategia por defecto (GPU/Metal/CPU).

    Se memoriza porque la TPU solo puede inicializarse una vez por proceso.
    """
    import tensorflow as tf

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
    except (ValueError, tf.errors.NotFoundError):
        return tf.distribute.get_strategy()


def run_name(data, ipc, img_size, seed):
    """Nombre único de una corrida, usado para checkpoints y logs."""
    ipc_part = "" if data == "full" else f"_ipc{ipc}"
    return f"{data}{ipc_part}_res{img_size}_s{seed}"
