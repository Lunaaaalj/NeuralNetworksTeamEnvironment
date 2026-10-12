"""Lectura de TFRecords de Petals to the Metal, normalización, augmentación y caché por clase.

Uso:
    python -m src.data --config configs/local.yaml --build-cache
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

from src.augment import dsa

AUTOTUNE = tf.data.AUTOTUNE
NATIVE_SIZE = 224

LABELED_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "class": tf.io.FixedLenFeature([], tf.int64),
}
UNLABELED_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "id": tf.io.FixedLenFeature([], tf.string),
}


def list_tfrecords(tfrecord_dir, split):
    files = sorted(tf.io.gfile.glob(str(Path(tfrecord_dir) / split / "*.tfrec")))
    if not files:
        raise FileNotFoundError(f"No hay archivos .tfrec en {Path(tfrecord_dir) / split}")
    return files


def decode_image(jpeg, img_size=NATIVE_SIZE):
    """JPEG -> float32 en [0, 1] (x / 255, ec. 3 del reporte), redimensionado si img_size != 224."""
    image = tf.io.decode_jpeg(jpeg, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    if img_size != NATIVE_SIZE:
        image = tf.image.resize(image, [img_size, img_size], antialias=True)
    return tf.ensure_shape(image, [img_size, img_size, 3])


def parse_labeled(example, img_size=NATIVE_SIZE):
    ex = tf.io.parse_single_example(example, LABELED_FEATURES)
    return decode_image(ex["image"], img_size), tf.cast(ex["class"], tf.int32)


def parse_unlabeled(example, img_size=NATIVE_SIZE):
    ex = tf.io.parse_single_example(example, UNLABELED_FEATURES)
    return decode_image(ex["image"], img_size), ex["id"]


def make_augmenter(seed=None):
    """Transformaciones moderadas de la sección 4.2: volteo, rotación, zoom y contraste."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),
            keras.layers.RandomRotation(0.05, seed=seed),
            keras.layers.RandomZoom(0.1, seed=seed),
            keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmenter",
    )


def build_dataset(tfrecord_dir, split, img_size=NATIVE_SIZE, batch_size=32,
                  augment=False, shuffle=False, seed=None):
    """tf.data para 'train' / 'val' (imagen, etiqueta) o 'test' (imagen, id).

    La augmentación solo debe activarse en train; val y test se leen siempre sin modificar
    y en orden determinista.
    """
    labeled = split != "test"
    parse = parse_labeled if labeled else parse_unlabeled
    files = list_tfrecords(tfrecord_dir, split)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE if shuffle else None)
    if not shuffle:
        options = tf.data.Options()
        options.deterministic = True
        ds = ds.with_options(options)
    ds = ds.map(lambda ex: parse(ex, img_size), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048, seed=seed)
    ds = ds.batch(batch_size)
    if augment:
        augmenter = make_augmenter(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def arrays_dataset(images, labels, batch_size=32, use_dsa=True, seed=0):
    """tf.data a partir de arreglos en memoria (subconjunto aleatorio o conjunto destilado).

    Con conjuntos tan pequeños se aplica DSA con una semilla distinta por imagen y por
    época, la misma familia de transformaciones que se usó al destilar.
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(labels), seed=seed)
    if use_dsa:
        seeds = tf.data.Dataset.random(seed=seed).batch(2)
        ds = tf.data.Dataset.zip((ds, seeds)).map(
            lambda xy, s: (tf.clip_by_value(dsa(xy[0][None], tf.cast(s, tf.int32))[0], 0.0, 1.0), xy[1]),
            num_parallel_calls=AUTOTUNE,
        )
    return ds.batch(min(batch_size, len(labels))).prefetch(AUTOTUNE)


# --- Caché por clase (necesaria para DM y para el subconjunto aleatorio) -------------------

def cache_path(processed_dir, img_size):
    return Path(processed_dir) / f"train_cache_res{img_size}.npz"


def build_class_cache(tfrecord_dir, img_size, out_path):
    """Guarda todo train como uint8 ordenado por etiqueta para muestrear lotes de una sola clase."""
    ds = tf.data.TFRecordDataset(list_tfrecords(tfrecord_dir, "train"))
    ds = ds.map(lambda ex: parse_labeled(ex, img_size), num_parallel_calls=AUTOTUNE).batch(256)
    images, labels = [], []
    for x, y in ds:
        images.append(np.round(x.numpy() * 255.0).astype(np.uint8))
        labels.append(y.numpy())
    images, labels = np.concatenate(images), np.concatenate(labels)
    order = np.argsort(labels, kind="stable")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, images=images[order], labels=labels[order])
    return out_path


class ClassCache:
    """Imágenes de entrenamiento (uint8) con índices por clase."""

    def __init__(self, path):
        data = np.load(path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.num_classes = int(self.labels.max()) + 1
        self.index = [np.flatnonzero(self.labels == c) for c in range(self.num_classes)]

    def sample(self, cls, n, rng):
        """n imágenes de la clase `cls` (con reemplazo si la clase tiene menos de n)."""
        idx = self.index[cls]
        return self.images[rng.choice(idx, n, replace=len(idx) < n)]

    def random_subset(self, ipc, rng):
        """Subconjunto real balanceado: ipc imágenes por clase (sin reemplazo cuando se puede)."""
        images, labels = [], []
        for c in range(self.num_classes):
            images.append(self.sample(c, ipc, rng))
            labels.append(np.full(ipc, c, dtype=np.int32))
        return np.concatenate(images).astype(np.float32) / 255.0, np.concatenate(labels)


def main():
    from src.config import load_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--res", type=int, help="resolución (por defecto data.img_size del config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_size = args.res or cfg["data"]["img_size"]
    if args.build_cache:
        out = build_class_cache(cfg["paths"]["tfrecords"], img_size,
                                cache_path(cfg["paths"]["processed"], img_size))
        cache = ClassCache(out)
        counts = [len(i) for i in cache.index]
        print(f"Caché guardada en {out}: {len(cache.labels)} imágenes, "
              f"{cache.num_classes} clases (mín {min(counts)}, máx {max(counts)} por clase)")


if __name__ == "__main__":
    main()
