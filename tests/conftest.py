"""Fixtures: TFRecords falsos con el mismo esquema que Petals to the Metal y un config diminuto."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import src  # noqa: F401  (antes que tensorflow: activa el shim de distutils, ver src/__init__.py)
import numpy as np
import pytest
import tensorflow as tf

NUM_CLASSES = 3
COLORS = np.array([[220, 40, 40], [40, 200, 40], [40, 40, 220]], dtype=np.float32)


def _bytes(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def fake_image(label, rng, size=224):
    """Imagen con un color dominante por clase + ruido, para que las clases sean separables."""
    img = COLORS[label] + rng.normal(0, 25, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def write_split(directory, split, n, rng, size=224):
    path = directory / split
    path.mkdir(parents=True)
    with tf.io.TFRecordWriter(str(path / f"00-{size}x{size}-{n}.tfrec")) as w:
        for i in range(n):
            label = i % NUM_CLASSES
            feature = {"image": _bytes(tf.io.encode_jpeg(fake_image(label, rng, size)).numpy())}
            if split == "test":
                feature["id"] = _bytes(f"id{i:03d}".encode())
            else:
                feature["class"] = _int(label)
            w.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())


@pytest.fixture(scope="session")
def tfrecord_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("tfrecords")
    rng = np.random.default_rng(0)
    write_split(root, "train", 24, rng)
    write_split(root, "val", 12, rng)
    write_split(root, "test", 6, rng)
    return root


@pytest.fixture
def tiny_cfg(tmp_path, tfrecord_dir):
    """Config equivalente a configs/*.yaml pero con 3 clases, 32 px y pocas iteraciones."""
    return {
        "paths": {
            "tfrecords": tfrecord_dir,
            "processed": tmp_path / "processed",
            "models": tmp_path / "models",
            "runs": tmp_path / "runs",
            "figures": tmp_path / "figures",
            "results": tmp_path / "resultados.csv",
        },
        "data": {"img_size": 32, "num_classes": NUM_CLASSES, "batch_size": 4},
        "distill": {"ipc": 2, "iterations": 5, "real_batch": 4, "classes_per_step": 3,
                    "lr_img": 0.01, "log_every": 1},
        "train": {"lr": 1e-3, "epochs_full": 2, "epochs_small": 3, "eval_every_small": 2, "patience": 5},
    }
