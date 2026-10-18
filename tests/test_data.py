import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import tensorflow as tf

from src.data import ClassCache, arrays_dataset, build_class_cache, build_dataset, dataset_cache_prefix


def test_train_batch_shape_dtype_and_range(tfrecord_dir):
    x, y = next(iter(build_dataset(tfrecord_dir, "train", 224, batch_size=4)))
    assert x.shape == (4, 224, 224, 3)
    assert x.dtype == tf.float32
    assert 0.0 <= float(tf.reduce_min(x)) and float(tf.reduce_max(x)) <= 1.0
    assert y.dtype == tf.int32 and set(y.numpy()) <= {0, 1, 2}


def test_resize_to_working_resolution(tfrecord_dir):
    x, _ = next(iter(build_dataset(tfrecord_dir, "val", 112, batch_size=2)))
    assert x.shape == (2, 112, 112, 3)


def test_val_is_deterministic_and_not_augmented(tfrecord_dir):
    ds = build_dataset(tfrecord_dir, "val", 32, batch_size=12)
    (x1, y1), (x2, y2) = next(iter(ds)), next(iter(ds))
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(x1, x2)


def test_train_augmentation_changes_images(tfrecord_dir):
    plain = next(iter(build_dataset(tfrecord_dir, "train", 32, batch_size=24)))[0]
    aug = next(iter(build_dataset(tfrecord_dir, "train", 32, batch_size=24, augment=True, seed=1)))[0]
    assert not np.allclose(plain, aug)
    assert 0.0 <= float(tf.reduce_min(aug)) and float(tf.reduce_max(aug)) <= 1.0


def test_test_split_returns_ids(tfrecord_dir):
    x, ids = next(iter(build_dataset(tfrecord_dir, "test", 32, batch_size=6)))
    assert x.shape[0] == 6
    assert ids.numpy()[0] == b"id000"


def test_class_cache_groups_by_label(tfrecord_dir, tmp_path):
    cache = ClassCache(build_class_cache(tfrecord_dir, 32, tmp_path / "cache.npz"))
    assert cache.images.dtype == np.uint8 and cache.images.shape == (24, 32, 32, 3)
    assert np.all(np.diff(cache.labels) >= 0)  # ordenado por etiqueta
    assert [len(i) for i in cache.index] == [8, 8, 8]

    rng = np.random.default_rng(0)
    images, labels = cache.random_subset(ipc=2, rng=rng)
    assert images.shape == (6, 32, 32, 3) and images.dtype == np.float32
    np.testing.assert_array_equal(np.bincount(labels), [2, 2, 2])
    # La clase 0 es roja en los datos falsos: el canal R domina.
    red = images[labels == 0].mean(axis=(0, 1, 2))
    assert red[0] > red[1] and red[0] > red[2]


def test_arrays_dataset_with_dsa(tfrecord_dir):
    images = np.random.default_rng(0).uniform(size=(6, 32, 32, 3)).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    x, y = next(iter(arrays_dataset(images, labels, batch_size=32)))
    assert x.shape == (6, 32, 32, 3)  # el batch se recorta al tamaño del conjunto
    assert 0.0 <= float(tf.reduce_min(x)) and float(tf.reduce_max(x)) <= 1.0
    assert sorted(y.numpy().tolist()) == labels.tolist()


def test_disk_cache_reuses_decoded_images_across_datasets(tfrecord_dir, tmp_path, monkeypatch):
    import src.data as data_module

    original = data_module.parse_labeled
    calls = tf.Variable(0)

    def counted_parse(example, img_size):
        with tf.control_dependencies([calls.assign_add(1)]):
            images, labels = original(example, img_size)
            return tf.identity(images), tf.identity(labels)

    monkeypatch.setattr(data_module, "parse_labeled", counted_parse)
    kwargs = dict(img_size=32, batch_size=4, cache="disk", cache_dir=tmp_path / "cache")
    first = list(build_dataset(tfrecord_dir, "train", **kwargs).as_numpy_iterator())
    assert int(calls.numpy()) == 24
    assert len(list((tmp_path / "cache").glob("*.index"))) == 1
    second = list(build_dataset(tfrecord_dir, "train", **kwargs).as_numpy_iterator())
    assert int(calls.numpy()) == 24  # segundo objeto lee disco sin volver a decodificar
    for (left, labels_left), (right, labels_right) in zip(first, second):
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(labels_left, labels_right)


def test_disk_cache_fingerprint_changes_with_source_and_resolution(tfrecord_dir, tmp_path):
    source = next((tfrecord_dir / "train").glob("*.tfrec"))
    copied = Path(shutil.copy2(source, tmp_path / source.name))
    initial = dataset_cache_prefix([copied], "train", 32, tmp_path)
    assert initial != dataset_cache_prefix([copied], "train", 64, tmp_path)
    assert initial != dataset_cache_prefix([copied], "val", 32, tmp_path)
    stat = copied.stat()
    os.utime(copied, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert initial != dataset_cache_prefix([copied], "train", 32, tmp_path)


def test_unknown_cache_mode_does_not_silently_allocate_ram(tfrecord_dir):
    with pytest.raises(ValueError, match="data.cache"):
        build_dataset(tfrecord_dir, "train", cache="typo")
