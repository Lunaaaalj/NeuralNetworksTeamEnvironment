import numpy as np
import tensorflow as tf

from src.augment import dsa
from src.data import ClassCache
from src.distill import DistributionMatcher, distill, load_synthetic, save_synthetic

from tests.conftest import fake_image


def make_cache(tmp_path, per_class=10, size=32):
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), per_class).astype(np.int32)
    images = np.stack([fake_image(c, rng, size) for c in labels])
    np.savez(tmp_path / "cache.npz", images=images, labels=labels)
    return ClassCache(tmp_path / "cache.npz")


def test_dsa_is_siamese_and_differentiable():
    x = tf.Variable(tf.random.uniform([2, 32, 32, 3], seed=0))
    seed = tf.constant([3, 4])
    np.testing.assert_array_equal(dsa(x, seed), dsa(x, seed))
    assert not np.allclose(dsa(x, seed), dsa(x, tf.constant([5, 6])))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(dsa(x, seed) ** 2)
    grad = tape.gradient(loss, x)
    assert grad is not None and float(tf.reduce_sum(tf.abs(grad))) > 0


def test_dm_from_noise_recovers_class_colors(tmp_path):
    # Datos falsos: clase 0 roja, 1 verde, 2 azul. Partiendo de gris, DM debe recuperar el color.
    cache = make_cache(tmp_path)
    images, labels, history, _ = distill(cache, ipc=2, iterations=60, real_batch=8, classes_per_step=3,
                                         lr_img=0.05, seed=0, log_every=1, init="noise")
    losses = [h["loss"] for h in history]
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])
    for c in range(3):
        assert images[labels == c].mean(axis=(0, 1, 2)).argmax() == c


def test_class_gradients_do_not_leak_between_classes():
    # Regresión: en tensorflow-metal el gradiente de x[i] se copiaba a todas las filas.
    rng = np.random.default_rng(0)
    dm = DistributionMatcher(rng.uniform(size=(3, 2, 16, 16, 3)).astype(np.float32))
    syn = tf.constant(dm.images())
    seeds = [tf.constant([i, 0]) for i in range(3)]
    with tf.GradientTape() as tape:
        tape.watch(syn)
        loss = tf.reduce_sum(dm._mean_features(syn, seeds)[0])  # solo depende de la clase 0
    grad = tape.gradient(loss, syn).numpy()
    assert np.abs(grad[0]).sum() > 0
    assert np.abs(grad[1:]).max() == 0


def test_distilled_output_shape_range_and_labels(tmp_path):
    cache = make_cache(tmp_path)
    images, labels, _, _ = distill(cache, ipc=2, iterations=3, real_batch=4, classes_per_step=2, seed=0)
    assert images.shape == (6, 32, 32, 3)
    assert images.min() >= 0.0 and images.max() <= 1.0
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])


def test_step_only_updates_selected_classes():
    rng = np.random.default_rng(0)
    init = rng.uniform(size=(3, 2, 16, 16, 3)).astype(np.float32)
    dm = DistributionMatcher(init)
    real = rng.integers(0, 256, size=(1, 4, 16, 16, 3), dtype=np.uint8)
    dm.step(real, np.array([1], np.int32), np.array([7, 8], np.int32))
    after = dm.images()
    np.testing.assert_array_equal(after[0], init[0])
    np.testing.assert_array_equal(after[2], init[2])
    assert not np.allclose(after[1], init[1])


def test_synthetic_npz_round_trip(tmp_path):
    images = np.random.default_rng(0).uniform(size=(4, 8, 8, 3)).astype(np.float32)
    labels = np.array([0, 0, 1, 1], np.int32)
    meta = {"ipc": 2, "img_size": 8, "iterations": 1, "seed": 0, "distill_seconds": 0.1}
    save_synthetic(tmp_path / "s.npz", images, labels, meta)
    images2, labels2, meta2 = load_synthetic(tmp_path / "s.npz")
    np.testing.assert_allclose(images2, images, atol=1e-3)  # se guarda en float16
    np.testing.assert_array_equal(labels2, labels)
    assert meta2 == meta
