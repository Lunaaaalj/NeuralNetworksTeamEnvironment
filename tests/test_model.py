import numpy as np
import pytest

from src.model import build_cnn, build_embedder, compile_cnn


@pytest.mark.parametrize("size", [224, 112])
def test_output_is_104_logits_at_any_resolution(size):
    model = build_cnn()
    out = model(np.zeros((2, size, size, 3), np.float32))
    assert out.shape == (2, 104)


def test_parameter_count_matches_table_3():
    # conv: 896 + 18,496 + 73,856 + 295,168 ; dense 256: 65,792 ; salida 104: 26,728
    assert build_cnn().count_params() == 480_936


def test_embedder_outputs_256_features():
    assert build_embedder()(np.zeros((3, 64, 64, 3), np.float32)).shape == (3, 256)


def test_training_reduces_loss_on_fixed_batch():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(16, 32, 32, 3)).astype(np.float32)
    y = rng.integers(0, 104, 16).astype(np.int32)
    model = compile_cnn(build_cnn(), lr=1e-3)
    before = model.evaluate(x, y, verbose=0)[0]
    model.fit(x, y, epochs=30, batch_size=16, verbose=0)
    after = model.evaluate(x, y, verbose=0)[0]
    assert after < before


def test_graph_matches_eager():
    # Regresión: en tensorflow-metal el grafo (fit/predict) omitía la ReLU de la capa densa.
    model = build_cnn()
    x = np.random.default_rng(0).uniform(size=(8, 64, 64, 3)).astype(np.float32)
    np.testing.assert_allclose(model.predict(x, verbose=0), model(x, training=False).numpy(), atol=1e-4)
