"""Comprobación independiente de orientación, aciertos y clases ausentes."""

import numpy as np

from src.confusion import confusion_tables


def test_confusion_keeps_correct_predictions_and_absent_classes():
    counts, normalized = confusion_tables([0, 0, 1, 2], [0, 1, 1, 0], 4)
    np.testing.assert_array_equal(counts, [[1, 1, 0, 0], [0, 1, 0, 0],
                                           [1, 0, 0, 0], [0, 0, 0, 0]])
    assert counts.sum() == 4
    assert counts.trace() / counts.sum() == 0.5
    np.testing.assert_allclose(normalized.sum(axis=1), [1, 1, 1, 0])
    np.testing.assert_allclose(normalized[0], [0.5, 0.5, 0, 0])
    assert np.isfinite(normalized).all()
