import numpy as np
from sklearn.metrics import f1_score

from src.evaluate import error_report
from src.metrics import f1_macro


def test_f1_macro_matches_sklearn_on_imbalanced_labels():
    # Clase 0 muy frecuente, clases 1 y 2 raras: la accuracy es alta pero el F1 macro no.
    y_true = np.array([0] * 18 + [1, 2])
    y_pred = np.array([0] * 20)
    assert f1_macro(y_true, y_pred) == f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert np.mean(y_true == y_pred) == 0.9
    assert f1_macro(y_true, y_pred) < 0.35


def test_error_report_finds_worst_class_and_confused_pair():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2])  # la clase 2 se confunde con la 1
    report = error_report(y_true, y_pred, num_classes=3, top_k=2)
    assert report["worst_classes"][0]["label"] == 2
    assert report["confused_pairs"][0] == {"true": "canterbury bells", "pred": "hard-leaved pocket orchid", "count": 2}
    assert report["accuracy"] == round(7 / 9, 4)
