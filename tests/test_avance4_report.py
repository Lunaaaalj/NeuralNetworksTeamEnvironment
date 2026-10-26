"""Derived report consistency, without training frameworks."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "reports/entregable_4"


class ReportEvidenceTest(unittest.TestCase):
    def test_per_class_metrics_reconcile_with_selected_matrix(self):
        d = R / "correcciones_etapa4"
        classes = pd.read_csv(d / "metricas_seleccionado.csv")
        selected = json.loads((d / "evaluacion_seleccionado.json").read_text(encoding="utf-8"))
        matrix = pd.read_csv(d / "matriz_conteos.csv", index_col=0).to_numpy()
        np.testing.assert_array_equal(classes.support, matrix.sum(1))
        tp = matrix.diagonal()
        precision = np.divide(tp, matrix.sum(0), out=np.zeros(104), where=matrix.sum(0) > 0)
        recall = tp / matrix.sum(1)
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(104), where=precision+recall > 0)
        np.testing.assert_allclose(classes.f1, f1, atol=1e-14)
        np.testing.assert_allclose(classes.precision, precision, atol=1e-14)
        np.testing.assert_allclose(classes.recall, recall, atol=1e-14)
        self.assertEqual(int(matrix.sum()), 3712)
        self.assertEqual(int(matrix.trace()), 2392)
        self.assertAlmostEqual(selected["f1_macro"], f1.mean())

    def test_selection_contains_only_original_a4_family(self):
        results = pd.read_csv(R / "correcciones_etapa4/resultados_a4_verificados.csv")
        self.assertEqual(len(results), 4)
        self.assertTrue(results.experiment_id.str.startswith("A4_").all())
        chosen = results.loc[results.f1_macro.idxmax()]
        self.assertEqual(chosen.variant, "cnn_regularizada")
        self.assertEqual(chosen.best_epoch, 61)
        self.assertEqual(chosen.epochs_run, 73)


if __name__ == "__main__":
    unittest.main()
