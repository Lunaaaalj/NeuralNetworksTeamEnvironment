"""No TensorFlow required: python -m unittest tests.test_avance4_regularization -v."""
import unittest
from reports.entregable_4.correcciones_etapa3.analizar_regularizacion import load_evidence, snapshot


class RegularizationEvidenceTest(unittest.TestCase):
    def test_selected_and_common_epochs_have_different_rankings(self):
        runs, table, common, _ = load_evidence()
        self.assertEqual(common, list(range(1, 47)))
        best = table[table.comparison == "selected_checkpoint"].set_index("experiment_id")
        epoch34 = table[table.comparison == "common_epoch_34"].set_index("experiment_id")
        self.assertAlmostEqual(best.loc["F/referencia", "val_f1_macro"], .578366870583084)
        self.assertGreater(best.loc["F/referencia", "val_f1_macro"], best.loc["F/sin_regularizacion", "val_f1_macro"])
        self.assertLess(epoch34.loc["F/referencia", "val_f1_macro"], epoch34.loc["F/sin_regularizacion", "val_f1_macro"])
        self.assertLess(epoch34.loc["F/referencia", "gap_f1"], epoch34.loc["F/sin_regularizacion", "gap_f1"])
        self.assertLess(epoch34.loc["F/referencia", "val_loss"], epoch34.loc["F/sin_regularizacion", "val_loss"])

    def test_missing_epoch_is_not_interpolated(self):
        runs, _, _, _ = load_evidence()
        h = runs["referencia"]["history"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            snapshot(h[h.epoch != 33], 34)

    def test_incomplete_clean_train_metrics_are_rejected(self):
        runs, _, _, _ = load_evidence()
        h = runs["referencia"]["history"].copy()
        h.loc[h.epoch == 33, "train_f1_macro"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            snapshot(h, 34)


if __name__ == "__main__":
    unittest.main()
