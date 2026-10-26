"""Lightweight historical-evidence checks; runnable without conftest/TensorFlow.

python -m unittest tests.test_avance4_evidence -v
These tests do not train models or create synthetic training results.
"""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.analyze_mlp_avance4 import classification_metrics, validate_saved_predictions

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/entregable_4/avance4_2026-10-21/runs/mlp_referencia/full_res112_s0_mlp_reference_directml"


class EvidenceValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(RUN / "evaluation/validation_predictions.csv")
        cls.matrix = pd.read_csv(RUN / "evaluation/confusion_matrix.csv", index_col=0).to_numpy()
        cls.report = json.loads((RUN / "eval_report.json").read_text(encoding="utf-8"))
        cls.summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
        cls.history = pd.read_csv(RUN / "history.csv")

    def test_historical_mlp_matches_individual_predictions_and_checkpoint(self):
        metrics = validate_saved_predictions(self.predictions, self.matrix, self.report, self.summary, self.history)
        self.assertAlmostEqual(metrics["accuracy"], 576 / 3712)
        self.assertAlmostEqual(metrics["f1_macro"], .028012612757152448)
        self.assertEqual(metrics["predicted_classes"], 21)
        self.assertEqual(metrics["zero_f1_classes"], 87)

    def test_macro_includes_classes_that_are_never_predicted(self):
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), 3)
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["f1_macro"], (2 / 3) / 3)
        self.assertEqual(result["zero_f1_classes"], 2)

    def test_altered_predictions_fail_against_saved_matrix(self):
        altered = self.predictions.copy()
        altered.loc[0, "predicted_label"] = (int(altered.loc[0, "predicted_label"]) + 1) % 104
        with self.assertRaisesRegex(ValueError, "matrix"):
            validate_saved_predictions(altered, self.matrix, self.report, self.summary, self.history)

    def test_metrics_from_a_different_experiment_are_rejected(self):
        report = copy.deepcopy(self.report)
        report["f1_macro"] = .5745
        with self.assertRaisesRegex(ValueError, "f1_macro"):
            validate_saved_predictions(self.predictions, self.matrix, report, self.summary, self.history)

    def test_last_epoch_cannot_be_substituted_for_selected_checkpoint(self):
        summary = copy.deepcopy(self.summary)
        summary["best_epoch"] = summary["epochs_run"]
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            validate_saved_predictions(self.predictions, self.matrix, self.report, summary, self.history)

    def test_invalid_labels_do_not_silently_disappear_from_macro(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 1, 104], [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
