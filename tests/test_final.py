"""Opciones agregadas para el reporte final: MLP, L2, label smoothing, pesos por clase y F1 de entrenamiento."""

import json

import keras
import numpy as np
import pandas as pd
import pytest

from scripts.run_final import EXPERIMENTS, merge
from src.data import balanced_class_weight, class_counts
from src.model import SparseSmoothedCrossentropy, build_cnn, build_mlp, build_model
from src.train import run


def test_mlp_matches_the_avance4_reference_size():
    # 112*112*3*256 + 256 (oculta) + 256*104 + 104 (salida)
    assert build_mlp(img_size=112).count_params() == 9_660_776


def test_build_model_reads_dropout_and_l2_from_config():
    cfg = {"data": {"num_classes": 104, "img_size": 112}, "model": {"dropout": 0.0, "l2": 1e-3}}
    model = build_model(cfg)
    assert model.get_layer("dropout").rate == 0.0
    assert len(model.losses) == 6  # 4 convoluciones + 2 densas; los sesgos no se penalizan
    assert build_model({"data": cfg["data"]}).count_params() == build_cnn().count_params()


def test_label_smoothing_matches_keras_one_hot_loss():
    logits = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    labels = np.array([0, 1, 2, 3, 1])
    ours = SparseSmoothedCrossentropy(0.1, num_classes=4)(labels, logits)
    expected = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)(
        np.eye(4)[labels], logits)
    np.testing.assert_allclose(float(ours), float(expected), rtol=1e-6)


def test_balanced_weights_give_the_average_class_weight_one():
    weights = balanced_class_weight([10, 30, 60])
    assert weights[0] == pytest.approx(100 / 30) and weights[2] == pytest.approx(100 / 180)
    # El total ponderado equivale al número de imágenes: no cambia la escala de la pérdida.
    assert sum(w * n for w, n in zip(weights.values(), [10, 30, 60])) == pytest.approx(100)


def test_class_counts_reads_labels_without_decoding(tiny_cfg):
    counts = class_counts(tiny_cfg["paths"]["tfrecords"], "train", tiny_cfg["data"]["num_classes"])
    assert counts.sum() > 0 and len(counts) == tiny_cfg["data"]["num_classes"]


def test_run_logs_train_f1_and_honors_ablation_switches(tiny_cfg):
    tiny_cfg["model"] = {"dropout": 0.0}
    tiny_cfg["data"]["augment"] = False
    tiny_cfg["train"].update(train_eval_images=6, class_weight="balanced", label_smoothing=0.1)
    summary = run(tiny_cfg, "full", seed=0)
    history = pd.read_csv(tiny_cfg["paths"]["runs"] / summary["run"] / "history.csv")
    assert {"train_f1_macro", "train_eval_accuracy", "train_eval_loss"} <= set(history.columns)
    assert summary["augment"] is False and summary["dropout"] == 0.0
    assert summary["label_smoothing"] == 0.1 and summary["class_weight"] == "balanced"
    # El checkpoint con la pérdida personalizada se puede volver a cargar para evaluar.
    keras.models.load_model(summary["model_path"], compile=False)


def test_each_final_experiment_changes_the_reference():
    labels = [label for label, *_ in EXPERIMENTS]
    assert len(labels) == len(set(labels)) and labels[0] == "referencia"
    for _, _, _, changes in EXPERIMENTS[1:]:
        assert changes, "cada experimento debe diferir de la referencia"
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    json.dumps([changes for *_, changes in EXPERIMENTS])  # serializable a YAML/JSON
