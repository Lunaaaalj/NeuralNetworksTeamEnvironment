"""Corridas de punta a punta de train.run con datos falsos: verifican que se escriben los artefactos."""

import json

import pandas as pd
import pytest

from src.classes import CLASSES
from src.data import ClassCache, build_class_cache, cache_path
from src.distill import distill, save_grid, save_synthetic, synthetic_path
from src.train import run


@pytest.fixture
def prepared_cfg(tiny_cfg):
    paths, img_size = tiny_cfg["paths"], tiny_cfg["data"]["img_size"]
    cache = ClassCache(build_class_cache(paths["tfrecords"], img_size, cache_path(paths["processed"], img_size)))
    images, labels, _, seconds = distill(cache, ipc=2, iterations=2, real_batch=4, classes_per_step=3)
    save_synthetic(synthetic_path(paths["processed"], 2, img_size), images, labels,
                   {"ipc": 2, "img_size": img_size, "iterations": 2, "seed": 0, "distill_seconds": seconds})
    save_grid(images, labels, paths["figures"] / "grid.png", CLASSES)
    assert (paths["figures"] / "grid.png").exists()
    return tiny_cfg


@pytest.mark.parametrize("data,ipc", [("distilled", 2), ("random", 2), ("full", None)])
def test_run_writes_checkpoint_history_and_summary(prepared_cfg, data, ipc):
    summary = run(prepared_cfg, data, ipc, seed=0)
    run_dir = prepared_cfg["paths"]["runs"] / summary["run"]

    assert (prepared_cfg["paths"]["models"] / f"{summary['run']}.keras").exists()
    history = pd.read_csv(run_dir / "history.csv")
    assert {"loss", "val_loss", "val_f1_macro"} <= set(history.columns)
    assert json.loads((run_dir / "summary.json").read_text())["run"] == summary["run"]
    assert 0.0 <= summary["val_f1_macro"] <= 1.0
    if data == "distilled":
        assert "distill_seconds" in summary
