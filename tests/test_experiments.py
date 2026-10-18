"""Los metadatos nuevos de DirectML no deben corromper los CSV ya existentes."""

import sys

import pandas as pd

from scripts import run_experiments


def test_append_new_checkpoint_metadata_preserves_existing_csv(tiny_cfg, monkeypatch):
    path = tiny_cfg["paths"]["results"]
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = {"data": "full", "ipc": None, "seed": 0, "val_f1_macro": 0.2,
                "val_accuracy": 0.3, "train_seconds": 10.0}
    pd.DataFrame([previous]).to_csv(path, index=False)
    current = {**previous, "seed": 1, "checkpoint_path": "best.pt", "export_seconds": 0.1,
               "checkpoint_seconds": 0.01, "input_pipeline": "dlpack_prefetch"}
    tiny_cfg["experiments"] = {"include_full": True, "seeds": [1], "data": [], "ipcs": []}
    monkeypatch.setattr(run_experiments, "load_config", lambda _: tiny_cfg)
    monkeypatch.setattr(run_experiments, "build_dataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_experiments, "run", lambda *args, **kwargs: current)
    monkeypatch.setattr(sys, "argv", ["run_experiments"])
    run_experiments.main()
    actual = pd.read_csv(path)
    assert list(actual.columns) == list(previous)
    assert actual["seed"].tolist() == [0, 1]
    assert actual["val_f1_macro"].tolist() == [0.2, 0.2]
