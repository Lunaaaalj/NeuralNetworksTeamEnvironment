"""The fully connected reference exports with the same pixel ordering and logits."""

import keras
import numpy as np
import torch

from src.train_mlp_reference import FlowerMLP, build_keras_mlp, save_portable_checkpoint


def test_mlp_checkpoint_export_preserves_spatial_order_and_logits(tmp_path):
    torch.manual_seed(11)
    model = FlowerMLP(img_size=8, num_classes=3).eval()
    exported = build_keras_mlp(img_size=8, num_classes=3)
    for name in ("dense", "logits"):
        layer = getattr(model, name)
        exported.get_layer(name).set_weights([
            layer.weight.detach().numpy().T, layer.bias.detach().numpy(),
        ])
    images = np.random.default_rng(11).random((4, 8, 8, 3), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(images).permute(0, 3, 1, 2)).numpy()
    path = tmp_path / "mlp.keras"
    exported.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_allclose(restored(images, training=False).numpy(), expected, atol=1e-5, rtol=1e-5)
    assert exported.count_params() == sum(p.numel() for p in model.parameters())


def test_checkpoint_survives_a_transient_windows_replacement_lock(tmp_path, monkeypatch):
    import os
    target = tmp_path / "best.pt"
    torch.save({"epoch": 1}, target)
    replace = os.replace
    calls = []

    def locked_once(source, destination):
        calls.append(destination)
        if len(calls) == 1:
            assert torch.load(target, weights_only=True)["epoch"] == 1
            raise PermissionError("simulated temporary reader lock")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", locked_once)
    save_portable_checkpoint({"epoch": 2}, target, retries=3, delay=0)
    assert len(calls) == 2
    assert torch.load(target, weights_only=True)["epoch"] == 2
    assert not list(tmp_path.glob("*.tmp"))
