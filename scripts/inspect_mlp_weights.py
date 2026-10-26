"""Inspect saved Keras MLP weights without TensorFlow, deserialization or training.

Requires numpy and h5py. Outputs only diagnostics under correcciones_etapa2/.
"""

import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/entregable_4/avance4_2026-10-21"
OUTPUT = ROOT / "reports/entregable_4/correcciones_etapa2"


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
        layers = config["config"]["layers"]
        kinds = [layer["class_name"] for layer in layers]
        assert kinds == ["InputLayer", "Flatten", "Dense", "Dropout", "Dense"], kinds
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
            arrays = [weights[f"layers/{layer}/vars/{index}"][()]
                      for layer in ("dense", "dense_1") for index in (0, 1)]
    kernel, bias, head, head_bias = [array.astype(np.float64) for array in arrays]
    assert kernel.shape == (37632, 256) and head.shape == (256, 104)
    assert all(np.isfinite(array).all() for array in arrays)
    # Exact interval bounds for x in [0,1]^37632, not a data-driven activation test.
    lower = np.minimum(kernel, 0).sum(axis=0) + bias
    upper = np.maximum(kernel, 0).sum(axis=0) + bias
    sums = kernel.sum(axis=0)
    return {
        "source": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "layer_types": kinds,
        "shapes": [list(array.shape) for array in arrays],
        "parameters": sum(array.size for array in arrays),
        "first_layer_parameters": kernel.size + bias.size,
        "dropout": next(layer["config"]["rate"] for layer in layers
                        if layer["class_name"] == "Dropout"),
        "negative_kernel_fraction": float((kernel < 0).mean()),
        "negative_incoming_sum_with_bias_units": int(((sums + bias) < 0).sum()),
        "kernel_column_sum_min_median_max": np.quantile(sums, [0, .5, 1]).tolist(),
        "guaranteed_inactive_units_on_unit_cube": int((upper <= 0).sum()),
        "guaranteed_active_units_on_unit_cube": int((lower > 0).sum()),
        "finite_weights": True,
        "limitation": "No real images, activations, gradients or logits were evaluated. "
                      "Negative column sums do not prove inactive ReLUs on real images.",
    }


def main():
    profiles = {
        "A4_MLP_lr1e4": BASE / "models/mlp_referencia/full_res112_s0_mlp_reference_directml.keras",
        "A4_MLP_lr1e3_failed": BASE / "diagnosticos/mlp_lr1e3/models/full_res112_s0_mlp_reference_directml.keras",
    }
    result = {"method": "Read-only Keras config and HDF5 arrays; no pickle/model loading",
              "python": sys.version.split()[0], "numpy": np.__version__, "h5py": h5py.__version__,
              "models": {key: inspect(path) for key, path in profiles.items()}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "pesos_mlp.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved read-only weight diagnostics:", OUTPUT / "pesos_mlp.json")


if __name__ == "__main__":
    main()
