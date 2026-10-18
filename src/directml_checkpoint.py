"""Checkpoint ligero de PyTorch y una sola exportación final a Keras."""

from pathlib import Path

import torch


def save_checkpoint(model, path, *, epoch, metrics, img_size):
    """Pesos portables en CPU; reemplazo atómico del mejor checkpoint completo."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "epoch": epoch,
        "metrics": dict(metrics),
        "img_size": img_size,
        "num_classes": model.logits.out_features,
    }
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_checkpoint(path, model_path):
    """Exporta los pesos del mejor F1, aunque la última época haya empeorado."""
    from src.model import build_cnn
    from src.train_directml import FlowerCNN, copy_to_keras

    model_path = Path(model_path)
    if model_path.suffix != ".keras":
        raise ValueError("El destino de exportación debe terminar en .keras")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FlowerCNN(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    exported = build_cnn(checkpoint["num_classes"])
    copy_to_keras(model, exported)
    temporary = model_path.with_name(model_path.stem + ".tmp.keras")
    try:
        exported.save(temporary)
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exporta el mejor checkpoint DirectML a Keras.")
    parser.add_argument("checkpoint", type=Path, help="archivo .pt guardado durante el entrenamiento")
    parser.add_argument("--output", type=Path, help="destino .keras; por defecto junto al .pt")
    args = parser.parse_args()
    output = args.output or args.checkpoint.with_suffix(".keras")
    export_checkpoint(args.checkpoint, output)
    print(output)
