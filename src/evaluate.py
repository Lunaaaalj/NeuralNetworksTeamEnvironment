"""Evalúa un modelo entrenado sobre val y, opcionalmente, genera submission.csv para Kaggle.

Uso:
    python -m src.evaluate --config configs/local.yaml --model models/distilled_ipc10_res112_s0.keras
    python -m src.evaluate --config configs/kaggle.yaml --model models/full_res224_s0.keras --make-submission
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score

from src.classes import CLASSES
from src.config import load_config
from src.data import build_dataset
from src.metrics import f1_macro, predict_labels


def error_report(y_true, y_pred, num_classes=104, top_k=10):
    """F1 macro, accuracy, las clases con peor F1 y los pares (verdadera -> predicha) más confundidos."""
    labels = np.arange(num_classes)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    present = np.isin(labels, y_true)
    worst = [int(c) for c in np.argsort(np.where(present, per_class, np.inf))[:top_k]]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:top_k]
    pairs = [np.unravel_index(i, cm.shape) for i in flat]

    return {
        "f1_macro": round(f1_macro(y_true, y_pred), 4),
        "accuracy": round(float(np.mean(y_true == y_pred)), 4),
        "worst_classes": [{"label": c, "name": CLASSES[c], "f1": round(float(per_class[c]), 4)} for c in worst],
        "confused_pairs": [
            {"true": CLASSES[t], "pred": CLASSES[p], "count": int(cm[t, p])}
            for t, p in pairs if cm[t, p] > 0
        ],
    }


def make_submission(model, tfrecord_dir, img_size, out_path):
    """Predice test en orden y escribe id,label."""
    ds = build_dataset(tfrecord_dir, "test", img_size, 128)
    ids, preds = [], []
    for x, batch_ids in ds:
        preds.append(tf.argmax(model(x, training=False), axis=-1).numpy())
        ids.extend(i.decode() for i in batch_ids.numpy())
    pd.DataFrame({"id": ids, "label": np.concatenate(preds)}).to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--submission-path", default="submission.csv")
    parser.add_argument("--confusion-matrix", action="store_true", help="exporta matriz completa a CSV, PNG y PDF")
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_size = cfg["data"]["img_size"]
    model = keras.models.load_model(args.model)

    val_ds = build_dataset(cfg["paths"]["tfrecords"], "val", img_size, 128)
    y_true, y_pred, val_loss = predict_labels(model, val_ds)
    report = error_report(y_true, y_pred, cfg["data"]["num_classes"])

    out = Path(cfg["paths"]["runs"]) / f"{Path(args.model).stem}" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.confusion_matrix:
        from src.confusion import save_confusion_artifacts

        details = save_confusion_artifacts(
            y_true, y_pred, CLASSES[:cfg["data"]["num_classes"]], out.parent / "evaluation",
            Path(cfg["paths"]["figures"]) / Path(args.model).stem,
            title=f"{model.name} — {Path(args.model).stem}")
        report.update(details)
        report.update(val_loss=float(val_loss), model_path=str(Path(args.model).resolve()),
                      split="val", img_size=img_size)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if args.make_submission:
        path = make_submission(model, cfg["paths"]["tfrecords"], img_size, args.submission_path)
        print(f"Submission escrito en {path}")


if __name__ == "__main__":
    main()
