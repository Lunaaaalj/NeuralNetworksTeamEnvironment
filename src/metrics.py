"""Métrica F1 macro (la de Kaggle) y callback de validación con checkpoint y parada temprana."""

import math
import time

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


def f1_macro(y_true, y_pred):
    """F1 macro sobre las clases presentes en y_true ∪ y_pred (igual que sklearn / Kaggle)."""
    return float(f1_score(y_true, y_pred, average="macro"))


def predict_labels(model, ds):
    """Etiquetas verdaderas, predichas y pérdida media sobre un dataset (imagen, etiqueta)."""
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum")
    y_true, y_pred, total_loss = [], [], 0.0
    for x, y in ds:
        logits = model(x, training=False)
        total_loss += float(loss_fn(y, logits))
        y_true.append(y.numpy())
        y_pred.append(tf.argmax(logits, axis=-1).numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return y_true, y_pred, total_loss / len(y_true)


class ValidationMonitor(keras.callbacks.Callback):
    """Evalúa en val cada `every` épocas: val_loss, val_accuracy y val_f1_macro.

    - Guarda el modelo con mejor F1 macro en `checkpoint_path`.
    - Detiene el entrenamiento tras `patience` evaluaciones sin mejora.
    - Escribe las métricas en `logs`, así CSVLogger (que debe ir después) las registra.
    - Lleva el tiempo de evaluación aparte para reportar el tiempo de entrenamiento puro.
    """

    def __init__(self, val_ds, checkpoint_path, every=1, patience=None, train_eval_ds=None):
        super().__init__()
        self.val_ds = val_ds
        # Subconjunto fijo de train sin aumentos ni dropout: las métricas de entrenamiento de
        # fit() se miden con aumentos y dropout activos y no son comparables con val.
        self.train_eval_ds = train_eval_ds
        self.checkpoint_path = str(checkpoint_path)
        self.every = every
        self.patience = patience
        self.best_f1 = -1.0
        self.best_accuracy = None
        self.best_epoch = -1
        self.eval_seconds = 0.0
        self._bad_evals = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        last = epoch + 1 == self.params.get("epochs")
        if not (epoch == 0 or (epoch + 1) % self.every == 0 or last):
            logs.update(val_loss=math.nan, val_accuracy=math.nan, val_f1_macro=math.nan)
            if self.train_eval_ds is not None:
                logs.update(train_eval_loss=math.nan, train_eval_accuracy=math.nan, train_f1_macro=math.nan)
            return

        start = time.perf_counter()
        y_true, y_pred, loss = predict_labels(self.model, self.val_ds)
        if self.train_eval_ds is not None:
            t_true, t_pred, t_loss = predict_labels(self.model, self.train_eval_ds)
            logs.update(train_eval_loss=t_loss, train_eval_accuracy=float(np.mean(t_true == t_pred)),
                        train_f1_macro=f1_macro(t_true, t_pred))
        self.eval_seconds += time.perf_counter() - start
        f1 = f1_macro(y_true, y_pred)
        logs.update(val_loss=loss, val_accuracy=float(np.mean(y_true == y_pred)), val_f1_macro=f1)
        print(f" - val_loss {loss:.4f} - val_acc {logs['val_accuracy']:.4f} - val_f1_macro {f1:.4f}")

        if f1 > self.best_f1:
            self.best_f1, self.best_epoch, self._bad_evals = f1, epoch + 1, 0
            self.best_accuracy = logs["val_accuracy"]
            self.model.save(self.checkpoint_path)
        else:
            self._bad_evals += 1
            if self.patience is not None and self._bad_evals >= self.patience:
                print(f"Parada temprana: {self.patience} evaluaciones sin mejorar F1 macro.")
                self.model.stop_training = True
