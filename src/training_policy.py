"""Opciones de optimización compartidas por TensorFlow y DirectML."""

import math

import keras


def optimizer_options(train):
    name = train.get("optimizer", "adam").lower()
    decay = float(train.get("weight_decay", 0.0))
    if name not in ("adam", "adamw"):
        raise ValueError("train.optimizer debe ser adam o adamw")
    if not math.isfinite(decay) or decay < 0 or (name == "adam" and decay != 0):
        raise ValueError("weight_decay debe ser >= 0 y requiere optimizer: adamw")
    return name, decay


def plateau_options(train):
    options = train.get("lr_schedule")
    if not options:
        return None
    if options.get("type", "plateau") != "plateau":
        raise ValueError("train.lr_schedule.type debe ser plateau")
    factor = float(options.get("factor", 0.5))
    patience = options.get("patience", 2)
    minimum = float(options.get("min_lr", 1e-5))
    if not 0 < factor < 1 or not isinstance(patience, int) or patience < 0:
        raise ValueError("El scheduler requiere factor en (0, 1) y patience entero >= 0")
    if not math.isfinite(minimum) or not 0 <= minimum <= train["lr"]:
        raise ValueError("min_lr debe estar entre 0 y train.lr")
    return {"factor": factor, "patience": patience, "min_lr": minimum}


def training_suffix(train):
    name, _ = optimizer_options(train)
    scheduled = plateau_options(train) is not None
    return "" if name == "adam" and not scheduled else f"_{name}" + ("_plateau" if scheduled else "")


def torch_scheduler(optimizer, train):
    options = plateau_options(train)
    if options is None:
        return None
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    return ReduceLROnPlateau(optimizer, mode="min", threshold=1e-4,
                             threshold_mode="abs", **options)


class ValidationPlateau(keras.callbacks.ReduceLROnPlateau):
    """Las épocas sin validación no consumen paciencia en conjuntos pequeños."""

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and math.isfinite(float(value)):
            super().on_epoch_end(epoch, logs)


def keras_scheduler(train):
    options = plateau_options(train)
    if options is None:
        return None
    # PyTorch reduce después de patience+1 evaluaciones malas. Keras usa >=;
    # ajustar este contador conserva el mismo criterio en ambos backends.
    options["patience"] += 1
    return ValidationPlateau(monitor="val_loss", mode="min", min_delta=1e-4, **options)
