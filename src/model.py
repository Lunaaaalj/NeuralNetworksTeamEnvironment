"""CNN propuesta en la Tabla 3 del reporte y el extractor de características usado por DM."""

import keras
from keras import layers

FILTERS = (32, 64, 128, 256)
EXTRA_CONV_STAGES = {"baseline": (), "deep": (3, 4), "batchnorm": ()}
BATCHNORM_EPSILON = 1e-3
BATCHNORM_MOMENTUM = 0.9  # Keras: proporción conservada de la media/varianza anterior.


def model_architecture(cfg):
    """La CNN original sigue siendo la opción por defecto de configuraciones antiguas."""
    architecture = cfg.get("model", {}).get("architecture", "baseline")
    if architecture not in EXTRA_CONV_STAGES:
        raise ValueError(f"model.architecture debe ser uno de {tuple(EXTRA_CONV_STAGES)}")
    return architecture


def _conv_stack(x, architecture="baseline"):
    """Cuatro poolings, con convoluciones adicionales o BatchNorm según la variante."""
    if architecture not in EXTRA_CONV_STAGES:
        raise ValueError(f"Arquitectura desconocida: {architecture}")
    for i, f in enumerate(FILTERS, start=1):
        normalized = architecture == "batchnorm"
        x = layers.Conv2D(f, 3, padding="same", activation=None if normalized else "relu",
                          use_bias=not normalized, name=f"conv{i}")(x)
        if normalized:
            x = layers.BatchNormalization(axis=-1, epsilon=BATCHNORM_EPSILON,
                                           momentum=BATCHNORM_MOMENTUM, name=f"bn{i}")(x)
            x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        if i in EXTRA_CONV_STAGES[architecture]:
            x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}_extra")(x)
    return layers.GlobalAveragePooling2D(name="gap")(x)


def build_cnn(num_classes=104, input_shape=(None, None, 3), dropout=0.5, *, architecture="baseline"):
    """CNN de la Tabla 3 o variantes deep/batchnorm. Devuelve logits.

    La entrada es de tamaño espacial libre: gracias a GAP la misma red sirve a 224x224
    o a 112x112 (resolución usada para destilar en local).
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = _conv_stack(inputs, architecture)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    name = "flower_cnn" if architecture == "baseline" else f"flower_cnn_{architecture}"
    return keras.Model(inputs, logits, name=name)


def build_embedder(input_shape=(None, None, 3)):
    """Solo la pila convolucional hasta GAP: vector de 256 características por imagen."""
    inputs = keras.Input(shape=input_shape, name="image")
    return keras.Model(inputs, _conv_stack(inputs), name="embedder")


def compile_cnn(model, lr=1e-3, *, optimizer="adam", weight_decay=0.0):
    from src.training_policy import optimizer_options

    name, decay = optimizer_options({"optimizer": optimizer, "weight_decay": weight_decay})
    if name == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=decay)
        optimizer.exclude_from_weight_decay(var_names=["bias"])
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
